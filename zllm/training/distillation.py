"""知识蒸馏训练。

核心思想：用教师模型的软标签（soft logits）指导学生模型学习。
- 软标签包含"暗知识"（dark knowledge）— 类间相似度信息
- Temperature > 1 使分布更平滑，暴露更多类间关系

Loss = α * CE(student, hard_labels) + (1-α) * T² * KL(teacher_soft || student_soft)

函数：
- distillation_loss: T² * KL 散度
- train_epoch: 蒸馏训练循环
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from zllm.training.utils import get_lr, Logger


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction="batchmean"):
    """计算蒸馏损失：T² * KL(teacher_soft || student_soft)。

    Args:
        student_logits: (batch, seq_len, vocab) 或 (N, vocab)
        teacher_logits: 同 shape
        temperature: 蒸馏温度（推荐 1.0-2.0）
        reduction: KL 散度 reduction

    Returns:
        scalar loss
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl


@dataclass
class DistillConfig:
    epochs: int = 6
    batch_size: int = 32
    learning_rate: float = 5e-6
    alpha: float = 0.5
    temperature: float = 1.5
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    max_seq_len: int = 340
    dtype: str = "bfloat16"
    num_workers: int = 1
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    save_weight: str = "full_dist"
    from_weight: str = "full_sft"
    from_resume: bool = False
    device: str = "cuda"


def train_epoch(student, teacher, loader, optimizer, scaler, cfg, epoch, device, start_step=0, total_steps=None):
    """蒸馏训练循环。

    Loss = α * CE + (1-α) * distill + aux_loss

    Args:
        student: ZLLMForCausalLM (训练)
        teacher: ZLLMForCausalLM (冻结) 或 None
        loader: DataLoader
        optimizer: AdamW
        scaler: GradScalerManager
        cfg: DistillConfig
        epoch: 当前 epoch
        device: 计算设备

    Returns:
        list[float]: 每个 step 的 loss
    """
    if total_steps is None:
        total_steps = len(loader)
    global_total = cfg.epochs * total_steps

    student.train()
    if teacher is not None:
        teacher.eval()
        teacher.requires_grad_(False)

    losses = []
    use_amp = scaler.enabled and torch.cuda.is_available()

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss_mask = (labels[..., 1:] != -100).float()

        lr = get_lr(epoch * total_steps + step, global_total, cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            res = student(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()

            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids).logits[..., :-1, :].contiguous()
                    vocab_student = student_logits.size(-1)
                    teacher_logits = teacher_logits[..., :vocab_student]

            shift_labels = labels[..., 1:].contiguous()
            loss_mask_flat = loss_mask.view(-1)
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
            ce_loss = ce_loss_raw + res.aux_loss if student.config.use_moe else ce_loss_raw

            if teacher is not None and cfg.alpha < 1.0:
                distill = distillation_loss(
                    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                    temperature=cfg.temperature,
                )
            else:
                distill = torch.tensor(0.0, device=device)

            loss = (cfg.alpha * ce_loss + (1 - cfg.alpha) * distill) / cfg.accumulation_steps

        scaler.scale(loss).backward()

        is_boundary = step % cfg.accumulation_steps == 0
        if is_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item() * cfg.accumulation_steps
        losses.append(loss_val)

        if step % cfg.log_interval == 0 or step == total_steps:
            Logger(
                f"Epoch:[{epoch + 1}/{cfg.epochs}]({step}/{total_steps}), "
                f"loss: {loss_val:.4f}, ce: {ce_loss_raw.item():.4f}, distill: {distill.item():.4f}, lr: {lr:.8f}"
            )

        del input_ids, labels, res, student_logits, ce_loss, distill, loss

    last_step = start_step + len(loader)
    if last_step > start_step and last_step % cfg.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return losses