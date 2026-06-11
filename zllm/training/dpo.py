"""DPO（Direct Preference Optimization）训练。

核心思想：直接用偏好数据训练，无需 reward model 或 RL。
- 双模型架构：policy model（训练）+ reference model（冻结）
- loss = -log_sigmoid(β * (π_θ(chosen)/π_ref(chosen) - π_θ(rejected)/π_ref(rejected)))

函数：
- logits_to_log_probs: 从 logits 提取每个 token 的 log 概率
- dpo_loss: 计算 DPO 偏好损失
- train_epoch: DPO 训练循环
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from zllm.training.utils import get_lr, Logger


def logits_to_log_probs(logits, labels):
    """从 logits 提取每个 token 位置的 log 概率。

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)

    Returns:
        log_probs: (batch, seq_len)
    """
    log_probs = F.log_softmax(logits, dim=2)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.15):
    """计算 DPO 偏好损失。

    输入已按 [chosen, rejected] 拼接，前半为 chosen，后半为 rejected。

    Args:
        ref_log_probs: (batch, seq_len) 参考模型的 log 概率
        policy_log_probs: (batch, seq_len) 策略模型的 log 概率
        mask: (batch, seq_len) 仅 assistant 区域的掩码
        beta: DPO 温度参数

    Returns:
        scalar loss
    """
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    batch_size = ref_log_probs.shape[0]
    chosen_ref = ref_log_probs[: batch_size // 2]
    reject_ref = ref_log_probs[batch_size // 2 :]
    chosen_policy = policy_log_probs[: batch_size // 2]
    reject_policy = policy_log_probs[batch_size // 2 :]

    pi_logratios = chosen_policy - reject_policy
    ref_logratios = chosen_ref - reject_ref
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


@dataclass
class DPOConfig:
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 4e-8
    beta: float = 0.15
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    max_seq_len: int = 1024
    dtype: str = "bfloat16"
    num_workers: int = 1
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    save_weight: str = "dpo"
    from_weight: str = "full_sft"
    from_resume: bool = False
    device: str = "cuda"


def train_epoch(model, ref_model, loader, optimizer, scaler, cfg, epoch, device, start_step=0, total_steps=None):
    """DPO 单 epoch 训练循环。

    每步：
    1. 取 chosen/rejected batch
    2. 拼接后分别过 ref_model（no_grad）和 policy model
    3. logits_to_log_probs → dpo_loss
    4. 反向传播
    """
    if total_steps is None:
        total_steps = len(loader)
    global_total = cfg.epochs * total_steps

    model.train()
    ref_model.eval()
    losses = []
    use_amp = scaler.enabled and torch.cuda.is_available()

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch["x_chosen"].to(device, non_blocking=True)
        x_rejected = batch["x_rejected"].to(device, non_blocking=True)
        y_chosen = batch["y_chosen"].to(device, non_blocking=True)
        y_rejected = batch["y_rejected"].to(device, non_blocking=True)
        mask_chosen = batch["mask_chosen"].to(device, non_blocking=True)
        mask_rejected = batch["mask_rejected"].to(device, non_blocking=True)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * total_steps + step, global_total, cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            with torch.no_grad():
                ref_logits = ref_model(x).logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            outputs = model(x)
            policy_log_probs = logits_to_log_probs(outputs.logits, y)

            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=cfg.beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / cfg.accumulation_steps

        scaler.scale(loss).backward()

        is_boundary = step % cfg.accumulation_steps == 0
        if is_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item() * cfg.accumulation_steps
        losses.append(loss_val)

        if step % cfg.log_interval == 0 or step == total_steps:
            Logger(
                f"Epoch:[{epoch + 1}/{cfg.epochs}]({step}/{total_steps}), "
                f"loss: {loss_val:.4f}, dpo: {dpo_loss_val.item():.4f}, lr: {lr:.8f}"
            )

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected
        del x, y, mask, ref_logits, ref_log_probs, outputs, policy_log_probs, loss

    last_step = start_step + len(loader)
    if last_step > start_step and last_step % cfg.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return losses
