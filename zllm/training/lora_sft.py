"""LoRA 微调训练脚本。

与 full_sft.py 的关键差异：
- learning_rate=1e-4（比 full_sft 高 10 倍，因为 LoRA 参数少需要更大 lr）
- apply_lora 注入低秩适配器
- freeze_non_lora 冻结基础模型
- 只优化 LoRA 参数
- save_lora 只保存 LoRA 权重
- rank=16 默认秩
"""

import torch
from dataclasses import dataclass

from zllm.training.utils import get_lr, Logger


@dataclass
class LoRAConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    rank: int = 16
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
    save_weight: str = "lora_medical"
    from_weight: str = "full_sft"
    from_resume: bool = False
    device: str = "cuda"


def train_epoch(model, loader, optimizer, scaler, cfg, epoch, device, lora_params, start_step=0, total_steps=None):
    """LoRA 单 epoch 训练循环。

    与 full_sft 的 train_epoch 几乎相同，差异：
    - 梯度裁剪只作用于 lora_params（而非全部参数）
    """
    if total_steps is None:
        total_steps = len(loader)
    global_total = cfg.epochs * total_steps

    model.train()
    losses = []
    use_amp = scaler.enabled and torch.cuda.is_available()

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        lr = get_lr(epoch * total_steps + step, global_total, cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            output = model(input_ids, labels=labels)
            loss = output.loss + output.aux_loss
            loss = loss / cfg.accumulation_steps

        scaler.scale(loss).backward()

        is_boundary = step % cfg.accumulation_steps == 0
        if is_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item() * cfg.accumulation_steps
        losses.append(loss_val)

        if step % cfg.log_interval == 0 or step == total_steps:
            Logger(
                f"Epoch:[{epoch + 1}/{cfg.epochs}]({step}/{total_steps}), "
                f"loss: {loss_val:.4f}, lr: {lr:.8f}"
            )

        del input_ids, labels, output, loss

    last_step = start_step + len(loader)
    if last_step > start_step and last_step % cfg.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return losses
