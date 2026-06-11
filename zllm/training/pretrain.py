"""预训练脚本。

train_epoch: 单 epoch 训练循环
PretrainConfig: 训练超参数配置
_format_duration: 时间格式化

支持 AMP、梯度累积、梯度裁剪、DDP、checkpoint 续训。
"""

import math
import time

import torch
from dataclasses import dataclass, field


@dataclass
class PretrainConfig:
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 5e-4
    accumulation_steps: int = 4
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    max_seq_len: int = 340
    dtype: str = "bfloat16"
    num_workers: int = 1
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    save_weight: str = "pretrain"
    from_weight: str = "none"
    from_resume: bool = False
    use_compile: bool = False
    device: str = "cuda"


def _format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes:.0f}m{secs:02.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours:.0f}h{mins:02.0f}m"


def train_epoch(model, loader, optimizer, scaler, cfg, epoch, device, start_step=0, total_steps=None):
    """执行单个 epoch 的训练。

    Args:
        model: ZLLMForCausalLM
        loader: DataLoader
        optimizer: AdamW
        scaler: GradScalerManager
        cfg: PretrainConfig
        epoch: 当前 epoch 编号
        device: 计算设备
        start_step: 起始 step（断点续训）
        total_steps: 总 step 数（None 则用 len(loader)）

    Returns:
        list[float]: 每个 step 的 loss
    """
    from zllm.training.utils import get_lr, Logger

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return losses
