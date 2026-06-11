"""监督微调（SFT）训练脚本。

SFT 与预训练的关键差异：
- learning_rate=1e-5（预训练的 1/50）
- max_seq_len=768（对话数据更长）
- from_weight='pretrain'（加载预训练权重）
- 使用 SFTDataset（只对 assistant 回复计算 loss）
- save_weight='full_sft'

train_epoch 与 pretrain 的循环结构相同：
lr 调度 → AMP 前向 → loss → 梯度累积 → clip → step → 日志
"""

import torch
from dataclasses import dataclass

from zllm.training.pretrain import _format_duration
from zllm.training.utils import get_lr, Logger


@dataclass
class SFTConfig:
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 1e-5
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    max_seq_len: int = 768
    dtype: str = "bfloat16"
    num_workers: int = 1
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    save_weight: str = "full_sft"
    from_weight: str = "pretrain"
    from_resume: bool = False
    use_compile: bool = False
    device: str = "cuda"


def train_epoch(model, loader, optimizer, scaler, cfg, epoch, device, start_step=0, total_steps=None):
    """SFT 单 epoch 训练循环。

    与 pretrain 的 train_epoch 结构相同，但使用 SFTConfig。

    Args:
        model: ZLLMForCausalLM
        loader: DataLoader（SFTDataset）
        optimizer: AdamW
        scaler: GradScalerManager
        cfg: SFTConfig
        epoch: 当前 epoch 编号
        device: 计算设备
        start_step: 起始 step（断点续训）
        total_steps: 总 step 数

    Returns:
        list[float]: 每个 step 的 loss
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
