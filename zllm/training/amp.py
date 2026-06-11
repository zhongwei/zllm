"""AMP 混合精度 + 梯度累积 + 梯度裁剪。

train_step 封装了完整的单步训练逻辑：
autocast → forward → loss/scaler → backward → unscale → clip → step → zero_grad
"""

import torch

from zllm.training.utils import is_main_process


class GradScalerManager:
    """封装 torch.amp.GradScaler，简化 fp16/bf16 选择。"""

    def __init__(self, enabled=True):
        self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        self.enabled = enabled

    def scale(self, loss):
        return self.scaler.scale(loss)

    def unscale_(self, optimizer):
        return self.scaler.unscale_(optimizer)

    def step(self, optimizer):
        return self.scaler.step(optimizer)

    def update(self):
        return self.scaler.update()


def train_step(
    model,
    optimizer,
    input_ids,
    labels,
    scaler,
    accumulation_steps=1,
    max_grad_norm=1.0,
    current_step=0,
    device="cuda",
):
    """执行单步训练。

    Args:
        model: ZLLMForCausalLM
        optimizer: AdamW
        input_ids, labels: batch tensors
        scaler: GradScalerManager
        accumulation_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪阈值
        current_step: 当前 step（用于判断是否该真正 step）
        device: 计算设备

    Returns:
        loss_value (float)
    """
    use_amp = scaler.enabled and torch.cuda.is_available()
    amp_dtype = torch.bfloat16

    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        output = model(input_ids, labels=labels)
        loss = output.loss + output.aux_loss
        loss = loss / accumulation_steps

    scaler.scale(loss).backward()

    is_accumulation_boundary = (current_step + 1) % accumulation_steps == 0
    if is_accumulation_boundary:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return loss.item() * accumulation_steps
