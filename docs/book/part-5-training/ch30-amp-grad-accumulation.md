---
part: 5
chapter: 30
title: 混合精度 AMP + 梯度累积 + GPU 优化
milestone: M6-b
source: zllm/training/amp.py
tests: tests/m06_training/test_161_amp.py
status: draft
---

# 第 30 章 混合精度 AMP + 梯度累积 + GPU 优化

Ch 29 搭好了训练基础设施，但还有两个现实问题：

1. **显存不够**——LLM 权重动辄几十亿参数，fp32 下每个参数占 4 字节，光存权重就要几十 GB。
2. **batch 太小**——显存有限意味着 batch_size 开不大，但小 batch 的梯度噪声大、训练不稳。

本章解决这两个问题：**混合精度（AMP）** 用 bf16 前向省一半显存，**梯度累积** 把多个小 batch 的梯度累加起来等效大 batch。再加上 TF32 / cuDNN benchmark / Flash SDPA 三个 GPU 硬件加速开关，训练就能「跑得动、跑得快」。

这三个技术合在一起，封装在 `train_step` 函数里——它是所有训练脚本共享的**单步训练原语**。

## 30.1 学习目标

读完本章，你应该能够：

- 解释 bf16 混合精度的原理：前向用 bf16（省显存），梯度累加用 fp32（保精度）；
- 说清 bf16 比 fp16 更适合 LLM 的原因（动态范围大，不需要 GradScaler 放大）；
- 默写出梯度累积的数学：$\nabla_{\text{累积}} = \frac{1}{N}\sum_{i=1}^{N} \nabla_i$，等效 batch_size = N × batch；
- 解释为什么 `unscale` 之后才能 `clip_grad_norm`（先还原真实尺度再裁剪）；
- 看懂 TF32 / cuDNN benchmark / Flash SDPA 各自加速什么环节。

## 30.2 原理回顾：精度、累积与加速

### 30.2.1 混合精度（回引 Ch 08）

Ch 08《数值稳定性》讲过浮点数的精度与范围。fp32（32 位）精度高但占显存；bf16（16 位）**动态范围和 fp32 一样**（指数位相同），但精度低一半。

**混合精度训练（AMP）** 的策略：**前向传播用 bf16**（省一半显存、算得快），**梯度累加和参数更新用 fp32**（保精度不溢出）。PyTorch 的 `torch.amp.autocast` 自动管理这个切换——在 `autocast` 上下文里，矩阵乘法自动降到 bf16，而 loss 计算和反传保持 fp32。

为什么 zllm 选 bf16 而非 fp16？因为 fp16 的动态范围小（最大约 65504），LLM 的梯度很容易**溢出**变成 inf。bf16 的动态范围和 fp32 相同（最大约 $3 \times 10^{38}$），几乎不会溢出，不需要 GradScaler 的放大补偿——代码更简单、更稳定。

### 30.2.2 梯度累积

理想情况下想用 batch_size=64，但显存只够 batch_size=8。**梯度累积**的做法：连续跑 8 个小 batch（batch=8），每个 batch 算完梯度后**不清零**，累加 8 次再一起更新。数学上：

$$
\nabla_{\text{累积}} \;=\; \frac{1}{N}\sum_{i=1}^{N} \nabla_i \;\approx\; \nabla_{\text{batch}=N\times 8}
$$

代码里每次 forward 前 `loss = loss / accumulation_steps`（`accumulation_steps=8`），这样 8 次累加后等效于一次大 batch 的平均梯度。**只有第 N 次才真正 `optimizer.step()`**，前 N-1 次只 `backward` 累加梯度。

### 30.2.3 GPU 硬件加速三件套

| 技术 | 加速什么 | 原理 |
|------|---------|------|
| TF32 | 矩阵乘法 | 用 19 位（而非 32 位）做乘法，精度几乎无损但快 2-3 倍 |
| cuDNN benchmark | 卷积/注意力 | 自动试几种 kernel 选最快的（适合固定输入形状） |
| Flash SDPA | 注意力 | 把 QK^T 和 softmax 融合，减少 HBM 读写（Ch 22 讲过） |

## 30.3 代码实现：train_step

完整实现见 `zllm/training/amp.py`（76 行）。`train_step` 是最核心的函数。

### 30.3.1 GradScalerManager：封装 GradScaler

> 完整实现见 `zllm/training/amp.py:12`

```python
class GradScalerManager:
    def __init__(self, enabled=True):
        self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        self.enabled = enabled
```

`GradScalerManager`（`:12-29`）：薄封装 `torch.amp.GradScaler`。注意：用 bf16 时 `enabled=False`（bf16 不需要放大），用 fp16 时 `enabled=True`。zllm 默认 bf16，所以 scaler 大多数时候只是个**空壳**——但保留接口，方便切 fp16。

### 30.3.2 train_step：单步训练原语

> 完整实现见 `zllm/training/amp.py:32`

```python
def train_step(model, optimizer, input_ids, labels, scaler,
               accumulation_steps=1, max_grad_norm=1.0, current_step=0, device="cuda"):
    use_amp = scaler.enabled and torch.cuda.is_available()
    amp_dtype = torch.bfloat16

    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        output = model(input_ids, labels=labels)
        loss = output.loss + output.aux_loss      # NTP loss + MoE 负载均衡 loss
        loss = loss / accumulation_steps           # 累积前先除 N

    scaler.scale(loss).backward()                  # 反传（累加梯度）

    is_accumulation_boundary = (current_step + 1) % accumulation_steps == 0
    if is_accumulation_boundary:                   # 只有累积边界才真正更新
        scaler.unscale_(optimizer)                 # 先还原梯度真实尺度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 再裁剪
        scaler.step(optimizer)                     # 更新参数
        scaler.update()                            # 更新 scaler 的缩放因子
        optimizer.zero_grad(set_to_none=True)      # 清零（为下一轮累积做准备）

    return loss.item() * accumulation_steps        # 返回未缩放的 loss 值
```

`train_step`（`:32-76`）五个阶段：

1. **AMP 前向**（`:61-62`）：`autocast(dtype=bfloat16)` 让矩阵乘法自动降到 bf16。
2. **loss 合并**（`:63-64`）：`output.loss`（NTP 交叉熵）+ `output.aux_loss`（MoE 负载均衡，Ch 24）。两者一起反传，MoE 的专家均衡自动生效。然后 `/ accumulation_steps` 为累积做准备。
3. **反传累积**（`:66`）：`scaler.scale(loss).backward()`。注意这里**不清零梯度**——梯度会累加到 `.grad` 上。
4. **边界更新**（`:68-74`）：只有 `(current_step+1) % accumulation_steps == 0` 时才执行更新四连：`unscale_` → `clip_grad_norm_` → `step` → `zero_grad`。
5. **返回 loss**（`:76`）：`loss.item() * accumulation_steps` 还原成单 batch 的 loss 值用于日志。

**为什么 unscale 在 clip 之前？** `scaler.scale` 把 loss 放大了（fp16 才需要），梯度也跟着放大了。`unscale_` 把梯度**还原成真实尺度**，然后 `clip_grad_norm` 才能在真实尺度上判断「梯度范数是否超 1.0」。如果先 clip 再 unscale，裁剪阈值就对不上真实梯度了。

> 对应测试 `tests/m06_training/test_161_amp.py:40`（基本 step loss>0）、`:48`（梯度裁剪有效——`max_grad_norm=0.01` 后梯度范数被限住）、`:57`（3 步累积，前 2 步不 step）、`:67`（CUDA 上 AMP）、`:76`（step 后梯度清零）。

## 30.4 GPU 性能开关：gpu.py

完整实现见 `zllm/training/gpu.py`（25 行），三个开关函数：

> 完整实现见 `zllm/training/gpu.py:22`

```python
def setup_gpu_performance():
    enable_tf32()              # matmul/cudnn 允许 TF32
    enable_cudnn_benchmark()   # 自动选最快卷积 kernel
    enable_flash_sdpa()        # 启用 Flash Attention
```

`setup_gpu_performance`（`:22-25`）一键全开。训练脚本开头调一次即可。`enable_tf32`（`:9-11`）设置 `torch.backends.cuda.matmul.allow_tf32 = True`；`enable_flash_sdpa`（`:18-19`）启用 Ch 22 讲过的 Flash Attention（`torch.backends.cuda.enable_flash_sdp(True)`）。

> 对应测试 `tests/m06_training/test_166_gpu.py:52-58`（三个开关都能正常调用，不报错）。

## 30.5 对应单元测试

> 对应测试 `tests/m06_training/test_161_amp.py`（83 行）

- **TestGradScalerManager**（`test_161_amp.py:27`）：初始化 `:28`、scale 改变值 `:32`。
- **TestTrainStep**（`test_161_amp.py:39`）：
  - `test_basic_step` `test_161_amp.py:40`：单步 loss 正且 finite。
  - `test_gradient_clipping` `test_161_amp.py:48`：`max_grad_norm=0.01` 后所有梯度范数 ≤ 阈值。
  - `test_accumulation_steps` `test_161_amp.py:57`：3 步累积，模拟「只有第 3 步才 step」。
  - `test_amp_on_cuda` `test_161_amp.py:67`：CUDA 上 AMP 跑通（CPU 跳过）。
  - `test_zero_grad_after_step` `test_161_amp.py:76`：step 后 `.grad` 为 None 或 0（`set_to_none=True`）。

## 30.6 动手验证

```bash
pytest tests/m06_training/test_161_amp.py tests/m06_training/test_166_gpu.py -v
```

预期：全部 PASSED（CUDA 相关测试在 CPU 上自动 skip）。手写验证梯度累积逻辑：

```bash
python -c "
import torch
from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.amp import train_step, GradScalerManager
cfg = ZLLMConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
model = ZLLMForCausalLM(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScalerManager(enabled=False)
ids = torch.randint(0, 100, (2, 16))
labels = ids.clone()
# 累积 3 步
for step in range(3):
    loss = train_step(model, opt, ids, labels, scaler, accumulation_steps=3, current_step=step)
    print(f'step {step}: loss={loss:.4f}, grad={model.lm_head.weight.grad is not None}')
"
```

## 30.7 本章小结 + 下章预告

本章要点：

1. **bf16 混合精度**：前向 bf16 省显存、梯度 fp32 保精度。bf16 动态范围大，不需要 GradScaler 放大。
2. **梯度累积**：连续 N 个小 batch 的梯度累加后更新，等效 batch_size = N × batch。`loss / N` 保证数学等价。
3. **train_step 五阶段**：autocast → loss+/accum → backward → 边界(unscale→clip→step→zero) → 返回。
4. **unscale 在 clip 前**：先还原真实梯度尺度再裁剪。
5. **GPU 三开关**：TF32（matmul 快 2-3 倍）、cuDNN benchmark（选最快 kernel）、Flash SDPA（注意力少读写 HBM）。

> **一句话带走**：AMP 省 显存、梯度累积等效大 batch、GPU 三开关加速——train_step 把它们封装成一行可调的原语。

**下章预告**：数据（Ch 27–28）+ 基础设施（Ch 29）+ 训练原语（Ch 30）都齐了，该把它们组装成完整的训练循环了。Ch 31《预训练：NTP 与训练循环》——`train_epoch` 把 DataLoader、模型、优化器、lr 调度、AMP、梯度累积全部串起来，跑出第一条 loss 曲线。
