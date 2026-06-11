# 第6章 训练基础设施 — 让模型可训练、可恢复

## 学习目标

掌握 PyTorch 训练基础设施：随机种子、学习率调度、混合精度、梯度累积、checkpoint。

## 本章概览

| 模块 | 文件 | 职责 |
|------|------|------|
| 训练工具 | `training/utils.py` | seed/lr/init_model/checkpoint/logger/sampler |
| 混合精度 | `training/amp.py` | autocast + GradScaler + train_step |
| GPU 优化 | `training/gpu.py` | TF32/cudnn/Flash SDPA |

## 6.1 学习率调度（余弦退火）

```
lr(step) = base_lr × (0.1 + 0.45 × (1 + cos(π × step / total)))
```

- 初始（step=0）：lr = base_lr × 1.0
- 中点（step=N/2）：lr = base_lr × 0.55
- 终点（step=N）：lr = base_lr × 0.1（最小学习率）

## 6.2 Checkpoint 系统

`lm_checkpoint()` 双重保存：
- **权重文件** `{weight}_{dim}.pth`：FP16 权重（用于推理）
- **续训文件** `{weight}_{dim}_resume.pth`：权重 + 优化器状态 + epoch + step

原子写入（`.tmp` → `os.replace`）防止写入中断导致文件损坏。

## 6.3 混合精度训练（AMP）

```
with autocast(dtype=bfloat16):
    loss = model(input_ids, labels)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), max_grad_norm)
scaler.step(optimizer)
scaler.update()
```

bfloat16 比 float16 更稳定（指数位与 float32 相同），推荐使用。

## 6.4 梯度累积

当 GPU 显存不足时，用小 batch 多次累积梯度后再 step：
```
for i, batch in enumerate(loader):
    loss = train_step(..., accumulation_steps=4, current_step=i)
    # 只有每 4 步才真正 optimizer.step()
```

## 6.5 SkipBatchSampler

断点续训时跳过已训练的 batch：
```python
sampler = SkipBatchSampler(base_sampler, batch_size=32, skip_batches=resumed_step)
```

## 验证

```bash
pytest tests/m06_training/ -v   # 33 个测试全绿
```
