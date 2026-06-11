# 第7章 预训练 — 从零训练语言模型

## 学习目标

掌握完整预训练流程：数据加载 → 模型初始化 → 训练循环 → checkpoint 保存/恢复。

## 核心概念

### Next-Token Prediction（NTP）

预训练目标：给定前 t 个 token，预测第 t+1 个 token。

```
输入: [BOS, t1, t2, t3, ..., tn]
目标: [t1, t2, t3, ..., tn, EOS]
Loss: CrossEntropy(logits[:-1], labels[1:], ignore_index=-100)
```

### 训练循环结构（9 步模板）

1. 初始化环境 + 随机种子
2. GPU 性能优化（TF32/cudnn/Flash）
3. 配置模型 + 检查 checkpoint
4. 混合精度配置
5. 定义模型 + 数据 + 优化器
6. 从 checkpoint 恢复
7. torch.compile + DDP 包装
8. 训练主循环（epoch 迭代）
9. 清理分布式进程

### train_epoch

每个 epoch 内：
- 动态学习率（余弦退火）
- AMP 混合精度前向
- 梯度累积
- unscale → clip → step → zero_grad
- 定期日志 + 保存

## PretrainConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 2 | 训练轮数 |
| batch_size | 64 | 批量大小 |
| learning_rate | 5e-4 | 初始学习率 |
| accumulation_steps | 4 | 梯度累积步数 |
| grad_clip | 1.0 | 梯度裁剪阈值 |
| max_seq_len | 340 | 最大序列长度 |

## 验证

```bash
pytest tests/m07_pretrain/ -v   # 12 个测试全绿
```
