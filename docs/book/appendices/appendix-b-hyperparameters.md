---
part: appendix
appendix: B
title: 超参数表
status: draft
---

# 附录 B 超参数表

本附录汇总 zllm 的模型架构超参和各训练阶段的超参对比。

## B.1 模型架构（ZLLMConfig）

> 来源：`zllm/config.py`

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `vocab_size` | 6400 | 词表大小 |
| `hidden_size` | 768 | 隐藏维度 |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `num_attention_heads` | 8 | 注意力头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA，Ch 22） |
| `intermediate_size` | 2432 | FFN 中间维度（π 缩放，Ch 23） |
| `head_dim` | 96 | 每头维度（hidden / num_heads） |
| `max_position_embeddings` | 32768 | 最大位置 |
| `rms_norm_eps` | 1e-6 | RMSNorm ε（Ch 20） |
| `rope_theta` | 1000000.0 | RoPE 基频（Ch 21） |
| `tie_word_embeddings` | True | Weight Tying（Ch 26） |
| `flash_attn` | True | Flash Attention（Ch 22） |
| `hidden_act` | "silu" | SwiGLU 激活（Ch 23） |

### MoE 配置（Ch 24）

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `use_moe` | False | 是否启用 MoE |
| `num_experts` | 4 | 专家数 |
| `num_experts_per_tok` | 1 | 每 token 激活专家数 |
| `moe_intermediate_size` | None | MoE 专家中间维度 |
| `norm_topk_prob` | True | 归一化 top-k 概率 |
| `router_aux_loss_coef` | 5e-4 | 负载均衡 loss 系数 |

### 参数量估算（默认配置）

- 总参数量：约 64M（密集）/ 约 64M-A（MoE，active 更少）
- embedding + lm_head（Weight Tying）：6400 × 768 ≈ 4.9M
- 每层 attention：约 1.77M（GQA：q=768²，k/v=768·384 各一，o=768²）
- 每层 FFN（SwiGLU）：约 5.6M
- 8 层合计（attention + FFN）：约 59M；加 embedding 约 4.9M，总计约 64M

## B.2 训练阶段超参对比

| 阶段 | 学习率 | batch | accum | epochs | 来源 |
|------|--------|-------|-------|--------|------|
| 预训练 | 5e-4 | 64 | 4 | 2 | `PretrainConfig` |
| SFT | 1e-5 | 16 | 1 | 2 | `SFTConfig` |
| LoRA | 1e-4 | 32 | 1 | 10 | `LoRAConfig` |
| DPO | 4e-8 | 4 | 1 | 1 | `DPOConfig` |
| PPO | 3e-7 | 2 | 1 | 1 | `PPOConfig` |
| GRPO | 3e-7 | 2 | 1 | 1 | `GRPOConfig` |
| 蒸馏 | 5e-6 | 32 | 1 | 6 | `DistillConfig` |
| Agent RL | 3e-7 | 2 | 1 | 1 | `AgentConfig` |

### 学习率变化规律

```
预训练   5e-4 ────────────  最大学率（从头学语言）
SFT      1e-5 ─            1/50（微调防遗忘）
LoRA     1e-4 ──           10× SFT（参数少需要更大）
DPO      4e-8 ─            1/250 SFT（极精细偏好调整）
PPO/GRPO 3e-7 ─            RL 级别（在线采样微调）
```

## B.3 算法特定超参

### 对齐方法

| 方法 | 关键超参 | 默认值 | 说明 |
|------|---------|--------|------|
| DPO | `beta` | 0.15 | 偏好温度 |
| PPO | `clip_epsilon` | 0.2 | policy clip |
| PPO | `vf_coef` | 0.5 | value loss 权重 |
| PPO | `kl_coef` | 0.02 | KL 惩罚 |
| PPO | `gamma/lam` | 1.0/0.95 | GAE 折扣/λ |
| GRPO | `num_generations` | 6 | 群体大小 |
| GRPO | `beta` | 0.1 | KL 系数 |
| CISPO | `epsilon_high` | 5.0 | 单边上界 |

### 蒸馏

| 超参 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | 0.5 | 硬标签 CE 权重（1-α 给软标签） |
| `temperature` | 1.5 | 软化温度 |

### 解码（Ch 41）

| 超参 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 0.85 | 采样温度 |
| `top_p` | 0.95 | nucleus 采样 |
| `max_new_tokens` | 128/8192 | 生成上限（API/CLI） |
| `repetition_penalty` | 1.0 | 重复惩罚（1=不惩罚） |
