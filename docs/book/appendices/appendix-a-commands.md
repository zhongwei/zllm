---
part: appendix
appendix: A
title: 命令速查
status: draft
---

# 附录 A 命令速查

本附录汇总全书涉及的训练、推理、部署命令。zllm 的训练脚本以 Python 模块形式调用（各训练阶段有对应的 Config dataclass + train_epoch），部署用 FastAPI 服务或 CLI。

## A.1 Tokenizer 训练（Ch 19）

```python
from zllm.tokenizer.trainer import train_tokenizer
tok = train_tokenizer(corpus, vocab_size=6400, save_dir="out/tok")
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocab_size` | 6400 | 词表大小 |
| `save_dir` | — | 输出目录 |

## A.2 预训练（Ch 31–32）

```python
from zllm.training.pretrain import PretrainConfig, train_epoch
cfg = PretrainConfig(epochs=2, batch_size=64, learning_rate=5e-4, accumulation_steps=4)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 2 | 训练轮数 |
| `batch_size` | 64 | 批大小 |
| `learning_rate` | 5e-4 | 基础学习率 |
| `accumulation_steps` | 4 | 梯度累积步数 |
| `max_seq_len` | 340 | 序列长度 |
| `from_weight` | "none" | 起始权重 |
| `save_weight` | "pretrain" | 保存名 |

## A.3 SFT 监督微调（Ch 33）

```python
from zllm.training.full_sft import SFTConfig, train_epoch
cfg = SFTConfig(epochs=2, learning_rate=1e-5, from_weight="pretrain")
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 1e-5 | 预训练的 1/50 |
| `max_seq_len` | 768 | 对话更长 |
| `from_weight` | "pretrain" | 加载预训练权重 |

## A.4 LoRA 微调（Ch 34）

```python
from zllm.training.lora_sft import LoRAConfig, train_epoch
cfg = LoRAConfig(rank=16, learning_rate=1e-4, from_weight="full_sft")
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rank` | 16 | 低秩维度 |
| `learning_rate` | 1e-4 | full_sft 的 10 倍 |
| `epochs` | 10 | 参数少学得慢 |

## A.5 DPO 偏好优化（Ch 36）

```python
from zllm.training.dpo import DPOConfig, train_epoch
cfg = DPOConfig(beta=0.15, learning_rate=4e-8, from_weight="full_sft")
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta` | 0.15 | DPO 温度 |
| `learning_rate` | 4e-8 | 极低，防灾难遗忘 |

## A.6 PPO 强化学习（Ch 37）

```python
from zllm.training.ppo import PPOConfig
cfg = PPOConfig(clip_epsilon=0.2, vf_coef=0.5, kl_coef=0.02, gamma=1.0, lam=0.95)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `clip_epsilon` | 0.2 | policy clip 范围 |
| `vf_coef` | 0.5 | value loss 权重 |
| `kl_coef` | 0.02 | KL 惩罚系数 |
| `lam` | 0.95 | GAE λ |

## A.7 GRPO / CISPO（Ch 38）

```python
from zllm.training.grpo import GRPOConfig
cfg = GRPOConfig(num_generations=6, beta=0.1, loss_type="cispo")
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_generations` | 6 | 每 prompt 生成数 |
| `loss_type` | "cispo" | CISPO 单边裁剪 |
| `epsilon_high` | 5.0 | CISPO 上界 |

## A.8 知识蒸馏（Ch 39）

```python
from zllm.training.distillation import DistillConfig, train_epoch
cfg = DistillConfig(alpha=0.5, temperature=1.5, from_weight="full_sft")
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | 0.5 | 硬/软标签权重 |
| `temperature` | 1.5 | 蒸馏温度 |

## A.9 Agent RL（Ch 40）

```python
from zllm.training.agent_rl import AgentConfig
cfg = AgentConfig(max_turns=3, max_gen_len=256)
```

## A.10 推理与部署（Ch 41–43）

```python
# 解码生成
from zllm.serving.generate import generate, generate_with_cache
out = generate_with_cache(model, input_ids, max_new_tokens=128, temperature=0.85, top_p=0.95)

# API 服务
from zllm.serving.api_server import create_app
app = create_app()
# uvicorn: uvicorn zllm.serving.api_server:create_app --factory --port 8000

# CLI 配置
from zllm.serving.cli import CLIConfig
cfg = CLIConfig(weight="full_sft", temperature=0.85)
```

## A.11 测试

```bash
# 全量测试
pytest tests/ -v

# 按里程碑
pytest tests/m03_model_components/ -v    # 模型组件
pytest tests/m07_pretrain/ -v            # 预训练
pytest tests/m12_serving/ -v             # 推理部署
```
