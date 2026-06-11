# zllm — 从零训练中文大语言模型

Step-by-step TDD 教学项目，从 Tokenizer 到 Agent RL 全流程。覆盖完整训练管线，GPU 可训练。

## 快速开始

```bash
pip install -e ".[dev]"
pytest
```

## 训练管线

```
Tokenizer → Pretrain → SFT → LoRA → DPO → PPO → GRPO → Distillation → Agent RL → Serving
```

## 文档

- [设计文档](docs/design.md) — 项目定位、架构、关键决策
- [实施计划](docs/plan.md) — 300 步、12 里程碑的完整路线图

## 模型架构

对齐 Qwen3 / minimind-3：GQA、RoPE、SwiGLU、MoE、Weight Tying。默认 ~64M 参数（dim=768, 8 layers, vocab=6400）。

## 开发约定

- 每步遵循 TDD：测试 → 实现 → 文档
- GPU-first（CUDA 必需）
- Python 3.14+，全部最新依赖
- 文档中文为主，保留英文专业术语
