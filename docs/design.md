# zllm 设计文档

> 日期：2026-06-11
> 状态：已批准

## 1. 项目定位

**zllm** — 从零训练中文大语言模型。一个 step-by-step TDD 教学项目，同时具备完整的真实训练能力。

- 独立仓库：`/home/zw/zllm`，不依赖 minimind 的代码或文档
- Python 3.14+，全部依赖使用最新版本
- GPU-first（CUDA 必需）
- 模型架构对齐 Qwen3/minimind-3，可加载 minimind 权重
- 覆盖完整训练管线：Tokenizer → Pretrain → SFT → LoRA → DPO → PPO → GRPO → Distillation → Agent RL → Serving

## 2. 方法论：渐进式里程碑构建

300 个步骤组织为 12 个里程碑。每个里程碑产出可运行、可测试的功能检查点。每步遵循 TDD（测试 → 实现 → 文档）。

替代方案及选择理由：

| 方案 | 选择理由 |
|------|----------|
| 渐进里程碑 ✅ | 每步可运行可验证；里程碑=功能检查点；TDD 自然融入 |
| 模块优先 | 工程感强但教学叙事不连贯 |
| 纯教程 | 轻量但不满足"完整可训练"需求 |

## 3. 目录结构

```
zllm/
├── pyproject.toml              # Python 3.14+, 最新 deps
├── README.md
├── zllm/                       # 主包（pip install -e .）
│   ├── __init__.py
│   ├── config.py               # ZLLMConfig: 模型/训练配置
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── bpe.py              # BPE 算法
│   │   ├── special_tokens.py   # 特殊 token 定义
│   │   └── chat_template.py    # Chat Template (Jinja2)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── norms.py            # RMSNorm
│   │   ├── rope.py             # RoPE 位置编码
│   │   ├── attention.py        # GQA 注意力
│   │   ├── ffn.py              # SwiGLU / MoE FFN
│   │   ├── block.py            # Transformer Block
│   │   ├── backbone.py         # Transformer 主体 (ZLLMModel)
│   │   ├── causal_lm.py        # ZLLMForCausalLM (含 generate)
│   │   └── lora.py             # LoRA 模块
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── pretrain.py
│   │   ├── sft.py
│   │   ├── dpo.py
│   │   ├── rlaif.py
│   │   └── agent.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── utils.py            # DDP, checkpoint, lr, logging
│   │   ├── pretrain.py
│   │   ├── full_sft.py
│   │   ├── lora_sft.py
│   │   ├── dpo.py
│   │   ├── ppo.py
│   │   ├── grpo.py
│   │   ├── distillation.py
│   │   ├── agent_rl.py
│   │   └── rollout.py          # Rollout 引擎 (Torch + SGLang)
│   └── serving/
│       ├── __init__.py
│       ├── generate.py         # 推理生成
│       ├── api_server.py       # OpenAI 兼容 API
│       └── cli.py              # CLI 推理入口
├── tests/
│   ├── conftest.py             # 共享 fixtures (device, small config)
│   ├── m01_foundations/
│   ├── m02_tokenizer/
│   ├── m03_model_components/
│   ├── m04_model_assembly/
│   ├── m05_data_pipeline/
│   ├── m06_training_infra/
│   ├── m07_pretrain/
│   ├── m08_sft/
│   ├── m09_lora/
│   ├── m10_alignment/
│   ├── m11_distill_agent/
│   └── m12_serving/
├── docs/
│   ├── plan.md                 # 总体实施计划 (300 步)
│   ├── design.md               # 本设计文档
│   └── steps/                  # 每步教学文档
│       ├── 001-project-init.md
│       └── ...
├── data/                       # 训练数据 (.gitignore)
├── out/                        # 模型权重 (.gitignore)
├── checkpoints/                # 断点续训 (.gitignore)
└── scripts/
    ├── download_data.py        # 数据下载脚本
    └── train.py                # 统一训练入口
```

## 4. 12 个里程碑

| # | 里程碑 | 步骤 | 步数 | 产出 |
|---|--------|------|------|------|
| M1 | 基础与工具 | 1-25 | 25 | 项目骨架、张量操作、autograd |
| M2 | Tokenizer | 26-50 | 25 | BPE 分词器、Chat Template |
| M3 | 模型组件 I | 51-80 | 30 | RMSNorm、RoPE、Attention(GQA) |
| M4 | 模型组件 II + 组装 | 81-110 | 30 | FFN/SwiGLU、MoE、Block、CausalLM |
| M5 | 数据流水线 | 111-140 | 30 | 5 种 Dataset + collate |
| M6 | 训练基础设施 | 141-170 | 30 | DDP、AMP、checkpoint、lr schedule |
| M7 | 预训练 | 171-200 | 30 | 完整 pretrain 循环 |
| M8 | 监督微调 (SFT) | 201-220 | 20 | SFT 训练 + 对话评估 |
| M9 | LoRA 微调 | 221-235 | 15 | LoRA 注入/训练/合并 |
| M10 | 对齐训练 | 236-270 | 35 | DPO + PPO + GRPO |
| M11 | 蒸馏与 Agent | 271-290 | 20 | 知识蒸馏 + Agent RL |
| M12 | 推理与部署 | 291-300 | 10 | API 服务、模型转换、评测 |

## 5. 每步标准结构（TDD 三件套）

每个步骤固定包含：

1. **教学文档** `docs/steps/NNN-name.md` — 概念讲解 + 数学原理 + 代码目标（中文，保留英文术语）
2. **测试文件** `tests/mXX/test_NNN_name.py` — TDD 红色阶段：先写测试定义预期行为
3. **实现代码** `zllm/...` — TDD 绿色阶段：写实现让测试通过

每步独立可运行：`pytest tests/mXX/test_NNN_name.py -v`

## 6. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 与 minimind 关系 | 独立重写，架构兼容 | 教学清晰；内部参数命名保持一致，`load_state_dict` 可直接加载 minimind 权重 |
| 测试环境 | GPU-first，小尺寸(dim=64) | 用户选择；小尺寸保证速度 |
| 文档语言 | 中文为主，保留英文术语 | 与 minimind walkthrough 一致 |
| 配置类 | 继承 PretrainedConfig | 兼容 Transformers 生态 |
| 命名 | zllm/ZLLMForCausalLM | 独立身份，区别于 MiniMind |
| Weight Tying | True | 与 minimind/Qwen3 对齐 |
| 默认模型规模 | dim=768, 8 layers, vocab=6400 | 与 minimind 对齐 (~64M params) |
| MoE | 支持(4 experts, top-1) | 与 minimind 对齐 |

## 7. 依赖（最新版本）

```
python >= 3.14
torch >= 2.7
transformers >= 4.52
datasets >= 3.6
tokenizers >= 0.21
pytest >= 8.4
swanlab >= 0.4
streamlit >= 1.45
fastapi + uvicorn
```

## 8. 本次会话交付范围

- 项目骨架（目录结构 + pyproject.toml + README + conftest + 空模块文件）
- 总体实施计划文档（展开到全部 ~300 步的里程碑/步骤清单）
- 不实现具体算法代码，后续会话逐步展开
