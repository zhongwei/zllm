# zllm — 从零训练中文大语言模型

**zllm** 是一个 step-by-step TDD 教学项目，同时具备完整的真实训练能力。从 Tokenizer 到 Agent RL，覆盖 LLM 训练全流程，3142 行代码、428 个测试、12 个里程碑。

模型架构对齐 Qwen3 / minimind-3（GQA、RoPE、SwiGLU、MoE、Weight Tying），默认 ~64M 参数。

## 训练管线全景

```
                    ┌──────────────┐
                    │   Tokenizer  │  BPE 分词 + Chat Template
                    │   (M2)       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Pretrain   │  Next Token Prediction
                    │   (M7)       │  lr=5e-4, epochs=2
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │     SFT      │  监督微调，只对 assistant 回复计算 loss
                    │   (M8)       │  lr=1e-5, from_weight=pretrain
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼─────┐ ┌───▼────┐ ┌────▼──────┐
       │    LoRA    │ │  DPO   │ │ PPO/GRPO  │
       │   (M9)     │ │ (M10)  │ │   (M10)   │
       │ rank=16    │ │ β=0.15 │ │ GAE/Group │
       └────────────┘ └────────┘ └───────────┘
                           │
              ┌────────────┼────────────┐
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │ Distillation│          │  Agent RL   │
       │   (M11)     │          │   (M11)     │
       │ T²·KL 蒸馏  │          │ 6 个模拟工具 │
       └─────────────┘          └─────────────┘
                           │
                    ┌──────▼───────┐
                    │   Serving    │  generate + KV Cache + OpenAI API
                    │   (M12)      │
                    └──────────────┘
```

## 快速开始

```bash
# 安装（开发模式）
pip install -e ".[dev]"

# 运行全部 428 个测试
pytest

# 运行特定里程碑的测试
pytest tests/m02_tokenizer/ -v
pytest tests/m10_alignment/ -v
```

### 5 行推理

```python
import torch
from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.serving.generate import generate

model = ZLLMForCausalLM(ZLLMConfig())          # 随机初始化的模型
input_ids = torch.randint(0, 6400, (1, 8))      # 随机输入
output = generate(model, input_ids, max_new_tokens=32, temperature=0.0)  # greedy
print(output.shape)  # (1, 40)
```

## 模型架构

### 默认参数（~64M Dense）

| 参数 | 值 | 说明 |
|------|----|------|
| `hidden_size` | 768 | 隐藏层维度 |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `vocab_size` | 6400 | 词表大小 |
| `num_attention_heads` | 8 | Query 头数（GQA） |
| `num_key_value_heads` | 4 | KV 头数（GQA，2:1 压缩） |
| `intermediate_size` | 2368 | SwiGLU 中间层（π 缩放，对齐 64 倍数） |
| `rope_theta` | 1,000,000 | RoPE 基频 |
| `max_position_embeddings` | 32768 | 最大序列长度 |
| `tie_word_embeddings` | True | 输入/输出 embedding 共享 |

### MoE 变体（~198M 总 / ~64M 活跃）

```python
config = ZLLMConfig(use_moe=True, num_experts=4, num_experts_per_tok=1)
model = ZLLMForCausalLM(config)
```

4 个 expert FFN，Router Top-1 选路 + 辅助损失，与 Qwen3-MoE 对齐。

## 12 里程碑导航

| # | 里程碑 | 核心内容 | 教学文档 | 测试数 |
|---|--------|---------|----------|--------|
| M1 | 基础与工具 | 项目骨架、ZLLMConfig、fixtures | [001-project-init](docs/steps/001-project-init.md) | — |
| M2 | Tokenizer | BPE 核心算法 + 生产级 trainer + Chat Template | [026-tokenizer](docs/steps/026-tokenizer-overview.md) | 72 |
| M3 | 模型组件 I | RMSNorm、RoPE（含 YaRN）、GQA Attention（QK-Norm + KV Cache） | [051-model-components](docs/steps/051-model-components-1.md) | 51 |
| M4 | 模型组装 | SwiGLU FFN、MoE Router、Block、Backbone、CausalLM | [081-model-components](docs/steps/081-model-components-2.md) | 60 |
| M5 | 数据流水线 | 5 种 Dataset + TokenizerAdapter + collate | [111-data-pipeline](docs/steps/111-data-pipeline.md) | 45 |
| M6 | 训练基础设施 | seed、lr schedule、checkpoint、AMP、GPU | [141-training-infra](docs/steps/141-training-infra.md) | 33 |
| M7 | 预训练 | PretrainConfig、train_epoch、NTP 验证 | [171-pretrain](docs/steps/171-pretrain.md) | 12 |
| M8 | SFT | SFTConfig、label masking、对话生成 | [201-sft](docs/steps/201-sft.md) | 17 |
| M9 | LoRA | inject/freeze/save/load/merge | [221-lora](docs/steps/221-lora.md) | 25 |
| M10 | 对齐训练 | DPO + PPO + GRPO | [236-alignment](docs/steps/236-alignment.md) | 40 |
| M11 | 蒸馏 + Agent | T²·KL 蒸馏 + 工具调用 RL | [271-distill-agent](docs/steps/271-distill-agent.md) | 33 |
| M12 | 推理部署 | generate、KV Cache、OpenAI API、CLI | [291-serving](docs/steps/291-serving.md) | 20 |

## 各模块用法

### Tokenizer

zllm 提供两套 BPE 实现：教学版（`bpe.py`，纯 Python，清晰展示算法）和生产版（`trainer.py`，基于 HuggingFace tokenizers 库）。

```python
from zllm.tokenizer.bpe import train_bpe, encode, decode

# 训练 BPE（教学版）
merges = train_bpe(["你好世界", "今天天气真好"], vocab_size=260)

# 编码/解码
ids = encode("你好", merges)
text = decode(ids, merges)
```

生产环境使用 `TokenizerAdapter` 包装 `tokenizers.Tokenizer`，提供 transformers 风格 API：

```python
from tokenizers import Tokenizer
from zllm.tokenizer.adapter import wrap

tok = Tokenizer.from_file("model/tokenizer.json")
tok = wrap(tok)  # → TokenizerAdapter

# 基础编码
result = tok("你好世界")
print(result["input_ids"])

# Chat Template
ids = tok.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=True,
    add_generation_prompt=True,
)
print(tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
```

### Model

```python
from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM

# Dense 模型（~64M）
config = ZLLMConfig(hidden_size=768, num_hidden_layers=8)
model = ZLLMForCausalLM(config)

# MoE 模型（~198M total / ~64M active）
config = ZLLMConfig(hidden_size=768, num_hidden_layers=8, use_moe=True)
model = ZLLMForCausalLM(config)

# 前向传播
import torch
input_ids = torch.randint(0, 6400, (2, 64))
output = model(input_ids)
print(output.logits.shape)    # (2, 64, 6400)
print(output.aux_loss)        # MoE 辅助损失（Dense 时为 0）

# 计算 loss
labels = input_ids.clone()
output = model(input_ids, labels=labels)
print(output.loss)            # cross_entropy

# KV Cache 推理
output = model(input_ids, use_cache=True)
print(output.past_key_values) # tuple of (k, v) per layer
```

### Dataset

5 种数据集，对应不同训练阶段：

| 数据集 | 输入格式 | 用途 |
|--------|---------|------|
| `PretrainDataset` | `{"text": "..."}` | 预训练（NTP） |
| `SFTDataset` | `{"conversations": [...]}` | 监督微调（只对 assistant 计算 loss） |
| `DPODataset` | `{"chosen": [...], "rejected": [...]}` | DPO 偏好训练 |
| `RLAIFDataset` | `{"conversations": [...]}` | PPO/GRPO 强化学习 |
| `AgentDataset` | `{"conversations": [...], "tools": [...]}` | Agent RL 工具调用 |

```python
from zllm.dataset.pretrain import PretrainDataset
from zllm.dataset.sft import SFTDataset

# 预训练数据
ds = PretrainDataset("data/pretrain_t2t.jsonl", tokenizer=tok, max_seq_len=340)
input_ids, labels = ds[0]

# SFT 数据（labels 中非 assistant 部分为 -100）
ds = SFTDataset("data/sft_t2t.jsonl", tokenizer=tok, max_seq_len=768)
input_ids, labels = ds[0]
```

### Training

所有训练脚本遵循统一的循环结构：lr 调度 → AMP 前向 → loss → 梯度累积 → clip → step → 日志。

```python
from zllm.training.pretrain import PretrainConfig, train_epoch
from zllm.training.full_sft import SFTConfig
from zllm.training.utils import get_lr

# 预训练配置
cfg = PretrainConfig(epochs=2, batch_size=64, learning_rate=5e-4, accumulation_steps=4)

# SFT 配置（lr 低 50 倍）
cfg = SFTConfig(epochs=2, batch_size=16, learning_rate=1e-5, from_weight="pretrain")

# Cosine lr schedule
lr = get_lr(current_step=500, total_steps=10000, max_lr=5e-4)
```

### LoRA

低秩适配器：A=高斯初始化，B=零初始化（初始时 LoRA 输出为 0，不破坏预训练权重）。

```python
from zllm.model.lora import apply_lora, freeze_non_lora, save_lora, load_lora, merge_lora
from zllm.training.lora_sft import LoRAConfig

cfg = LoRAConfig(rank=16, learning_rate=1e-4, from_weight="full_sft")

# 注入 LoRA 到模型
lora_params = apply_lora(model, rank=cfg.rank)
freeze_non_lora(model)

# 训练后保存/加载
save_lora(model, "out/lora_medical_768.pth")
load_lora(model, "out/lora_medical_768.pth")

# 合并到基础权重（推理用）
merge_lora(model)
```

### 对齐训练（DPO / PPO / GRPO）

**DPO** — 直接偏好优化，无需 reward model：

```python
from zllm.training.dpo import dpo_loss, logits_to_log_probs, DPOConfig

cfg = DPOConfig(learning_rate=4e-8, beta=0.15)
# loss = -log_sigmoid(β * (π_θ_chosen/π_ref - π_θ_rejected/π_ref))
```

**PPO** — 带 Critic 的近端策略优化：

```python
from zllm.training.ppo import CriticModel, compute_gae, ppo_policy_loss, ppo_value_loss

critic = CriticModel(config)
# GAE: δ_t = r_t + γV(s_{t+1}) - V(s_t)
# Clipped surrogate: min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

**GRPO** — 无 Critic，群体相对优势：

```python
from zllm.training.grpo import grpo_loss, cispo_loss, compute_group_advantages, per_token_kl

# 组内标准化: advantages = (reward - mean) / std
# CISPO: 单边裁剪（ratio ≤ ε_high，允许低质量 token 降权）
```

### Distillation

知识蒸馏：用教师模型的软标签指导学生模型。

```python
from zllm.training.distillation import distillation_loss, DistillConfig

cfg = DistillConfig(alpha=0.5, temperature=1.5)
# loss = α * CE(student, hard_labels) + (1-α) * T² * KL(teacher_soft || student_soft)
```

### Agent RL

工具调用强化学习，包含 6 个模拟工具（天气/时间/汇率/翻译/计算/单位换算）。

```python
from zllm.training.agent_rl import (
    execute_tool, parse_tool_calls, calculate_agent_reward, TOOLS
)

# 解析模型生成的工具调用
calls = parse_tool_calls('```json\n{"name": "calculate_math", "arguments": {"expression": "2+3"}}\n```')

# 执行工具
result = execute_tool("calculate_math", {"expression": "2+3"})  # {"result": "5"}

# 计算多维度奖励（长度 + 工具正确性 + GT 匹配 + 重复惩罚）
reward = calculate_agent_reward(response, gt_answer="5", tool_calls=calls)
```

### Serving

```python
from zllm.serving.generate import generate, generate_with_cache
from zllm.serving.cli import CLIConfig

# 基础生成（支持 greedy/temperature/top-k/top-p/repetition_penalty）
output = generate(model, input_ids, max_new_tokens=128, temperature=0.85, top_p=0.95)

# KV Cache 加速（首次处理 prompt，后续只处理新 token）
output = generate_with_cache(model, input_ids, max_new_tokens=128, temperature=0.0)

# CLI 配置
cfg = CLIConfig(weight="full_sft", temperature=0.85, top_p=0.95)
```

OpenAI 兼容 API 服务器：

```python
from zllm.serving.api_server import create_app

app = create_app()
# 端点：
# GET  /v1/models            — 模型列表
# POST /v1/chat/completions  — 对话生成
```

```bash
# 启动服务（需要 uvicorn）
uvicorn zllm.serving.api_server:create_app --factory --host 0.0.0.0 --port 8000
```

## 训练命令速查

```bash
pip install -e ".[dev]"

# 1. 预训练（必须）
python -m zllm.training.pretrain

# 2. SFT（必须）
python -m zllm.training.full_sft

# 3. LoRA 微调（可选）
python -m zllm.training.lora_sft

# 4. 对齐训练（可选，三选一或顺序执行）
python -m zllm.training.dpo
python -m zllm.training.ppo
python -m zllm.training.grpo

# 5. 蒸馏（可选）
python -m zllm.training.distillation

# 6. Agent RL（可选）
python -m zllm.training.agent_rl
```

多 GPU (DDP)：

```bash
torchrun --nproc_per_node 4 -m zllm.training.pretrain
```

断点续训：

```bash
python -m zllm.training.pretrain --from_resume 1
```

## 项目结构

```
zllm/
├── pyproject.toml
├── README.md
├── zllm/
│   ├── __init__.py
│   ├── config.py                   # ZLLMConfig (PretrainedConfig)
│   ├── tokenizer/
│   │   ├── bpe.py                  # BPE 核心算法（教学版）
│   │   ├── special_tokens.py       # <|im_start|>, <|im_end|>, <|pad|>
│   │   ├── trainer.py              # 生产级 BPE trainer
│   │   ├── chat_template.py        # ChatML 模板渲染
│   │   └── adapter.py              # TokenizerAdapter + wrap()
│   ├── model/
│   │   ├── norms.py                # RMSNorm
│   │   ├── rope.py                 # RoPE + YaRN 位置缩放
│   │   ├── attention.py            # GQA Attention (Flash + 手动 KV Cache)
│   │   ├── ffn.py                  # SwiGLU + MoE (Router Top-K)
│   │   ├── block.py                # Transformer Block
│   │   ├── backbone.py             # ZLLMModel (embed + layers + norm)
│   │   ├── causal_lm.py            # ZLLMForCausalLM (PreTrainedModel)
│   │   └── lora.py                 # LoRA inject/freeze/save/load/merge
│   ├── dataset/
│   │   ├── utils.py                # collate + padding
│   │   ├── pretrain.py             # PretrainDataset
│   │   ├── sft.py                  # SFTDataset (label masking)
│   │   ├── dpo.py                  # DPODataset
│   │   ├── rlaif.py                # RLAIFDataset
│   │   └── agent.py                # AgentDataset
│   ├── training/
│   │   ├── utils.py                # seed, get_lr, lm_checkpoint, Logger
│   │   ├── amp.py                  # GradScalerManager
│   │   ├── gpu.py                  # GPU 工具
│   │   ├── pretrain.py             # PretrainConfig + train_epoch
│   │   ├── full_sft.py             # SFTConfig + train_epoch
│   │   ├── lora_sft.py             # LoRAConfig + train_epoch
│   │   ├── dpo.py                  # DPOConfig + train_epoch
│   │   ├── ppo.py                  # CriticModel, GAE, PPO loss
│   │   ├── grpo.py                 # GRPO/CISPO loss + group advantages
│   │   ├── distillation.py         # DistillConfig + distillation_loss
│   │   └── agent_rl.py             # AgentConfig + 6 个模拟工具
│   └── serving/
│       ├── generate.py             # generate + generate_with_cache
│       ├── api_server.py           # FastAPI /v1/chat/completions
│       └── cli.py                  # CLIConfig
├── tests/
│   ├── conftest.py                 # small_config (dim=64), default_config, device
│   ├── m01_foundations/
│   ├── m02_tokenizer/
│   ├── m03_model_components/
│   ├── m04_model_assembly/
│   ├── m05_data_pipeline/
│   ├── m06_training/
│   ├── m07_pretrain/
│   ├── m08_sft/
│   ├── m09_lora/
│   ├── m10_alignment/
│   ├── m11_distill_agent/
│   └── m12_serving/
├── docs/
│   ├── plan.md                     # 300 步实施计划
│   ├── design.md                   # 设计文档
│   └── steps/                      # 每个里程碑的教学文档
├── data/                           # 训练数据（.gitignore）
├── out/                            # 模型权重（.gitignore）
├── checkpoints/                    # 断点续训（.gitignore）
└── scripts/
    ├── download_data.py
    └── train.py
```

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 与 minimind 关系 | 独立重写，架构兼容 | 教学清晰；内部参数命名保持一致，`load_state_dict` 可直接加载 minimind 权重 |
| 配置类 | 继承 `PretrainedConfig` | 兼容 Transformers 生态 |
| Weight Tying | `True` | 与 Qwen3/minimind 对齐，减少参数量 |
| π 缩放 intermediate_size | `ceil(hidden_size * π / 64) * 64` | 对齐 64 倍数，提升 Tensor Core 利用率 |
| Attention 双路径 | Flash + 手动 KV Cache | Flash 快但不可 inspect；手动路径支持 KV Cache 推理 |
| LoRA 初始化 | A=高斯, B=零 | 初始输出为 0，不破坏预训练权重 |
| Checkpoint 写入 | `.tmp` → `os.replace` | 原子写入，防止断电损坏 |
| lr 调度 | Cosine with warmup | `train_epoch` 用 `cfg.learning_rate` 而非 optimizer 的 lr |
| Tokenizer 适配 | `wrap()` → `TokenizerAdapter` | raw `tokenizers.Tokenizer` 不可调用，必须 wrap |
| MoE 路由 | Top-1 + 空专家梯度 + 辅助损失 | 防止 expert collapse |

## 依赖 & 环境

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.14+ | |
| PyTorch | 2.7+ | CUDA 版本 |
| Transformers | 4.52+ | PretrainedConfig, PreTrainedModel |
| tokenizers | 0.21+ | BPE trainer |
| datasets | 3.6+ | 数据加载 |
| FastAPI | 0.115+ | API 服务 |
| pytest | 8.4+ | 测试 |

```bash
pip install -e ".[dev]"
```

**GPU 必需** — 测试默认使用 `device='cuda'`。

## 测试

```bash
# 全部测试
pytest

# 特定里程碑
pytest tests/m03_model_components/ -v

# 特定功能
pytest tests/m10_alignment/test_240_dpo.py -v

# 带覆盖率
pytest --cov=zllm --cov-report=term-missing
```

测试 fixtures（`tests/conftest.py`）：
- `small_config` — dim=64, 2 layers, 4 heads, 2 kv_heads, vocab=100（快速测试）
- `default_config` — dim=768（完整配置）
- `device` — `'cuda'` if available, else `'cpu'`

## 开发约定

- **TDD**：每步遵循 红（写测试）→ 绿（写实现）→ 重构
- **GPU-first**：CUDA 必需，测试使用 `device='cuda'`
- **Python 3.14+**，全部依赖使用最新版本
- **文档中文为主**，保留英文专业术语
- **每个里程碑配套教学文档** `docs/steps/NNN-name.md`
- **无注释** — 代码即文档，命名自解释
- **权重命名** `<type>_<dim>.pth`（如 `full_sft_768.pth`、`pretrain_768.pth`、`lora_medical_768.pth`）
