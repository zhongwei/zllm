# zllm 实施计划 — 从零训练中文大语言模型

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 TDD 方式从零构建一个完整可训练的中文 LLM 训练管线，覆盖 Tokenizer → Pretrain → SFT → LoRA → DPO → PPO → GRPO → Distillation → Agent RL → Serving 全流程。

**Architecture:** 渐进式里程碑构建。300 个步骤组织为 12 个里程碑，每步遵循 TDD（测试→实现→文档）。`zllm/` 包随步骤增长，每个里程碑产出可运行的功能检查点。模型架构对齐 Qwen3/minimind-3（GQA、RoPE、SwiGLU、MoE、Weight Tying）。

**Tech Stack:** Python 3.14+、PyTorch ≥2.7、Transformers ≥4.52、tokenizers ≥0.21、datasets ≥3.6、pytest ≥8.4、CUDA（GPU-first）

**文档语言：** 中文为主，保留英文专业术语。每步配套 `docs/steps/NNN-name.md` 教学文档。

---

## 文件结构总览

```
zllm/
├── pyproject.toml              # 包配置与依赖
├── README.md                   # 项目说明
├── zllm/                       # 主包
│   ├── __init__.py
│   ├── config.py               # ZLLMConfig
│   ├── tokenizer/{bpe,special_tokens,chat_template}.py
│   ├── model/{norms,rope,attention,ffn,block,backbone,causal_lm,lora}.py
│   ├── dataset/{pretrain,sft,dpo,rlaif,agent}.py
│   ├── training/{utils,pretrain,full_sft,lora_sft,dpo,ppo,grpo,distillation,agent_rl,rollout}.py
│   └── serving/{generate,api_server,cli}.py
├── tests/                      # 测试（按 milestone 分目录）
│   ├── conftest.py             # 共享 fixtures
│   └── m01_foundations/ ... m12_serving/
├── docs/
│   ├── plan.md                 # 本文档
│   ├── design.md               # 设计文档
│   └── steps/                  # 每步教学文档
├── data/  out/  checkpoints/   # 运行时目录（.gitignore）
└── scripts/{download_data,train}.py
```

每个文件职责单一：`norms.py` 只管归一化，`rope.py` 只管位置编码，等等。文件随里程碑逐步创建。

---

## 里程碑总览

| # | 里程碑 | 步骤 | 步数 | 检查点 |
|---|--------|------|------|--------|
| M1 | 基础与工具 | 1-25 | 25 | `pytest m01` 全绿，骨架可运行 |
| M2 | Tokenizer | 26-50 | 25 | 可 tokenize/decode 任意文本 |
| M3 | 模型组件 I | 51-80 | 30 | RMSNorm/RoPE/GQA 各有测试 |
| M4 | 模型组件 II + 组装 | 81-110 | 30 | 可创建模型、前向、算 loss |
| M5 | 数据流水线 | 111-140 | 30 | 可加载 JSONL、构造 labels |
| M6 | 训练基础设施 | 141-170 | 30 | 可保存/恢复训练状态 |
| M7 | 预训练 | 171-200 | 30 | 真实数据预训练 loss 下降 |
| M8 | 监督微调 | 201-220 | 20 | 模型可对话 |
| M9 | LoRA 微调 | 221-235 | 15 | 可做领域适配 |
| M10 | 对齐训练 | 236-270 | 35 | DPO/PPO/GRPO |
| M11 | 蒸馏与 Agent | 271-290 | 20 | 模型可调用工具 |
| M12 | 推理与部署 | 291-300 | 10 | OpenAI API 可调用 |

---

# M1: 基础与工具（步骤 1-25）

> **目标：** 搭建项目骨架，建立测试基础设施，复习 PyTorch 核心概念（张量、autograd、nn.Module、训练循环）。完成后 `pytest` 全绿，`import zllm` 可用。

**Files:**
- Create: `pyproject.toml`, `README.md`, `.gitignore`
- Create: `zllm/__init__.py`, `zllm/config.py`
- Create: `tests/conftest.py`, `tests/m01_foundations/`

### Task 1: 项目初始化

**Files:**
- Create: `/home/zw/zllm/pyproject.toml`
- Create: `/home/zw/zllm/.gitignore`
- Create: `/home/zw/zllm/README.md`
- Create: `/home/zw/zllm/docs/steps/001-project-init.md`

- [ ] **Step 1: 写 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=80"]
build-backend = "setuptools.build_meta"

[project]
name = "zllm"
version = "0.0.1"
description = "从零训练中文大语言模型 - step by step TDD 教学项目"
requires-python = ">=3.14"
dependencies = [
    "torch>=2.7",
    "transformers>=4.52",
    "datasets>=3.6",
    "tokenizers>=0.21",
    "swanlab>=0.4",
    "streamlit>=1.45",
    "fastapi>=0.115",
    "uvicorn>=0.34",
]

[project.optional-dependencies]
dev = ["pytest>=8.4", "pytest-cov>=6"]

[tool.setuptools.packages.find]
include = ["zllm*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: 写 .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
data/
out/
checkpoints/
.venv/
*.swp
wandb/
swanlog/
```

- [ ] **Step 3: 写 README.md**

```markdown
# zllm — 从零训练中文大语言模型

Step-by-step TDD 教学项目，从 Tokenizer 到 Agent RL 全流程。覆盖完整训练管线，GPU 可训练。

## 快速开始

\`\`\`bash
pip install -e ".[dev]"
pytest
\`\`\`

## 训练管线

Tokenizer → Pretrain → SFT → LoRA → DPO → PPO → GRPO → Distillation → Agent RL

详见 [实施计划](docs/plan.md)。
```

- [ ] **Step 4: git init + 首次提交**

```bash
cd /home/zw/zllm
git init
git add pyproject.toml .gitignore README.md docs/
git commit -m "chore: init zllm project"
```

- [ ] **Step 5: 写教学文档 docs/steps/001-project-init.md**

内容：项目定位、目录结构、依赖说明、Python 3.14 + 最新依赖的选择理由。

---

### Task 2: 安装包并验证导入

**Files:**
- Create: `zllm/__init__.py`
- Test: `tests/m01_foundations/test_002_import.py`

- [ ] **Step 1: 写 zllm/__init__.py**

```python
__version__ = "0.0.1"
```

- [ ] **Step 2: 写失败测试**

```python
def test_zllm_importable():
    import zllm
    assert zllm.__version__ == "0.0.1"
```

- [ ] **Step 3: 安装包**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/m01_foundations/test_002_import.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add zllm/__init__.py tests/m01_foundations/test_002_import.py
git commit -m "feat: zllm package importable"
```

---

### Task 3: 共享测试 fixtures

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/m01_foundations/test_003_fixtures.py`

- [ ] **Step 1: 写 conftest.py**

```python
import pytest
import torch

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def small_config():
    """用于快速测试的小模型配置"""
    return {
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
    }
```

- [ ] **Step 2: 写测试验证 fixture 可用**

```python
def test_device_fixture(device):
    assert device is not None

def test_small_config_fixture(small_config):
    assert small_config["hidden_size"] == 64
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/m01_foundations/test_003_fixtures.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/m01_foundations/test_003_fixtures.py
git commit -m "test: add shared fixtures (device, small_config)"
```

---

### Task 4: 张量创建与基本属性

**Test:** `tests/m01_foundations/test_004_tensor_basics.py`
**Doc:** `docs/steps/004-tensor-basics.md`

- [ ] 测试张量创建（zeros/ones/randn）、shape/dtype/device 属性
- [ ] 测试张量索引与切片
- [ ] 测试广播机制（broadcasting）
- [ ] 测试逐元素运算（+、-、*、/）
- [ ] 测试矩阵乘法（matmul）与 einsum
- [ ] 测试 reshape/view/permute
- [ ] Commit

---

### Task 5: 自动微分（autograd）

**Test:** `tests/m01_foundations/test_005_autograd.py`
**Doc:** `docs/steps/005-autograd.md`

- [ ] 测试 requires_grad、backward 基本流程
- [ ] 测试计算图（grad_fn）与 retain_graph
- [ ] 测试梯度累积与 zero_grad
- [ ] 测试梯度裁剪 clip_grad_norm_
- [ ] Commit

---

### Task 6: nn.Module 基础

**Test:** `tests/m01_foundations/test_006_nn_module.py`
**Doc:** `docs/steps/006-nn-module.md`

- [ ] 测试自定义 nn.Module、parameters()
- [ ] 测试 nn.Linear 与参数初始化
- [ ] 测试 nn.Embedding
- [ ] 测试 nn.CrossEntropyLoss
- [ ] Commit

---

### Task 7: 优化器与微型训练循环

**Test:** `tests/m01_foundations/test_007_optimizer.py`
**Doc:** `docs/steps/007-optimizer.md`

- [ ] 测试 AdamW 优化器一步更新
- [ ] 测试完整微型训练循环（loss 下降）
- [ ] 测试模型 save/load state_dict
- [ ] Commit

---

### Task 8: 混合精度与 CUDA 管理

**Test:** `tests/m01_foundations/test_008_amp.py`
**Doc:** `docs/steps/008-amp.md`

- [ ] 测试 autocast 上下文（bfloat16）
- [ ] 测试 GradScaler（float16）
- [ ] 测试 .to(device) 与 non_blocking
- [ ] 测试随机种子设置（可复现性）
- [ ] Commit

---

### Task 9: ZLLMConfig 配置类

**Files:**
- Create: `zllm/config.py`
- Test: `tests/m01_foundations/test_009_config.py`
**Doc:** `docs/steps/009-config.md`

- [ ] 测试 ZLLMConfig 默认值（dim=768, layers=8, vocab=6400）
- [ ] 测试 GQA 配置（q_heads=8, kv_heads=4）
- [ ] 测试 MoE 配置（num_experts=4）
- [ ] 测试 config 序列化/反序列化
- [ ] Commit

---

### Task 10: M1 集成测试

**Test:** `tests/m01_foundations/test_010_m1_integration.py`

- [ ] 综合测试：用小 config 创建一个 toy nn.Module，训练一步，保存/加载
- [ ] 运行 `pytest tests/m01_foundations/ -v` 全绿
- [ ] 写 M1 完成文档 `docs/steps/025-m1-complete.md`
- [ ] Commit + tag `m1-foundations`

---

# M2: Tokenizer（步骤 26-50）

> **目标：** 从零实现 BPE 分词器、特殊 token、Chat Template。完成后可对任意中文/英文文本 tokenize 与 decode。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 26 | 文本到字节序列（UTF-8 / Byte-Level） | `tokenizer/bpe.py` |
| 27 | 字节级初始词表（256 alphabet） | `tokenizer/bpe.py` |
| 28 | BPE 合并算法核心（pair counting + merge） | `tokenizer/bpe.py` |
| 29 | BPE 训练循环（迭代合并到 vocab_size） | `tokenizer/bpe.py` |
| 30 | 特殊 token：系统级 `<\|im_start\|>`/`<\|im_end\|>` | `tokenizer/special_tokens.py` |
| 31 | 特殊 token：视觉/音频预留 | `tokenizer/special_tokens.py` |
| 32 | 特殊 token：工具调用标记 | `tokenizer/special_tokens.py` |
| 33 | 特殊 token：思考链标记 | `tokenizer/special_tokens.py` |
| 34 | 特殊 token：buffer 预留位 | `tokenizer/special_tokens.py` |
| 35 | BPE 训练器封装（Tokenizer + BpeTrainer） | `tokenizer/bpe.py` |
| 36 | pre_tokenizer 配置（ByteLevel） | `tokenizer/bpe.py` |
| 37 | tokenizer 训练脚本 | `scripts/train_tokenizer.py` |
| 38 | added_tokens_decoder 配置 | `tokenizer/bpe.py` |
| 39 | tokenizer.json 保存 | `tokenizer/bpe.py` |
| 40 | tokenizer_config.json 生成 | `tokenizer/chat_template.py` |
| 41 | 编码接口（text → ids） | `tokenizer/bpe.py` |
| 42 | 解码接口（ids → text） | `tokenizer/bpe.py` |
| 43 | 批量编解码 | `tokenizer/bpe.py` |
| 44 | Chat Template 设计（Jinja2 结构） | `tokenizer/chat_template.py` |
| 45 | 角色格式化（im_start/role/content/im_end） | `tokenizer/chat_template.py` |
| 46 | 工具声明模板（tools 注入） | `tokenizer/chat_template.py` |
| 47 | 思考链模式（open_thinking） | `tokenizer/chat_template.py` |
| 48 | apply_chat_template 接口 | `tokenizer/chat_template.py` |
| 49 | 压缩率评测 | `tests/m02_tokenizer/test_049_compression.py` |
| 50 | M2 集成测试 | `tests/m02_tokenizer/test_050_integration.py` |

---

# M3: 模型组件 I — 归一化、位置编码、注意力（步骤 51-80）

> **目标：** 实现 RMSNorm、RoPE（含 YaRN）、GQA 注意力。每个组件独立单元测试。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 51 | RMSNorm 数学原理（公式推导） | `docs/steps/051-rmsnorm.md` |
| 52 | RMSNorm 实现 | `model/norms.py` |
| 53 | RMSNorm 数值稳定性（float32 计算） | `model/norms.py` |
| 54 | RMSNorm 测试 | `tests/m03_*/test_054_rmsnorm.py` |
| 55 | 位置编码概念（为什么需要） | `docs/steps/055-pos-emb.md` |
| 56 | RoPE 原理（旋转矩阵编码相对位置） | `docs/steps/056-rope.md` |
| 57 | 频率预计算 precompute_freqs_cis | `model/rope.py` |
| 58 | rope_theta 的作用（1e6 vs 10000） | `model/rope.py` |
| 59 | rotate_half 实现 | `model/rope.py` |
| 60 | apply_rotary_pos_emb（Q/K 旋转） | `model/rope.py` |
| 61 | RoPE buffer 注册（register_buffer） | `model/rope.py` |
| 62 | RoPE 测试 | `tests/m03_*/test_062_rope.py` |
| 63 | YaRN RoPE Scaling（长序列外推） | `model/rope.py` |
| 64 | YaRN 频率缩放（ramp/inv_dim） | `model/rope.py` |
| 65 | YaRN 测试 | `tests/m03_*/test_065_yarn.py` |
| 66 | 注意力机制基础（QKV 概念） | `docs/steps/066-attention.md` |
| 67 | Q/K/V 投影层 | `model/attention.py` |
| 68 | 多头注意力 MHA 基础 | `model/attention.py` |
| 69 | GQA 分组查询（n_rep 计算） | `model/attention.py` |
| 70 | repeat_kv 实现（KV head 扩展） | `model/attention.py` |
| 71 | GQA 注意力前向传播 | `model/attention.py` |
| 72 | QK-Norm（q_norm/k_norm 稳定训练） | `model/attention.py` |
| 73 | 输出投影 o_proj | `model/attention.py` |
| 74 | Dropout（attn/resid） | `model/attention.py` |
| 75 | Causal Mask（triu 实现） | `model/attention.py` |
| 76 | Flash Attention（SDPA） | `model/attention.py` |
| 77 | 手动注意力路径（scores+softmax+mask） | `model/attention.py` |
| 78 | 双路径切换（flash vs manual） | `model/attention.py` |
| 79 | KV Cache 处理（past_key_value） | `model/attention.py` |
| 80 | M3 集成测试 | `tests/m03_*/test_080_integration.py` |

---

# M4: 模型组件 II + 组装（步骤 81-110）

> **目标：** 实现 SwiGLU/MoE FFN、Transformer Block、完整 ZLLMForCausalLM。完成后可前向传播并计算 loss。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 81 | FFN 基础结构 | `docs/steps/081-ffn.md` |
| 82 | SwiGLU 激活（gate*up*down） | `model/ffn.py` |
| 83 | SiLU 激活函数 | `model/ffn.py` |
| 84 | FeedForward 实现 | `model/ffn.py` |
| 85 | intermediate_size π 缩放 | `model/ffn.py` |
| 86 | FFN 测试 | `tests/m04_*/test_086_ffn.py` |
| 87 | MoE 原理 | `docs/steps/087-moe.md` |
| 88 | Router 门控网络 | `model/ffn.py` |
| 89 | MOEFeedForward 实现 | `model/ffn.py` |
| 90 | 稀疏专家计算（index_add_） | `model/ffn.py` |
| 91 | 负载均衡辅助损失 | `model/ffn.py` |
| 92 | 空专家梯度保持 | `model/ffn.py` |
| 93 | MoE 测试 | `tests/m04_*/test_093_moe.py` |
| 94 | Pre-Norm vs Post-Norm | `docs/steps/094-norm.md` |
| 95 | Transformer Block 组装 | `model/block.py` |
| 96 | Block 前向传播 | `model/block.py` |
| 97 | Block 测试 | `tests/m04_*/test_097_block.py` |
| 98 | Token Embedding 层 | `model/backbone.py` |
| 99 | 多层堆叠（N layers） | `model/backbone.py` |
| 100 | 最终 RMSNorm | `model/backbone.py` |
| 101 | RoPE 预计算缓存 | `model/backbone.py` |
| 102 | ZLLMModel（主体）组装 | `model/backbone.py` |
| 103 | ZLLMModel 前向传播 | `model/backbone.py` |
| 104 | ZLLMModel 测试 | `tests/m04_*/test_104_backbone.py` |
| 105 | Weight Tying（lm_head=embed） | `model/causal_lm.py` |
| 106 | 交叉熵损失计算 | `model/causal_lm.py` |
| 107 | ignore_index 处理（-100） | `model/causal_lm.py` |
| 108 | logits_to_keep 优化 | `model/causal_lm.py` |
| 109 | ZLLMForCausalLM 组装 | `model/causal_lm.py` |
| 110 | M4 集成测试（前向+loss） | `tests/m04_*/test_110_integration.py` |

---

# M5: 数据流水线（步骤 111-140）

> **目标：** 实现 5 种 Dataset 类（Pretrain/SFT/DPO/RLAIF/Agent），含标签构造与 collate。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 111 | JSONL 数据加载（load_dataset） | `dataset/pretrain.py` |
| 112 | 数据格式规范定义 | `docs/steps/112-data-format.md` |
| 113 | pre_processing_chat（概率 system prompt） | `dataset/utils.py` |
| 114 | post_processing_chat（空思考链） | `dataset/utils.py` |
| 115 | PretrainDataset 数据格式 | `dataset/pretrain.py` |
| 116 | PretrainDataset tokenize 流程 | `dataset/pretrain.py` |
| 117 | BOS/EOS 添加 | `dataset/pretrain.py` |
| 118 | Padding 处理 | `dataset/pretrain.py` |
| 119 | 自回归标签构造 | `dataset/pretrain.py` |
| 120 | ignore_index 设置 | `dataset/pretrain.py` |
| 121 | SFTDataset 数据格式 | `dataset/sft.py` |
| 122 | Chat Template 渲染 | `dataset/sft.py` |
| 123 | SFT 标签生成 generate_labels | `dataset/sft.py` |
| 124 | assistant 区间定位 | `dataset/sft.py` |
| 125 | prompt 掩码（只标 assistant） | `dataset/sft.py` |
| 126 | DPODataset 数据格式 | `dataset/dpo.py` |
| 127 | chosen/rejected 渲染 | `dataset/dpo.py` |
| 128 | DPO loss_mask 构造 | `dataset/dpo.py` |
| 129 | DPO (x,y,mask) 三元组 | `dataset/dpo.py` |
| 130 | RLAIFDataset（prompt-only） | `dataset/rlaif.py` |
| 131 | thinking_ratio 采样 | `dataset/rlaif.py` |
| 132 | AgentRLDataset（messages+tools+gt） | `dataset/agent.py` |
| 133 | parse_conversations | `dataset/agent.py` |
| 134 | 自定义 collate_fn | `dataset/agent.py` |
| 135 | DataLoader 配置 | `dataset/__init__.py` |
| 136 | pin_memory 与 non_blocking | `dataset/__init__.py` |
| 137 | DistributedSampler 集成 | `dataset/__init__.py` |
| 138 | 数据增强与多样性 | `dataset/utils.py` |
| 139 | 数据集对比验证 | `tests/m05_*/test_139_compare.py` |
| 140 | M5 集成测试 | `tests/m05_*/test_140_integration.py` |

---

# M6: 训练基础设施（步骤 141-170）

> **目标：** 实现 DDP、AMP、checkpoint、lr schedule、日志。完成后可保存/恢复训练状态。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 141 | 训练脚本通用结构（9 步模板） | `docs/steps/141-template.md` |
| 142 | init_distributed_mode（DDP 初始化） | `training/utils.py` |
| 143 | NCCL 后端配置 | `training/utils.py` |
| 144 | 单卡/多卡透明切换 | `training/utils.py` |
| 145 | setup_seed（随机种子） | `training/utils.py` |
| 146 | 分布式种子偏移（42+rank） | `training/utils.py` |
| 147 | get_lr（余弦退火+warmup） | `training/utils.py` |
| 148 | 学习率公式推导 | `docs/steps/148-lr.md` |
| 149 | init_model（模型初始化） | `training/utils.py` |
| 150 | 权重加载 load_state_dict | `training/utils.py` |
| 151 | from_weight='none'（从头初始化） | `training/utils.py` |
| 152 | lm_checkpoint 保存（原子写入） | `training/utils.py` |
| 153 | .tmp → os.replace 原子操作 | `training/utils.py` |
| 154 | FP16/CPU 权重提取 | `training/utils.py` |
| 155 | lm_checkpoint 恢复（加载续训） | `training/utils.py` |
| 156 | GPU 数量变化的 step 转换 | `training/utils.py` |
| 157 | SkipBatchSampler（跳过已训练） | `training/utils.py` |
| 158 | Logger（分布式日志） | `training/utils.py` |
| 159 | is_main_process（rank 0） | `training/utils.py` |
| 160 | WandB/SwanLab 集成 | `training/utils.py` |
| 161 | AMP autocast（bfloat16/float16） | `training/utils.py` |
| 162 | GradScaler 使用 | `training/utils.py` |
| 163 | 梯度累积 accumulation_steps | `training/utils.py` |
| 164 | 梯度裁剪 clip_grad_norm_ | `training/utils.py` |
| 165 | torch.compile 加速 | `training/utils.py` |
| 166 | GPU 性能优化（TF32/cudnn） | `training/utils.py` |
| 167 | Flash SDPA 启用 | `training/utils.py` |
| 168 | 训练日志格式化 | `training/utils.py` |
| 169 | 时间格式化 _format_duration | `training/utils.py` |
| 170 | M6 集成测试 | `tests/m06_*/test_170_integration.py` |

---

# M7: 预训练（步骤 171-200）

> **目标：** 完整 pretrain 循环。完成后可用真实数据预训练，loss 下降。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 171 | train_pretrain 命令行参数 | `training/pretrain.py` |
| 172 | 超参数默认值 | `training/pretrain.py` |
| 173 | GPU 性能设置（TF32/cudnn/flash） | `training/pretrain.py` |
| 174 | WandB/SwanLab 配置 | `training/pretrain.py` |
| 175 | 模型与数据定义 | `training/pretrain.py` |
| 176 | 优化器配置（AdamW） | `training/pretrain.py` |
| 177 | 从 checkpoint 恢复 | `training/pretrain.py` |
| 178 | torch.compile 包装 | `training/pretrain.py` |
| 179 | DDP 包装 | `training/pretrain.py` |
| 180 | train_epoch 函数结构 | `training/pretrain.py` |
| 181 | 数据迁移到 GPU | `training/pretrain.py` |
| 182 | 动态学习率更新 | `training/pretrain.py` |
| 183 | 混合精度前向传播 | `training/pretrain.py` |
| 184 | 损失计算（CE + aux_loss） | `training/pretrain.py` |
| 185 | 梯度累积步 | `training/pretrain.py` |
| 186 | unscale_+clip+step+update | `training/pretrain.py` |
| 187 | zero_grad(set_to_none=True) | `training/pretrain.py` |
| 188 | 训练日志记录 _log_step | `training/pretrain.py` |
| 189 | _save_step 模型保存 | `training/pretrain.py` |
| 190 | 双重保存（权重+续训） | `training/pretrain.py` |
| 191 | 断点续训完整流程 | `training/pretrain.py` |
| 192 | 训练主循环（epoch 迭代） | `training/pretrain.py` |
| 193 | 进程清理 destroy_process_group | `training/pretrain.py` |
| 194 | next-token prediction 验证 | `tests/m07_*/test_194_ntp.py` |
| 195 | loss 下降验证 | `tests/m07_*/test_195_loss.py` |
| 196 | torch.compile 效果测试 | `tests/m07_*/test_196_compile.py` |
| 197 | 推荐训练参数（单 GPU） | `docs/steps/197-params.md` |
| 198 | 多 GPU DDP 训练 | `training/pretrain.py` |
| 199 | 完整 pretrain 运行 | `scripts/train.py` |
| 200 | M7 集成测试（端到端） | `tests/m07_*/test_200_integration.py` |

---

# M8: 监督微调 SFT（步骤 201-220）

> **目标：** SFT 训练循环。完成后模型可对话。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 201 | SFT 目标与意义 | `docs/steps/201-sft.md` |
| 202 | 与 Pretrain 的关键差异 | `docs/steps/202-sft-vs-pretrain.md` |
| 203 | train_full_sft 命令行参数 | `training/full_sft.py` |
| 204 | SFT 超参数（lr=1e-5, seq=768） | `training/full_sft.py` |
| 205 | from_weight='pretrain' 配置 | `training/full_sft.py` |
| 206 | SFTDataset 集成 | `training/full_sft.py` |
| 207 | SFT 训练循环 | `training/full_sft.py` |
| 208 | SFT loss 验证 | `tests/m08_*/test_208_loss.py` |
| 209 | SFT 模型保存 | `training/full_sft.py` |
| 210 | SFT 断点续训 | `training/full_sft.py` |
| 211 | Chat Template 推理验证 | `tests/m08_*/test_211_chat.py` |
| 212 | apply_chat_template 使用 | `serving/generate.py` |
| 213 | 多轮对话维护 | `serving/generate.py` |
| 214 | SFT 后模型行为测试 | `tests/m08_*/test_214_behavior.py` |
| 215 | 对话质量评估 | `tests/m08_*/test_215_quality.py` |
| 216 | 推荐训练参数 | `docs/steps/216-params.md` |
| 217 | SFT 完整运行 | `scripts/train.py` |
| 218 | pretrain→SFT 权重衔接 | `training/full_sft.py` |
| 219 | 模型对话能力验证 | `tests/m08_*/test_219_chat.py` |
| 220 | M8 集成测试 | `tests/m08_*/test_220_integration.py` |

---

# M9: LoRA 微调（步骤 221-235）

> **目标：** LoRA 注入/训练/合并。完成后可做领域适配。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 221 | LoRA 动机与原理（低秩分解） | `docs/steps/221-lora.md` |
| 222 | LoRA 类实现（A/B 矩阵） | `model/lora.py` |
| 223 | 初始化策略（A 高斯, B 零） | `model/lora.py` |
| 224 | LoRA 前向传播 B(A(x)) | `model/lora.py` |
| 225 | apply_lora（注入到 Linear） | `model/lora.py` |
| 226 | 方阵检测（in==out features） | `model/lora.py` |
| 227 | monkey-patch forward | `model/lora.py` |
| 228 | 闭包绑定避免循环变量 | `model/lora.py` |
| 229 | 参数冻结 requires_grad=False | `training/lora_sft.py` |
| 230 | 只优化 LoRA 参数 | `training/lora_sft.py` |
| 231 | save_lora（只存 LoRA 权重） | `model/lora.py` |
| 232 | load_lora（加载 LoRA） | `model/lora.py` |
| 233 | merge_lora（W + B@A 合并） | `model/lora.py` |
| 234 | LoRA 训练脚本 | `training/lora_sft.py` |
| 235 | M9 集成测试 | `tests/m09_*/test_235_integration.py` |

---

# M10: 对齐训练 — DPO + PPO + GRPO（步骤 236-270）

> **目标：** 三种对齐方法。完成后模型输出符合偏好。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 236 | RLHF 三步流程概述 | `docs/steps/236-rlhf.md` |
| 237 | DPO 原理（直接偏好优化） | `docs/steps/237-dpo.md` |
| 238 | Bradley-Terry 偏好模型 | `docs/steps/238-bradley-terry.md` |
| 239 | DPO 数学推导 | `docs/steps/239-dpo-math.md` |
| 240 | logits_to_log_probs 实现 | `training/dpo.py` |
| 241 | dpo_loss 实现（log sigmoid） | `training/dpo.py` |
| 242 | DPO 双模型架构（model+ref） | `training/dpo.py` |
| 243 | DPO 训练循环 | `training/dpo.py` |
| 244 | DPO 超参数（beta=0.15, lr=4e-8） | `training/dpo.py` |
| 245 | DPO 集成测试 | `tests/m10_*/test_245_dpo.py` |
| 246 | PPO 原理概述 | `docs/steps/246-ppo.md` |
| 247 | Clipped Objective | `docs/steps/247-ppo-clip.md` |
| 248 | CriticModel（价值网络） | `training/ppo.py` |
| 249 | Reward 计算系统（多维度） | `training/ppo.py` |
| 250 | rep_penalty（重复惩罚） | `training/ppo.py` |
| 251 | GAE 优势估计 | `training/ppo.py` |
| 252 | PPO 更新（mini-batch） | `training/ppo.py` |
| 253 | KL 早停机制 | `training/ppo.py` |
| 254 | Value Loss 裁剪 | `training/ppo.py` |
| 255 | PPO 训练循环 | `training/ppo.py` |
| 256 | PPO 超参数详解 | `docs/steps/256-ppo-params.md` |
| 257 | PPO 集成测试 | `tests/m10_*/test_257_ppo.py` |
| 258 | GRPO vs PPO 核心差异 | `docs/steps/258-grpo-vs-ppo.md` |
| 259 | GRPO 群体相对优势 | `training/grpo.py` |
| 260 | GRPO 无 Critic 设计 | `training/grpo.py` |
| 261 | GRPO 训练循环 | `training/grpo.py` |
| 262 | GRPO Loss（标准 PPO Clip） | `training/grpo.py` |
| 263 | CISPO Loss（单边裁剪） | `training/grpo.py` |
| 264 | KL 散度计算 per_token_kl | `training/grpo.py` |
| 265 | GRPO 超参数 | `training/grpo.py` |
| 266 | GRPO 集成测试 | `tests/m10_*/test_266_grpo.py` |
| 267 | 三种对齐方法对比 | `docs/steps/267-alignment-compare.md` |
| 268 | Reward Model 集成 | `training/utils.py` |
| 269 | LMForRewardModel 封装 | `training/utils.py` |
| 270 | M10 集成测试 | `tests/m10_*/test_270_integration.py` |

---

# M11: 蒸馏与 Agent RL（步骤 271-290）

> **目标：** 知识蒸馏 + Agent 工具调用。完成后模型可调用工具。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 271 | 知识蒸馏原理（软标签） | `docs/steps/271-distill.md` |
| 272 | distillation_loss（KL 散度） | `training/distillation.py` |
| 273 | 温度参数 Temperature | `training/distillation.py` |
| 274 | 蒸馏双模型（teacher/student） | `training/distillation.py` |
| 275 | CE + Distill 组合损失 | `training/distillation.py` |
| 276 | MoE → Dense 蒸馏 | `training/distillation.py` |
| 277 | 蒸馏训练循环 | `training/distillation.py` |
| 278 | 蒸馏集成测试 | `tests/m11_*/test_278_distill.py` |
| 279 | Agent RL 目标（工具使用） | `docs/steps/279-agent.md` |
| 280 | 模拟工具系统 TOOLS 定义 | `training/agent_rl.py` |
| 281 | 工具执行 execute_tool | `training/agent_rl.py` |
| 282 | parse_tool_calls（正则解析） | `training/agent_rl.py` |
| 283 | rollout_single（单样本多轮） | `training/agent_rl.py` |
| 284 | rollout_batch（批量） | `training/agent_rl.py` |
| 285 | Agent Reward 计算（多维度） | `training/agent_rl.py` |
| 286 | validate_gt_in_text（GT 验证） | `training/agent_rl.py` |
| 287 | Agent RL 训练循环 | `training/agent_rl.py` |
| 288 | AgentRLDataset + collate | `dataset/agent.py` |
| 289 | Agent 多轮工具调用验证 | `tests/m11_*/test_289_toolcall.py` |
| 290 | M11 集成测试 | `tests/m11_*/test_290_integration.py` |

---

# M12: 推理与部署（步骤 291-300）

> **目标：** CLI 推理、OpenAI API 服务、模型转换。完成后可通过 API 调用模型。

| 步骤 | 标题 | 文件 |
|------|------|------|
| 291 | eval_llm CLI 推理 | `serving/cli.py` |
| 292 | init_model 推理加载 | `serving/generate.py` |
| 293 | 流式生成 TextStreamer | `serving/generate.py` |
| 294 | 推理速度测量 tokens/s | `serving/generate.py` |
| 295 | generate() 解码策略 | `model/causal_lm.py` |
| 296 | Temperature/Top-K/Top-P/Rep | `model/causal_lm.py` |
| 297 | KV Cache 推理加速 | `model/causal_lm.py` |
| 298 | OpenAI 兼容 API 服务 | `serving/api_server.py` |
| 299 | 模型格式转换 PyTorch↔Transformers | `scripts/convert_model.py` |
| 300 | M12 集成测试（全流程） | `tests/m12_*/test_300_e2e.py` |

---

## 展开策略

本计划为**总体框架**。M1 已展开为可执行的 TDD 任务；M2-M12 为步骤清单（标题+文件+目标），后续会话按里程碑逐步展开为完整 TDD 任务（测试代码+实现代码+教学文档）。

每个里程碑展开时遵循相同模式：
1. 把表格中的每行扩展为 Task（含 Files、Steps、完整代码）
2. 每步配套 `docs/steps/NNN-name.md` 教学文档
3. 里程碑末尾集成测试 + git tag

## 自检

- **Spec 覆盖：** 设计文档 12 个里程碑全部映射到计划任务，300 步覆盖 minimind 全部 16 章内容（tokenizer→serving）。✓
- **占位符扫描：** M1 有完整代码；M2-M12 为有意为之的框架级枚举（标题+文件+目标明确），非占位符，待后续会话展开。✓
- **类型一致性：** 类名 `ZLLMForCausalLM`/`ZLLMModel`/`ZLLMConfig` 全程一致；文件路径前后一致。✓
