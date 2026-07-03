# 设计文档：从零训练大语言模型（教学书）

> 日期：2026-07-03
> 状态：已批准（待用户最终审阅）
> 关联项目：zllm（`/home/zw/zllm`）

## 1. 目标与定位

基于 zllm 项目（12 里程碑 / 300 步骤 / 428 测试 / 3142 行代码，覆盖 Tokenizer → Pretrain → SFT → LoRA → DPO → PPO → GRPO → Distillation → Agent RL → Serving 全流程）编写一本**完整的、可操作的、理论先行**的教学书。

读者读完本书后能够：
- 理解大语言模型背后的数学与深度学习理论；
- 按章节一步一步复现 zllm 项目，从环境搭建到训练出可对话的中文 LLM；
- 把书中每一步操作关联到项目中的实现代码与单元测试。

### 关键决策（已与用户确认）

| 维度 | 决策 |
|------|------|
| 输出位置与结构 | `docs/book/` 分章子目录，每章一个 `.md` + `README.md`/`SUMMARY.md` 索引 |
| 目标读者 | Python 开发者 / ML 初学者；每个概念从直觉讲起 + 数学推导 + 配图 |
| 代码/测试关联 | 核心摘录（5-20 行）+ `file_path:line` 完整引用；不内嵌整文件 |
| 图示来源 | Mermaid + LaTeX + ASCII 为主；外链图下载到本地 `docs/book/assets/` |
| 整体结构方案 | 方案 C：理论型教科书（厚重数学/理论篇在前，实战篇在后） |
| 拆分粒度 | 更细粒度（43 章 + 4 附录） |
| 章节模板 | 理论篇与实战篇用**不同模板** |

## 2. 整体章节框架（TOC）

共 **7 个部分（Part）+ 1 个序言 + 附录**，43 章 + 4 附录。理论部分 15 章（Part I–II），实战部分 28 章（Part III–VII），实战章与里程碑 1:1 映射，便于关联源码与测试。

### 第 0 部分 序言与导读
- **Ch 00 关于本书** —— 为什么写、给谁看、怎么读、前置要求、全书路线图

### 第 I 部分 数学基础（纯理论，8 章）
- **Ch 01 线性代数：向量与矩阵**
- **Ch 02 线性代数：分解与几何**（特征分解 / SVD / 投影 / 正交）
- **Ch 03 概率论基础**（分布 / 贝叶斯 / 条件概率）
- **Ch 04 统计推断**（期望 / 方差 / 协方差 / MLE / MAP）
- **Ch 05 信息论**（熵 / 交叉熵 / KL / JS / 互信息）← 导出 NTP 与 CE 损失
- **Ch 06 最优化基础**（凸优化 / 梯度下降 / SGD / 动量 / Adam）
- **Ch 07 微积分与链式法则**（雅可比 / 海森 / 反向传播的数学基础）
- **Ch 08 张量计算与 PyTorch 自动微分**（计算图 / autograd / GPU/CUDA）

### 第 II 部分 深度学习与 Transformer 理论（纯理论，7 章）
- **Ch 09 神经网络基础**（MLP / 激活 / 损失）
- **Ch 10 反向传播与训练动力学**（推导 / 初始化 / 正则化）
- **Ch 11 序列建模：从 RNN/LSTM 到瓶颈**
- **Ch 12 注意力机制**（对齐 → 自注意力，QKV 推导）
- **Ch 13 Transformer 架构详解**（原始论文逐模块）
- **Ch 14 解码策略理论**（greedy / beam / sampling，概率视角）
- **Ch 15 现代语言模型全景**（GPT 演进 / decoder-only / Scaling Law / 涌现）

### 第 III 部分 基石与分词（实战，M1 + M2，4 章）
- **Ch 16 项目初始化与开发环境（M1）** → `zllm/config.py` · `tests/m01_foundations`
- **Ch 17 分词理论：BPE/WordPiece/SentencePiece**
- **Ch 18 教学版 BPE 实现（M2-a）** → `zllm/tokenizer/bpe.py`
- **Ch 19 生产版 Tokenizer + 特殊 Token + Chat Template（M2-b）** → `zllm/tokenizer/{trainer,special_tokens,chat_template,adapter}.py` · `tests/m02_tokenizer`

### 第 IV 部分 模型架构（实战，M3 + M4，7 章）
- **Ch 20 RMSNorm 归一化（M3-a）** → `model/norms.py`
- **Ch 21 RoPE 旋转位置编码 + YaRN（M3-b）** → `model/rope.py`
- **Ch 22 GQA 注意力 + QK-Norm + KV Cache（M3-c）** → `model/attention.py`
- **Ch 23 SwiGLU 前馈网络（M4-a）** → `model/ffn.py`
- **Ch 24 MoE 混合专家（M4-b）** → `model/ffn.py`
- **Ch 25 Block + Backbone 组装（M4-c）** → `model/{block,backbone}.py`
- **Ch 26 CausalLM 头 + Weight Tying + Loss（M4-d）** → `model/causal_lm.py` · `tests/m03_model_components`、`tests/m04_model_assembly`

### 第 V 部分 数据与训练（实战，M5 + M6 + M7，6 章）
- **Ch 27 数据流水线总览与 TokenizerAdapter（M5-a）** → `dataset/utils.py`
- **Ch 28 五种 Dataset 实现（M5-b）** → `dataset/{pretrain,sft,dpo,rlaif,agent}.py` · `tests/m05_data_pipeline`
- **Ch 29 训练基础设施：种子/学习率/checkpoint（M6-a）** → `training/utils.py`
- **Ch 30 混合精度 AMP + 梯度累积 + GPU 优化（M6-b）** → `training/{amp,gpu}.py` · `tests/m06_training`
- **Ch 31 预训练：NTP 与训练循环（M7 理论+实现）** → `training/pretrain.py` · `tests/m07_pretrain`
- **Ch 32 预训练实战：数据准备/训练/loss 监控（M7 实操）**

### 第 VI 部分 微调与对齐（实战，M8-M11，8 章）
- **Ch 33 监督微调 SFT + Label Masking（M8）** → `training/full_sft.py` · `tests/m08_sft`
- **Ch 34 LoRA 低秩适配（M9）** → `model/lora.py`、`training/lora_sft.py` · `tests/m09_lora`
- **Ch 35 RLHF 框架与对齐总论（背景理论）**
- **Ch 36 DPO 直接偏好优化（M10-a）** → `training/dpo.py`
- **Ch 37 PPO + GAE + Critic（M10-b）** → `training/ppo.py`
- **Ch 38 GRPO + CISPO（M10-c）** → `training/grpo.py` · `tests/m10_alignment`
- **Ch 39 知识蒸馏（M11-a）** → `training/distillation.py`
- **Ch 40 Agent RL 工具调用（M11-b）** → `training/agent_rl.py` · `tests/m11_distill_agent`

### 第 VII 部分 推理与部署（实战，M12，3 章）
- **Ch 41 解码算法实现（greedy/temp/top-k/top-p/rep）** → `serving/generate.py`
- **Ch 42 KV Cache 加速推理** → `serving/generate.py` · `model/attention.py`
- **Ch 43 OpenAI 兼容 API + CLI 部署** → `serving/{api_server,cli}.py` · `tests/m12_serving`

### 附录
- **附录 A 命令速查** · **附录 B 超参数表** · **附录 C 术语表** · **附录 D 参考文献**

## 3. 章节模板

### 3.1 理论章模板（Part I–II，无代码，重推导）
1. 学习目标
2. 直觉与动机（生活化类比 + Mermaid 概念图）
3. 数学定义（LaTeX：定义 / 定理）
4. 推导（LaTeX 逐步推导，配几何图示）
5. 与 LLM / 本项目的联系（"这个概念将在 Ch22 用于…"）—— 钩子，前向引用实战章
6. 本章小结 + 思考题

### 3.2 实战章模板（Part III–VII，1:1 映射代码/测试）
1. 学习目标
2. 原理回顾（精简版，链接到对应理论章）
3. 代码实现（核心摘录 5-20 行 + `file_path:line` 完整引用）
4. 对应单元测试（测试意图说明 + `file_path:line`）
5. 动手验证（`pytest` 命令 + 预期输出）
6. 本章小结 + 下章预告

## 4. 目录结构与命名约定

```
docs/book/
├── README.md                    # 本书入口 + 完整目录（含进度勾选）
├── SUMMARY.md                   # 章节索引（链接到各章）
├── assets/                      # 所有图片（下载的外链图 + 自生成 SVG）
├── part-0-preface/ch00-about.md
├── part-1-math/ch01..ch08
├── part-2-dl-transformer/ch09..ch15
├── part-3-tokenizer/ch16..ch19
├── part-4-architecture/ch20..ch26
├── part-5-data-training/ch27..ch32
├── part-6-finetune-alignment/ch33..ch40
├── part-7-serving/ch41..ch43
└── appendices/appendix-a..d
```

- **文件名**：`ch<NN>-<kebab-name>.md`（NN 与里程碑步骤号尽量对齐，如 ch18 对应 steps 26-29）。
- **元数据**：每章文件头使用 YAML front-matter：
  ```yaml
  ---
  part: 4
  chapter: 22
  title: GQA 注意力 + QK-Norm + KV Cache
  milestone: M3
  source: zllm/model/attention.py
  tests: tests/m03_model_components/test_067_attention.py
  status: draft
  ---
  ```
- **子目录名**：`part-<N>-<kebab>`。

## 5. 格式规范

- **数学公式**：行内 `$...$`，块级 `$$...$$`，多步推导用 `\begin{aligned}...\end{aligned}` 对齐。
- **Mermaid**：流程图 `graph TD/LR`、时序图 `sequenceDiagram`、类图 `classDiagram`。
- **代码引用**：核心摘录用 ```python 围栏（5-20 行），紧跟引用行，例：
  > 完整实现见 `zllm/model/rope.py:17`
- **测试引用**：说明测试意图 + 路径行号，例：
  > 对应测试 `tests/m03_model_components/test_057_rope.py:13`
- **验证命令**：```bash pytest ... ``` + 预期输出块。
- **图片**：外链图下载到 `assets/`，正文用相对路径 `![alt](../assets/xxx.png)`。
- **术语**：首次出现中英对照（如"交叉熵（Cross-Entropy）"），附录 C 汇总。
- **语言**：中文为主，保留英文专业术语（与项目约定一致）。

## 6. 分阶段交付计划（7 个 Phase）

| Phase | 范围 | 产物 | 章数 |
|-------|------|------|------|
| 1 | Part 0 + I | 序言 + 数学基础 | 9 |
| 2 | Part II | DL/Transformer 理论 | 7 |
| 3 | Part III | 项目初始化 + Tokenizer | 4 |
| 4 | Part IV | 模型架构 | 7 |
| 5 | Part V | 数据与训练 | 6 |
| 6 | Part VI | 微调与对齐 | 8 |
| 7 | Part VII + 附录 | 推理部署 + 附录 | 7 |

每个 Phase 内部步骤：先搭骨架（各章 front-matter + 标题 + 学习目标）→ 逐章填充正文 → 交叉引用与图示校正。每个 Phase 完成后更新 `README.md` 进度勾选，并在本书中可独立审阅。

## 7. 范围与非目标

**范围内**：
- 7 部分的全部 43 章 + 4 附录正文。
- Mermaid/LaTeX/ASCII 图示与推导。
- 代码核心摘录 + file:line 引用。
- 外链图下载到 `assets/`。

**非目标（YAGNI）**：
- 不内嵌完整源码文件（用引用代替）。
- 不新增/修改 zllm 项目源码与测试（只读取引用）。
- 不构建 PDF/HTML 发布产物（仅维护 Markdown 源）。
- 不覆盖或替换现有 `docs/steps/`（本书是独立教学叙事）。

## 8. 验收标准

- 每章符合对应模板的 6 段结构。
- 实战章正确引用真实存在的 `file_path:line`（实现 + 测试），引用可通过 `grep`/文件读取校验。
- 每个 Phase 结束后 `README.md` 进度表更新。
- 数学公式 LaTeX 语法正确；Mermaid 图语法正确可渲染。
- 章节间前向/后向引用（"将在 Ch22…"/"见 Ch05"）一致，无悬空引用。

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 引用的 `file_path:line` 随源码变动失效 | 每个 Phase 校验引用；引用行号可放宽到文件级并在正文给出函数名锚点 |
| 理论篇过重导致读者流失 | 每章结尾"与本项目联系"钩子 + 可选"快速路径"提示 |
| 外链图片失效 | 一律下载到本地 `assets/`，正文只引用本地路径 |
| 内容量大难以一次性完成 | 7 Phase 分段交付，每 Phase 独立可审阅 |

## 10. 下一步

本设计批准并经用户审阅后，转入 **writing-plans** 技能，产出逐章生成的实施计划（按 7 个 Phase 拆解为可执行任务）。
