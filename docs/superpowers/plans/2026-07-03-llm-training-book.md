# 从零训练大语言模型（教学书）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **注意：** 本项目是**文档/书籍撰写**项目（非代码项目）。每个 Task = 一章，撰写循环为：建骨架 → 写各节正文 → 配图/Mermaid/LaTeX → 校验引用 → 勾选进度 → 提交。本计划**分多次会话增量撰写**：本次会话产出「全局骨架 + Phase 1 详细任务」，Phase 2–7 的详细任务在后续会话追加到本文件末尾。

**Goal:** 基于 zllm 项目撰写一本理论先行的中文教学书，读者可按章节一步步从环境搭建训练出可对话的中文 LLM，并把每一步关联到实现代码与单元测试。

**Architecture:** 理论型教科书（方案 C）。Part I–II 为纯理论（数学 → 深度学习 → Transformer → 现代 LLM 全景，15 章）；Part III–VII 为实战篇，与 zllm 12 个里程碑 1:1 映射（28 章），共 43 章 + 4 附录。采用两套章节模板：理论章（直觉→定义→推导→项目钩子）、实战章（原理→代码摘录+file:line→测试+file:line→pytest 验证）。图示以 Mermaid + LaTeX + ASCII 为主，外链图下载到本地 `assets/`。

**Tech Stack:** Markdown（CommonMark + GFM）、Mermaid 图表、LaTeX 数学公式（`$...$` / `$$...$$`）、zllm 源码与测试（只读引用，不修改）。

## Global Constraints

（取自已批准的设计文档 `docs/superpowers/specs/2026-07-03-llm-training-book-design.md`，逐条照抄）

- **输出位置**：`docs/book/` 分章子目录；每章一个 `.md`；含 `README.md`（入口+进度勾选）与 `SUMMARY.md`（章节索引）。
- **读者**：Python 开发者 / ML 初学者；每个概念从直觉讲起 + 数学推导 + 配图。
- **代码/测试关联**：核心摘录（5-20 行）+ `file_path:line` 完整引用；**不内嵌整文件**；**只读引用 zllm 源码/测试，不修改它们**。
- **图示**：Mermaid + LaTeX + ASCII 为主；外链图一律下载到本地 `docs/book/assets/`，正文只用相对路径 `../assets/xxx.png`。
- **语言**：中文为主，保留英文专业术语；术语首次出现中英对照（附录 C 汇总）。
- **命名**：文件 `ch<NN>-<kebab-name>.md`；子目录 `part-<N>-<kebab>`；每章文件头 YAML front-matter（part/chapter/title/milestone/source/tests/status）。
- **数学公式**：行内 `$...$`，块级 `$$...$$`，多步推导用 `\begin{aligned}...\end{aligned}`。
- **章节模板**：理论章 6 段（学习目标→直觉与动机→数学定义→推导→与本项目联系→小结+思考题）；实战章 6 段（学习目标→原理回顾→代码实现→对应单元测试→动手验证→小结+下章预告）。
- **交付**：7 个 Phase 分段；每 Phase 完成更新 `README.md` 进度勾选。
- **非目标**：不新增/修改 zllm 源码与测试；不构建 PDF/HTML 发布产物；不替换现有 `docs/steps/`；无占位符（TBD/TODO 等）。

---

## File Structure

```
docs/book/
├── README.md                    # 本书入口 + 完整目录（含进度勾选 ☐/✅）
├── SUMMARY.md                   # 章节索引（链接到各章 .md）
├── assets/                      # 所有图片（下载的外链图 + 自生成）
│   └── chNN-*.png
├── part-0-preface/
│   └── ch00-about.md
├── part-1-math/
│   ├── ch01-linear-algebra-vectors.md
│   ├── ch02-linear-algebra-decomposition.md
│   ├── ch03-probability.md
│   ├── ch04-statistics-inference.md
│   ├── ch05-information-theory.md
│   ├── ch06-optimization.md
│   ├── ch07-calculus-chain-rule.md
│   └── ch08-autograd.md
├── part-2-dl-transformer/       # Phase 2 追加
├── part-3-tokenizer/            # Phase 3 追加
├── part-4-architecture/         # Phase 4 追加
├── part-5-data-training/        # Phase 5 追加
├── part-6-finetune-alignment/   # Phase 6 追加
├── part-7-serving/              # Phase 7 追加
└── appendices/                  # Phase 7 追加
```

每个文件单一职责：一章一文件。`README.md` 是唯一的入口与进度看板；`SUMMARY.md` 是导航；`assets/` 集中所有图片避免散落。

---

## Phase 总览（骨架；详细任务随会话追加）

| Phase | 范围 | Task 范围 | 状态 |
|-------|------|-----------|------|
| 1 | Part 0 + Part I（序言 + 数学基础） | Task 1–10（含索引/骨架） | 本次会话详细撰写 |
| 2 | Part II（DL/Transformer 理论） | 待追加 | 后续会话 |
| 3 | Part III（M1+M2 分词） | 待追加 | 后续会话 |
| 4 | Part IV（M3+M4 模型架构） | 待追加 | 后续会话 |
| 5 | Part V（M5+M6+M7 数据与训练） | 待追加 | 后续会话 |
| 6 | Part VI（M8-M11 微调与对齐） | 待追加 | 后续会话 |
| 7 | Part VII + 附录（M12 推理部署 + 附录） | 待追加 | 后续会话 |

---

# Phase 1：序言 + 数学基础（Part 0 + Part I）

**范围：** Ch 00 关于本书；Ch 01–Ch 08 数学基础（纯理论，8 章）。完成后读者具备理解后续 Transformer/LLM 所需的数学语言。

**理论章通用撰写循环（每章 Task 内步骤模板）：**
1. 建文件 + YAML front-matter + 6 段标题骨架
2. 写「学习目标」+「直觉与动机」（含至少 1 个 Mermaid 概念图）
3. 写「数学定义」+「推导」（LaTeX，含几何/图示）
4. 写「与本项目联系」钩子 + 「小结 + 思考题」
5. 校验：LaTeX/Mermaid 语法、无占位符、交叉引用一致
6. 更新 `README.md`/`SUMMARY.md` 进度 + 提交

---

### Task 1: 搭建 docs/book/ 骨架与索引

**Files:**
- Create: `docs/book/README.md`
- Create: `docs/book/SUMMARY.md`
- Create: `docs/book/assets/.gitkeep`

**Interfaces:**
- Produces: 本书入口、完整目录（43 章 + 4 附录，含 ☐/✅ 勾选）、导航索引；后续所有 Task 都向 `README.md` 登记。

- [ ] **Step 1: 创建目录与占位**

```bash
mkdir -p docs/book/assets docs/book/part-0-preface docs/book/part-1-math
touch docs/book/assets/.gitkeep
```

- [ ] **Step 2: 写 README.md（入口 + 目录 + 进度看板）**

写入 `docs/book/README.md`，内容包含：书名「从零训练大语言模型」、一句话定位、读者画像、7 个 Part 的大纲，以及**完整的 43 章 + 4 附录进度表**（表格列：Part / Ch / 标题 / 里程碑 / 状态），状态用 ☐（未写）/ ✅（完成）。此时全部为 ☐。

- [ ] **Step 3: 写 SUMMARY.md（章节索引）**

按 Part 分组，每章一行 Markdown 链接（即使目标文件尚未创建，先写好相对路径，如 `[Ch 01 线性代数：向量与矩阵](part-1-math/ch01-linear-algebra-vectors.md)`）。

- [ ] **Step 4: 校验目录与链接格式**

```bash
ls -R docs/book/
grep -c "☐" docs/book/README.md   # 应为 47（43 章 + 4 附录）
```
Expected: 目录树出现 part-0-preface、part-1-math、assets；README 含 47 个 ☐。

- [ ] **Step 5: 提交**

```bash
git add docs/book/
git commit -m "docs(book): scaffold docs/book/ with README index and SUMMARY"
```

---

### Task 2: Ch 00 关于本书（Part 0）

**Files:**
- Create: `docs/book/part-0-preface/ch00-about.md`

**Interfaces:**
- Produces: 序言章，定义全书定位/读者/前置要求/路线图；后续章节可回引「见 Ch 00 路线图」。

- [ ] **Step 1: 建 front-matter + 6 段骨架**

```markdown
---
part: 0
chapter: 0
title: 关于本书
milestone: null
source: null
tests: null
status: draft
---

# 第 0 章 关于本书

## 0.1 为什么写这本书
## 0.2 这本书给谁看
## 0.3 前置要求
## 0.4 如何阅读（理论篇 vs 实战篇 / 快速路径）
## 0.5 全书路线图
## 0.6 勘误与配套资源
```

- [ ] **Step 2: 写 0.1–0.2（动机 + 读者画像）**

说明 zllm 是 step-by-step TDD 教学项目（12 里程碑/300 步/428 测试），本书是其教学叙事版。读者画像：懂 Python、ML 基础薄弱、想亲手炼一个 LLM。

- [ ] **Step 3: 写 0.3–0.4（前置 + 阅读方法）**

前置：Python 3.14+、PyTorch 2.7+、CUDA GPU、pip install。阅读方法：理论篇可跳读；实战篇必须动手。插入一张 Mermaid「全书阅读路径图」（`graph LR`：数学基础→DL 理论→分词→架构→数据训练→微调对齐→部署）。

- [ ] **Step 4: 写 0.5–0.6（路线图 + 资源）**

路线图：复用 README 的 Part 大纲，配 Mermaid `graph TD` 训练管线全景图（Tokenizer→Pretrain→SFT→{LoRA/DPO/PPO/GRPO/Distill/Agent}→Serving，对齐项目 README 中的管线图）。资源：指向 `docs/steps/`、项目 `README.md`、`tests/`。

- [ ] **Step 5: 校验 + 勾选 + 提交**

校验 Mermaid 语法、无占位符；将 README 中 Ch 00 的 ☐ 改为 ✅。

```bash
git add docs/book/part-0-preface/ch00-about.md docs/book/README.md
git commit -m "docs(book): write Ch00 preface (about this book)"
```

---

### Task 3: Ch 01 线性代数：向量与矩阵

**Files:**
- Create: `docs/book/part-1-math/ch01-linear-algebra-vectors.md`

**Interfaces:**
- Produces: 向量/矩阵/张量/点积/矩阵乘法/范数 的定义与几何；Ch 02 依赖本章程向量/点积；Ch 12 注意力（$QK^T$）回引本章矩阵乘法。

- [ ] **Step 1: front-matter + 6 段骨架**

front-matter `part:1 chapter:1 title:线性代数：向量与矩阵 status:draft`。六段：学习目标 / 直觉与动机 / 数学定义 / 推导与几何 / 与本项目联系 / 小结+思考题。

- [ ] **Step 2: 直觉与动机（含 Mermaid）**

类比：向量=箭头（方向+长度）；矩阵=变换/表格。Mermaid `graph LR` 概念图：标量→向量→矩阵→张量（维度递增）。

- [ ] **Step 3: 数学定义（LaTeX）**

定义并给出 LaTeX：向量 $\mathbf{x}\in\mathbb{R}^n$；点积 $\mathbf{x}\cdot\mathbf{y}=\sum_i x_i y_i$；矩阵乘法 $(AB)_{ij}=\sum_k A_{ik}B_{kj}$；范数 $L_1/L_2/L_\infty$；向量夹角 $\cos\theta=\frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$。

- [ ] **Step 4: 推导与几何（含图示）**

用 ASCII/几何图说明：点积=投影长度、正交=点积为 0；矩阵乘法作为线性变换（旋转/缩放）。给出「矩阵乘法计算」逐步示意。

- [ ] **Step 5: 与本项目联系 + 小结 + 思考题**

钩子：「点积和矩阵乘法是注意力的核心——Ch 12 将用 $QK^T$ 计算词与词的相关度；嵌入向量（Ch 16）就是 $\mathbb{R}^d$ 中的向量」。3 道思考题（如：两个单位向量的点积范围？）。

- [ ] **Step 6: 校验 + 勾选 + 提交**

```bash
git add docs/book/part-1-math/ch01-linear-algebra-vectors.md docs/book/README.md
git commit -m "docs(book): write Ch01 linear algebra - vectors and matrices"
```

---

### Task 4: Ch 02 线性代数：分解与几何

**Files:**
- Create: `docs/book/part-1-math/ch02-linear-algebra-decomposition.md`

**Interfaces:**
- Produces: 特征分解、SVD、投影、正交基；LoRA（Ch 34）的低秩近似 $\Delta W=BA$ 回引本章程秩与 SVD；softmax/注意力残差子空间。

- [ ] **Step 1: front-matter + 骨架**（part:1 chapter:2）

- [ ] **Step 2: 直觉（Mermaid）** —— 类比：特征向量=变换中方向不变的轴；SVD=把任意矩阵拆成「旋转-缩放-旋转」。Mermaid 概念图。

- [ ] **Step 3: 数学定义（LaTeX）**

特征分解 $A\mathbf{v}=\lambda\mathbf{v}$、$A=Q\Lambda Q^{-1}$；SVD $A=U\Sigma V^T$；矩阵秩 $\text{rank}(A)$；投影矩阵 $P=A(A^TA)^{-1}A^T$；正交 $\mathbf{x}^T\mathbf{y}=0$。

- [ ] **Step 4: 推导与几何（图示）**

SVD 几何分解图（旋转→缩放→旋转）；秩与信息量的关系（低秩≈可压缩）。

- [ ] **Step 5: 与本项目联系 + 小结 + 思考题**

钩子：「Ch 34 LoRA 的核心思想就是假设权重更新 $\Delta W$ 是**低秩**的，用 $B_{d\times r}A_{r\times d}$（$r\ll d$）近似；这背后正是 SVD 低秩近似的直觉（Eckart-Young 定理）」。

- [ ] **Step 6: 校验 + 勾选 + 提交**

```bash
git commit -m "docs(book): write Ch02 linear algebra - decomposition and geometry"
```

---

### Task 5: Ch 03 概率论基础

**Files:**
- Create: `docs/book/part-1-math/ch03-probability.md`

**Interfaces:**
- Produces: 概率分布/条件概率/贝叶斯/联合与边缘；Ch 14 解码采样（temperature/top-k/top-p）与 Ch 36-38 强化学习回引本章。

- [ ] **Step 1: front-matter + 骨架**（part:1 chapter:3）

- [ ] **Step 2: 直觉（Mermaid）** —— 频率派 vs 贝叶斯派直觉；离散/连续分布示意。

- [ ] **Step 3: 数学定义（LaTeX）**

概率公理；条件概率 $P(A|B)=P(A,B)/P(B)$；贝叶斯 $P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}$；期望 $E[X]$；常见分布：伯努利、类别分布（categorical）、均匀、高斯 $\mathcal{N}(\mu,\sigma^2)$。

- [ ] **Step 4: 推导与几何（图示）**

贝叶斯公式逐项解读（先验/似然/后验/证据）；softmax 作为类别分布参数化（钩向前 Ch 14）。

- [ ] **Step 5: 与本项目联系 + 小结 + 思考题**

钩子：「语言模型每一步输出的是一个**类别分布**（在词表上的概率）——Ch 14 的采样、Ch 36 DPO 的 $\pi_\theta$、Ch 37 PPO 的 ratio 都是概率分布上的运算」。

- [ ] **Step 6: 校验 + 勾选 + 提交**

```bash
git commit -m "docs(book): write Ch03 probability foundations"
```

---

### Task 6: Ch 04 统计推断

**Files:**
- Create: `docs/book/part-1-math/ch04-statistics-inference.md`

**Interfaces:**
- Produces: 期望/方差/协方差/相关/MLE/MAP；Ch 20 RMSNorm（均值/方差）与所有 loss 推导回引 MLE。

- [ ] **Step 1: front-matter + 骨架**（part:1 chapter:4）

- [ ] **Step 2: 直觉（Mermaid）** —— 「最可能 vs 最合理」：MLE 找最可能参数，MAP 加先验。

- [ ] **Step 3: 数学定义（LaTeX）**

方差 $\text{Var}(X)=E[(X-E[X])^2]$；协方差 $\text{Cov}(X,Y)$；相关系数；MLE $\hat\theta=\arg\max_\theta P(D|\theta)$；MAP $\hat\theta=\arg\max_\theta P(\theta|D)$；对数似然 $\log P(D|\theta)$。

- [ ] **Step 4: 推导与几何（图示）**

推导「高斯噪声下 MLE 等价于最小化均方误差」；「交叉熵 = 伯努利/类别分布的负对数似然」（桥接 Ch 05）。

- [ ] **Step 5: 与本项目联系 + 小结 + 思考题**

钩子：「Ch 20 的 RMSNorm 本质是对每个 token 的 hidden vector 做均值/方差归一；预训练的交叉熵损失（Ch 31）就是类别分布 MLE 的负对数似然」。

- [ ] **Step 6: 校验 + 勾选 + 提交**

```bash
git commit -m "docs(book): write Ch04 statistical inference"
```

---

### Task 7: Ch 05 信息论

**Files:**
- Create: `docs/book/part-1-math/ch05-information-theory.md`

**Interfaces:**
- Produces: 熵/交叉熵/KL/JS/互信息；**本章是 NTP 与 CE 损失的直接来源**，Ch 31 预训练、Ch 39 蒸馏（KL）回引。

- [ ] **Step 1: front-matter + 骨架**（part:1 chapter:5）

- [ ] **Step 2: 直觉（Mermaid）** —— 「信息=意外程度」：罕见事件信息量大。熵=平均意外。配一张熵随概率变化的 ASCII/几何图。

- [ ] **Step 3: 数学定义（LaTeX）**

自信息 $I(x)=-\log p(x)$；香农熵 $H(p)=-\sum p(x)\log p(x)$；交叉熵 $H(p,q)=-\sum p(x)\log q(x)$；KL 散度 $D_{KL}(p\|q)=\sum p(x)\log\frac{p(x)}{q(x)}$；$H(p,q)=H(p)+D_{KL}(p\|q)$；互信息 $I(X;Y)$。

- [ ] **Step 4: 推导与几何（图示）**

推导「最小化交叉熵 ⟺ 最小化 KL（当 p 固定）」；为什么用自然对数（nat）vs log2（bit）。

- [ ] **Step 5: 与本项目联系 + 小结 + 思考题**

钩子（关键）：「预训练就是在最小化模型分布 $q_\theta$ 与数据经验分布 $p$ 的交叉熵（Ch 31）；蒸馏用温度缩放后的 KL 把教师分布传给学生（Ch 39，$T^2\cdot KL$）」。

- [ ] **Step 6: 校验 + 勾选 + 提交**

```bash
git commit -m "docs(book): write Ch05 information theory (entropy/CE/KL)"
```

---

### Task 8: Ch 06 最优化基础

**Files:**
- Create: `docs/book/part-1-math/ch06-optimization.md`

**Interfaces:**
- Produces: 梯度下降/SGD/动量/Adam；Ch 29 学习率调度、Ch 15 pretrain 的 lr 选择回引。

- [ ] **Step 1: front-matter + 骨架**（part:1 chapter:6）

- [ ] **Step 2: 直觉（Mermaid）** —— 下山类比（梯度=最陡上升方向，负梯度=最陡下降）。Mermaid 画出 loss 曲面 + 梯度下降轨迹。

- [ ] **Step 3: 数学定义（LaTeX）**

梯度 $\nabla f$；梯度下降 $\theta_{t+1}=\theta_t-\eta\nabla f(\theta_t)$；SGD；动量 $v_t=\beta v_{t-1}+\nabla f$；Adam 一阶/二阶矩估计 $\hat m,\hat v$ 与更新式。

- [ ] **Step 4: 推导与几何（图示）**

学习率过大/过小的几何示意；动量如何穿越狭窄谷底；Adam 自适应步长原理。

- [ ] **Step 5: 与本项目联系 + 小结 + 思考题**

钩子：「本项目用 AdamW + 余弦学习率退火（Ch 29）：lr 从 $5\times10^{-4}$ 余弦降到 $5\times10^{-5}$；为何 SFT 用小得多的 lr（$10^{-5}$）将在 Ch 33 解释」。

- [ ] **Step 6: 校验 + 勾选 + 提交**

```bash
git commit -m "docs(book): write Ch06 optimization (GD/SGD/Adam)"
```

---

### Task 9: Ch 07 微积分与链式法则

**Files:**
- Create: `docs/book/part-1-math/ch07-calculus-chain-rule.md`

**Interfaces:**
- Produces: 偏导/雅可比/海森/链式法则/泰勒展开；是反向传播（Ch 10）与 autograd（Ch 08）的数学基础。

- [ ] **Step 1: front-matter + 骨架**（part:1 chapter:7）

- [ ] **Step 2: 直觉（Mermaid）** —— 复合函数=流水线，链式法则=把总变化量拆给每一级。Mermaid 画计算图节点。

- [ ] **Step 3: 数学定义（LaTeX）**

偏导 $\partial f/\partial x_i$；雅可比 $J_{ij}=\partial f_i/\partial x_j$；海森 $H_{ij}=\partial^2 f/\partial x_i\partial x_j$；链式法则 $\frac{df}{dx}=\frac{df}{dg}\cdot\frac{dg}{dx}$；多元链式 $\frac{\partial L}{\partial x}=\sum_k\frac{\partial L}{\partial y_k}\frac{\partial y_k}{\partial x}$；泰勒 $f(x)\approx f(a)+f'(a)(x-a)+\tfrac12f''(a)(x-a)^2$。

- [ ] **Step 4: 推导与几何（图示）**

用一个两层复合函数手算链式法则全过程；雅可比作为局部线性近似；梯度裁剪（clip）= 在大曲率处限制步长的泰勒一阶动机。

- [ ] **Step 5: 与本项目联系 + 小结 + 思考题**

钩子：「反向传播就是链式法则在计算图上的高效执行（Ch 08/Ch 10）；本项目 `grad_clip=1.0`（Ch 29）防止梯度爆炸，其原理正是本章的梯度模长」。

- [ ] **Step 6: 校验 + 勾选 + 提交**

```bash
git commit -m "docs(book): write Ch07 calculus and chain rule"
```

---

### Task 10: Ch 08 张量计算与 PyTorch 自动微分

**Files:**
- Create: `docs/book/part-1-math/ch08-autograd.md`

**Interfaces:**
- Produces: 张量/broadcasting/计算图/autograd/GPU/CUDA；从本章起读者要能跑代码，是 Part II 与所有实战章的工程桥梁。
- Consumes: Ch 01–07 的数学（链式法则→反向传播实现）。

- [ ] **Step 1: front-matter + 骨架**（part:1 chapter:8）

- [ ] **Step 2: 直觉（Mermaid）** —— 张量=多维数组；计算图=有向无环图（前向建图、反向求导）。Mermaid 画一个前向+反向的计算图示例。

- [ ] **Step 3: 数学定义（LaTeX）**

张量形状记法 $x\in\mathbb{R}^{B\times T\times d}$；broadcasting 规则；计算图 $L=g_n\circ\dots\circ g_1$；反向模式 $\bar{x_i}=\sum_j \bar{y_j}\frac{\partial y_j}{\partial x_i}$。

- [ ] **Step 4: 推导与代码示意（图示 + PyTorch 摘录）**

手算一个小计算图的反向传播；给出 PyTorch autograd 最小示例（`requires_grad_` / `backward` / `.grad`），但**不引用项目源码**（本章为理论章）。可放一段 5-8 行通用 PyTorch 代码说明 `loss.backward()`。

- [ ] **Step 5: 与本项目联系 + 小结 + 思考题**

钩子：「zllm 所有模型都用 `nn.Module` + autograd（Ch 20+）；`ZLLMConfig` 的张量形状约定（hidden=768, 8 层）将在 Ch 26 组装成完整计算图；混合精度（bfloat16）是 Ch 30 的工程优化」。思考题：为什么反向模式比正向模式高效？

- [ ] **Step 6: 校验 + 勾选 + 提交**

```bash
git add docs/book/part-1-math/ch08-autograd.md docs/book/README.md docs/book/SUMMARY.md
git commit -m "docs(book): write Ch08 tensors and PyTorch autograd (completes Part I)"
```

---

## Phase 1 完成标准（DoD）✅ 已完成（已合并 main）

- `docs/book/README.md`：Part 0 + Part I 全部 9 章 ☐→✅。
- `docs/book/SUMMARY.md`：9 个链接全部指向已存在文件。
- 每章满足理论章 6 段模板；含 ≥1 个 Mermaid 图；LaTeX 公式语法正确；每章有「与本项目联系」钩子。
- 无 TBD/TODO 占位符（`grep -rniE "TBD|TODO|FIXME|待定|占位" docs/book/` 为空）。
- 10 个提交（Task 1 骨架 + Task 2–10 各一章）；总计 ~3218 行，已逐章控制器复查通过。

---

# Phase 2：深度学习与 Transformer 理论（Part II）

**范围：** Ch 09–Ch 15（纯理论，7 章）。完成后读者掌握从 MLP 到 Transformer 的完整理论，可进入 Part III 实战。所有章均为**理论章模板**（不引用 zllm file:line，只在「与本项目联系」节做前向钩子）。

**撰写循环：** 与 Phase 1 相同（front-matter + 6 段 → Mermaid/LaTeX → 钩子 → 校验 → README 勾选 → 提交）。

---

### Task 11: Ch 09 神经网络基础

**Files:**
- Create: `docs/book/part-2-dl-transformer/ch09-neural-network-basics.md`

**Interfaces:**
- Consumes: Ch 01（矩阵乘法=线性变换）、Ch 06（梯度下降）、Ch 07（链式法则）、Ch 08（autograd）。
- Produces: MLP/激活函数/损失函数理论；Ch 10（反传）依赖本章网络结构，Ch 23（SwiGLU）回引激活函数。

- [ ] **Step 1: front-matter（part:2 chapter:9）+ 6 段骨架**
- [ ] **Step 2: 直觉（Mermaid）** —— 神经元=加权求和+激活；MLP=多层堆叠的函数复合。Mermaid 画一个 3 层 MLP。
- [ ] **Step 3: 数学定义（LaTeX）** —— 线性层 $\mathbf{h}=W\mathbf{x}+\mathbf{b}$；激活 sigmoid/tanh/ReLU $\max(0,x)$/SiLU $x\sigma(x)$；损失：MSE（回归）、交叉熵（分类，回引 Ch 05）。
- [ ] **Step 4: 推导与几何（图示）** —— 每个激活函数的形状与梯度（sigmoid 梯度消失、ReLU 死神经元、SiLU 平滑）；为什么需要非线性激活（否则多层仍等价单层线性）。
- [ ] **Step 5: 与本项目联系（钩子）** —— **Ch 23 SwiGLU** 用 SiLU 做门控（$FFN(x)=down(silu(gate(x))\cdot up(x))$）；**Ch 26 CausalLM** 用交叉熵损失；MLP 是 Transformer FFN 的前身。
- [ ] **Step 6: 校验 + 勾选 Ch09 + 提交** `docs(book): write Ch09 neural network basics`

---

### Task 12: Ch 10 反向传播与训练动力学

**Files:**
- Create: `docs/book/part-2-dl-transformer/ch10-backprop-training-dynamics.md`

**Interfaces:**
- Consumes: Ch 07（链式法则）、Ch 08（autograd）、Ch 09（MLP）。
- Produces: 反传推导 + 初始化 + 正则化；后续所有训练章（Ch 31/33）依赖。

- [ ] **Step 1: front-matter（part:2 chapter:10）+ 骨架**
- [ ] **Step 2: 直觉（Mermaid）** —— 反传=链式法则在计算图上的高效调度（前向存中间值，反向逐节点求导）。Mermaid 画前向/反向计算图。
- [ ] **Step 3: 数学定义（LaTeX）** —— 逐层梯度公式；权重初始化 Xavier $Var(W)=1/n_{in}$、He $Var(W)=2/n_{in}$；正则化 $L_2$、Dropout。
- [ ] **Step 4: 推导与几何（图示）** —— 手推一个 2 层 MLP 的反传全过程（forward 存 $a^{(l)}$，backward 算 $\delta^{(l)}$）；解释 Xavier/He 为何稳定（保持各层激活/梯度方差）。
- [ ] **Step 5: 与本项目联系（钩子）** —— 初始化动机呼应 Ch 04 方差；**Ch 31 预训练** 每步 `loss.backward()`；Dropout 在注意力（Ch 22）出现；残差连接（Ch 25）缓解梯度消失（呼应 Ch 07 的 +1）。
- [ ] **Step 6: 校验 + 勾选 Ch10 + 提交** `docs(book): write Ch10 backprop and training dynamics`

---

### Task 13: Ch 11 序列建模：从 RNN/LSTM 到瓶颈

**Files:**
- Create: `docs/book/part-2-dl-transformer/ch11-sequence-modeling-rnn.md`

**Interfaces:**
- Consumes: Ch 09–10。
- Produces: 序列建模动机与瓶颈，为 Ch 12 注意力铺路。

- [ ] **Step 1: front-matter（part:2 chapter:11）+ 骨架**
- [ ] **Step 2: 直觉（Mermaid）** —— 语言是序列；RNN 逐步传递隐状态。Mermaid 画 RNN 时间展开图。
- [ ] **Step 3: 数学定义（LaTeX）** —— RNN $\mathbf{h}_t=\tanh(W_h\mathbf{h}_{t-1}+W_x\mathbf{x}_t)$；LSTM 门控（输入/遗忘/输出门）；梯度随时间步连乘。
- [ ] **Step 4: 推导与几何（图示）** —— RNN 的 BPTT 梯度连乘 → 梯度消失/爆炸；LSTM 门控如何缓解；**核心瓶颈**：无法并行（时序依赖）+ 长程依赖弱。
- [ ] **Step 5: 与本项目联系（钩子）** —— 这些瓶颈正是 Transformer/注意力（Ch 12）要解决的：并行 + 长程依赖；zllm 完全不用 RNN，是 decoder-only Transformer。
- [ ] **Step 6: 校验 + 勾选 Ch11 + 提交** `docs(book): write Ch11 sequence modeling - RNN/LSTM`

---

### Task 14: Ch 12 注意力机制

**Files:**
- Create: `docs/book/part-2-dl-transformer/ch12-attention.md`

**Interfaces:**
- Consumes: Ch 01（点积/矩阵乘法）、Ch 03（softmax）、Ch 05（缩放动机）、Ch 11（RNN 瓶颈）。
- Produces: 注意力核心理论；Ch 22（GQA）实战回引。

- [ ] **Step 1: front-matter（part:2 chapter:12）+ 骨架**
- [ ] **Step 2: 直觉（Mermaid）** —— 注意力=「查询词去检索相关的键值对」。Mermaid 画 Q/K/V 检索示意。
- [ ] **Step 3: 数学定义（LaTeX）** —— 缩放点积注意力 $\text{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^\top}{\sqrt{d_k}})V$；多头注意力（投影到 $h$ 个子空间分别做）。
- [ ] **Step 4: 推导与几何（图示）** —— 为什么除以 $\sqrt{d_k}$（点积方差随维度增长，softmax 饱和，回引 Ch 03/05）；逐对相关度矩阵的可视化。
- [ ] **Step 5: 与本项目联系（钩子）** —— **Ch 22 GQA**（8 Q 头/4 KV 头）、**QK-Norm**、**KV Cache**；多头→分组查询；Flash Attention（Ch 22）。
- [ ] **Step 6: 校验 + 勾选 Ch12 + 提交** `docs(book): write Ch12 attention mechanism`

---

### Task 15: Ch 13 Transformer 架构详解

**Files:**
- Create: `docs/book/part-2-dl-transformer/ch13-transformer-architecture.md`

**Interfaces:**
- Consumes: Ch 09（FFN）、Ch 12（注意力）。
- Produces: 完整 Transformer 架构；Ch 25（Block 组装）回引。

- [ ] **Step 1: front-matter（part:2 chapter:13）+ 骨架**
- [ ] **Step 2: 直觉（Mermaid）** —— Transformer=注意力+FFN+残差+归一化的堆叠。Mermaid 画原始 Encoder/Decoder Block。
- [ ] **Step 3: 数学定义（LaTeX）** —— Pre-Norm vs Post-Norm；残差 $x'=x+\text{Sublayer}(Norm(x))$；Position Encoding；Encoder-Decoder 交叉注意力。
- [ ] **Step 4: 推导与几何（图示）** —— 原始论文 "Attention Is All You Need" 架构图逐模块拆解；为什么 decoder-only 成为主流（GPT 路线）。
- [ ] **Step 5: 与本项目联系（钩子）** —— zllm 是 **decoder-only + Pre-Norm**（Ch 25 Block）；用 RoPE 替代绝对位置编码（Ch 21）；RMSNorm 替代 LayerNorm（Ch 20）。
- [ ] **Step 6: 校验 + 勾选 Ch13 + 提交** `docs(book): write Ch13 transformer architecture`

---

### Task 16: Ch 14 解码策略理论

**Files:**
- Create: `docs/book/part-2-dl-transformer/ch14-decoding-theory.md`

**Interfaces:**
- Consumes: Ch 03（概率分布/softmax）、Ch 05（熵）。
- Produces: 解码理论；Ch 41（解码实现）回引。

- [ ] **Step 1: front-matter（part:2 chapter:14）+ 骨架**
- [ ] **Step 2: 直觉（Mermaid）** —— 从分布到文本：贪心 vs 采样的权衡（质量 vs 多样性）。
- [ ] **Step 3: 数学定义（LaTeX）** —— greedy=argmax；temperature $p_i=\mathrm{softmax}(z_i/T)$；top-k、top-p（nucleus）$ \sum_{i\in S}p_i\ge p$；beam search（简述）；repetition penalty。
- [ ] **Step 4: 推导与几何（图示）** —— T 如何改变分布尖锐度（回引 Ch 03/05）；top-k vs top-p 的自适应差别（概率集中时少取/分散时多取）；熵与多样性的关系。
- [ ] **Step 5: 与本项目联系（钩子）** —— **Ch 41 解码实现**（zllm 默认 temperature=0.85, top_p=0.95, top_k=50）；采样在 GRPO/PPO rollout（Ch 37/38）里用于生成多个 response。
- [ ] **Step 6: 校验 + 勾选 Ch14 + 提交** `docs(book): write Ch14 decoding strategy theory`

---

### Task 17: Ch 15 现代语言模型全景

**Files:**
- Create: `docs/book/part-2-dl-transformer/ch15-llm-landscape.md`

**Interfaces:**
- Consumes: Ch 13（Transformer）、Ch 05（NTP/CE）。
- Produces: 现代 LLM 全景，Part II 收官，过渡到 Part III 实战。

- [ ] **Step 1: front-matter（part:2 chapter:15）+ 骨架**
- [ ] **Step 2: 直觉（Mermaid）** —— GPT 演进时间线（GPT-1→4 / Llama / Qwen）；decoder-only 成主流。Mermaid 画演进 + 训练三阶段（pretrain→SFT→align）。
- [ ] **Step 3: 数学定义（LaTeX）** —— NTP 目标 $\mathcal{L}=-\sum_t\log P(x_t|x_{<t})$；Scaling Law $\mathcal{L}(N)\approx (N_c/N)^{\alpha}$；涌现能力（简述）。
- [ ] **Step 4: 推导与几何（图示）** —— Scaling Law 幂律曲线；参数/数据/算力的权衡；为什么 decoder-only + NTP 胜出。
- [ ] **Step 5: 与本项目联系（钩子）** —— zllm 对齐 **Qwen3/minimind-3**（GQA/RoPE/SwiGLU/MoE/Weight Tying），~64M 参数；缩放版完整管线 pretrain→SFT→{LoRA/DPO/PPO/GRPO}→serving；为 Part III（Ch 16 项目初始化）开篇。
- [ ] **Step 6: 校验 + 勾选 Ch15 + 提交** `docs(book): write Ch15 modern LLM landscape`

---

## Phase 2 完成标准（DoD）

- `docs/book/README.md`：Part II 全部 7 章（Ch 09–15）☐→✅。
- 创建 `docs/book/part-2-dl-transformer/` 目录及 7 个章节文件。
- 每章理论章 6 段模板；≥1 Mermaid；≥3 `$$`；含「与本项目联系」钩子；无占位符。
- 7 个提交（Task 11–17 各一）。

---

## Phase 3–7 详细任务（待后续会话追加）

> 占位说明（计划本身的进度标记，非内容占位）：以下 Phase 将在后续会话按同样 Task 粒度追加。Phase 3 起为实战章，Task 增加 `grep` 校验引用的 zllm 源码/测试行号真实存在的步骤。

- **Phase 3（Part III，Ch 16–19，4 章）**：M1+M2。引用 `zllm/tokenizer/*.py`、`tests/m01_foundations`、`tests/m02_tokenizer`。
- **Phase 4（Part IV，Ch 20–26，7 章）**：M3+M4。引用 `zllm/model/*.py`、`tests/m03_model_components`、`tests/m04_model_assembly`。
- **Phase 5（Part V，Ch 27–32，6 章）**：M5+M6+M7。
- **Phase 6（Part VI，Ch 33–40，8 章）**：M8-M11。
- **Phase 7（Part VII + 附录，Ch 41–43 + 附录 A-D）**：M12。
