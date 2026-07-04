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
├── part-5-training/             # Phase 5 追加
├── part-6-alignment/             # Phase 6 追加
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
| 3 | Part III（M1+M2 分词） | 已追加（Task 18–21） | 进行中 |
| 4 | Part IV（M3+M4 模型架构） | 已追加（Task 22–28） | 已完成 |
| 5 | Part V（M5+M6+M7 数据与训练） | 已追加（Task 29–34） | 已完成 |
| 6 | Part VI（M8-M11 微调与对齐） | 已追加（Task 35–42） | 进行中 |
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

## Phase 3 详细任务（Part III，Ch 16–19，实战篇开篇）

**范围：** Ch 16–Ch 19（4 章）。里程碑 M1（项目初始化）+ M2（分词器）。本 Phase 起为**实战章**（模板见 spec 3.2），必须引用真实的 zllm 源码/测试 `file:line`。每个 Task 完成后，校验步骤增加：`grep -cE "\.py:[0-9]"` 必须 > 0（Ch 17 纯理论章除外），且所有引用的文件:行号必须真实存在（用 `read`/`grep` 抽查）。

**前置事实（Phase 3 全部任务共用，源自 2026-07-03 探源，确保 file:line 准确）：**

- `zllm/config.py`（73 行）：`ZLLMConfig.__init__` 在 `config.py:15-69`；GQA（`num_attention_heads=8`/`num_key_value_heads=4` 在 `config.py:20-21`）；π 缩放 `intermediate_size = ceil(hidden_size*π/64)*64` 在 `config.py:48-52`（hidden=768 → 2368）；`head_dim` property 在 `config.py:71-73`（= 768//8 = 96）；MoE 参数 `config.py:29-35`；`tie_word_embeddings=True`、`rope_theta=1000000.0` 在 `config.py:26-27`。
- `zllm/__init__.py`：`from zllm.config import ZLLMConfig`、`__version__ = "0.0.1"`。
- `tests/conftest.py`：`device` fixture `conftest.py:7-10`；`small_config`（dim=64,2 层）`conftest.py:13-25`；`default_config`（dim=768,8 层,vocab=6400）`conftest.py:28-33`。
- `tests/m01_foundations/test_002_import.py`：`test_zllm_importable`（`__version__=="0.0.1"`）`:4-7`；`test_config_importable` `:10-12`。
- `tests/m01_foundations/test_003_fixtures.py`：GQA 断言 `:22-24`；π 缩放断言 `:27-31`（`math.ceil(768*π/64)*64`）；`head_dim==96` `:34-35`。
- `zllm/tokenizer/bpe.py`（99 行）：`byte_level_encode` `:8-14`；`get_pair_counts` `:17-27`；`merge` `:30-44`；`encode` `:47-56`；`decode`（含递归 `expand`）`:59-75`；`train_bpe` `:78-99`（`num_merges=vocab_size-256`，`new_id=256+i`，`max(counts,...)`）。
- `zllm/tokenizer/trainer.py`（57 行）：`train_tokenizer` `:18-43`（`Tokenizer(BPE())`、`ByteLevel` pre-tokenizer `:31`、`ByteLevelDecoder` `:33`、`BpeTrainer(special_tokens=ALL_SPECIAL_TOKENS, initial_alphabet=ByteLevel.alphabet())` `:34-38`、`train_from_iterator` `:39`、保存 `tokenizer.json` `:42`）；`load_tokenizer` `:46-57`。
- `zllm/tokenizer/special_tokens.py`（52 行）：`SPECIAL_TOKENS` 列表 `:12-34`（im_start/im_end/pad/vision/toolcall `📞`/reasoning）；`BUFFER_TOKENS = [<|buffer1..8|>]` `:37`；`ALL_SPECIAL_TOKENS` `:40`；常量 `IM_START` 等 `:43-52`。
- `zllm/tokenizer/chat_template.py`（94 行）：`render_messages` `:25-72`（tools 注入 system `:46-55`、open_thinking 包裹 `:64-65`、add_generation_prompt `:69-70`）；Jinja2 `CHAT_TEMPLATE` `:76-94`。
- `zllm/tokenizer/adapter.py`（78 行）：`TokenizerAdapter` `:10-65`（bos/eos/pad id `:13-18`、`__call__` 加 special token `:44-52`、`apply_chat_template` `:54-65`）；`wrap` `:74-78`。
- `tests/m02_tokenizer/`：`test_026_bpe_core.py`（TestByteLevelEncode/TestGetPairCounts/TestMerge/TestTrainBPE）、`test_030_special_tokens.py`（test_no_duplicates `:24-25`、toolcall `📞` `:39-44`）、`test_035_production_tokenizer.py`（corpus fixture `:11-20`、special token 单 token `:45-54`、roundtrip `:72-80`、压缩 `:99-103`）、`test_041_encode_decode.py`（TestEncode/TestDecode/TestRoundtrip）、`test_044_chat_template.py`（generation prompt `:53-56`、open_thinking `:65-71`、tools `:87-93`）、`test_050_integration.py`（全管线 `:25-34`）。
- 依赖/环境：`pyproject.toml`（`requires-python>=3.14`、`torch>=2.7`、`tokenizers>=0.21`、`transformers>=4.52`）；安装 `pip install -e ".[dev]"`；运行 `pytest`（428 测试）。

---

### Task 18: Ch 16 项目初始化与开发环境（M1）

**Files:**
- Create: `docs/book/part-3-tokenizer/ch16-project-setup.md`
- Modify: `docs/book/README.md`（Ch 16 行 ☐→✅）
- Read-only refs: `zllm/config.py`、`zllm/__init__.py`、`tests/conftest.py`、`tests/m01_foundations/test_002_import.py`、`tests/m01_foundations/test_003_fixtures.py`、`README.md`、`pyproject.toml`

**Interfaces:**
- Consumes: Ch 08（张量/PyTorch）、Ch 13（Transformer 架构，需解释 config 里每个超参的由来）、Ch 15（全景，config 对齐 Qwen3）
- Produces: 实战章首篇，定调「原理回顾→代码摘录→测试→pytest 验证」节奏；后续 Ch 17–19 的前置环境；为 Ch 20（RMSNorm，会用 `small_config`/`default_config`）铺路

- [ ] **Step 1: 创建 front-matter + 6 节实战模板**

YAML front-matter（7 键）：`part:3, chapter:16, title:项目初始化与开发环境, milestone:M1, source:zllm/config.py, tests:tests/m01_foundations, status:draft`。6 节按实战模板（spec 3.2）：学习目标 / 原理回顾 / 代码实现 / 对应单元测试 / 动手验证 / 小结+下章预告。

- [ ] **Step 2: 第 1–2 节 学习目标 + 原理回顾**

学习目标：装好环境、读懂 `ZLLMConfig` 每个参数、跑通 M1 测试。原理回顾：精简回引 Ch 13（Transformer 超参：层数/头数/维度）+ Ch 15（对齐 Qwen3），说明 config 就是把 Transformer 架构参数化。≥1 Mermaid：config 参数 → 架构组件映射图（vocab→embedding、hidden→各层、layers→深度、GQA 头数→注意力）。

- [ ] **Step 3: 第 3 节 代码实现（核心摘录 + file:line）**

摘录 `ZLLMConfig.__init__` 签名（`config.py:15-39`，5–20 行）并逐参数解释：`vocab_size=6400`、`hidden_size=768`、`num_hidden_layers=8`、GQA（`num_attention_heads=8`/`num_key_value_heads=4`，2:1）、`rope_theta=1e6`、`tie_word_embeddings=True`。重点讲 π 缩放（`config.py:48-52`，`ceil(hidden_size*π/64)*64` → 2368，对齐 64 倍数提升 Tensor Core）。摘录 `head_dim` property（`config.py:71-73`）。每段摘录后跟引用行如「完整实现见 `zllm/config.py:48`」。

- [ ] **Step 4: 第 4 节 对应单元测试（file:line）**

讲 `tests/conftest.py` 的三个 fixture（device `:7-10`、small_config `:13-25`、default_config `:28-33`）。讲 `test_002_import.py:4-7`（`__version__`）、`test_003_fixtures.py` 的 GQA 断言（`:22-24`）、π 缩放断言（`:27-31`）、head_dim（`:34-35`）。引用行格式「对应测试 `tests/m01_foundations/test_003_fixtures.py:27`」。

- [ ] **Step 5: 第 5 节 动手验证（pytest + 预期输出）**

```bash
pip install -e ".[dev]"
pytest tests/m01_foundations/ -v
```
预期输出：4 个 test 全 PASSED。配一句说明：这 4 个测试是「地基」，确保包可导入、config 默认值正确、fixture 可用。

- [ ] **Step 6: 第 6 节 小结 + 下章预告**

小结：环境就绪、config 读懂。下章预告：进入分词——Ch 17 分词理论（BPE/WordPiece/SentencePiece）。

- [ ] **Step 7: 校验 + commit**

校验：front-matter 7 键；`grep -cE "\.py:[0-9]"` ≥ 5；引用的 file:line 全部真实（抽查 config.py:48、test_003_fixtures.py:27）；无 TBD/TODO；README Ch 16 → ✅。Commit: `docs(book): write Ch16 project setup and environment (M1)`。

---

### Task 19: Ch 17 分词理论：BPE/WordPiece/SentencePiece

**Files:**
- Create: `docs/book/part-3-tokenizer/ch17-tokenization-theory.md`
- Modify: `docs/book/README.md`（Ch 17 行 ☐→✅）
- Read-only refs: 仅 `zllm/tokenizer/bpe.py`、`trainer.py` 作「将在 Ch 18/19 实现」的前向钩子（不展开 file:line）

**Interfaces:**
- Consumes: Ch 03（概率/频率）、Ch 05（熵/编码——子词是「压缩」）
- Produces: 为 Ch 18（bpe.py）和 Ch 19（trainer.py）提供算法理论；说明 zllm 为何选 BPE

**注：** 本章是 Part III 里唯一的**理论/原理章**（无独立源码文件，类比 Part VI 的 Ch 35「RLHF 总论」）。用「原理」为主的 6 节结构，不强制 file:line。

- [ ] **Step 1: front-matter + 6 节**

YAML：`part:3, chapter:17, title:分词理论：BPE/WordPiece/SentencePiece, milestone:M2, source:null, tests:null, status:draft`。

- [ ] **Step 2: 学习目标 + 直觉**

讲清三件事：为什么要分词（文本→token id 序列）、字符级/词级/子词级的取舍（OOV 问题）、子词为何胜出。≥1 Mermaid：文本→分词→token id 流水线。

- [ ] **Step 3: 三种算法（LaTeX 推导）**

- **BPE**：从字符（字节）起，每步贪心合并最高频相邻对。形式：$(a^*,b^*)=\arg\max_{(a,b)} f(a,b)$，$new\_id\leftarrow 256+k$。
- **WordPiece**：按似然增益打分 $s(A,B)=\dfrac{f(AB)}{f(A)\cdot f(B)}$，选使语言模型似然提升最大的对合并（BERT 用）。
- **Unigram / SentencePiece**：反向——从大词表逐步删除使总似然下降最小的 token；SentencePiece 是把 BPE/Unigram 做成语言无关（直接吃原始字节/字符，不预切空格）。
配 ≥1 ASCII：三种算法对「low/lowest/newest」的分词对比。

- [ ] **Step 4: 对比 + 为何选 BPE**

表格：确定性 / 训练方向 / 是否需预分词 / 压缩率 / 实现复杂度。说明 zllm 选 BPE 的理由：确定性、易教学、HF tokenizers 原生支持、与 Qwen3/LLaMA 一致。

- [ ] **Step 5: 与项目联系（前向钩子）**

钩子：Ch 18 从零实现 BPE（`bpe.py` 的 `train_bpe`/`merge`/`encode`）；Ch 19 用 HF tokenizers 生产级实现（`trainer.py`）+ 特殊 token + chat template。回引 Ch 05（子词 = 更短编码 = 更低交叉熵目标长度）。

- [ ] **Step 6: 小结 + 思考题 + 下章预告**

3 思考题（如：BPE 对未登录中文为何不崩？WordPiece 和 BPE 在合并判据上的本质区别？）。预告 Ch 18。

- [ ] **Step 7: 校验 + commit**

校验：≥3 个 `$$`；≥1 mermaid；无 TBD；README Ch 17 → ✅。Commit: `docs(book): write Ch17 tokenization theory (BPE/WordPiece/SentencePiece)`。

---

### Task 20: Ch 18 教学版 BPE 实现（M2-a）

**Files:**
- Create: `docs/book/part-3-tokenizer/ch18-bpe-teaching-impl.md`
- Modify: `docs/book/README.md`（Ch 18 行 ☐→✅）
- Read-only refs: `zllm/tokenizer/bpe.py`、`tests/m02_tokenizer/test_026_bpe_core.py`、`tests/m02_tokenizer/test_041_encode_decode.py`

**Interfaces:**
- Consumes: Ch 17（BPE 理论：合并最高频对）、Ch 05（字节/编码）
- Produces: 读懂 `bpe.py` 全部 6 个函数；为 Ch 19（生产版）做对比铺垫

- [ ] **Step 1: front-matter + 6 节**

YAML：`part:3, chapter:18, title:教学版 BPE 实现, milestone:M2, source:zllm/tokenizer/bpe.py, tests:tests/m02_tokenizer/test_026_bpe_core.py, status:draft`。

- [ ] **Step 2: 原理回顾**

精简回引 Ch 17 的 BPE 算法（合并最高频对）。≥1 Mermaid：`train_bpe` 主循环（统计对→选最高频→合并→分配 new_id→重复）。

- [ ] **Step 3: 代码实现（6 函数逐一摘录 + file:line）**

- `byte_level_encode`（`bpe.py:8-14`）：`list(text.encode("utf-8"))`，解释为何字节级（无 OOV，中文 3 字节）。
- `get_pair_counts`（`bpe.py:17-27`）：扫描相邻对。
- `merge`（`bpe.py:30-44`）：从左到右，已合并不重复参与——配 ASCII 讲 `[1,2,1,2]→[99,99]`。
- `train_bpe`（`bpe.py:78-99`）：`num_merges=vocab_size-256`，`new_id=256+i`，`max(counts,...)`。
- `encode`（`bpe.py:47-56`）：按 new_id 升序逐条应用 merge。
- `decode`（`bpe.py:59-75`）：递归 `expand` 展开回字节。
每段 5–20 行摘录 + 引用行。

- [ ] **Step 4: 对应单元测试（file:line）**

讲 `test_026_bpe_core.py` 四组测试意图：TestByteLevelEncode（中文 3 字节 `:19-24`）、TestGetPairCounts（最高频对 `:53-55`）、TestMerge（相邻不重复合并 `:65-67`、重叠对 `:69-71`）、TestTrainBPE（首合并是最高频 `(97,98)` `:81-86`）。讲 `test_041_encode_decode.py` TestRoundtrip（`:49-61`，中英混合往返）。引用行格式准确。

- [ ] **Step 5: 动手验证**

```bash
pytest tests/m02_tokenizer/test_026_bpe_core.py tests/m02_tokenizer/test_041_encode_decode.py -v
```
预期：全 PASSED。配一段交互（可选）：在 Python 里 `train_bpe(["abab"], vocab_size=258)` 看 merges。

- [ ] **Step 6: 小结 + 下章预告**

小结：纯 Python BPE 读懂了，但大规模语料太慢。预告 Ch 19：生产级 HF tokenizers 实现。

- [ ] **Step 7: 校验 + commit**

校验：`grep -cE "\.py:[0-9]"` ≥ 6（6 个函数各一引用）；抽查 `bpe.py:78`、`test_026_bpe_core.py:65`、`test_041_encode_decode.py:49` 真实；无 TBD；README Ch 18 → ✅。Commit: `docs(book): write Ch18 teaching BPE implementation (M2-a)`。

---

### Task 21: Ch 19 生产版 Tokenizer + 特殊 Token + Chat Template（M2-b）

**Files:**
- Create: `docs/book/part-3-tokenizer/ch19-production-tokenizer.md`
- Modify: `docs/book/README.md`（Ch 19 行 ☐→✅，Part III 完成）
- Read-only refs: `zllm/tokenizer/{trainer,special_tokens,chat_template,adapter}.py`、`tests/m02_tokenizer/{test_030_special_tokens,test_035_production_tokenizer,test_044_chat_template,test_050_integration}.py`

**Interfaces:**
- Consumes: Ch 17（BPE 理论）、Ch 18（教学版，对比）、Ch 15（对话格式 im_start/im_end）
- Produces: Part III 收官；为 Ch 27（TokenizerAdapter 在数据流水线）、Ch 33（SFT 用 chat template）、Ch 40（Agent RL 用 toolcall token）铺路

- [ ] **Step 1: front-matter + 6 节**

YAML：`part:3, chapter:19, title:生产版 Tokenizer + 特殊 Token + Chat Template, milestone:M2, source:zllm/tokenizer/trainer.py, tests:tests/m02_tokenizer/test_035_production_tokenizer.py, status:draft`。

- [ ] **Step 2: 原理回顾 + 四大组件总览**

回引 Ch 18 教学版（慢），引出生产版四大件：训练器（trainer）、特殊 token（special_tokens）、对话模板（chat_template）、适配器（adapter）。≥1 Mermaid：`train_tokenizer → special_tokens 注入 → chat_template 渲染 → adapter 包装 → 供训练/推理调用`。

- [ ] **Step 3: 代码实现（4 文件摘录 + file:line）**

- **trainer**：`train_tokenizer`（`trainer.py:18-43`，`Tokenizer(BPE())`、`ByteLevel` pre-tokenizer `:31`、`BpeTrainer(special_tokens=ALL_SPECIAL_TOKENS, initial_alphabet=ByteLevel.alphabet())` `:34-38`、`train_from_iterator` `:39`、存 `tokenizer.json` `:42`）；`load_tokenizer`（`:46-57`）。
- **special_tokens**：`SPECIAL_TOKENS` 列表（`special_tokens.py:12-34`，对话边界/pad/多模态预留/toolcall `📞`/reasoning）；`BUFFER_TOKENS`（`:37`，8 个预留位）；`ALL_SPECIAL_TOKENS`（`:40`）。
- **chat_template**：`render_messages`（`chat_template.py:25-72`，tools 注入 system `:46-55`、open_thinking 包裹 `:64-65`、add_generation_prompt `:69-70`）；Jinja2 `CHAT_TEMPLATE`（`:76-94`）。
- **adapter**：`TokenizerAdapter`（`adapter.py:10-65`，bos/eos/pad id `:13-18`、`__call__` 加 special `:44-52`、`apply_chat_template` `:54-65`）；`wrap`（`:74-78`）。
每文件 5–20 行摘录 + 引用。

- [ ] **Step 4: 对应单元测试（file:line）**

讲 4 个测试文件意图：`test_030_special_tokens.py`（无重复 `:24-25`、toolcall `📞` `:39-44`、buffer 计数 `:52-53`）；`test_035_production_tokenizer.py`（special token 单 token `:45-54`、roundtrip 中文 `:72-75`、压缩 `:99-103`）；`test_044_chat_template.py`（generation prompt `:53-56`、open_thinking `:65-71`、tools `:87-93`）；`test_050_integration.py`（全管线 train→save→load→render→encode→decode `:25-34`）。

- [ ] **Step 5: 动手验证**

```bash
pytest tests/m02_tokenizer/ -v
```
预期：M2 全部测试 PASSED。配一段说明：特殊 token ID 占词表前部（`test_035:56-59`，`vocab["<|im_start|>"] < 50`）。

- [ ] **Step 6: 小结 + 下章预告 + Part III 收官**

小结：tokenizer 管线全通。Part III 收官语：从 config 到 tokenizer，基石铺好。下章预告：Part IV Ch 20（RMSNorm）开始搭模型。

- [ ] **Step 7: 校验 + commit**

校验：`grep -cE "\.py:[0-9]"` ≥ 8（4 文件各 ≥2 引用）；抽查 `trainer.py:34`、`special_tokens.py:12`、`chat_template.py:46`、`adapter.py:54`、`test_044_chat_template.py:65`、`test_050_integration.py:25` 真实；无 TBD；README Ch 19 → ✅（Part III 全 ✅）。Commit: `docs(book): write Ch19 production tokenizer + special tokens + chat template (M2-b)`。

---

### Phase 3 完工标准（DoD）

- Ch 16–19 全部写完，README 第 16–19 行全 ✅。
- 实战章（16/18/19）`grep -cE "\.py:[0-9]"` ≥ 5/6/8；Ch 17（理论章）≥3 `$$`。
- 全 Phase 无 TBD/TODO/占位符。
- 所有引用的 zllm file:line 经 `read`/`grep` 抽查真实存在（不可臆造行号）。
- Phase 3 完成后合并 main，更新本计划第 68 行进度表为「已完成」。

---

## Phase 4 详细任务（Part IV，Ch 20–26，模型架构 M3+M4）

**范围：** Ch 20–Ch 26（7 章）。里程碑 M3（模型组件：RMSNorm/RoPE+YaRN/GQA 注意力）+ M4（模型组装：SwiGLU/MoE/Block+Backbone/CausalLM）。全部为**实战章**（模板 spec 3.2），引用 `zllm/model/*.py` + `tests/m03_model_components` + `tests/m04_model_assembly`。

**前置事实（Phase 4 全部任务共用，源自 2026-07-04 探源，确保 file:line 准确）：**

- `zllm/model/norms.py`（21 行）：`RMSNorm(nn.Module)` `:11`；`__init__(dim, eps=1e-5)` `:12-15`（`weight=ones(dim)`）；`norm(x)` `:17-18`（`x*rsqrt(x.pow(2).mean(-1,keepdim)+eps)`，**不乘 weight**）；`forward` `:20-21`（`(weight*self.norm(x.float())).type_as(x)`——float32 内部计算防溢出）。
- `zllm/model/rope.py`（69 行）：`precompute_freqs_cis(dim,end=32768,rope_base=1e6,rope_scaling=None)` `:17-56`（freqs=`1/rope_base^(2k/d)` `:23-28`；YaRN ramp 混合 `:30-51`；`outer(t,freqs)` `:53`；cos/sin 各 `cat` 复制一份 `:54-55`）；`rotate_half(x)` `:59-63`（`cat(-x[d//2:], x[:d//2])`）；`apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1)` `:66-69`（`q*cos + rotate_half(q)*sin`）。
- `zllm/model/attention.py`（125 行）：`repeat_kv(x,n_rep)` `:22-30`（expand+reshape 复制 KV 头）；`Attention.__init__` `:33-69`（`n_rep=num_attention_heads//num_key_value_heads` `:43`；q/k/v/o_proj 无 bias；`q_norm/k_norm=RMSNorm(head_dim)` `:60-61`；`flash` 标志 `:66-69`）；`forward` `:71-125`（q/k/v proj→reshape→**q_norm/k_norm** `:81`→RoPE `:84`→**KV cache concat** `:86-89`→repeat_kv `:92-93`→flash SDPA `:95-109` 或手动 `scores/sqrt(head_dim)`+causal mask triu `:110-121`→o_proj `:124`）。
- `zllm/model/ffn.py`（67 行）：`FeedForward` `:17-27`（gate_proj/up_proj/down_proj 无 bias `:21-23`，act_fn=`ACT2FN[hidden_act]` `:24`；forward `down(silu(gate(x))*up(x))` `:26-27`）；`MOEFeedForward` `:30-67`（gate=`Linear(hidden,num_experts)` `:34`；experts=ModuleList of FeedForward `:35-37`；forward：softmax gate `:43`、topk `:44-46`、norm_topk_prob 归一 `:47-48`、per-expert `index_add_` `:50-55`、空专家梯度保持 `:56-57`、aux_loss 负载均衡 `:58-66`）。
- `zllm/model/block.py`（31 行）：`ZLLMBlock.__init__` `:15-21`（self_attn/input_layernorm/post_attention_layernorm=RMSNorm/mllp=`FeedForward if not use_moe else MOEFeedForward` `:21`）；`forward` `:23-31`（**Pre-Norm**：`attn(input_norm(x))+residual` `:24-29`，`mlp(post_norm(x))+residual` `:30`）。
- `zllm/model/backbone.py`（77 行）：`ZLLMModel.__init__` `:17-36`（embed_tokens `:23`、layers=ModuleList of ZLLMBlock `:25-27`、norm `:28`、预计算并 register_buffer freqs_cos/sin `:29-36`）；`forward` `:38-77`（embed→按 start_pos 切片 position_embeddings `:43-61`→循环 layer `:63-71`→final norm `:72`→汇总 MoE aux_loss `:73-76`）。
- `zllm/model/causal_lm.py`（54 行）：`ZLLMForCausalLM(PreTrainedModel,GenerationMixin)` `:20`；`_tied_weights_keys={"lm_head.weight":"model.embed_tokens.weight"}` `:22`；`__init__` `:24-31`（model=ZLLMModel、lm_head=`Linear(hidden,vocab)` `:28`、weight tying `:29-30`）；`forward` `:33-54`（hidden,past,aux=model(...) `:34-36`；**logits_to_keep 切片** `:37-42`；labels 时交叉熵 `x[...,:-1] vs y[...,1:]`、`ignore_index=-100` `:44-47`；返回 `MoeCausalLMOutputWithPast` `:48-54`）。
- 测试：`test_052_rmsnorm.py`（weight ones `:17`、identity `:37`、scaling fill_2.0 `:44`、float32 内部 `:65`、norm 方法不乘 weight `:80`）；`test_057_rope.py`（shapes `:14`、pos0 cos=1/sin=0 `:19-25`、rope_base 影响周期 `:33`、cat 复制 `:40`、rotate_half `:52-58`、pos0 identity `:76`）；`test_063_yarn.py`（缩放频率 `:27`、**低于 orig_max 不生效** `:33`、attn_factor `:40`、pos0 不变 `:53`）；`test_067_attention.py`（repeat_kv `:15-32`、linear shapes `:36`、n_rep `:51`、qk_norm 存在 `:55`、**kv cache concat** `:90-99`、causal `:101`、flash vs manual close `:120`）；`test_082_ffn.py`（linear shapes `:15`、**swiglu 公式手算** `:49-58`、π 缩放 `:33-39`）；`test_093_moe.py`（moe_config fixture `:13-25`、gate shape `:29`、sparse top1 `:51`、aux_loss 训练非零 `:69`、eval 为零 `:77`、aux_loss backward `:102`）；`test_097_block.py`（has self_attn `:15`、双 norm `:19`、moe 切换 `:29`、residual `:48`、moe block `:73`）；`test_104_backbone.py`（embed `:15`、layer count `:19`、rope buffers `:27`、**kv cache 增量 4→6** `:58-65`、moe aux_loss `:67`、attention_mask `:84`）；`test_109_causal_lm.py`（lm_head shape `:15`、**weight tying equal** `:19`、tying disabled `:23`、loss with labels `:48`、ignore_index -100 `:56`、logits_to_keep `:73`、aux_loss moe `:85`、past_key_values 长度=层数 `:95`）。

---

### Task 22: Ch 20 RMSNorm 归一化（M3-a）
Create `docs/book/part-4-architecture/ch20-rmsnorm.md`。YAML：`part:4,chapter:20,title:RMSNorm 归一化,milestone:M3,source:zllm/model/norms.py,tests:tests/m03_model_components/test_052_rmsnorm.py,status:draft`。6 节实战模板。原理回顾回引 Ch 13（LayerNorm）+ 讲 RMSNorm 省「减均值」的效率。代码：`norms.py:17-21`（norm 方法 + forward float32）。重点推导 RMS 公式 $\text{RMS}(x)=\sqrt{\frac{1}{d}\sum x_i^2}$，对比 LayerNorm $\frac{x-\mu}{\sigma}$ 少算 $\mu$、省一次同步。测试：test_052 identity `:37`、scaling `:44`、float32 内部 `:65`、norm 方法 `:80`。pytest `tests/m03_model_components/test_052_rmsnorm.py -v`。校验 `grep -cE "\.py:[0-9]"`≥4。README Ch20→✅。Commit `docs(book): write Ch20 RMSNorm normalization (M3-a)`。

### Task 23: Ch 21 RoPE 旋转位置编码 + YaRN（M3-b）
Create `ch21-rope-yarn.md`。YAML `source:zllm/model/rope.py,tests:tests/m03_model_components/test_057_rope.py`。原理：回引 Ch 12/13（位置编码必要性）。核心推导——复数旋转：把 head_dim 两两配对视为 2D 向量，乘旋转矩阵 $R_\theta=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$，$\theta_i=m/\theta^{2i/d}$；证明 $\langle R_m q, R_n k\rangle$ 只依赖相对位置 $m-n$（相对位置编码的本质）。代码：`precompute_freqs_cis` `:17-56`、`rotate_half` `:59-63`、`apply_rotary_pos_emb` `:66-69`。YaRN：`rope.py:30-51` 的 ramp 混合，长序列外推（仅 end>orig_max 生效）。测试 test_057（pos0 identity `:76`、rope_base 周期 `:33`）+ test_063（低于 orig_max 不变 `:33`）。校验 `grep "\.py:[0-9]"`≥5。README Ch21→✅。Commit `docs(book): write Ch21 RoPE + YaRN (M3-b)`。

### Task 24: Ch 22 GQA 注意力 + QK-Norm + KV Cache（M3-c）
Create `ch22-gqa-qknorm-kv-cache.md`。YAML `source:zllm/model/attention.py,tests:tests/m03_model_components/test_067_attention.py`。原理回引 Ch 12（点积注意力 $\sqrt{d_k}$）+ Ch 13（GQA 2:1）。代码：`repeat_kv` `:22-30`（KV 头复制 n_rep）、`Attention.__init__` qk_norm `:60-61`、forward KV cache concat `:86-89`、flash/manual 双路径 `:95-121`（causal mask triu(1) `:113-118`）。重点：GQA 省 KV cache、QK-Norm 稳训练、KV cache 推理加速。测试 test_067（repeat_kv `:21-24`、kv cache concat 4→6 `:90-99`、flash vs manual `:120`）。pytest test_067。校验 `grep "\.py:[0-9]"`≥6。README Ch22→✅。Commit `docs(book): write Ch22 GQA attention + QK-Norm + KV Cache (M3-c)`。

### Task 25: Ch 23 SwiGLU 前馈网络（M4-a）
Create `ch23-swiglu.md`。YAML `source:zllm/model/ffn.py,tests:tests/m04_model_assembly/test_082_ffn.py`。原理：回引 Ch 13（FFN）+ Ch 09（激活函数）。SwiGLU 公式 $\text{down}(\text{SiLU}(\text{gate}(x))\odot\text{up}(x))$，SiLU=$x\sigma(x)$；对比 ReLU FFN 的门控机制。代码：`FeedForward` `:17-27`（gate/up/down_proj）。测试 test_082（**swiglu 公式手算验证** `:49-58`、π 缩放 intermediate `:33-39` 回引 Ch16）。pytest test_082。校验 `grep "\.py:[0-9]"`≥4。README Ch23→✅。Commit `docs(book): write Ch23 SwiGLU FFN (M4-a)`。

### Task 26: Ch 24 MoE 混合专家（M4-b）
Create `ch24-moe.md`。YAML `source:zllm/model/ffn.py,tests:tests/m04_model_assembly/test_093_moe.py`。原理：稀疏激活——每个 token 只路由到 top-k 专家，参数多但计算量可控。推导：门控 $g=\text{softmax}(W_g x)$，top-k 选择 + 归一；aux_loss 负载均衡 $L_{aux}=N\cdot\text{coef}\sum_i f_i\cdot P_i$（$f$=实际负载，$P$=平均概率）。代码：`MOEFeedForward` `:30-67`（gate `:34`、topk `:44-46`、index_add_ `:50-55`、空专家梯度保持 `:56-57`、aux_loss `:58-66`）。测试 test_093（sparse top1 `:51`、aux_loss 训练非零/eval 零 `:69/77`、aux backward `:102`）。pytest test_093。校验 `grep "\.py:[0-9]"`≥5。README Ch24→✅。Commit `docs(book): write Ch24 MoE mixture of experts (M4-b)`。

### Task 27: Ch 25 Block + Backbone 组装（M4-c）
Create `ch25-block-backbone.md`。YAML `source:zllm/model/{block,backbone}.py,tests:tests/m04_model_assembly/test_097_block.py`。原理回引 Ch 13（Pre-Norm Transformer block）+ Ch 10（残差连接）。代码：`ZLLMBlock` Pre-Norm 双残差 `:23-31`、`ZLLMModel` embed+layers+norm+freqs buffers `:17-36`、forward 位置切片 `:43-61`。重点：Pre-Norm 训练稳定（vs Post-Norm）、残差让深网络可训（回引 Ch10）。测试 test_097（双 norm `:19`、moe 切换 `:29`、residual `:48`）+ test_104（layer count `:19`、rope buffers `:27`、**kv cache 增量** `:58-65`）。pytest test_097 test_104。校验 `grep "\.py:[0-9]"`≥6。README Ch25→✅。Commit `docs(book): write Ch25 Block + Backbone assembly (M4-c)`。

### Task 28: Ch 26 CausalLM 头 + Weight Tying + Loss（M4-d）
Create `ch26-causal-lm-head.md`。YAML `source:zllm/model/causal_lm.py,tests:tests/m04_model_assembly/test_109_causal_lm.py`。原理回引 Ch 05（交叉熵=NTP 目标）+ Ch 15（Weight Tying 参数效率）。代码：`ZLLMForCausalLM` weight tying `:22/29-30`、forward `:33-54`（logits_to_keep 切片 `:37-42`、**NTP loss** `x[:-1] vs y[1:]` + `ignore_index=-100` `:44-47`）。重点：Weight Tying 省 $V\times d$ 参数、logits_to_keep 推理优化、ignore_index 服务 SFT label masking（Ch33）。测试 test_109（weight tying equal `:19`、loss with labels `:48`、ignore_index `:56`、logits_to_keep `:73`、aux_loss moe `:85`）。pytest test_109。Part IV 收官。校验 `grep "\.py:[0-9]"`≥5。README Ch26→✅。Commit `docs(book): write Ch26 CausalLM head + weight tying + loss (M4-d)`。

### Phase 4 完工标准（DoD）
- Ch 20–26 全部写完，README 第 20–26 行全 ✅（Part IV 完成）。
- 每章 `grep -cE "\.py:[0-9]"` 达标（见各 Task）。
- 全 Phase 无 TBD/TODO/占位符；所有引用 file:line 经 `sed -n` 抽查真实。
- Phase 4 完成后更新本计划进度表第 69 行为「已完成」。

---

## Phase 5–7 详细任务（待后续会话追加）

> 占位说明（计划本身的进度标记，非内容占位）：以下 Phase 将在后续会话按同样 Task 粒度追加。

- **Phase 5（Part V，Ch 27–32，6 章）**：M5+M6+M7。引用 `zllm/dataset/*.py`、`zllm/training/{utils,amp,gpu,pretrain}.py`、`tests/m05_data_pipeline`、`tests/m06_training`、`tests/m07_pretrain`。

### 前置事实块（Phase 5 引用源，已逐行核实）

**`zllm/dataset/utils.py`（37 行）**：`SYSTEM_PROMPTS` 10 条中英 `:10-21`；`pre_processing_chat` 概率加 system prompt `:24-30`（有 tools 直接返回 `:25-26`、已有 system 不重复 `:27`、`random.random() < ratio` `:28-29`）；`post_processing_chat` 概率移除空 think 标签 `:33-37`（`<reasoningchain_start>\n\n<reasoningchain_end>` `:34`、`random.random() > ratio` 移除 `:35-36`）。

**`zllm/dataset/pretrain.py`（41 行）**：`PretrainDataset.__init__` `:18-23`（`wrap(tokenizer)` `:21`、`load_dataset("json")` `:23`）；`__getitem__` `:28-41`（encode `add_special_tokens=False` + `max_length-2` 截断 `:30-35`、加 BOS/EOS `:36`、pad 补齐 `:37`、`labels = input_ids.clone()` `:39`、**pad 位置 → -100** `:40`）。

**`zllm/dataset/sft.py`（81 行）**：`SFTDataset.__init__` `:20-36`（features schema `:25-33`、`bos_id`/`eos_id` 编码 `:35-36`）；`create_chat_prompt` `:41-53`（tools 解析 `:46-47`、`apply_chat_template` `:51-53`）；**`generate_labels`** `:55-71`（全 -100 起始 `:56`、找 `bos_id` 锚点 `:59`、找 `eos_id` 结束 `:63`、start..end+eos 标记真实 id `:66-67`）；`__getitem__` `:73-81`（pre/post processing `:75-77`、encode+pad `:78-79`）。

**`zllm/dataset/dpo.py`（70 行）**：`DPODataset.__init__` `:19-26`；`__getitem__` `:31-52`（chosen/rejected 各渲染+编码 `:33-42`、**错位一位** `ids[:-1]`/`ids[1:]` `:46-51`、返回 6 个张量 dict `:45-52`）；`generate_loss_mask` `:54-70`（与 sft generate_labels 同构，但 mask=0/1 而非 -100/id）。

**`zllm/dataset/rlaif.py`（42 行）**：`RLAIFDataset.__init__` `:18-24`（`thinking_ratio=0.5` `:23`）；`create_chat_prompt` `:29-37`（`pre_processing_chat` `:30`、**`use_thinking = random < ratio`** `:31`、`conversations[:-1]` 去掉答案 `:33`、`add_generation_prompt=True` `:36`）；`__getitem__` `:39-42`（只返回 prompt，answer=""）。

**`zllm/dataset/agent.py`（40 行）**：`AgentRLDataset.__init__` `:14-22`（手读 JSONL `:20-22`）；`parse_conversations` `:27-35`（tools 解析 `:32-33`、**`messages[:-1]` 去最后一条 assistant** `:35`）；`__getitem__` `:37-40`（返回 messages+tools+gt）。

**`zllm/training/utils.py`（165 行）**：`get_lr` **余弦退火** `:46-47`（`base*(0.1+0.45*(1+cos(π·step/total)))`，起点=base、终点=0.1·base）；`setup_seed` `:59-64`（random/np/torch/cuda 四重种子）；`init_model` `:67-78`（权重路径 `f"{from_weight}_{hidden}{_moe}.pth"` `:72-73`、`load_state_dict(strict=False)` `:75`）；**`lm_checkpoint`** `:81-139`（原子写 `.tmp`→`os.replace` `:96-98/125-127`、half().cpu() `:94`、resume 含 optimizer/epoch/step/world_size/wandb_id `:108-115`、读时 world_size 变化自动换算 step `:133-137`）；`SkipBatchSampler` `:142-165`（断点续训跳过已训 batch）。

**`zllm/training/amp.py`（76 行）**：`GradScalerManager` `:12-29`（封装 `torch.amp.GradScaler`）；**`train_step`** `:32-76`（autocast bf16 `:61`、`loss = output.loss + aux_loss` `:63`、`/ accumulation_steps` `:64`、`scaler.scale().backward()` `:66`、累积边界判断 `:68`、unscale→clip→step→update→zero_grad `:70-74`）。

**`zllm/training/gpu.py`（25 行）**：`enable_tf32` `:9-11`、`enable_cudnn_benchmark` `:14-15`、`enable_flash_sdpa` `:18-19`、`setup_gpu_performance` `:22-25`。

**`zllm/training/pretrain.py`（120 行）**：`PretrainConfig` dataclass `:17-36`（epochs=2/batch=64/lr=5e-4/accum=4/grad_clip=1.0/max_seq_len=340）；**`train_epoch`** `:51-120`（余弦 lr 调度 `:82-84`、autocast+loss+accum `:86-89`、scaler 反传 `:91`、边界 unscale→clip→step `:93-99`、log `:104-108`、末尾补齐残余累积 `:112-118`）；`_format_duration` `:39-48`。

**测试**：`test_113_utils`（pre_processing ratio `:13/19`、tools 保留 `:34`、empty think 移除/保留 `:49/55`）；`test_115_pretrain`（BOS at start `:51`、pad→-100 `:55-59`、non-pad labels==input `:61-65`）；`test_121_sft`（labels 有 valid `:53`、prompt masked `:58`、pad masked `:63`、generate_labels 无 assistant 全 -100 `:70`、assistant region marked `:75`）；`test_126_dpo`（6 keys `:49-56`、错位 shape `(63,)` `:60`、mask has ones `:69`、chosen≠rejected `:74`）；`test_130_rlaif_agent`（prompt+generation_prompt `:50-52`、agent messages/tools/gt `:88/93/98`、drop last `:103`）；`test_145_utils`（lr 起点 `:40-43`、终点 `:45-48`、中点 `:55-58`、seed 可复现 `:24-29`、checkpoint save/load `:100-139`、moe suffix `:141-152`）；`test_161_amp`（basic step `:40`、grad clip `:48`、accumulation `:57`、amp cuda `:67`、zero_grad `:76`）；`test_166_gpu`（SkipBatchSampler skip `:32-37`、tf32/flash `:52-55`）；`test_194_loss_ntp`（**loss 下降** `:42-57`、**过拟合** `:61-73`、**下一 token 预测 >10%** `:75-102`）。

### Task 29: Ch 27 数据流水线总览与 TokenizerAdapter（M5-a）
Create `part-5-training/ch27-data-pipeline-adapter.md`。YAML `source:zllm/dataset/utils.py,tests:tests/m05_data_pipeline/test_113_utils.py`。原理：回引 Ch 19（tokenizer）+ Ch 16（数据是训练燃料），讲数据流水线全景（原始文本→预处理→tokenize→batch→labels/mask），Mermaid 画五种 Dataset 的关系图。代码：`dataset/utils.py` 的 `pre_processing_chat`（概率加 system prompt `:24-30`）+ `post_processing_chat`（概率移除空 think 标签 `:33-37`）+ `wrap()` adapter 介绍。重点：概率增强为什么有益（数据多样性）、empty think 移除让模型学会「该想就想、不想就不写空标签」。测试 test_113（ratio 0/1 `:13/19`、tools 保留 `:34`、think 移除/保留 `:49/55`）。pytest test_113。校验 `grep "\.py:[0-9]"`≥4。README Ch27→✅。Commit `docs(book): write Ch27 data pipeline + TokenizerAdapter (M5-a)`。

### Task 30: Ch 28 五种 Dataset 实现（M5-b）
Create `part-5-training/ch28-dataset-implementations.md`。YAML `source:zllm/dataset/{pretrain,sft,dpo,rlaif,agent}.py,tests:tests/m05_data_pipeline/test_115_pretrain.py`。原理：回引 Ch 05（NTP）+ Ch 26（ignore_index=-100），讲五种 Dataset 各自产出什么——Pretrain（input_ids, labels 全保留）、SFT（labels 只留 assistant 区域）、DPO（chosen/rejected 错位 + loss_mask）、RLAIF（prompt-only）、Agent（messages+tools+gt）。Mermaid 对比五种 Dataset 的输入输出。代码：PretrainDataset（BOS/EOS/pad/-100 `:36-40`）、SFTDataset（**generate_labels** bos→eos 锚点 `:55-71`）、DPODataset（错位 `:46-51` + loss_mask `:54-70`）、RLAIFDataset（thinking_ratio `:31` + prompt-only `:39-42`）、AgentRLDataset（drop last assistant `:35`）。重点：SFT label masking 是「只对回复算 loss」的核心、DPO 与 SFT 的 mask 差异（-100 vs 0/1）、RLAIF 为什么不需要 labels。测试 test_115（BOS `:51`、pad→-100 `:55`）+ test_121（labels valid `:53`、prompt masked `:58`）+ test_126（6 keys `:49`、错位 shape `:60`）。pytest test_115 test_121 test_126 test_130。校验 `grep "\.py:[0-9]"`≥8。README Ch28→✅。Commit `docs(book): write Ch28 five Dataset implementations (M5-b)`。

### Task 31: Ch 29 训练基础设施：种子/学习率/checkpoint（M6-a）
Create `part-5-training/ch29-training-infrastructure.md`。YAML `source:zllm/training/utils.py,tests:tests/m06_training/test_145_utils.py`。原理：回引 Ch 09（SGD）+ Ch 11（学习率调度），讲种子可复现、**余弦退火**学习率曲线、checkpoint 原子写入与断点续训。LaTeX 画余弦退火公式 $\eta_t = \eta_0(0.1+0.45(1+\cos(\pi t/T)))$。代码：`get_lr` `:46-47`（起点 base、终点 0.1·base）、`setup_seed` `:59-64`（四重种子）、`init_model` `:67-78`（权重路径命名 `:72-73`）、**`lm_checkpoint`** `:81-139`（原子写 `.tmp`→`replace` `:96-98`、half().cpu() `:94`、resume dict `:108-115`、world_size 变化自动换算 `:133-137`）、`SkipBatchSampler` `:142-165`。重点：余弦退火为什么有效（大步快学→小步精调）、原子写防崩溃、断点续训的工程价值、MoE 参数统计 `A{active}M` 含义。测试 test_145（lr 三点 `:40/45/55`、seed 复现 `:24`、checkpoint save/load `:100`、moe suffix `:141`）。pytest test_145 test_166（SkipBatchSampler）。校验 `grep "\.py:[0-9]"`≥7。README Ch29→✅。Commit `docs(book): write Ch29 training infrastructure (M6-a)`。

### Task 32: Ch 30 混合精度 AMP + 梯度累积 + GPU 优化（M6-b）
Create `part-5-training/ch30-amp-grad-accumulation.md`。YAML `source:zllm/training/amp.py,tests:tests/m06_training/test_161_amp.py`。原理：回引 Ch 08（数值稳定性）+ Ch 10（梯度），讲 bf16 混合精度（前向 bf16、梯度累加 fp32）、梯度累积（小 batch 模拟大 batch）、梯度裁剪（防爆炸）、TF32/cuDNN/Flash SDPA 硬件加速。LaTeX 梯度累积公式。代码：**`train_step`** `:32-76`（autocast `:61`、`loss+aux_loss` `:63`、`/accum` `:64`、scaler 反传 `:66`、边界 unscale→clip→step `:68-74`）、`GradScalerManager` `:12-29`、`gpu.py` 的 TF32/Flash `:9-25`。重点：bf16 vs fp16（动态范围 vs 精度）、为什么 unscale 后才 clip、梯度累积等价大 batch 的数学、GPU 三大开关各自加速什么。测试 test_161（basic `:40`、clip `:48`、accumulation `:57`、amp cuda `:67`、zero_grad `:76`）+ test_166（tf32 `:52`）。pytest test_161 test_166。校验 `grep "\.py:[0-9]"`≥6。README Ch30→✅。Commit `docs(book): write Ch30 AMP + grad accumulation + GPU optimization (M6-b)`。

### Task 33: Ch 31 预训练：NTP 与训练循环（M7 理论+实现）
Create `part-5-training/ch31-pretraining-ntp.md`。YAML `source:zllm/training/pretrain.py,tests:tests/m07_pretrain/test_194_loss_ntp.py`。原理：回引 Ch 05（NTP 损失推导）+ Ch 15（训练目标），把 Ch 26 的 CausalLM loss 接上 Ch 30 的 train_step，讲完整训练循环的五个阶段（forward→loss→backward→step→log）。Mermaid 画训练循环状态机。代码：**`train_epoch`** `:51-120`（余弦 lr 调度 `:82-84`、autocast+loss+accum `:86-89`、scaler 反传 `:91`、边界 step `:93-99`、log `:104-108`、末尾补齐 `:112-118`）、`PretrainConfig` dataclass `:17-36`。重点：训练循环 = 前四章零件的总装、余弦 lr 每 step 更新、accumulation 末尾不整除的补齐逻辑、log_interval 的监控意义。测试 test_194（**loss 下降** `:42-57`、**过拟合** `:61-73`、**下一 token 预测 >10%** `:75-102`）。pytest test_194。校验 `grep "\.py:[0-9]"`≥6。README Ch31→✅。Commit `docs(book): write Ch31 pretraining NTP + training loop (M7)`。

### Task 34: Ch 32 预训练实战：数据准备/训练/loss 监控（M7 实操）
Create `part-5-training/ch32-pretraining-practice.md`。YAML `source:zllm/training/pretrain.py,zllm/dataset/pretrain.py,tests:tests/m07_pretrain/test_200_integration.py`。原理：实操章，把 Ch 27-31 串起来，给出端到端的预训练流程（数据格式→tokenizer→Dataset→DataLoader→模型→训练循环→loss 监控→checkpoint）。重点在「怎么做」而非新代码，大量引用前 5 章的 file:line。代码：组装 `PretrainDataset` + `DataLoader` + `ZLLMForCausalLM` + `train_epoch`，给出完整可跑的 20 行脚本片段。重点：数据量与 epoch 的经验、loss 曲线的健康形态（先快降后缓降→平台）、何时该 SFT（预训练 loss 平台期）、小卡训练的 batch/accum 折衷。动手验证：跑 test_200_integration。校验 `grep "\.py:[0-9]"`≥4。README Ch32→✅，Part V 收官。Commit `docs(book): write Ch32 pretraining practice (M7); completes Part V`。

### Phase 5 完工标准（DoD）
- Ch 27–32 全部写完，README 第 27–32 行全 ✅（Part V 完成）。
- 每章 `grep -cE "\.py:[0-9]"` 达标（见各 Task）。
- 全 Phase 无 TBD/TODO/占位符；所有引用 file:line 经 `sed -n` 抽查真实。
- Phase 5 完成后更新本计划进度表第 70 行为「已完成」。
- **Phase 6（Part VI，Ch 33–40，8 章）**：M8-M11。引用 `zllm/training/{full_sft,dpo,ppo,grpo,distillation,agent_rl,lora_sft}.py`、`zllm/model/lora.py`、`tests/m08_sft`、`tests/m09_lora`、`tests/m10_alignment`、`tests/m11_distill_agent`。

### 前置事实块（Phase 6 引用源，已逐行核实）

**`zllm/model/lora.py`（130 行）**：`LoRA` 类 `:21-39`（A 降维 `:33`、B 升维 `:34`、A 高斯 `:35`、**B 零初始化** `:36`、`forward = B(A(x))` `:38-39`）；`apply_lora` `:42-60`（只注入方阵 `in==out` `:52`、monkey-patch `forward(x) = original + lora(x)` `:57-60`）；`freeze_non_lora` `:69-79`（非 lora `requires_grad=False`）；`save_lora` `:82-94`（只存 A/B）；`load_lora` `:97-112`；**`merge_lora`** `:115-130`（`W_merged = W + B@A` `:128-129`）。

**`zllm/training/full_sft.py`（112 行）**：`SFTConfig` `:22-40`（**lr=1e-5**（预训练 1/50）`:25`、max_seq_len=768 `:30`、from_weight='pretrain' `:37`、save_weight='full_sft' `:36`）；`train_epoch` `:43-112`（与 pretrain 同结构，用 SFTDataset + SFTConfig）。

**`zllm/training/lora_sft.py`（96 行）**：`LoRAConfig` `:19-37`（**lr=1e-4**（full_sft 的 10 倍）`:22`、rank=16 `:23`、epochs=10 `:20`、from_weight='full_sft' `:35`）；`train_epoch` `:40-96`（**梯度裁剪只作用于 lora_params** `:72/91`）。

**`zllm/training/dpo.py`（162 行）**：`logits_to_log_probs` `:20-31`（log_softmax + gather）；**`dpo_loss`** `:34-61`（`(ref·policy).sum(mask)` `:48-49`、前半 chosen 后半 rejected `:52-55`、`logits = π_logratios - ref_logratios` `:57-59`、`loss = -logsigmoid(β·logits)` `:60`）；`train_epoch` `:86-162`（双模型 policy+ref `:99-100`、ref no_grad `:121-122`、cat chosen+rejected `:112-114`）。

**`zllm/training/ppo.py`（157 行）**：`CriticModel` `:29-44`（继承 ZLLMForCausalLM、value_head Linear(hidden,1) `:38`、forward 返回 values `:40-44`）；**`compute_gae`** `:47-81`（`δ_t = r + γV_{t+1} - V_t` `:70`、`A = δ + γλ·last_gae` `:71`、标准化 `:75-78`、returns = adv + values `:80`）；`ppo_policy_loss` `:84-103`（ratio=exp(new-old) `:99`、`min(ratio·A, clip·A)` `:100-102`）；`ppo_value_loss` `:106-125`（clipped value loss）；`PPOConfig` `:128-157`（clip_epsilon=0.2 `:134`、vf_coef=0.5 `:135`、kl_coef=0.02 `:136`、gamma=1.0/lam=0.95 `:137-138`）。

**`zllm/training/grpo.py`（133 行）**：`per_token_kl` `:24-37`（`exp(ref-policy) - (ref-policy) - 1` `:36`）；**`compute_group_advantages`** `:40-56`（组内 `(r - mean)/std` `:56`）；`grpo_loss` `:59-85`（ratio clip [1-ε,1+ε] `:80`、`min - β·KL` `:84`）；**`cispo_loss`** `:88-106`（单边裁剪 `clamp(ratio, max=ε_high)` `:103`、`clamped·adv·policy_logp` `:105`）；`GRPOConfig` `:109-133`（num_generations=6 `:118`、loss_type='cispo' `:117`、epsilon_high=5.0 `:116`）。

**`zllm/training/distillation.py`（164 行）**：**`distillation_loss`** `:21-38`（teacher softmax/T `:34`、student log_softmax/T `:36`、KL `:37`、**×T²** `:38`）；`DistillConfig` `:42-61`（alpha=0.5 `:46`、temperature=1.5 `:47`、lr=5e-6 `:45`）；`train_epoch` `:64-164`（**loss = α·CE + (1-α)·distill + aux** `:133`、teacher eval+freeze `:87-89`、词表对齐 `:110-111`）。

**`zllm/training/agent_rl.py`（165 行）**：`TOOLS` 6 个工具 `:21-28`；`MOCK_RESULTS` 模拟执行 `:36-43`；`execute_tool` `:46-62`；`parse_tool_calls` `:65-85`（正则提取 ```json``` `:77`）；`validate_gt_in_text` `:88-98`；**`calculate_agent_reward`** `:101-142`（长度 `:120-123`、工具正确 `:125-130`、GT 匹配 `:132-133`、n-gram 重复惩罚 `:135-140`）；`AgentConfig` `:145-165`（max_turns=3 `:150`）。

**测试**：`test_208_sft_loss`（labels only assistant `:68`、pad masked `:86`、loss 下降 `:99`、过拟合 `:116`、SFT vs pretrain `:132`）；`test_222_lora_module`（A 高斯/B 零 `:31/36`、init=0 不改输出 `:40/83`、方阵注入 `:71`、非方阵不注入 `:77`、参数量<10% `:125`）；`test_231_save_load_merge`（save only lora `:37`、load 恢复 `:47`、merge 一致 `:85`、lora train loss 下降 `:126`、只有 lora 有梯度 `:165`）；`test_240_dpo`（log_probs shape `:26`、chosen 优势 loss 低 `:56`、梯度流 `:67`、train epoch `:129`、loss 下降 `:140`）；`test_246_ppo`（critic shape `:27`、GAE 零奖励零优势 `:50`、正奖励正优势 `:58`、returns=adv+val `:67`、标准化 `:75`、clip `:86`、value loss `:104`）；`test_258_grpo`（KL 同分布=0 `:16`、group 标准化 mean=0 `:35`、高奖励高优势 `:43`、clip `:81`、cispo 单边 `:96`）；`test_272_distillation`（相同=0 `:25`、T 越大越平滑 `:43`、梯度流 `:50`、alpha 平衡 `:66`、train 下降 `:110`、无教师纯 CE `:124`）；`test_280_agent_rl`（6 工具 `:24`、execute `:45`、parse `:76`、validate `:107`、reward `:121`、config `:145`）。

### Task 35: Ch 33 监督微调 SFT + Label Masking（M8）
Create `part-6-alignment/ch33-sft-label-masking.md`。YAML `source:zllm/training/full_sft.py,tests:tests/m08_sft/test_208_sft_loss.py`。原理：回引 Ch 28（SFTDataset label masking）+ Ch 32（预训练平台期转 SFT），讲 SFT 与预训练的三个差异（lr 1/50、只对 assistant 算 loss、加载预训练权重）。Mermaid 画 label masking 对比。代码：`SFTConfig` `:22-40`（lr=1e-5、from_weight='pretrain'）、`train_epoch` `:43-112`（与 pretrain 同结构）。重点：label masking 让模型学「回答」而非「复述问题」、lr 大幅降低避免遗忘预训练知识、from_weight='pretrain' 接力。测试 test_208（labels only assistant `:68`、loss 下降 `:99`、过拟合 `:116`、SFT<random `:132`）。pytest test_208。校验 `grep "\.py:[0-9]"`≥5。README Ch33→✅。

### Task 36: Ch 34 LoRA 低秩适配（M9）
Create `part-6-alignment/ch34-lora.md`。YAML `source:zllm/model/lora.py,tests:tests/m09_lora/test_222_lora_module.py`。原理：回引 Ch 15（参数效率）+ Ch 07（低秩分解 SVD），讲 LoRA 的 ΔW=B@A 低秩分解、B 零初始化保证初始不改输出、只注入方阵（q_proj/o_proj）。LaTeX 推导 ΔW=BA 的参数量节省。Mermaid 画注入+合并流程。代码：`LoRA` 类 `:21-39`、`apply_lora` `:42-60`、`freeze_non_lora` `:69-79`、`save/load/merge_lora` `:82-130`。重点：B 零初始化、只注入方阵、参数量<10%、merge 后 W+BA 推理一致、lr 比 full_sft 高 10 倍。测试 test_222（A 高斯/B 零 `:31/36`、init=0 `:83`、参数<10% `:125`）+ test_231（merge 一致 `:85`、train loss 下降 `:126`、只有 lora 有梯度 `:165`）。pytest test_222 test_231。校验 `grep "\.py:[0-9]"`≥8。README Ch34→✅。

### Task 37: Ch 35 RLHF 框架与对齐总论（背景理论）
Create `part-6-alignment/ch35-rlhf-framework.md`。**理论章**（无独立源码文件）。YAML `source:none,tests:none`。原理：讲 RLHF 三阶段（SFT→RM→PPO）经典框架、DPO 绕过 RM 的简化、GRPO 群体相对优势去 Critic、对齐的目标（让模型有用/无害/诚实）。Mermaid 画 RLHF vs DPO vs GRPO 对比图。LaTeX 推导 RLHF 目标 $\max \mathbb{E}[r(x,y)] - \beta\text{KL}(\pi\|\pi_{\text{ref}})$ 与 DPO 的等价转化。重点：对齐税、奖励黑客、KL 惩罚防偏离。无 file:line 要求（理论章）。README Ch35→✅。

### Task 38: Ch 36 DPO 直接偏好优化（M10-a）
Create `part-6-alignment/ch36-dpo.md`。YAML `source:zllm/training/dpo.py,tests:tests/m10_alignment/test_240_dpo.py`。原理：回引 Ch 35（DPO 绕过 RM），讲双模型架构（policy 训练 + reference 冻结）、 Bradley-Terry 偏好模型、loss=-logsigmoid(β·(π_θ(c)/π_ref(c) - π_θ(r)/π_ref(r)))。LaTeX 推导 DPO loss 从 RLHF 目标的闭式解。代码：`logits_to_log_probs` `:20-31`、**`dpo_loss`** `:34-61`（chosen/rejected 拼接 `:52-55`、logsigmoid `:60`）、`train_epoch` `:86-162`（ref no_grad `:121`）。重点：β 温度参数、lr=4e-8 极低（防灾难遗忘）、ref 锚定防漂移。测试 test_240（log_probs `:26`、chosen 优势 `:56`、梯度流 `:67`、loss 下降 `:140`）。pytest test_240。校验 `grep "\.py:[0-9]"`≥6。README Ch36→✅。

### Task 39: Ch 37 PPO + GAE + Critic（M10-b）
Create `part-6-alignment/ch37-ppo-gae-critic.md`。YAML `source:zllm/training/ppo.py,tests:tests/m10_alignment/test_246_ppo.py`。原理：回引 Ch 35（RLHF 经典 PPO），讲 Actor-Critic 双模型、GAE 优势估计、clipped surrogate loss、value function loss。LaTeX 推导 GAE 的 δ_t 和 A_t 递推、PPO clip 公式。代码：`CriticModel` `:29-44`（共享 backbone + value_head）、**`compute_gae`** `:47-81`（δ→A 递推 `:70-71`、标准化 `:75-78`）、`ppo_policy_loss` `:84-103`（ratio clip）、`ppo_value_loss` `:106-125`。重点：GAE 的 λ 平衡偏差/方差、clip 防策略崩溃、value loss 也 clip、vf_coef=0.5 权衡。测试 test_246（critic shape `:27`、GAE `:50/58/67`、clip `:86`、value `:104`）。pytest test_246。校验 `grep "\.py:[0-9]"`≥7。README Ch37→✅。

### Task 40: Ch 38 GRPO + CISPO（M10-c）
Create `part-6-alignment/ch38-grpo-cispo.md`。YAML `source:zllm/training/grpo.py,tests:tests/m10_alignment/test_258_grpo.py`。原理：回引 Ch 35（GRPO 去 Critic）+ Ch 37（PPO 对比），讲群体相对优势（同 prompt 生成 N 个 response 组内标准化）、KL 惩罚、CISPO 单边裁剪。LaTeX 推导 group advantage 和 KL。Mermaid 画 GRPO 流程（prompt→N 生成→组内排序→reward）。代码：`per_token_kl` `:24-37`、**`compute_group_advantages`** `:40-56`、`grpo_loss` `:59-85`（双边 clip）、**`cispo_loss`** `:88-106`（单边 clip `:103`）。重点：去 Critic 省一半模型、num_generations=6、CISPO 允许低概率 token 进一步降低（鼓励修剪低质内容）。测试 test_258（KL=0 `:16`、group mean=0 `:35`、高奖励高优势 `:43`、clip `:81`、cispo `:96`）。pytest test_258。校验 `grep "\.py:[0-9]"`≥6。README Ch38→✅。

### Task 41: Ch 39 知识蒸馏（M11-a）
Create `part-6-alignment/ch39-distillation.md`。YAML `source:zllm/training/distillation.py,tests:tests/m11_distill_agent/test_272_distillation.py`。原理：回引 Ch 06（softmax 温度）+ Ch 05（KL 散度），讲软标签暗知识、温度 T 使分布平滑暴露类间关系、loss=α·CE+(1-α)·T²·KL。LaTeX 推导蒸馏 loss。Mermaid 画 teacher→student 传递。代码：**`distillation_loss`** `:21-38`（teacher softmax/T、student log_softmax/T、KL、×T²）、`train_epoch` `:64-164`（α·CE+(1-α)·distill `:133`、teacher freeze `:87-89`、词表对齐 `:110-111`）。重点：T² 补偿梯度尺度、α 平衡硬/软标签、词表对齐（teacher/student vocab 可能不同）。测试 test_272（相同=0 `:25`、T 大平滑 `:43`、梯度流 `:50`、alpha `:66`、train 下降 `:110`、无教师纯CE `:124`）。pytest test_272。校验 `grep "\.py:[0-9]"`≥5。README Ch39→✅。

### Task 42: Ch 40 Agent RL 工具调用（M11-b）
Create `part-6-alignment/ch40-agent-rl-tools.md`。YAML `source:zllm/training/agent_rl.py,tests:tests/m11_distill_agent/test_280_agent_rl.py`。原理：回引 Ch 28（AgentRLDataset）+ Ch 35（RL），讲 Agent 的多轮工具调用循环（生成→解析→执行→反馈→再生成）、多维度奖励（长度/工具正确/GT 匹配/重复惩罚）。Mermaid 画 Agent 多轮交互流程。代码：`TOOLS` `:21-28`、`execute_tool` `:46-62`、`parse_tool_calls` `:65-85`、**`calculate_agent_reward`** `:101-142`（四维度）。重点：工具调用的 JSON 格式解析、奖励设计的多目标平衡、max_turns=3 多轮交互。测试 test_280（6 工具 `:24`、execute `:45`、parse `:76`、validate `:107`、reward `:121`）。pytest test_280。校验 `grep "\.py:[0-9]"`≥5。Part VI 收官。README Ch40→✅。

### Phase 6 完工标准（DoD）
- Ch 33–40 全部写完，README 第 33–40 行全 ✅（Part VI 完成）。
- 每章 `grep -cE "\.py:[0-9]"` 达标（见各 Task；Ch 35 理论章豁免）。
- 全 Phase 无 TBD/TODO/占位符；所有引用 file:line 经 `sed -n` 抽查真实。
- Phase 6 完成后更新本计划进度表第 71 行为「已完成」。
- **Phase 7（Part VII + 附录，Ch 41–43 + 附录 A-D）**：M12。引用 `zllm/serving/{generate,api_server,cli}.py`、`tests/m12_serving`。
