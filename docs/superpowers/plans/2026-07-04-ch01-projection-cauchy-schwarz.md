# Ch01 §1.4 投影重写 + 柯西–施瓦茨小节 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `docs/book/part-1-math/ch01-linear-algebra-vectors.md` 的 §1.4「点积 = 投影长度」去重、让投影成为领起，并新增独立的「柯西–施瓦茨不等式」子小节。

**Architecture:** 单文件定点重写。删掉 §1.4 里与 §1.3 重复的 17 行余弦定理推导，替换为「回指 → 投影定义 → 三档表 → 投影图 → 方向一致性桥接」，紧接新增「### 柯西–施瓦茨不等式」子小节；最后给 §1.6 小结补一短句。

**Tech Stack:** Markdown + LaTeX 数学（`$$...$$` / `$...$` / `\boxed{}`），无代码、无 SVG 改动。

## Global Constraints

（抄自 spec「写作约定」+「非目标」，逐字适用每个任务）

- 数学记号同全书一致：向量 `$\mathbf{x}$`、范数 `$\Vert\mathbf{x}\Vert$`、夹角 `$\theta$`、box 用 `$\boxed{...}$`。
- 段落宜短（每块 1–4 句），关键结论用 `\boxed{}` 或 `>` 收口。
- 中文标点、`—` 破折号、`→` 等排版同全书。
- **不改** §1.3、§1.5、思考题、任何 SVG、其它章节。
- §1.3 的余弦定理推导（现 lines 200–230）是本书该推导的唯一权威副本，**保持不动**；本计划只是让 §1.4 不再重复它。
- 提交信息用祈使句英文，匹配近期 commit 风格（如 "Rewrite cosine similarity subsection..."）。
- 仅当用户明确要求时才 commit；本计划的 commit 步骤在执行时由人确认。

## File Structure

| 文件 | 责任 | 本计划动作 |
|---|---|---|
| `docs/book/part-1-math/ch01-linear-algebra-vectors.md` | 第 1 章正文 | Task 1 改 §1.4 前半（现 lines 258–294）；Task 2 改 §1.6 小结第 2 条（现 line 388） |

仅 1 个文件。无新文件、无 SVG、无源码。

---

### Task 1: 重写 §1.4「点积 = 投影长度」并新增「柯西–施瓦茨不等式」子小节

**Files:**
- Modify: `docs/book/part-1-math/ch01-linear-algebra-vectors.md`，替换现 lines 258–294（从 `### 点积 = 投影长度` 标题行，到「方向一致性」段末「...原因（Ch 12）。」为止）。

**Interfaces:**
- Consumes: §1.3 已确立的恒等式 `$\mathbf{x}\cdot\mathbf{y}=\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert\cos\theta$`（现 line 230）与三档描述（现 lines 186–198）；本任务用一行回指引用，不复制其推导。
- Produces: 两个新/改的 `###` 子小节，紧接其后是未改动的 `### 矩阵乘法 = 线性变换`（现 line 296）。Task 2 依赖本任务不改动 §1.6。

**前提确认（执行前先读一遍，确认行号未漂移）：**

- [ ] **Step 1: 读现行 §1.4 前半，确认替换边界**

  Run: 用 Read 工具读 `docs/book/part-1-math/ch01-linear-algebra-vectors.md` 的 lines 254–296。
  Expected: 254 行为 `## 1.4 推导与几何`；258 行为 `### 点积 = 投影长度`；260–276 行为待删的重复推导（含两个 `$$...$$` 块和一个 `\boxed{ \mathbf{x} \cdot \mathbf{y} = ... \cos\theta }`）；278–288 行为「三层含义」三点；290–292 行为投影图引用；294 行为「方向一致性」段；296 行为 `### 矩阵乘法 = 线性变换`。
  若行号已漂移，以**内容锚点**（上述标题/公式/图引用）定位，不要按绝对行号机械替换。

**去重 + 重写：**

- [ ] **Step 2: 用以下完整内容替换「### 点积 = 投影长度」到「方向一致性」段末（现 258–294 行）**

  `oldString`（精确匹配现行 258–294 行整段，从 `### 点积 = 投影长度` 起、到「...原因（Ch 12）。」止——执行时按文件实际内容取整段）。

  `newString`（逐字粘贴，不要改动任何记号/空格/标点）：

  ````markdown
  ### 点积 = 投影长度

  §1.3 已经推出了点积的几何形式 $\mathbf{x}\cdot\mathbf{y}=\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert\cos\theta$ 。这一节换一个角度看它：不谈夹角，只谈**投影**。

  把向量 $\mathbf{y}$ 沿 $\mathbf{x}$ 的方向「拍扁」，得到它在 $\mathbf{x}$ 方向上的分量 $\Vert\mathbf{y}\Vert\cos\theta$ 。这是一个带正负号的标量——沿 $\mathbf{x}$ 正方向为正、反方向为负——叫作 $\mathbf{y}$ 在 $\mathbf{x}$ 上的**有向投影长度**。于是点积可以读作：

  $$
  \boxed{ \; \mathbf{x}\cdot\mathbf{y} \;=\; \Vert\mathbf{x}\Vert \times (\text{$\mathbf{y}$ 在 $\mathbf{x}$ 方向上的有向投影长度}) \; }
  $$

  换谁都一样：点积也可以写成 $\Vert\mathbf{y}\Vert \times (\mathbf{x}\text{ 在 }\mathbf{y}\text{ 方向上的投影})$ ，它不分「谁投到谁」。

  投影的正负，正好对应 §1.3 开头讲过的三种夹角：

  | 夹角 | $\mathbf{y}$ 的有向投影 | 点积 $\mathbf{x}\cdot\mathbf{y}$ |
  |---|:---:|:---:|
  | 锐角（ $\theta<90^\circ$ ） | $>0$ | $>0$ |
  | 直角（ $\theta=90^\circ$ ） | $=0$ | $=0$ |
  | 钝角（ $\theta>90^\circ$ ） | $<0$ | $<0$ |

  直角那一档尤其重要：**几何上「垂直」，代数上「点积为零」，两者是同一件事**（ $\theta=90^\circ\Rightarrow\cos\theta=0\Rightarrow\mathbf{x}\cdot\mathbf{y}=0$ ）。这就是后面理解「无关」「解耦」的根基。

  下面这张示意图把「投影」和「夹角」的关系画出来（为清晰起见画在 2 维平面）：

  ![点积 = 投影长度](figs/ch01-dot-product-projection_anim.svg)

  可以看到，随着夹角 $\theta$ 从锐变钝，有向投影从正变负，点积也跟着同号变化。两个极端最容易记： $\mathbf{y}$ 与 $\mathbf{x}$ **同向**（ $\theta=0$ ）时点积最大、为 $\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert$ ；**反向**（ $\theta=\pi$ ）时最小、为 $-\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert$ 。**点积天然就是「方向一致性」的度量**——这正是注意力机制用它来衡量「词与词有多相关」的原因（Ch 12）。

  那么这个上下界是严格的吗？由下面这条不等式一锤定音。

  ### 柯西–施瓦茨不等式

  **柯西–施瓦茨不等式（Cauchy–Schwarz inequality）**：

  $$
  \boxed{ \; |\mathbf{x}\cdot\mathbf{y}| \;\leq\; \Vert\mathbf{x}\Vert\,\Vert\mathbf{y}\Vert \; }
  $$

  等号当且仅当两向量**共线**（同向或反向）时成立。

  **证明只需一行**：把 §1.3 的恒等式 $\mathbf{x}\cdot\mathbf{y}=\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert\cos\theta$ 代入，再用基本事实 $|\cos\theta|\leq 1$ ，立刻得到 $|\mathbf{x}\cdot\mathbf{y}|\leq\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert$ ；等号对应 $|\cos\theta|=1$ ，即 $\theta\in\{0,\pi\}$ ，正是两向量共线。

  **几何读法**：不等式两边除以 $\Vert\mathbf{x}\Vert$ ，得到 $|\Vert\mathbf{y}\Vert\cos\theta|\leq\Vert\mathbf{y}\Vert$ ——一个向量在任一方向上的**投影长度，绝不会超过它自身的长度**。柯西–施瓦茨本质上就是这句话的代数化身。

  **两条直接推论**：

  - 把 $\mathbf{x},\mathbf{y}$ 都取成**单位向量**（ $\Vert\mathbf{x}\Vert=\Vert\mathbf{y}\Vert=1$ ），点积落在 $[-1,1]$ ——这正是 §1.3 余弦相似度值域 $[-1,1]$ 的代数依据。
  - 前文刚看到的「同向取最大、反向取最小」，其严格上下界正是 $\pm\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert$ 。

  **它不只是个摆设**：Ch 06（最优化）会用它严格证明**「负梯度是最速下降方向」**——在单位步长 $\Vert\mathbf{d}\Vert=1$ 的约束下， $\nabla f(\theta)^\top\mathbf{d}\ge -\Vert\nabla f(\theta)\Vert$ ，等号当 $\mathbf{d}$ 取负梯度方向时成立。换句话说，要让损失下降最快，朝负梯度走是数学上最优的——而这条结论的根基，正是柯西–施瓦茨。
  ````

  注意：上面是 markdown 嵌套展示，实际写入文件时去掉外层 ```` ```` 围栏，只保留 `### 点积 = 投影长度` 起的内容。

**验证（去重 + 结构 + 数学）：**

- [ ] **Step 3: 确认去重——余弦定理推导在 §1.4 已消失**

  Run: `rg -n "另一方面，直接展开左边|比较两式，立刻得到点积的几何形式" docs/book/part-1-math/ch01-linear-algebra-vectors.md`
  Expected: **0 条命中**（这两句是旧 §1.4 重推导的标志句，应已被删）。

- [ ] **Step 4: 确认 §1.3 的推导仍在（没误删权威副本）**

  Run: `rg -n "比较这两个展开式" docs/book/part-1-math/ch01-linear-algebra-vectors.md`
  Expected: **1 条命中**，且位于 §1.3（line 224 附近，即现 line 224 的 `> 比较这两个展开式`）。若 0 条，说明误删了 §1.3，必须回滚。

- [ ] **Step 5: 确认新子小节标题就位**

  Run: `rg -n "^### 柯西–施瓦茨不等式|^### 点积 = 投影长度|^### 矩阵乘法 = 线性变换" docs/book/part-1-math/ch01-linear-algebra-vectors.md`
  Expected: **3 条命中，且顺序为**：`点积 = 投影长度` → `柯西–施瓦茨不等式` → `矩阵乘法 = 线性变换`。

- [ ] **Step 6: 确认投影图引用仍在（且唯一）**

  Run: `rg -n "ch01-dot-product-projection_anim" docs/book/part-1-math/ch01-linear-algebra-vectors.md`
  Expected: **1 条命中**，位于新的「点积 = 投影长度」小节内。

- [ ] **Step 7: 通读新写的两段，核对数学一致性**

  用 Read 读 §1.4 从 `### 点积 = 投影长度` 到 `### 矩阵乘法 = 线性变换` 之前。逐条核对：
  - 投影 box `$\mathbf{x}\cdot\mathbf{y} = \Vert\mathbf{x}\Vert \times (\text{有向投影长度})$` 与 §1.3 恒等式（line 230）相容（把 $\Vert\mathbf{y}\Vert\cos\theta$ 记作投影即得）。
  - 三档表的符号（`>0` / `=0` / `<0`）与 §1.3 开头三档（现 lines 188–198）一致。
  - 柯西–施瓦茨 box `$|\mathbf{x}\cdot\mathbf{y}| \leq \Vert\mathbf{x}\Vert\,\Vert\mathbf{y}\Vert$` 可由 §1.3 恒等式 + $|\cos\theta|\leq 1$ 推出。
  - 柯西–施瓦茨的钩子（Ch 06 负梯度最速下降）与 `ch06-optimization.md` 不矛盾（ch06:205、370、386 已引用 Ch01 的 Cauchy–Schwarz）。

- [ ] **Step 8: （可选）渲染检查**

  若本地有 markdown 预览（如 `mdbook serve` / VS Code 预览），打开本章，确认：表格正常渲染、两个 `\boxed{}` 显示正常、`$\text{中文}$` 不乱码。
  若无预览环境，跳过此步——前 7 步已足够保证正确性。

- [ ] **Step 9: 提交（仅当用户确认要 commit 时）**

  ```bash
  git add docs/book/part-1-math/ch01-linear-algebra-vectors.md
  git commit -m "Rewrite dot-product projection subsection and add Cauchy-Schwarz subsection"
  ```

---

### Task 2: §1.6 小结同步——为柯西–施瓦茨补一短句

**Files:**
- Modify: `docs/book/part-1-math/ch01-linear-algebra-vectors.md`，§1.6 小结第 2 条（现 line 388）。

**Interfaces:**
- Consumes: Task 1 已新增的柯西–施瓦茨子小节（本条小结需与之呼应）。
- Produces: 无下游依赖（本章末尾）。

- [ ] **Step 1: 读现行 §1.6 小结第 2 条，确认锚点**

  Run: Read `docs/book/part-1-math/ch01-linear-algebra-vectors.md` 的 lines 386–392。
  Expected: line 388 为 `2. **点积**： ...也是「长度 × 投影」的几何度量；**正交 ⟺ 点积为零**。`（以句号结尾）。若漂移，按内容定位这句。

- [ ] **Step 2: 在「正交 ⟺ 点积为零」之后追加柯西–施瓦茨短句**

  用 Edit：
  - `oldString`：`也是「长度 × 投影」的几何度量；**正交 ⟺ 点积为零**。`
  - `newString`：`也是「长度 × 投影」的几何度量；**正交 ⟺ 点积为零**；柯西–施瓦茨不等式 $|\mathbf{x}\cdot\mathbf{y}|\leq\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert$ （Ch 06 证明负梯度方向要用）。`

- [ ] **Step 3: 确认追加成功且唯一**

  Run: `rg -n "柯西–施瓦茨不等式 \$\\\|\\\\mathbf{x}" docs/book/part-1-math/ch01-linear-algebra-vectors.md`
  （若该正则难写，改用：`rg -n "Ch 06 证明负梯度方向要用" docs/book/part-1-math/ch01-linear-algebra-vectors.md`）
  Expected: **1 条命中**，位于 §1.6 小结第 2 条。

- [ ] **Step 4: 通读 §1.6 小结第 2 条，确认通顺**

  用 Read 读改动行。预期整句为：
  > 2. **点积**： $\mathbf{x}\cdot\mathbf{y} = \sum_i x_i y_i = \Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert\cos\theta$ 。它既是「逐位相乘求和」的代数运算，也是「长度 × 投影」的几何度量；**正交 ⟺ 点积为零**；柯西–施瓦茨不等式 $|\mathbf{x}\cdot\mathbf{y}|\leq\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert$ （Ch 06 证明负梯度方向要用）。

- [ ] **Step 5: 提交（仅当用户确认要 commit 时）**

  ```bash
  git add docs/book/part-1-math/ch01-linear-algebra-vectors.md
  git commit -m "Add Cauchy-Schwarz clause to ch01 summary"
  ```

  （若 Task 1 尚未提交、且用户希望两步合一，可合并为一次 commit，消息用 "Rewrite dot-product projection subsection, add Cauchy-Schwarz subsection and summary clause"。）

---

## 全局回读（两个 Task 都完成后）

- [ ] **Final: 从 §1.3 末尾连读到 §1.4「矩阵乘法 = 线性变换」之前**

  预期叙事链一气呵成、无重复、无跳跃：
  **§1.3 推导出恒等式 → §1.4 用恒等式谈投影 → 投影正负扣回三档（含正交） → 方向一致性引出上下界 → 柯西–施瓦茨给上下界严格证明 + Ch06 钩子**。
  特别确认：§1.4 里**不再出现**第二份余弦定理推导。
