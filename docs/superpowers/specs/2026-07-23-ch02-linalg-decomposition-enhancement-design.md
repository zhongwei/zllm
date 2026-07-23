# 第 2 章《线性代数：分解与几何》整体增强设计

- **日期**：2026-07-23
- **目标文件**：`docs/book/part-1-math/ch02-linear-algebra-decomposition.md`
- **新增资产**：`docs/book/part-1-math/figs/ch02-*.svg`（5 个）
- **方案**：B（中度重写，保留骨架，插入新小节，ASCII 全替换为动态 SVG）
- **参考风格**：`figs/ch01-matrix-multiplication_anim.svg`（SMIL 动画、`@media (prefers-color-scheme: dark)`、system-ui 字体、color-coded、monospace 计算过程）

## 1. 背景与动机

第 1 章已有 12 个 SVG 图（含 8 个动画），行数 425；第 2 章当前**零 SVG**，仅有 1 张 mermaid 概念图 + 2 处 ASCII 画（SVD 三步几何、奇异值谱），行数 388。此外章节里存在 6 处 LaTeX 渲染 bug（`\VertA`/`\VertQ` 缺空格），且缺少手算数值例题与常见误区提示，与 Ch01 的成熟度存在差距。

本次目标：在保留作者口吻与既有骨架（2.1 学习目标 → 2.2 直觉 → 2.3 定义 → 2.4 推导 → 2.5 钩子 → 2.6 小结）的前提下，补齐 SVG 图示、手算例题、FAQ、定理证明与新小节，使 Ch02 与 Ch01 风格统一。

## 2. SVG 图设计（5 个，全部动画）

统一规范：`viewBox`、`width="100%"`、SMIL `<animate>` + `repeatCount="indefinite"`、深浅色双套配色（沿用 ch01 的 `#2563eb/#ea580c/#7c3aed` 调色）、`font-family="system-ui, ..."`、monospace 计算行、`<title>/<desc>` 无障碍标注。全部文件名以 `_anim.svg` 结尾。

### 2.1 `ch02-eigenvector-transform_anim.svg`（落位 2.3.1 末「几何演示」）
- **内容**：单位圆上均匀分布 8 个向量被 $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$ 作用。非特征向量被拧转方向；沿特征轴 $\mathbf{q}_1=\frac{1}{\sqrt2}(1,1)^\top$（红）、$\mathbf{q}_2=\frac{1}{\sqrt2}(1,-1)^\top$（蓝）的两根向量**方向不变**，分别拉伸 $\lambda_1=3$、$\lambda_2=1$ 倍。
- **动画**：原向量淡入 → 变换后的向量箭头生长 → 高亮两个特征向量并标注 $\lambda$ → 循环。
- **下方**：公式行 `A·q₁ = 3·q₁`、`A·q₂ = 1·q₂` 高亮闪烁。

### 2.2 `ch02-svd-three-step_anim.svg`（落位 2.4.1，**替换 ASCII 圆→椭圆图**）
- **内容**：单位圆 → ① $V^\top$ 旋转（仍是圆，但坐标轴箭头转向）→ ② $\Sigma$ 沿轴拉伸成长短半轴 $\sigma_1,\sigma_2$ 的椭圆 → ③ $U$ 把椭圆整体旋转到最终姿态。
- **动画**：四帧分步（输入 / 旋转后 / 拉伸后 / 最终），每步顶部标签切换 `① Vᵀ` / `② Σ` / `③ U`，底部显示当前矩阵名与「保长」「拉伸 σ₁=…」等说明。椭圆半轴随 σ 值真实变化（取 $\sigma_1=2,\sigma_2=0.8$）。

### 2.3 `ch02-spectrum-truncation_anim.svg`（落位 2.4.4，**替换 ASCII 谱图**）
- **内容**：8 根奇异值柱 $\sigma_1\geq\dots\geq\sigma_8$ 从高到低（如 $3.0,2.0,1.2,0.6,0.2,0.08,0.03,0.01$）。
- **动画**：$k$ 在 $1\to3\to5$ 间切换；前 $k$ 根高亮（紫色），$k{+}1$ 起淡化为灰色（误差）；右侧实时显示 $\lVert A-A_k\rVert_F=\sqrt{\sum_{i>k}\sigma_i^2}$ 与「能量保留率 $\sum_{i\le k}\sigma_i^2/\sum\sigma_i^2$」。
- **目的**：直观呈现 Eckart-Young「截断 = 丢弃低能量方向」。

### 2.4 `ch02-lowrank-rebuild_anim.svg`（落位 2.5 钩子一 LoRA 内，与 §3.5 一致）
- **内容**：一个 $16\times16$ 灰度「图像」由 $A=U\Sigma V^\top$ 表达，随 $k=1,2,4,8,16$ 逐步重建，从模糊→清晰。
- **下方**：进度条对比「完整存储 $16\times16=256$ 个数」vs「$A_k$ 仅需 $k(16+16)=32k$ 个数」，$k=4$ 时压缩 2 倍、$k=8$ 时 4 倍。
- **目的**：把「低秩 ≈ 可压缩」与 LoRA 的 $2dr$ 参数量直接可视对应。

### 2.5 `ch02-projection_anim.svg`（落位 2.3.5 末「几何演示」）
- **内容**：$\mathbb{R}^2$ 中向量 $\mathbf{b}=(3,4)^\top$（橙），列空间 $\text{span}\{(1,1)^\top\}$（蓝色过原点直线），从 $\mathbf{b}$ 端点画垂线到直线上的投影点 $P\mathbf{b}$（紫），残差 $\mathbf{b}-P\mathbf{b}$（灰虚线），投影点处标直角符号。
- **动画**：$\mathbf{b}$ 先出现 → 垂线下降到直线 → $P\mathbf{b}$ 落定 → 公式 $P=A(A^\top A)^{-1}A^\top$ 与数值结果淡入。

## 3. 文本增强

### 3.1 手算数值例题（4 个）
| 落位 | 矩阵 | 要点 |
|------|------|------|
| 2.3.1 末（接 SVG #1） | $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$ | 求 $\det(A-\lambda I)=(2-\lambda)^2-1=0$ → $\lambda=3,1$；解特征向量；写出 $Q\Lambda Q^\top$ 并**反乘验证还原 $A$** |
| 2.3.3 末 | $A=\begin{pmatrix}1&1\\0&1\end{pmatrix}$（剪切） | 由 $A^\top A=\begin{pmatrix}1&1\\1&2\end{pmatrix}$ 求右奇异向量；$\sigma_i=\sqrt{\lambda_i}$；$\mathbf{u}_i=A\mathbf{v}_i/\sigma_i$；显式写出 $U,\Sigma,V$，呼应「剪切=旋转-拉伸-旋转」 |
| 2.3.5 末（接 SVG #5） | $\mathbf{b}=(3,4)^\top$ 投影到 $\text{span}\{(1,1)^\top\}$ | 算 $P$，验证 $P^2=P$、$P^\top=P$、$P\mathbf{b}=(3.5,3.5)^\top$ |
| 2.4.4 例（接 SVG #3） | 用 #3 同款 $\sigma$ 谱 | 手算 $k=3$ 时的 Frobenius 误差 $\sqrt{0.6^2+0.2^2+\cdots}$ 与能量保留率 |

### 3.2 新增小节
- **2.4.5 伪逆与最小二乘**：定义 $A^+=V\Sigma^+U^\top$（$\Sigma^+$ 把非零 $\sigma_i$ 取倒数、零奇异值置零）；几何上 $A^+$ 把 $A$ 的「方向+缩放」全部求逆、零空间方向丢弃；最小二乘解 $\hat{\mathbf{x}}=A^+\mathbf{b}$，$\hat{\mathbf{x}}$ 的像 $A\hat{\mathbf{x}}=P\mathbf{b}$ 正是 $\mathbf{b}$ 到列空间的投影——**把 SVD↔投影矩阵闭环**。
- **2.4.6 SVD 与 PCA 的严格关系**：数据矩阵 $X$ 中心化（去均值）后，协方差 $\frac1{n}X^\top X$ 的特征分解给出 SVD 的 $V$；主成分 = 右奇异向量；各主方向方差 = $\sigma_i^2/n$；方差累积贡献率 = $\sum_{i\le k}\sigma_i^2/\sum\sigma_i^2$（呼应 SVG #3）。回应 2.5 钩子四。

### 3.3 FAQ 提示框（沿用 `> ` 引用块 + 「⚠️ 常见误区」前缀）
1. 「特征值一定是实数吗？」→ 一般方阵未必（如 $B=\begin{pmatrix}0&1\\-1&0\end{pmatrix}$ 特征值 $\pm i$）；**实对称矩阵一定实数**。
2. 「SVD 的 $U,V$ 唯一吗？」→ 不唯一：重奇异值处正交基可任选，单奇异值处仅差整体符号。
3. 「秩亏损为什么可怕？」→ $A^{-1}$ 不存在、条件数 $\to\infty$、数值病态、梯度方向不可靠。
4. 「正交矩阵 $\det Q=1$？」→ $\det Q=\pm1$；$-1$ 对应**反射**（如 $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$），纯旋转才 $\det=+1$。

### 3.4 定理证明补充
- **对称矩阵特征向量两两正交**：由 $A\mathbf{q}_i=\lambda_i\mathbf{q}_i$ 与 $A^\top=A$ 推 $(\lambda_i-\lambda_j)\mathbf{q}_i^\top\mathbf{q}_j=0$，$\lambda_i\ne\lambda_j$ 时内积为零。
- **Eckart-Young 证明草图**：Courant–Fischer 极小极大 + 维数论证（任意秩 $\le k$ 矩阵 $B$ 的零空间维数 $\ge n-k$，必与某 $\mathbf{v}_i$（$i\le k+1$）相交，故 $\lVert A-B\rVert_2\ge\sigma_{k+1}$）。

### 3.5 加厚 2.5 钩子一（LoRA）
- 在现有 $d=768,r=16$ 之外补一张「不同 $r$ 下的参数量/压缩比」对比表（$r=4,8,16,32,64$）。
- 补「为什么 $\Delta W$ 低秩」的经验依据：Hu et al. 2021 实证「预训练权重本身有很长的有效秩，但**任务适配方向**集中在极少数主轴」；引用 Aghajanyan 2020（内禀维度）。
- 把 SVG #4 嵌在此处作直接图示。

## 4. Bug 修复清单
- 6 处 `\VertA`/`\VertQ` → `\Vert A`/`\Vert Q`：line 139（`\VertQ`）、232（`\VertA`）、273（`\VertA`×2）、279（`\VertA`）、373（`\VertA`）。
- 提交前重跑 AGENTS.md 三条自检命令，确认零违规：
  - `grep -rnE '\\[,:!]' docs/book/`
  - `grep -rn '\\text{[^}]*\$' docs/book/`
  - `grep -rnP '[、，。；：）」』]\$' docs/book/`
- 检查思考题第 2、3 题的 `\begin{pmatrix}...\end{pmatrix}` 行内渲染（确认 `\cr` 一致）。

## 5. 学习目标同步更新
2.1 节学习目标补两条，与新内容对齐：
- 写出伪逆 $A^+=V\Sigma^+U^\top$ 并解释它为何给出最小二乘解；
- 说出 SVD 与 PCA 的关系（主成分 = 右奇异向量，方差 = $\sigma_i^2/n$）。

## 6. 小结与预告微调
2.6 节小结补 2 条（伪逆、PCA），保持「5 条 → 7 条」结构；「前方预告」段（概率论）保留不动。

## 7. 实施顺序（writing-plans 将据此细化）
1. 创建 5 个 SVG 文件（每个独立，可并行）。
2. 修复 6 处 `\Vert` bug（独立小提交）。
3. 按章节顺序扩写 prose：2.3.1（例题+SVG#1）→ 2.3.3（例题）→ 2.3.5（例题+SVG#5）→ 2.4.1（SVG#2）→ 2.4.3/2.4.4（用 SVG#3 的 σ 谱做手算）→ 新增 2.4.5/2.4.6 → 2.5 钩子一加厚 + 嵌 SVG#4 → 2.1/2.6 同步 → FAQ 与证明散落插入。
4. 全章自检三条 grep + 目测公式渲染。

## 8. 非目标（YAGNI）
- 不改动 2.2 节的两个生活类比（橡皮膜、复印机）——已是优点。
- 不重排章节顺序（如把秩前移）——风险大于收益。
- 不引入外部图片/数据集；SVG 全部自包含、手画。
- 不写 PyTorch 代码（留给 Ch34 实战章）。
