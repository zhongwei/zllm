# ch01 ASCII → 引用式 SVG 图 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 2 个引用式静态 SVG 替换 `ch01-linear-algebra-vectors.md` 中的 2 张 ASCII 几何图，并额外保存 2 个 `_anim.svg` 动画版。

**Architecture:** 在 `docs/book/part-1-math/figs/` 下新增 4 个手写 SVG 文件（图1/图2 各一对「静态+动画」），再用标准 markdown 图片语法 `![alt](figs/x.svg)` 替换 ch01 里对应的两个 ASCII 代码块。不引入构建管线、不改 mermaid 图、不改正文文字。

**Tech Stack:** 纯 SVG 1.1（含 SMIL `<animate>` 用于动画版）；Markdown 图片引用；Python（仅用于 XML 合法性校验）。

## Global Constraints

（摘自 spec `docs/superpowers/specs/2026-07-03-ch01-ascii-to-svg-figures-design.md`，每个任务隐式遵守）

- **范围**：只改 `docs/book/part-1-math/ch01-linear-algebra-vectors.md` 与新增 `docs/book/part-1-math/figs/` 下 4 个 SVG。不触碰其它章节、不碰 mermaid 概念图、不改正文文字/公式。
- **风格 B「彩色简洁」**：向量 x = `#2563eb`（蓝，`stroke-width="2.6"`）；向量 y / A·x = `#ea580c`（橙，`stroke-width="2.6"`）；投影/弧 = `#9ca3af` / `#6b7280`，虚线 `stroke-dasharray="6 4"`；坐标轴细参考线 `#cbd5e1`/`#e5e7eb`，`stroke-width="1"`；原点黑色小圆点；字体 `Cambria, Georgia, 'Times New Roman', serif`，变量斜体；画布 `viewBox="0 0 380 250"` 白底。
- **命名约定**：`<chapter>-<topic>.svg` 静态；`<chapter>-<topic>_anim.svg` 动画。书内只引用静态版。
- **编码**：SVG 文件 UTF-8 保存（含 `‖·θα` 等字符），首行 `<?xml version="1.0" encoding="UTF-8"?>`。
- **提交规则**：本仓库默认**不自动提交**。计划中保留了 commit 步骤作为「检查点」，但执行时**只有当用户明确同意提交**才执行 `git add` / `git commit`；否则跳过提交步骤、仅完成文件改动与验证。

---

## File Structure

| 路径 | 动作 | 职责 |
|------|------|------|
| `docs/book/part-1-math/figs/ch01-dot-product-projection.svg` | 新建 | 图1 静态：点积 = 投影（入书引用） |
| `docs/book/part-1-math/figs/ch01-dot-product-projection_anim.svg` | 新建 | 图1 动画：y 扫动 θ（不入书） |
| `docs/book/part-1-math/figs/ch01-matrix-transform.svg` | 新建 | 图2 静态：矩阵 = 旋转+缩放（入书引用） |
| `docs/book/part-1-math/figs/ch01-matrix-transform_anim.svg` | 新建 | 图2 动画：x→A·x 形变（不入书） |
| `docs/book/part-1-math/ch01-linear-algebra-vectors.md` | 修改 | 用 2 处图片引用替换 2 个 ASCII 代码块（第 208–221 行、第 245–256 行） |

---

## Task 1: 创建 figs 目录与 2 张静态 SVG

**Files:**
- Create: `docs/book/part-1-math/figs/ch01-dot-product-projection.svg`
- Create: `docs/book/part-1-math/figs/ch01-matrix-transform.svg`

**Interfaces:**
- Produces: 两个静态 SVG 文件，被 Task 3 的图片引用按文件名 `figs/ch01-dot-product-projection.svg`、`figs/ch01-matrix-transform.svg` 引用。

- [ ] **Step 1: 创建目录并写入图1 静态 SVG**

用文件写入工具创建 `docs/book/part-1-math/figs/ch01-dot-product-projection.svg`，内容**完整**如下（不要省略、不要改坐标）：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 250" font-family="Cambria, Georgia, 'Times New Roman', serif">
  <title>点积 = 投影长度</title>
  <desc>向量 x 与 y，夹角 θ，y 在 x 方向上的投影 ‖y‖·cosθ。</desc>
  <defs>
    <marker id="ahX" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#2563eb"/></marker>
    <marker id="ahY" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#ea580c"/></marker>
  </defs>
  <rect width="380" height="250" fill="#ffffff"/>
  <line x1="20" y1="205" x2="360" y2="205" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="40" y1="205" x2="335" y2="205" stroke="#2563eb" stroke-width="2.6" marker-end="url(#ahX)"/>
  <text x="342" y="200" font-size="16" font-style="italic" fill="#2563eb">x</text>
  <line x1="40" y1="205" x2="183.4" y2="104.7" stroke="#ea580c" stroke-width="2.6" marker-end="url(#ahY)"/>
  <text x="189" y="100" font-size="16" font-style="italic" fill="#ea580c">y</text>
  <line x1="183.4" y1="205" x2="183.4" y2="104.7" stroke="#9ca3af" stroke-width="1.6" stroke-dasharray="6 4"/>
  <path d="M176.4,205 L176.4,197 L183.4,197" fill="none" stroke="#9ca3af" stroke-width="1"/>
  <path d="M86,205 A46,46 0 0,0 77.7,178.6" fill="none" stroke="#6b7280" stroke-width="1.3"/>
  <text x="88" y="196" font-size="14" font-style="italic" fill="#374151">θ</text>
  <text x="150" y="222" font-size="12.5" fill="#6b7280" font-style="italic">proj = ‖y‖·cosθ</text>
  <circle cx="40" cy="205" r="2.8" fill="#1f2937"/>
  <text x="26" y="222" font-size="12" fill="#6b7280">O</text>
</svg>
```

- [ ] **Step 2: 写入图2 静态 SVG**

创建 `docs/book/part-1-math/figs/ch01-matrix-transform.svg`，内容**完整**如下：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 250" font-family="Cambria, Georgia, 'Times New Roman', serif">
  <title>矩阵乘法 = 旋转 + 缩放</title>
  <desc>输入向量 x 经矩阵 A 变换为 A·x：旋转角 α 并缩放。</desc>
  <defs>
    <marker id="mX" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#2563eb"/></marker>
    <marker id="mY" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#ea580c"/></marker>
    <marker id="mArc" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7280"/></marker>
  </defs>
  <rect width="380" height="250" fill="#ffffff"/>
  <line x1="30" y1="210" x2="360" y2="210" stroke="#e5e7eb" stroke-width="1"/>
  <line x1="60" y1="30" x2="60" y2="235" stroke="#e5e7eb" stroke-width="1"/>
  <line x1="60" y1="210" x2="163.4" y2="172.4" stroke="#2563eb" stroke-width="2.6" marker-end="url(#mX)"/>
  <text x="168" y="170" font-size="16" font-style="italic" fill="#2563eb">x</text>
  <line x1="60" y1="210" x2="148" y2="57.6" stroke="#ea580c" stroke-width="2.6" marker-end="url(#mY)"/>
  <text x="153" y="54" font-size="16" font-style="italic" fill="#ea580c">A·x</text>
  <path d="M102.3,194.6 A45,45 0 0,0 82.5,171.03" fill="none" stroke="#6b7280" stroke-width="1.4" marker-end="url(#mArc)"/>
  <text x="98" y="186" font-size="14" font-style="italic" fill="#374151">α</text>
  <path d="M170,172.4 A110,110 0 0,1 60,100" fill="none" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3 4"/>
  <circle cx="60" cy="210" r="2.8" fill="#1f2937"/>
  <text x="44" y="227" font-size="12" fill="#6b7280">O</text>
  <text x="120" y="240" font-size="12.5" fill="#6b7280" font-style="italic">旋转 α · 缩放 ×1.6</text>
</svg>
```

- [ ] **Step 3: 校验两个静态 SVG 是合法 XML**

运行（仓库根目录）：
```bash
py -c "import glob,xml.etree.ElementTree as ET; [ET.parse(f) or print('OK', f) for f in glob.glob('docs/book/part-1-math/figs/*.svg')]"
```
Expected 输出：列出 4 个文件中已存在的（本任务后应为 2 个）`OK docs/book/part-1-math/figs/ch01-dot-product-projection.svg` 与 `.../ch01-matrix-transform.svg`，且无 `ParseError`。

- [ ] **Step 4: 人工目测静态图**

在浏览器直接打开这两个 `.svg`（或用 IDE 预览），确认：图1 有蓝向量 x、橙向量 y、灰色虚线投影、θ 弧、`proj = ‖y‖·cosθ` 标注；图2 有蓝 x、橙 A·x、带箭头的 α 旋转弧、淡虚线弧、`旋转 α · 缩放 ×1.6` 标注。色彩/比例与脑暴确认的风格 B 一致。

- [ ] **Step 5: （可选）提交检查点**

> 仅当用户同意提交时执行。
```bash
git add docs/book/part-1-math/figs/ch01-dot-product-projection.svg docs/book/part-1-math/figs/ch01-matrix-transform.svg
git commit -m "docs(book): ch01 用静态 SVG 替换 ASCII 几何图（点积投影 / 矩阵变换）"
```

---

## Task 2: 创建 2 张动画 SVG

**Files:**
- Create: `docs/book/part-1-math/figs/ch01-dot-product-projection_anim.svg`
- Create: `docs/book/part-1-math/figs/ch01-matrix-transform_anim.svg`

**Interfaces:**
- Produces: 两个 `_anim.svg` 文件；不被 ch01 引用，仅供直接打开 / 未来 web 用。

- [ ] **Step 1: 写入图1 动画 SVG**

创建 `docs/book/part-1-math/figs/ch01-dot-product-projection_anim.svg`，内容**完整**如下（注意每个被动画的属性都给了 base 值）：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 250" font-family="Cambria, Georgia, 'Times New Roman', serif">
  <title>点积 = 投影长度（动画）</title>
  <desc>向量 y 在来回扫动夹角 θ，投影垂线与 θ 弧实时跟随，演示投影随夹角变化。</desc>
  <defs>
    <marker id="ahXa" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#2563eb"/></marker>
    <marker id="ahYa" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#ea580c"/></marker>
  </defs>
  <rect width="380" height="250" fill="#ffffff"/>
  <line x1="20" y1="205" x2="360" y2="205" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="40" y1="205" x2="335" y2="205" stroke="#2563eb" stroke-width="2.6" marker-end="url(#ahXa)"/>
  <text x="342" y="200" font-size="16" font-style="italic" fill="#2563eb">x</text>
  <line x1="206.44" y1="205" x2="206.44" y2="150.93" stroke="#9ca3af" stroke-width="1.6" stroke-dasharray="6 4">
    <animate attributeName="x1" values="206.44;163.74;94.07;163.74;206.44" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="x2" values="206.44;163.74;94.07;163.74;206.44" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="y2" values="150.93;81.26;38.56;81.26;150.93" dur="6s" repeatCount="indefinite"/>
  </line>
  <line x1="40" y1="205" x2="206.44" y2="150.93" stroke="#ea580c" stroke-width="2.6" marker-end="url(#ahYa)">
    <animate attributeName="x2" values="206.44;163.74;94.07;163.74;206.44" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="y2" values="150.93;81.26;38.56;81.26;150.93" dur="6s" repeatCount="indefinite"/>
  </line>
  <text x="212" y="146" font-size="16" font-style="italic" fill="#ea580c">y</text>
  <path d="M86,205 A46,46 0 0,0 83.75,190.79" fill="none" stroke="#6b7280" stroke-width="1.3">
    <animate attributeName="d"
      values="M86,205 A46,46 0 0,0 83.75,190.79;M86,205 A46,46 0 0,0 72.53,172.47;M86,205 A46,46 0 0,0 54.21,161.25;M86,205 A46,46 0 0,0 72.53,172.47;M86,205 A46,46 0 0,0 83.75,190.79"
      dur="6s" repeatCount="indefinite"/>
  </path>
  <text x="84" y="196" font-size="14" font-style="italic" fill="#374151">θ
    <animate attributeName="x" values="84;76;62;76;84" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="y" values="196;188;180;188;196" dur="6s" repeatCount="indefinite"/>
  </text>
  <circle cx="40" cy="205" r="2.8" fill="#1f2937"/>
  <text x="26" y="222" font-size="12" fill="#6b7280">O</text>
  <text x="40" y="240" font-size="12.5" fill="#6b7280" font-style="italic">proj = ‖y‖·cosθ</text>
</svg>
```

- [ ] **Step 2: 写入图2 动画 SVG**

创建 `docs/book/part-1-math/figs/ch01-matrix-transform_anim.svg`，内容**完整**如下：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 250" font-family="Cambria, Georgia, 'Times New Roman', serif">
  <title>矩阵乘法 = 旋转 + 缩放（动画）</title>
  <desc>橙色向量在 x 与 A·x 之间连续形变，演示矩阵把 x 旋转 α 又拉伸。</desc>
  <defs>
    <marker id="aX2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#2563eb"/></marker>
    <marker id="aY2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#ea580c"/></marker>
  </defs>
  <rect width="380" height="250" fill="#ffffff"/>
  <line x1="30" y1="210" x2="360" y2="210" stroke="#e5e7eb" stroke-width="1"/>
  <line x1="60" y1="30" x2="60" y2="235" stroke="#e5e7eb" stroke-width="1"/>
  <line x1="60" y1="210" x2="163.4" y2="172.4" stroke="#2563eb" stroke-width="2" stroke-opacity="0.35" marker-end="url(#aX2)"/>
  <text x="168" y="170" font-size="16" font-style="italic" fill="#2563eb" fill-opacity="0.5">x</text>
  <line x1="60" y1="210" x2="163.4" y2="172.4" stroke="#ea580c" stroke-width="2.8" marker-end="url(#aY2)">
    <animate attributeName="x2" values="163.4;148;163.4" dur="4s" repeatCount="indefinite"/>
    <animate attributeName="y2" values="172.4;57.6;172.4" dur="4s" repeatCount="indefinite"/>
  </line>
  <text x="153" y="54" font-size="16" font-style="italic" fill="#ea580c">A·x</text>
  <circle cx="60" cy="210" r="2.8" fill="#1f2937"/>
  <text x="44" y="227" font-size="12" fill="#6b7280">O</text>
  <text x="100" y="240" font-size="12.5" fill="#6b7280" font-style="italic">矩阵把 x 旋转 α 又拉伸</text>
</svg>
```

- [ ] **Step 3: 校验全部 4 个 SVG 是合法 XML**

运行：
```bash
py -c "import glob,xml.etree.ElementTree as ET; [ET.parse(f) or print('OK', f) for f in sorted(glob.glob('docs/book/part-1-math/figs/*.svg'))]"
```
Expected：恰好打印 4 行 `OK ...`，无 `ParseError`。

- [ ] **Step 4: 人工目测动画**

在浏览器直接打开两个 `_anim.svg`：图1 橙色 y 应在 ~6s 一轮来回扫动 θ、灰色虚线投影与 θ 弧同步跟随；图2 橙色向量应在 ~4s 一轮于 x 与 A·x 间形变，半透明蓝 x 始终作为参照。

- [ ] **Step 5: （可选）提交检查点**

> 仅当用户同意提交时执行。
```bash
git add docs/book/part-1-math/figs/ch01-dot-product-projection_anim.svg docs/book/part-1-math/figs/ch01-matrix-transform_anim.svg
git commit -m "docs(book): ch01 几何图追加动画版 SVG（_anim）"
```

---

## Task 3: 替换 ch01 的两处 ASCII 块为图片引用并验收

**Files:**
- Modify: `docs/book/part-1-math/ch01-linear-algebra-vectors.md`（第 208–221 行 图1 块；第 245–256 行 图2 块）

**Interfaces:**
- Consumes: Task 1 产出的两个静态 SVG 的相对路径 `figs/ch01-dot-product-projection.svg`、`figs/ch01-matrix-transform.svg`（从该 .md 看，相对路径前缀就是 `figs/`）。

- [ ] **Step 1: 替换图1 ASCII 块**

在 `docs/book/part-1-math/ch01-linear-algebra-vectors.md` 中，定位到紧随句子「下面这张示意图把「投影」和「夹角」的关系画出来（为清晰起见画在 2 维平面）：」之后的那个 **无语言标注的围栏代码块**（以 ``` 开始、``` 结束，内容是含 `y  (终点)`、`proj = ‖y‖·cosθ`、`├─────┘` 的 ASCII 图）。把**整个围栏代码块（含两个 ``` 围栏行）**替换为下面这一行：

```markdown
![点积 = 投影长度](figs/ch01-dot-product-projection.svg)
```

其上方的引导句「下面这张示意图…」与下方的段落「一个直观的推论：当 $\mathbf{y}$ 与 $\mathbf{x}$ 同向…」**保持原样不动**。

- [ ] **Step 2: 替换图2 ASCII 块**

同理，定位到「LLM 里每一层做的 `x @ W`…」段落之后、以 ``` 包裹、内容含 `变换前`、`A·x`、`○──────────→` 的 ASCII 块。把**整个围栏代码块（含两个 ``` 围栏行）**替换为：

```markdown
![矩阵乘法 = 线性变换（旋转 + 缩放）](figs/ch01-matrix-transform.svg)
```

其前后段落保持原样。

- [ ] **Step 3: 校验 ASCII 残影已清除、引用已就位**

运行（预期：两条 `figs/...` 各匹配 1 次；两条 ASCII 特征字符均 0 匹配）：
```bash
rg -n "figs/ch01-(dot-product-projection|matrix-transform)\.svg" docs/book/part-1-math/ch01-linear-algebra-vectors.md
rg -nU "↗|├─────┘|○──────────→" docs/book/part-1-math/ch01-linear-algebra-vectors.md
```
Expected：第一条命令输出 2 行（图1、图2 引用各 1）；第二条命令**无输出**（ASCII 图已彻底移除）。

- [ ] **Step 4: 校验未误伤其它内容**

确认 mermaid 概念图与公式仍在（用 `-F` 固定字符串，避免正则/反引号转义问题）：
```bash
rg -nF "```mermaid" docs/book/part-1-math/ch01-linear-algebra-vectors.md
rg -nF "\\mathbf{x}" docs/book/part-1-math/ch01-linear-algebra-vectors.md
```
Expected：第一条至少 1 行命中（mermaid 块在）；第二条多处命中（点积等公式里的 `\mathbf{x}` 完好）。同时人工浏览该文件，确认除两处图片引用外，正文文字、章节结构、公式编号无变化。

- [ ] **Step 5: 最终目测渲染**

在 GitHub/GitLab 网页（或本地 markdown 预览）打开 ch01：两张图片应在原 ASCII 位置正常渲染为静态 SVG；点开图片可放大、清晰无锯齿。

- [ ] **Step 6: （可选）提交检查点**

> 仅当用户同意提交时执行。
```bash
git add docs/book/part-1-math/ch01-linear-algebra-vectors.md
git commit -m "docs(book): ch01 正文引用新 SVG，移除 ASCII 几何图"
```

---

## 验收标准（全部满足即完成）

1. `docs/book/part-1-math/figs/` 下恰好 4 个 `.svg`，且全部通过 `xml.etree.ElementTree.parse` 无报错。
2. ch01 中两处原 ASCII 块已被 `![…](figs/ch01-*.svg)` 取代；`rg "↗|├─────┘|○──────────→"` 无命中。
3. ch01 的 mermaid 概念图与正文公式未受影响。
4. 2 张静态 SVG 在浏览器/GitHub 渲染正确；2 张 `_anim.svg` 在浏览器直接打开时动画正常。
5. 无新增依赖、无构建脚本、未改动其它章节。
