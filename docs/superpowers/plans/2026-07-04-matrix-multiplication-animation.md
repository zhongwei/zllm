# 矩阵乘法动画实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建矩阵乘法 SVG 动画并插入到 markdown 文档中

**Architecture:** 参考 ch01-dot-product-calc_anim.svg 的风格，创建展示 2×3 和 3×2 矩阵乘法的动画 SVG，使用 SMIL 动画技术实现逐格计算过程的可视化

**Tech Stack:** SVG, SMIL Animation, CSS

## Global Constraints

- 矩阵维度：A(2×3) × B(3×2) = C(2×2)
- 动画时长：10秒循环
- 颜色方案：蓝色(A)、橙色(B)、绿色(计算过程)、紫色(结果)
- 支持暗色模式
- 与参考动画风格一致

---

### Task 1: 创建 SVG 文件

**Files:**
- Create: `docs/book/part-1-math/figs/ch01-matrix-multiplication_anim.svg`

**Interfaces:**
- Consumes: 设计规范 `docs/superpowers/specs/2026-07-04-matrix-multiplication-animation-design.md`
- Produces: 完整的 SVG 动画文件

- [ ] **Step 1: 创建 SVG 基础结构和样式**

创建文件 `docs/book/part-1-math/figs/ch01-matrix-multiplication_anim.svg`，包含基础结构：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 300" width="100%" height="100%" font-family="system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
  <title>矩阵乘法计算过程（动画）</title>
  <desc>逐步展示矩阵乘法 A(2×3) × B(3×2) = C(2×2) 的计算过程</desc>
  <style>
    .axis{color:#cbd5e1} .a{color:#2563eb} .b{color:#ea580c} .calc{color:#059669} 
    .gray{color:#9ca3af} .ink{color:#1f2937} .lbl{color:#374151} .dim{color:#6b7280} 
    .bracket{color:#64748b} .result{color:#7c3aed} .op{color:#374151}
    @media (prefers-color-scheme: dark){
      .axis{color:#475569} .a{color:#60a5fa} .b{color:#fb923c} .calc{color:#34d399} 
      .gray{color:#64748b} .ink{color:#e5e7eb} .lbl{color:#e5e7eb} .dim{color:#9ca3af} 
      .bracket{color:#94a3b8} .result{color:#a78bfa} .op{color:#e5e7eb}
    }
  </style>
  
  <defs>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" stroke-width="0.3" class="gray" opacity="0.3"/>
    </pattern>
  </defs>
  
  <!-- 背景与网格 -->
  <rect width="100%" height="100%" fill="url(#grid)" class="gray"/>
  
  <!-- 辅助分割线 -->
  <line class="gray" x1="240" y1="30" x2="240" y2="270" stroke="currentColor" stroke-width="0.5" stroke-dasharray="4 4" opacity="0.4"/>
  <line class="gray" x1="440" y1="30" x2="440" y2="270" stroke="currentColor" stroke-width="0.5" stroke-dasharray="4 4" opacity="0.4"/>
  
  <!-- 顶部标题 -->
  <text class="lbl" x="120" y="35" font-size="14" font-weight="600" fill="currentColor" text-anchor="middle">矩阵 A (2×3)</text>
  <text class="lbl" x="340" y="35" font-size="14" font-weight="600" fill="currentColor" text-anchor="middle">矩阵 B (3×2)</text>
  <text class="lbl" x="590" y="35" font-size="14" font-weight="600" fill="currentColor" text-anchor="middle">结果 C (2×2)</text>
</svg>
```

- [ ] **Step 2: 添加矩阵 A 的括号和元素**

在 `<svg>` 标签内，标题之后添加矩阵 A (2×3)：

```xml
<!-- ================= 矩阵 A = [1 2 3; 4 5 6] ================= -->
<g id="matrix-A">
  <!-- 矩阵括号 -->
  <path d="M 45 70 L 35 70 L 35 170 L 45 170" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>
  <path d="M 195 70 L 205 70 L 205 170 L 195 170" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>
  
  <!-- 第1行: 1, 2, 3 -->
  <text class="ink" x="70" y="105" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">1</text>
  <text class="ink" x="120" y="105" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">2</text>
  <text class="ink" x="170" y="105" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">3</text>
  
  <!-- 第2行: 4, 5, 6 -->
  <text class="ink" x="70" y="145" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">4</text>
  <text class="ink" x="120" y="145" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">5</text>
  <text class="ink" x="170" y="145" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">6</text>
  
  <text class="a" x="120" y="195" font-size="14" font-style="italic" fill="currentColor" text-anchor="middle">A</text>
</g>
```

- [ ] **Step 3: 添加矩阵 B 的括号和元素**

在矩阵 A 之后添加矩阵 B (3×2)：

```xml
<!-- ================= 矩阵 B = [7 8; 9 10; 11 12] ================= -->
<g id="matrix-B">
  <!-- 矩阵括号 -->
  <path d="M 275 70 L 265 70 L 265 170 L 275 170" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>
  <path d="M 405 70 L 415 70 L 415 170 L 275 170" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>
  
  <!-- 第1行: 7, 8 -->
  <text class="ink" x="310" y="105" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">7</text>
  <text class="ink" x="370" y="105" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">8</text>
  
  <!-- 第2行: 9, 10 -->
  <text class="ink" x="310" y="135" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">9</text>
  <text class="ink" x="370" y="135" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">10</text>
  
  <!-- 第3行: 11, 12 -->
  <text class="ink" x="310" y="165" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">11</text>
  <text class="ink" x="370" y="165" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">12</text>
  
  <text class="b" x="340" y="195" font-size="14" font-style="italic" fill="currentColor" text-anchor="middle">B</text>
</g>
```

- [ ] **Step 4: 添加结果矩阵 C 的括号和元素**

在矩阵 B 之后添加结果矩阵 C (2×2)：

```xml
<!-- ================= 结果矩阵 C ================= -->
<g id="matrix-C">
  <!-- 矩阵括号 -->
  <path d="M 495 80 L 485 80 L 485 160 L 495 160" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>
  <path d="M 605 80 L 615 80 L 615 160 L 605 160" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>
  
  <!-- C11 = 58 -->
  <text class="ink" x="530" y="115" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">58</text>
  
  <!-- C12 = 64 -->
  <text class="ink" x="580" y="115" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">64</text>
  
  <!-- C21 = 139 -->
  <text class="ink" x="530" y="145" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">139</text>
  
  <!-- C22 = 154 -->
  <text class="ink" x="580" y="145" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">154</text>
  
  <text class="result" x="550" y="195" font-size="14" font-style="italic" fill="currentColor" text-anchor="middle">C = A×B</text>
</g>
```

- [ ] **Step 5: 添加计算过程显示区域**

在矩阵 C 之后添加计算过程显示：

```xml
<!-- ================= 计算过程 ================= -->
<g id="calculation">
  <!-- C11 的计算: 1×7 + 2×9 + 3×11 = 58 -->
  <text x="460" y="220" font-size="16" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
    <tspan class="a">1</tspan><tspan class="op">×</tspan><tspan class="b">7</tspan><tspan class="op"> + </tspan><tspan class="a">2</tspan><tspan class="op">×</tspan><tspan class="b">9</tspan><tspan class="op"> + </tspan><tspan class="a">3</tspan><tspan class="op">×</tspan><tspan class="b">11</tspan><tspan class="op"> = </tspan><tspan class="result" font-weight="bold">58</tspan>
    <animate attributeName="opacity" values="0;1;0;0;0;0;0;0;0" keyTimes="0;0.1;0.25;0.3;0.4;0.55;0.7;0.85;1" dur="10s" repeatCount="indefinite"/>
  </text>
  
  <!-- C12 的计算: 1×8 + 2×10 + 3×12 = 64 -->
  <text x="460" y="240" font-size="16" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
    <tspan class="a">1</tspan><tspan class="op">×</tspan><tspan class="b">8</tspan><tspan class="op"> + </tspan><tspan class="a">2</tspan><tspan class="op">×</tspan><tspan class="b">10</tspan><tspan class="op"> + </tspan><tspan class="a">3</tspan><tspan class="op">×</tspan><tspan class="b">12</tspan><tspan class="op"> = </tspan><tspan class="result" font-weight="bold">64</tspan>
    <animate attributeName="opacity" values="0;0;0;1;0;0;0;0;0" keyTimes="0;0.1;0.25;0.3;0.4;0.55;0.7;0.85;1" dur="10s" repeatCount="indefinite"/>
  </text>
  
  <!-- C21 的计算: 4×7 + 5×9 + 6×11 = 139 -->
  <text x="460" y="260" font-size="16" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
    <tspan class="a">4</tspan><tspan class="op">×</tspan><tspan class="b">7</tspan><tspan class="op"> + </tspan><tspan class="a">5</tspan><tspan class="op">×</tspan><tspan class="b">9</tspan><tspan class="op"> + </tspan><tspan class="a">6</tspan><tspan class="op">×</tspan><tspan class="b">11</tspan><tspan class="op"> = </tspan><tspan class="result" font-weight="bold">139</tspan>
    <animate attributeName="opacity" values="0;0;0;0;0;1;0;0;0" keyTimes="0;0.1;0.25;0.3;0.4;0.55;0.7;0.85;1" dur="10s" repeatCount="indefinite"/>
  </text>
  
  <!-- C22 的计算: 4×8 + 5×10 + 6×12 = 154 -->
  <text x="460" y="280" font-size="16" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
    <tspan class="a">4</tspan><tspan class="op">×</tspan><tspan class="b">8</tspan><tspan class="op"> + </tspan><tspan class="a">5</tspan><tspan class="op">×</tspan><tspan class="b">10</tspan><tspan class="op"> + </tspan><tspan class="a">6</tspan><tspan class="op">×</tspan><tspan class="b">12</tspan><tspan class="op"> = </tspan><tspan class="result" font-weight="bold">154</tspan>
    <animate attributeName="opacity" values="0;0;0;0;0;0;0;1;0" keyTimes="0;0.1;0.25;0.3;0.4;0.55;0.7;0.85;1" dur="10s" repeatCount="indefinite"/>
  </text>
</g>

<text class="dim" x="375" y="295" font-size="13" fill="currentColor" text-anchor="middle" font-style="italic">
  矩阵乘法 = A的行 × B的列 = 点积
</text>
```

- [ ] **Step 6: 验证 SVG 文件**

在浏览器中打开 SVG 文件，验证：
- 三个矩阵正确显示
- 动画循环正常
- 颜色和布局符合设计

---

### Task 2: 更新 Markdown 文档

**Files:**
- Modify: `docs/book/part-1-math/ch01-linear-algebra-vectors.md:138-142`

**Interfaces:**
- Consumes: 创建的 SVG 文件
- Produces: 更新后的 markdown 文档，包含 SVG 引用

- [ ] **Step 1: 在 markdown 中添加 SVG 引用**

在 ch01-linear-algebra-vectors.md 文件中，找到以下位置（约第 140 行）：

```markdown
> **矩阵乘法 = 「取 $A$ 的一行、取 $B$ 的一列、做点积」填进结果矩阵的对应位置。**

这也解释了为什么内层维度 $p$ 必须匹配：因为那正是做点积时求和的下标范围。
```

在 `> **矩阵乘法 = ...**` 这一行之后，插入：

```markdown

下面这张动画展示了矩阵乘法的**具体计算过程**（以 2×3 和 3×2 矩阵为例）：

![矩阵乘法计算过程](figs/ch01-matrix-multiplication_anim.svg)

可以看到，矩阵乘法就是「行×列做点积」——每一格结果都是 A 的一行与 B 的一列的点积。
```

- [ ] **Step 2: 验证 markdown 渲染**

在 markdown 预览器中打开文档，验证 SVG 图片正确显示。

- [ ] **Step 3: 提交更改**

```bash
git add docs/book/part-1-math/figs/ch01-matrix-multiplication_anim.svg
git add docs/book/part-1-math/ch01-linear-algebra-vectors.md
git commit -m "feat: 添加矩阵乘法计算过程动画"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ 布局：三栏展示矩阵 A、B、C
- ✅ 动画：10秒循环，逐格计算展示
- ✅ 示例数据：1-12 数字
- ✅ 视觉风格：与参考动画一致的颜色方案
- ✅ 插入位置：在「矩阵乘法」口诀之后

**2. Placeholder scan:**
- 无 "TBD"、"TODO" 等占位符
- 所有代码步骤都包含完整实现

**3. Type consistency:**
- SVG 文件路径一致
- markdown 引用路径正确
