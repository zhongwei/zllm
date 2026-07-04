# Dot Product Calculation Animation SVG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an elegant SVG animation showing the step-by-step calculation of dot product for 3D vectors, in the style of 3Blue1Brown.

**Architecture:** Single SVG file with SMIL animations, three-column layout (row vector x | column vector y | calculation process), progressive highlighting with smooth transitions.

**Tech Stack:** SVG with SMIL animations, CSS for styling, dark mode support via media queries.

## Global Constraints

- Follow existing SVG style in `docs/book/part-1-math/figs/` (color scheme, font, dimensions)
- Support both light and dark mode via `prefers-color-scheme`
- Use CSS classes for colors consistent with existing figures
- Font: Cambria, Georgia, 'Times New Roman', serif
- Dimensions: 500×280 (wider than existing to fit three columns)

---

### Task 1: Create SVG Skeleton and Static Layout

**Files:**
- Create: `docs/book/part-1-math/figs/ch01-dot-product-calc_anim.svg`

**Interfaces:**
- Consumes: Existing color scheme from `ch01-dot-product-projection_anim.svg`
- Produces: Complete animated SVG file

- [ ] **Step 1: Create SVG file with base structure and three-column layout**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 280" width="500" height="280" font-family="Cambria, Georgia, 'Times New Roman', serif">
  <title>点积计算过程（动画）</title>
  <desc>逐步展示点积 x·y = x₁y₁ + x₂y₂ + x₃y₃ 的计算过程</desc>
  <style>
    .axis{color:#cbd5e1}.x{color:#2563eb}.y{color:#ea580c}.calc{color:#059669}.gray{color:#9ca3af}.ink{color:#1f2937}.lbl{color:#374151}.dim{color:#6b7280}.bracket{color:#64748b}
    @media (prefers-color-scheme: dark){
      .axis{color:#475569}.x{color:#60a5fa}.y{color:#fb923c}.calc{color:#34d399}.gray{color:#64748b}.ink{color:#e5e7eb}.lbl{color:#e5e7eb}.dim{color:#9ca3af}.bracket{color:#94a3b8}
    }
  </style>
  
  <!-- 背景网格（可选，增加3b1b风格） -->
  <defs>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" stroke-width="0.3" class="gray" opacity="0.3"/>
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="url(#grid)" class="gray"/>
  
  <!-- 三列分隔线 -->
  <line class="gray" x1="140" y1="30" x2="140" y2="250" stroke="currentColor" stroke-width="0.5" stroke-dasharray="4 4" opacity="0.4"/>
  <line class="gray" x1="280" y1="30" x2="280" y2="250" stroke="currentColor" stroke-width="0.5" stroke-dasharray="4 4" opacity="0.4"/>
  
  <!-- 列标题 -->
  <text class="lbl" x="70" y="35" font-size="14" font-weight="500" fill="currentColor" text-anchor="middle">x (行向量)</text>
  <text class="lbl" x="210" y="35" font-size="14" font-weight="500" fill="currentColor" text-anchor="middle">y (列向量)</text>
  <text class="lbl" x="390" y="35" font-size="14" font-weight="500" fill="currentColor" text-anchor="middle">计算过程</text>
</svg>
```

- [ ] **Step 2: Add row vector x (left column)**

```xml
  <!-- 左列：行向量 x = [1, 2, 3] -->
  <g id="row-vector-x">
    <!-- 左括号 -->
    <text class="bracket" x="25" y="130" font-size="60" fill="currentColor">[</text>
    
    <!-- 元素 -->
    <text class="x" id="x1" x="55" y="105" font-size="26" fill="currentColor" text-anchor="middle">1</text>
    <text class="x" id="x2" x="85" y="105" font-size="26" fill="currentColor" text-anchor="middle">2</text>
    <text class="x" id="x3" x="115" y="105" font-size="26" fill="currentColor" text-anchor="middle">3</text>
    
    <!-- 右括号 -->
    <text class="bracket" x="130" y="130" font-size="60" fill="currentColor">]</text>
    
    <!-- 向量名 -->
    <text class="x" x="70" y="155" font-size="15" font-style="italic" fill="currentColor" text-anchor="middle">xᵀ</text>
  </g>
```

- [ ] **Step 3: Add column vector y (middle column)**

```xml
  <!-- 中列：列向量 y = [4, 5, 6]ᵀ -->
  <g id="col-vector-y">
    <!-- 左括号 -->
    <text class="bracket" x="175" y="145" font-size="70" fill="currentColor">[</text>
    
    <!-- 元素（垂直排列） -->
    <text class="y" id="y1" x="210" y="75" font-size="26" fill="currentColor" text-anchor="middle">4</text>
    <text class="y" id="y2" x="210" y="115" font-size="26" fill="currentColor" text-anchor="middle">5</text>
    <text class="y" id="y3" x="210" y="155" font-size="26" fill="currentColor" text-anchor="middle">6</text>
    
    <!-- 右括号 -->
    <text class="bracket" x="235" y="145" font-size="70" fill="currentColor">]</text>
    
    <!-- 向量名 -->
    <text class="y" x="210" y="190" font-size="15" font-style="italic" fill="currentColor" text-anchor="middle">y</text>
  </g>
```

- [ ] **Step 4: Add calculation area (right column)**

```xml
  <!-- 右列：计算过程 -->
  <g id="calculation">
    <!-- 步骤1: x₁ × y₁ -->
    <text class="calc" id="calc1" x="310" y="75" font-size="22" fill="currentColor" opacity="0">
      <tspan class="x">1</tspan>×<tspan class="y">4</tspan> = <tspan class="ink">4</tspan>
    </text>
    
    <!-- 步骤2: x₂ × y₂ -->
    <text class="calc" id="calc2" x="310" y="115" font-size="22" fill="currentColor" opacity="0">
      <tspan class="x">2</tspan>×<tspan class="y">5</tspan> = <tspan class="ink">10</tspan>
    </text>
    
    <!-- 步骤3: x₃ × y₃ -->
    <text class="calc" id="calc3" x="310" y="155" font-size="22" fill="currentColor" opacity="0">
      <tspan class="x">3</tspan>×<tspan class="y">6</tspan> = <tspan class="ink">18</tspan>
    </text>
    
    <!-- 求和线 -->
    <line id="sum-line" class="ink" x1="305" y1="168" x2="480" y2="168" stroke="currentColor" stroke-width="1.5" opacity="0"/>
    
    <!-- 最终结果 -->
    <text class="ink" id="final-sum" x="310" y="200" font-size="24" font-weight="500" fill="currentColor" opacity="0">
      x·y = <tspan font-size="28">32</tspan>
    </text>
  </g>
  
  <!-- 底部公式说明 -->
  <text class="dim" x="250" y="260" font-size="13" fill="currentColor" text-anchor="middle" font-style="italic">
    点积 = 对应元素相乘再求和
  </text>
</svg>
```

---

### Task 2: Add Highlight Effects

**Files:**
- Modify: `docs/book/part-1-math/figs/ch01-dot-product-calc_anim.svg`

**Interfaces:**
- Consumes: Element IDs from Task 1
- Produces: Highlight animation definitions

- [ ] **Step 1: Add highlight glow filter in defs section**

```xml
  <defs>
    <!-- ... existing grid pattern ... -->
    
    <!-- 发光效果（3b1b风格） -->
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    
    <!-- 高亮圆圈（用于标记当前计算元素） -->
    <circle id="highlight-circle" r="18" fill="none" stroke-width="2.5" opacity="0"/>
  </defs>
```

- [ ] **Step 2: Add highlight circles for vector elements**

```xml
  <!-- 高亮指示器（放在向量元素下方） -->
  <g id="highlights">
    <!-- x 向量的高亮圆圈 -->
    <circle class="x" id="hl-x1" cx="55" cy="98" r="18" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0"/>
    <circle class="x" id="hl-x2" cx="85" cy="98" r="18" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0"/>
    <circle class="x" id="hl-x3" cx="115" cy="98" r="18" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0"/>
    
    <!-- y 向量的高亮圆圈 -->
    <circle class="y" id="hl-y1" cx="210" cy="67" r="18" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0"/>
    <circle class="y" id="hl-y2" cx="210" cy="107" r="18" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0"/>
    <circle class="y" id="hl-y3" cx="210" cy="147" r="18" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0"/>
  </g>
```

---

### Task 3: Add SMIL Animations

**Files:**
- Modify: `docs/book/part-1-math/figs/ch01-dot-product-calc_anim.svg`

**Interfaces:**
- Consumes: Element IDs from Tasks 1-2
- Produces: Complete animated SVG

- [ ] **Step 1: Add animation for step 1 (x₁ × y₁)**

Add to the highlights group after the circle definitions:

```xml
    <!-- 步骤1动画：高亮 x₁ 和 y₁ -->
    <animate href="#hl-x1" attributeName="opacity" values="0;1;1;0" keyTimes="0;0.05;0.15;0.2" dur="10s" repeatCount="indefinite"/>
    <animate href="#hl-x1" attributeName="fill-opacity" values="0;0.2;0.2;0" keyTimes="0;0.05;0.15;0.2" dur="10s" repeatCount="indefinite"/>
    <animate href="#hl-y1" attributeName="opacity" values="0;1;1;0" keyTimes="0;0.05;0.15;0.2" dur="10s" repeatCount="indefinite"/>
    <animate href="#hl-y1" attributeName="fill-opacity" values="0;0.2;0.2;0" keyTimes="0;0.05;0.15;0.2" dur="10s" repeatCount="indefinite"/>
    
    <!-- 显示计算过程1 -->
    <animate href="#calc1" attributeName="opacity" values="0;1" keyTimes="0;0.1" dur="10s" repeatCount="indefinite"/>
```

- [ ] **Step 2: Add animation for step 2 (x₂ × y₂)**

```xml
    <!-- 步骤2动画：高亮 x₂ 和 y₂ -->
    <animate href="#hl-x2" attributeName="opacity" values="0;1;1;0" keyTimes="0.2;0.25;0.35;0.4" dur="10s" repeatCount="indefinite"/>
    <animate href="#hl-x2" attributeName="fill-opacity" values="0;0.2;0.2;0" keyTimes="0.2;0.25;0.35;0.4" dur="10s" repeatCount="indefinite"/>
    <animate href="#hl-y2" attributeName="opacity" values="0;1;1;0" keyTimes="0.2;0.25;0.35;0.4" dur="10s" repeatCount="indefinite"/>
    <animate href="#hl-y2" attributeName="fill-opacity" values="0;0.2;0.2;0" keyTimes="0.2;0.25;0.35;0.4" dur="10s" repeatCount="indefinite"/>
    
    <!-- 显示计算过程2 -->
    <animate href="#calc2" attributeName="opacity" values="0;1" keyTimes="0;0.3" dur="10s" repeatCount="indefinite"/>
```

- [ ] **Step 3: Add animation for step 3 (x₃ × y₃)**

```xml
    <!-- 步骤3动画：高亮 x₃ 和 y₃ -->
    <animate href="#hl-x3" attributeName="opacity" values="0;1;1;0" keyTimes="0.4;0.45;0.55;0.6" dur="10s" repeatCount="indefinite"/>
    <animate href="#hl-x3" attributeName="fill-opacity" values="0;0.2;0.2;0" keyTimes="0.4;0.45;0.55;0.6" dur="10s" repeatCount="indefinite"/>
    <animate href="#hl-y3" attributeName="opacity" values="0;1;1;0" keyTimes="0.4;0.45;0.55;0.6" dur="10s" repeatCount="indefinite"/>
    <animate href="#hl-y3" attributeName="fill-opacity" values="0;0.2;0.2;0" keyTimes="0.4;0.45;0.55;0.6" dur="10s" repeatCount="indefinite"/>
    
    <!-- 显示计算过程3 -->
    <animate href="#calc3" attributeName="opacity" values="0;1" keyTimes="0;0.5" dur="10s" repeatCount="indefinite"/>
```

- [ ] **Step 4: Add animation for final sum**

```xml
    <!-- 显示求和线和最终结果 -->
    <animate href="#sum-line" attributeName="opacity" values="0;1" keyTimes="0;0.65" dur="10s" repeatCount="indefinite"/>
    <animate href="#final-sum" attributeName="opacity" values="0;1" keyTimes="0;0.7" dur="10s" repeatCount="indefinite"/>
    
    <!-- 最终结果高亮闪烁 -->
    <animate href="#final-sum" attributeName="fill-opacity" values="1;0.6;1" keyTimes="0.75;0.85;0.95" dur="10s" repeatCount="indefinite"/>
```

---

### Task 4: Update Markdown File Reference

**Files:**
- Modify: `docs/book/part-1-math/ch01-linear-algebra-vectors.md:208`

**Interfaces:**
- Consumes: New SVG file
- Produces: Updated documentation reference

- [ ] **Step 1: Add reference to new SVG after the existing dot product projection animation**

After line 208, add a new figure reference:

```markdown
下面这张动画展示了点积的**具体计算过程**（以 3 维向量为例）：

![点积计算过程](figs/ch01-dot-product-calc_anim.svg)

可以看到，点积就是「对应位置相乘、再求和」——这个简单的操作正是注意力机制的核心。
```

- [ ] **Step 2: Verify the SVG renders correctly**

Open the markdown file in a browser or markdown viewer that supports SVG to confirm the animation plays as expected.

---

## Verification

After implementation:
1. Open `docs/book/part-1-math/figs/ch01-dot-product-calc_anim.svg` in a browser
2. Verify animation sequence: step1 → step2 → step3 → sum
3. Test dark mode by toggling browser/system preference
4. Check the SVG in the context of the markdown document
