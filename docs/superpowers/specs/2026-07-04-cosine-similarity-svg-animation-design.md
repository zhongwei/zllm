# Cosine Similarity SVG Animation Design

**Date**: 2026-07-04  
**Context**: Add SVG animations to "向量夹角与余弦相似度" section in `ch01-linear-algebra-vectors.md`

## Overview

Create two separate SVG animation files to illustrate cosine similarity:
1. Calculation process animation
2. Geometric relationship animation

## Design Rationale

**Two-file approach** chosen for:
- Single responsibility: each animation focuses on one concept
- Reusability: can be referenced independently in different chapters
- Reasonable duration: 6-10 seconds per animation
- Easier maintenance: modifications don't affect other files

## File 1: Calculation Animation

**Filename**: `ch01-cosine-similarity-calc_anim.svg`  
**Duration**: 10 seconds (loop)  
**Layout**: Three columns (left-middle-right), similar to `ch01-dot-product-calc_anim.svg`

### Visual Elements

**Left column**: Row vector xᵀ = [3, 4]
- Bracket notation (using path elements)
- Two elements: 3, 4
- Blue color scheme

**Middle column**: Column vector y = [6, 8]ᵀ
- Bracket notation (using path elements)
- Two elements: 6, 8
- Orange color scheme

**Right column**: Calculation process
- Step-by-step formula evaluation
- Color-coded results

### Animation Sequence

**Phase 1 (0-3s)**: Dot product calculation
- Highlight x₁=3 and y₁=6
- Show: 3 × 6 = 18
- Highlight x₂=4 and y₂=8
- Show: 4 × 8 = 32
- Show sum: 18 + 32 = 50

**Phase 2 (3-6s)**: Norm calculation
- Show: ‖x‖₂ = √(3² + 4²) = 5
- Show: ‖y‖₂ = √(6² + 8²) = 10

**Phase 3 (6-8s)**: Division
- Show: cos θ = 50 / (5 × 10)

**Phase 4 (8-10s)**: Final result
- Emphasize: cos θ = 1
- Interpretation: vectors are parallel (same direction)

### Color Scheme

Follow existing pattern from `ch01-dot-product-calc_anim.svg`:
- Blue (`#2563eb`): vector x elements
- Orange (`#ea580c`): vector y elements
- Green (`#059669`): intermediate results
- Purple (`#7c3aed`): final result
- Dark mode support via `prefers-color-scheme`

## File 2: Geometry Animation

**Filename**: `ch01-cosine-similarity-geometry_anim.svg`  
**Duration**: 8 seconds (loop)  
**Layout**: Single geometric figure with annotation panel

### Visual Elements

**Main figure**:
- Fixed vector x along horizontal axis (blue arrow)
- Rotating vector y (orange arrow)
- Angle arc θ between vectors
- Projection dashed line from y to x

**Annotation panel** (top-right or bottom):
- Real-time cos θ value display
- Color-coded: green (≈1) → yellow (≈0) → red (≈-1)

### Animation Sequence

**Full cycle (0-8s)**: Vector y rotates
- Start: θ = 0° (vectors aligned, cos θ = 1)
- Sweep: θ increases to 180°
- Return: θ decreases back to 0°

**Synchronized elements**:
- Angle arc follows rotation
- cos θ value updates in real-time
- Special markers at key angles:
  - θ = 0°: "方向相同 (cos = 1)"
  - θ = 90°: "正交 (cos = 0)"
  - θ = 180°: "方向相反 (cos = -1)"

### Technical Details

- Use `<animate>` on `x2`, `y2` attributes of vector y's line element
- Calculate intermediate positions for smooth rotation
- Update cos θ display synchronously
- Similar pattern to `ch01-dot-product-projection_anim.svg`

## Markdown Integration

**Location**: Section "向量夹角与余弦相似度" (lines 184-193)

**Changes**:
1. After formula definition (line 190), insert:
   ```markdown
   下面这张动画展示了余弦相似度的**计算过程**：
   
   ![余弦相似度计算过程](figs/ch01-cosine-similarity-calc_anim.svg)
   ```

2. After geometric explanation (line 192), insert:
   ```markdown
   下面这张动画展示了夹角 θ 与 cos θ 的**几何关系**：
   
   ![余弦相似度几何意义](figs/ch01-cosine-similarity-geometry_anim.svg)
   ```

## Implementation Notes

### Style Consistency

Both SVG files must follow existing patterns:
- XML declaration and namespace
- `viewBox` for responsive sizing
- CSS classes for colors with dark mode support
- System font stack for cross-platform rendering
- Grid background pattern (optional)

### Animation Technique

Use SMIL `<animate>` elements:
- `attributeName` for target attribute
- `values` for keyframe values
- `keyTimes` for timing (semicolon-separated 0-1 values)
- `dur` for duration
- `repeatCount="indefinite"` for looping

### Accessibility

- Include `<title>` and `<desc>` elements
- Ensure color contrast in both light/dark modes
- Avoid red-green only distinctions (add labels)

## File Locations

- SVG files: `docs/book/part-1-math/figs/`
- Markdown file: `docs/book/part-1-math/ch01-linear-algebra-vectors.md`

## Verification

After implementation:
1. Open SVG files in browser to verify animations play correctly
2. Check dark mode rendering
3. Verify markdown renders SVGs properly
4. Test responsive sizing at different viewport widths
