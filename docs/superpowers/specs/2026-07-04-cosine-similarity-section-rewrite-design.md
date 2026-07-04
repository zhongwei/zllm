# Cosine Similarity Section Rewrite Design

**Date**: 2026-07-04
**Context**: Rewrite the "向量夹角与余弦相似度" subsection (§1.3) of `docs/book/part-1-math/ch01-linear-algebra-vectors.md` and fix its two SVGs.
**Supersedes**: `2026-07-04-cosine-similarity-svg-animation-design.md` (that spec produced the SVGs this rewrite corrects).

## Problem Statement

The current cosine similarity subsection is hard to learn from, and its SVGs are misleading or weak:

1. **Order is backwards.** The three value-regimes (cos = 1 / 0 / −1) only appear at line 218, *after* a heavy law-of-cosines derivation. Intuition should lead; algebra should follow.
2. **Geometry SVG lies.** `ch01-cosine-similarity-geometry_anim.svg` line 48 hardcodes the readout to `1.00` with only an opacity blink, while its `<desc>` claims "实时显示 cos θ 值变化". As **y** rotates through 90° and 180°, the number still says 1.00. This is a factual error shown to the reader.
3. **Geometry SVG also changes ‖y‖ mid-rotation.** Its keyframes `(218,80)→(150,70)→(40,200)` do not keep **y** at constant length, which undercuts the whole "cosine only depends on direction" message.
4. **Calc SVG is degenerate.** It uses **x**=(3,4), **y**=(6,8). Since **y**=2**x**, the example is locked to cos=1, so the reader never sees why dividing by norms matters.
5. **Derivation duplicated.** The law-of-cosines proof appears in both §1.3 (192–214) and §1.4 (242–258). (Out of scope to remove from §1.4 — see Non-Goals.)

## Goals

- Lead §1.3 with the three angle regimes (acute / right / obtuse → cos sign) before any algebra, anchored by a new static figure.
- Keep the law-of-cosines derivation (user choice), but move it to *after* the intuition and frame it more clearly.
- Fix the geometry SVG so the cos readout is truthful and ‖y‖ stays constant.
- Fix the calc SVG with a non-parallel example so the full computation (including the divide-by-norms step) is meaningful.
- Follow existing SVG conventions exactly (palette, dark-mode, SMIL, font stack, viewBox).

## Non-Goals

- Touching §1.4 "点积 = 投影长度" (keeps its own copy of the derivation — accepted duplication).
- Changing §1.5 / §1.6 / the chapter summary.
- Modifying any other SVG (dot-product, matrix, norm figures).
- Adding interactivity beyond SMIL (no JS).

## Design

### Part A — New narrative order for §1.3 (lines 184–234)

Rewrite the subsection in this order. Each bullet is one logical block; existing good prose (e.g. "为什么不用点积直接当相似度") is preserved verbatim where possible.

1. **Opening intuition (no formulas).** State the core idea: to compare *directions*, look at the angle θ between the two vectors. Three regimes:
   - θ < 90° → directions are close → "similar"
   - θ = 90° → directions are unrelated → "orthogonal"
   - θ > 90° → directions diverge → "opposing"
   Immediately reference the new 3-panel SVG.
2. **Why cos θ, not θ.** θ grows 0°→180° ("bigger = less similar") which fights the "similarity bigger = closer" convention. cos θ maps the angle to [−1, 1] with the right polarity. State the three landmarks here, tying each back to one panel:
   - cos θ = 1 ⇔ θ = 0° (same direction)
   - cos θ = 0 ⇔ θ = 90° (orthogonal)
   - cos θ = −1 ⇔ θ = 180° (opposite)
3. **The question.** We only have coordinates of **x**, **y** — how do we compute cos θ?
4. **Derivation (kept, moved here, clearer framing).** Law of cosines on ‖**x**−**y**‖²; expand the left side coordinate-wise; compare the two expansions to get **x**·**y** = ‖**x**‖‖**y**‖ cos θ. Add a one-line cue before the comparison: "比较两个展开式，含 cos θ 的那一项必须等于含 x_iy_i 的那一项". Box the final formula: cos θ = (**x**·**y**) / (‖**x**‖₂‖**y**‖₂).
5. **Value range.** Because |cos θ| ≤ 1 (a basic property of cosine), the result lies in [−1, 1]. Keep this short — the three landmarks already live in step 2, so here just state the bound itself. (Cauchy–Schwarz is introduced later in §1.4; do not forward-reference it here.)
6. **Why not just use the dot product.** Preserve lines 222 (length-vs-direction paragraph) essentially verbatim.
7. **Coordinate calculation.** Reference the fixed calc SVG with the new non-parallel example.
8. **Geometry.** Reference the fixed geometry SVG.

### Part B — NEW static SVG: three cases

**File**: `docs/book/part-1-math/figs/ch01-cosine-similarity-three-cases.svg` (static — no `_anim` suffix, matching `ch01-dot-product-projection.svg` convention).

**Layout**: viewBox `0 0 720 260`. Three mini coordinate systems side by side, each ~220 wide, sharing baseline y=200. Each panel: a small origin dot O, a fixed blue arrow **x** pointing right along the axis, and an orange arrow **y** at the panel's angle. An angle arc + "θ" label between them. Below each: a result chip.

| Panel | Angle | x tip | y tip | cos θ sign | Result chip text |
|---|---|---|---|---|---|
| 1 急角 (acute) | ~45° | (right) | up-right | > 0 | cos θ ≈ 0.7 · 方向相近 |
| 2 直角 (right) | 90° | (right) | straight up | = 0 | cos θ = 0 · 正交 / 无关 |
| 3 钝角 (obtuse) | ~135° | (right) | up-left | < 0 | cos θ ≈ −0.7 · 方向偏离 |

Header above the three panels: "夹角决定方向相似度". Color follows existing palette classes (`.x` blue, `.y` orange, `.dim`, `.result`) with dark-mode media query. The sign of cos is reinforced by color: green-ish for >0, gray for =0, red-ish for <0 (all already in palette; keep contrasts accessible — labels carry meaning, not color alone).

### Part C — FIX geometry SVG (`...-geometry_anim.svg`)

Redesign the animation so it is truthful and direction-focused.

**Coordinate system**: viewBox `0 0 380 280`. Origin O=(180,210). **x** is a fixed blue arrow from O to (320,210) along the horizontal. **y** is an orange arrow of **constant pixel length R=140**, rotating through 7 discrete poses so each is readable (not a continuous smear). tip(θ) = (180 + 140·cos θ, 210 − 140·sin θ) (SVG y-down).

**Keyframes** (forward sweep 0°→180°, then reverse 180°→0°, looped). The cos readout is rendered as 7 separate `<text>` elements whose `opacity` is staggered to show exactly one value per pose (same technique already used in the calc SVG).

| θ | y tip (x2,y2) | cos θ display | readout color |
|---|---|---|---|
| 0° | (320, 210) | 1.00 | green (`.calc`) |
| 30° | (301.2, 140) | 0.87 | green |
| 60° | (250, 88.8) | 0.50 | green |
| 90° | (180, 70) | 0.00 | gray (`.dim`) |
| 120° | (110, 88.8) | −0.50 | red (`.norm`) |
| 150° | (58.8, 140) | −0.87 | red |
| 180° | (40, 210) | −1.00 | red |

**Other synced elements**:
- The θ arc path `d` animates through matching keyframes (arc grows with θ; sweep flag flips at >180° — here max is 180° so a single sweep direction suffices).
- The "θ" label position tracks the arc midpoint.
- The three interpretation labels ("方向相同 / 正交 / 方向相反") get opacity timings that match the pose where θ is actually near 0° / 90° / 180° (current timings are misaligned).
- ‖**y**‖ is constant by construction (R=140 at every keyframe).

**Timing**: animate `x2`/`y2` with `values` listing the 7 anchor tips forward then the reverse, over a ~7s loop. The vector moves smoothly between anchors; the *perceptual* dwelling at each regime comes from the stepped readout (one opaque value per anchor) — that is what makes 0°/90°/180° readable, not the motion itself.

### Part D — FIX calc SVG (`...-calc_anim.svg`)

Replace the degenerate parallel pair with a non-parallel, clean-arithmetic pair.

**Example**: **x** = (3, 4), **y** = (−3, 4). Mirror-symmetric (clean figure), equal norms (simplifies), all-integer arithmetic, result clearly ≠ 1.

**Recomputed steps** (every animated number in the SVG is updated to these):

- Phase 1 — dot product:
  - 3 × (−3) = −9
  - 4 × 4 = 16
  - −9 + 16 = **7**
- Phase 2 — norms:
  - ‖**x**‖ = √(3² + 4²) = √(9 + 16) = **5**
  - ‖**y**‖ = √((−3)² + 4²) = √(9 + 16) = **5**
- Phase 3 — division:
  - cos θ = 7 / (5 × 5) = **0.28**
- Conclusion line: change "x 与 y 方向相同" → "**x** 与 **y** 方向相近但不一致（θ ≈ 74°）".

The y column's two cells become `−3` and `4` (handle the minus sign in the `<text>` / `<tspan>`; reuse existing `.y` class). Keep the rest of the layout, brackets, grid, phase timings, and color scheme unchanged.

## Files Touched

| File | Action |
|---|---|
| `docs/book/part-1-math/ch01-linear-algebra-vectors.md` | Rewrite §1.3 (lines 184–234) per Part A; add reference to the new 3-panel SVG |
| `docs/book/part-1-math/figs/ch01-cosine-similarity-three-cases.svg` | **New** static 3-panel figure (Part B) |
| `docs/book/part-1-math/figs/ch01-cosine-similarity-geometry_anim.svg` | Rewrite per Part C (truthful readout, constant ‖y‖, fixed label timings) |
| `docs/book/part-1-math/figs/ch01-cosine-similarity-calc_anim.svg` | Update numbers per Part D (new non-parallel example) |

No other files change.

## Implementation Notes (conventions to follow)

- XML declaration + namespace on every SVG.
- `viewBox` for responsive sizing; `width`/`height` as in siblings.
- CSS classes for color + `@media (prefers-color-scheme: dark)` block — copy the class set used in the existing cosine SVGs.
- Font stack: the two existing cosine SVGs differ (one serif `Cambria…`, one `system-ui…`). Match each file's *own* existing stack to stay internally consistent.
- SMIL `<animate>` only; `attributeName` / `values` / `keyTimes` / `dur` / `repeatCount="indefinite"`. Stepped text via opacity stagger (already the chapter's pattern).
- `<title>` and `<desc>` on every SVG; keep them accurate (fix the geometry SVG's now-false desc).
- Mathematical correctness is non-negotiable: every displayed number must equal the actual value at that keyframe.

## Verification

1. Open each SVG in a browser; confirm animation loops with no jitter and the readout matches the drawn angle at every pose.
2. For the geometry SVG: pause-equivalent check — at θ=90° the readout must read 0.00 and **y** must be vertical; at θ=180° readout −1.00 and **y** points left along the axis; ‖**y**‖ identical at all poses.
3. For the calc SVG: hand-verify −9 + 16 = 7, √(9+16)=5, 7/25 = 0.28.
4. Toggle OS dark mode; confirm contrast and that labels remain legible.
5. Render the markdown; confirm the new 3-panel SVG appears right after the opening intuition paragraph and the other two SVGs stay in their (reordered) positions.
6. Re-read §1.3 end-to-end: the three regimes must appear before the derivation; the derivation must appear before the calc SVG.

## Open Questions

None — all design decisions confirmed with the user (scope = §1.3 only; keep derivation but reorder; 3-panel + fix both SVGs; calc example **x**=(3,4), **y**=(−3,4)).
