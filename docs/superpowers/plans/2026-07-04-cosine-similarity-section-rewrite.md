# Cosine Similarity Section Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the "向量夹角与余弦相似度" subsection (§1.3) of `ch01-linear-algebra-vectors.md` so intuition leads the algebra, and fix/replace its three SVGs (one new static, two corrected) so they are mathematically truthful and pedagogically clear.

**Architecture:** Four independent file-level tasks done in order — (1) create a new static 3-panel SVG, (2) rewrite the geometry animation with a truthful live cos readout and constant ‖y‖, (3) rewrite the calc animation with a non-parallel example, (4) rewrite the §1.3 markdown prose to the new order and reference all three SVGs. Each task ends with a commit.

**Tech Stack:** Hand-written SVG (SMIL `<animate>`), Markdown + LaTeX (`$...$`, `$$...$$`). No JS, no build step.

## Global Constraints

- **Palette / classes** (copy verbatim, used across all three SVGs): light → `.x #2563eb`, `.y #ea580c`, `.arc #6b7280`, `.ink #1f2937`, `.lbl #374151`, `.dim #6b7280`, `.pos #059669`, `.neg #dc2626`; dark via `@media (prefers-color-scheme: dark)` → `.x #60a5fa`, `.y #fb923c`, `.arc #94a3b8`, `.ink #e5e7eb`, `.lbl #e5e7eb`, `.dim #9ca3af`, `.pos #34d399`, `.neg #f87171`.
- **Font stack**: serif `Cambria, Georgia, 'Times New Roman', serif` for the geometry and 3-panel SVGs (they are geometry-type figures, matching `ch01-dot-product-projection_anim.svg`).
- Every SVG has `<?xml version="1.0" encoding="UTF-8"?>`, the `xmlns` namespace, a `viewBox`, a `<title>`, and an accurate `<desc>`.
- SMIL only (`<animate>` with `attributeName`/`values`/`keyTimes`/`dur`/`repeatCount="indefinite"`). No JavaScript.
- **Math correctness is non-negotiable**: every displayed number equals the actual value at that keyframe.
- Negative numbers in SVG `<text>` use the Unicode minus `−` (U+2212), matching the book's typographic style.
- Do NOT modify §1.4, §1.5, §1.6, or any SVG other than the three named below.
- Each task commits with a descriptive message (no `feat:`/`fix:` prefix required — match recent cosine commits which are plain English).

**Spec:** `docs/superpowers/specs/2026-07-04-cosine-similarity-section-rewrite-design.md`

---

### Task 1: New static 3-panel SVG (three angle regimes)

**Files:**
- Create: `docs/book/part-1-math/figs/ch01-cosine-similarity-three-cases.svg`

**Interfaces:**
- Produces: a static SVG referenced by §1.3 markdown as `figs/ch01-cosine-similarity-three-cases.svg`.

- [ ] **Step 1: Write the SVG file**

Create `docs/book/part-1-math/figs/ch01-cosine-similarity-three-cases.svg` with exactly this content:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 260" width="720" height="260" font-family="Cambria, Georgia, 'Times New Roman', serif">
  <title>余弦相似度三种情形</title>
  <desc>三个迷你坐标系并列：锐角(cos&gt;0 方向相近)、直角(cos=0 正交)、钝角(cos&lt;0 方向偏离)。</desc>
  <style>
    .axis{color:#cbd5e1}.x{color:#2563eb}.y{color:#ea580c}.arc{color:#6b7280}.ink{color:#1f2937}.lbl{color:#374151}.dim{color:#6b7280}.pos{color:#059669}.zero{color:#6b7280}.neg{color:#dc2626}
    @media (prefers-color-scheme: dark){
      .axis{color:#475569}.x{color:#60a5fa}.y{color:#fb923c}.arc{color:#94a3b8}.ink{color:#e5e7eb}.lbl{color:#e5e7eb}.dim{color:#9ca3af}.pos{color:#34d399}.zero{color:#9ca3af}.neg{color:#f87171}
    }
  </style>
  <defs>
    <marker id="ax1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path class="x" d="M0,0 L8,3 L0,6 Z" fill="currentColor"/></marker>
    <marker id="ay1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path class="y" d="M0,0 L8,3 L0,6 Z" fill="currentColor"/></marker>
  </defs>

  <text class="lbl" x="360" y="28" font-size="15" fill="currentColor" text-anchor="middle" font-weight="600">夹角决定方向相似度</text>

  <!-- Panel 1: acute, O=(110,200), x→(210,200), y at 45°→(181,129) -->
  <g id="panel-acute">
    <line class="axis" x1="60" y1="200" x2="225" y2="200" stroke="currentColor" stroke-width="1"/>
    <line class="x" x1="110" y1="200" x2="210" y2="200" stroke="currentColor" stroke-width="2.6" marker-end="url(#ax1)"/>
    <line class="y" x1="110" y1="200" x2="181" y2="129" stroke="currentColor" stroke-width="2.6" marker-end="url(#ay1)"/>
    <path class="arc" d="M140,200 A30,30 0 0,0 131.21,178.79" fill="none" stroke="currentColor" stroke-width="1.3"/>
    <text class="lbl" x="132" y="194" font-size="13" font-style="italic" fill="currentColor">θ</text>
    <text class="x" x="214" y="196" font-size="14" font-style="italic" fill="currentColor">x</text>
    <text class="y" x="186" y="124" font-size="14" font-style="italic" fill="currentColor">y</text>
    <circle class="ink" cx="110" cy="200" r="2.6" fill="currentColor"/>
    <text class="dim" x="100" y="216" font-size="11" fill="currentColor">O</text>
    <text class="lbl" x="142" y="76" font-size="12" fill="currentColor" text-anchor="middle">锐角 θ &lt; 90°</text>
    <text class="pos" x="142" y="240" font-size="13" fill="currentColor" text-anchor="middle" font-weight="600">cos θ ≈ 0.7 &gt; 0</text>
    <text class="pos" x="142" y="256" font-size="12" fill="currentColor" text-anchor="middle">方向相近</text>
  </g>

  <!-- Panel 2: right angle, O=(360,200), x→(460,200), y up→(360,100) -->
  <g id="panel-right">
    <line class="axis" x1="310" y1="200" x2="475" y2="200" stroke="currentColor" stroke-width="1"/>
    <line class="x" x1="360" y1="200" x2="460" y2="200" stroke="currentColor" stroke-width="2.6" marker-end="url(#ax1)"/>
    <line class="y" x1="360" y1="200" x2="360" y2="100" stroke="currentColor" stroke-width="2.6" marker-end="url(#ay1)"/>
    <path class="arc" d="M390,200 A30,30 0 0,0 360,170" fill="none" stroke="currentColor" stroke-width="1.3"/>
    <text class="lbl" x="382" y="192" font-size="13" font-style="italic" fill="currentColor">θ</text>
    <text class="x" x="464" y="196" font-size="14" font-style="italic" fill="currentColor">x</text>
    <text class="y" x="365" y="96" font-size="14" font-style="italic" fill="currentColor">y</text>
    <circle class="ink" cx="360" cy="200" r="2.6" fill="currentColor"/>
    <text class="dim" x="350" y="216" font-size="11" fill="currentColor">O</text>
    <text class="lbl" x="392" y="76" font-size="12" fill="currentColor" text-anchor="middle">直角 θ = 90°</text>
    <text class="zero" x="392" y="240" font-size="13" fill="currentColor" text-anchor="middle" font-weight="600">cos θ = 0</text>
    <text class="zero" x="392" y="256" font-size="12" fill="currentColor" text-anchor="middle">正交 / 无关</text>
  </g>

  <!-- Panel 3: obtuse, O=(600,200), x→(700,200), y at 135°→(529,129) -->
  <g id="panel-obtuse">
    <line class="axis" x1="490" y1="200" x2="710" y2="200" stroke="currentColor" stroke-width="1"/>
    <line class="x" x1="600" y1="200" x2="700" y2="200" stroke="currentColor" stroke-width="2.6" marker-end="url(#ax1)"/>
    <line class="y" x1="600" y1="200" x2="529" y2="129" stroke="currentColor" stroke-width="2.6" marker-end="url(#ay1)"/>
    <path class="arc" d="M630,200 A30,30 0 0,0 578.79,178.79" fill="none" stroke="currentColor" stroke-width="1.3"/>
    <text class="lbl" x="600" y="178" font-size="13" font-style="italic" fill="currentColor">θ</text>
    <text class="x" x="704" y="196" font-size="14" font-style="italic" fill="currentColor">x</text>
    <text class="y" x="514" y="124" font-size="14" font-style="italic" fill="currentColor">y</text>
    <circle class="ink" cx="600" cy="200" r="2.6" fill="currentColor"/>
    <text class="dim" x="590" y="216" font-size="11" fill="currentColor">O</text>
    <text class="lbl" x="610" y="76" font-size="12" fill="currentColor" text-anchor="middle">钝角 θ &gt; 90°</text>
    <text class="neg" x="610" y="240" font-size="13" fill="currentColor" text-anchor="middle" font-weight="600">cos θ ≈ −0.7 &lt; 0</text>
    <text class="neg" x="610" y="256" font-size="12" fill="currentColor" text-anchor="middle">方向偏离 / 相反</text>
  </g>
</svg>
```

- [ ] **Step 2: Verify XML well-formedness**

Run (from repo root):
```bash
node -e "const fs=require('fs');const s=fs.readFileSync('docs/book/part-1-math/figs/ch01-cosine-similarity-three-cases.svg','utf8');new (require('xml2js')).Parser({explicitArray:false}).parseString(s,(e,r)=>{if(e){console.error('PARSE ERROR',e.message);process.exit(1)}console.log('OK well-formed')});"
```
If `xml2js` is unavailable, fall back to a regex sanity check that all three `<g id="panel-*">` blocks exist and all three `cos θ` result texts are present:
```bash
node -e "const s=require('fs').readFileSync('docs/book/part-1-math/figs/ch01-cosine-similarity-three-cases.svg','utf8');console.log('panels',s.match(/panel-(acute|right|obtuse)/g));console.log('cosines',s.match(/cos θ[^<]*/g));"
```
Expected: panels `['panel-acute','panel-right','panel-obtuse']`; cosines include `≈ 0.7 > 0`, `= 0`, `≈ −0.7 < 0`.

- [ ] **Step 3: Commit**

```bash
git add docs/book/part-1-math/figs/ch01-cosine-similarity-three-cases.svg
git commit -m "Add static 3-panel SVG for cosine similarity angle regimes"
```

---

### Task 2: Rewrite geometry SVG (truthful live cos readout, constant ‖y‖)

**Files:**
- Modify (full overwrite): `docs/book/part-1-math/figs/ch01-cosine-similarity-geometry_anim.svg`

**Interfaces:**
- Produces: corrected animation referenced by §1.3 markdown as `figs/ch01-cosine-similarity-geometry_anim.svg`.

**Design notes** (origin O=(180,210), ‖y‖ pixel length R=140, 13 keyframes sweep 0°→180°→0° over 7s):

| θ° | y tip (x2,y2) | cos display | arc end (r=46) | θ-label mid (r=30, θ/2) |
|---|---|---|---|---|
| 0 | (320,210) | 1.00 | (226,210) | (210,210) |
| 30 | (301.24,140) | 0.87 | (219.84,187) | (208.98,202.24) |
| 60 | (250,88.76) | 0.50 | (203,170.16) | (205.98,195) |
| 90 | (180,70) | 0.00 | (180,164) | (201.21,188.79) |
| 120 | (110,88.76) | −0.50 | (157,170.16) | (195,184.02) |
| 150 | (58.76,140) | −0.87 | (140.16,187) | (187.76,181.02) |
| 180 | (40,210) | −1.00 | (134,210) | (180,180) |

Shared `keyTimes` for all 13-keyframe animations: `0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1`.

- [ ] **Step 1: Overwrite the SVG file**

Replace the entire contents of `docs/book/part-1-math/figs/ch01-cosine-similarity-geometry_anim.svg` with:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 280" width="380" height="280" font-family="Cambria, Georgia, 'Times New Roman', serif">
  <title>余弦相似度几何意义（动画）</title>
  <desc>向量 y 保持定长绕原点旋转，夹角 θ 从 0° 到 180° 再返回；读数面板按当前夹角实时显示对应的 cos θ 值。</desc>
  <style>
    .axis{color:#cbd5e1}.x{color:#2563eb}.y{color:#ea580c}.gray{color:#9ca3af}.arc{color:#6b7280}.ink{color:#1f2937}.lbl{color:#374151}.dim{color:#6b7280}.pos{color:#059669}.neg{color:#dc2626}
    @media (prefers-color-scheme: dark){
      .axis{color:#475569}.x{color:#60a5fa}.y{color:#fb923c}.gray{color:#64748b}.arc{color:#94a3b8}.ink{color:#e5e7eb}.lbl{color:#e5e7eb}.dim{color:#9ca3af}.pos{color:#34d399}.neg{color:#f87171}
    }
  </style>
  <defs>
    <marker id="ahXg" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path class="x" d="M0,0 L8,3 L0,6 Z" fill="currentColor"/></marker>
    <marker id="ahYg" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path class="y" d="M0,0 L8,3 L0,6 Z" fill="currentColor"/></marker>
  </defs>

  <line class="axis" x1="30" y1="210" x2="350" y2="210" stroke="currentColor" stroke-width="1"/>
  <line class="x" x1="180" y1="210" x2="320" y2="210" stroke="currentColor" stroke-width="2.6" marker-end="url(#ahXg)"/>
  <text class="x" x="328" y="205" font-size="16" font-style="italic" fill="currentColor">x</text>

  <line class="y" x1="180" y1="210" x2="320" y2="210" stroke="currentColor" stroke-width="2.6" marker-end="url(#ahYg)">
    <animate attributeName="x2" values="320;301.24;250;180;110;58.76;40;58.76;110;180;250;301.24;320" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/>
    <animate attributeName="y2" values="210;140;88.76;70;88.76;140;210;140;88.76;70;88.76;140;210" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/>
  </line>
  <text class="y" x="258" y="98" font-size="16" font-style="italic" fill="currentColor">y</text>

  <path class="arc" d="M226,210 A46,46 0 0,0 226,210" fill="none" stroke="currentColor" stroke-width="1.3">
    <animate attributeName="d"
      values="M226,210 A46,46 0 0,0 226,210;M226,210 A46,46 0 0,0 219.84,187;M226,210 A46,46 0 0,0 203,170.16;M226,210 A46,46 0 0,0 180,164;M226,210 A46,46 0 0,0 157,170.16;M226,210 A46,46 0 0,0 140.16,187;M226,210 A46,46 0 0,0 134,210;M226,210 A46,46 0 0,0 140.16,187;M226,210 A46,46 0 0,0 157,170.16;M226,210 A46,46 0 0,0 180,164;M226,210 A46,46 0 0,0 203,170.16;M226,210 A46,46 0 0,0 219.84,187;M226,210 A46,46 0 0,0 226,210"
      keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/>
  </path>
  <text class="lbl" x="210" y="210" font-size="14" font-style="italic" fill="currentColor">θ
    <animate attributeName="x" values="210;208.98;205.98;201.21;195;187.76;180;187.76;195;201.21;205.98;208.98;210" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/>
    <animate attributeName="y" values="210;202.24;195;188.79;184.02;181.02;180;181.02;184.02;188.79;195;202.24;210" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/>
  </text>

  <circle class="ink" cx="180" cy="210" r="2.8" fill="currentColor"/>
  <text class="dim" x="166" y="228" font-size="12" fill="currentColor">O</text>

  <g id="readout">
    <rect x="250" y="40" width="118" height="56" rx="8" fill="none" stroke="currentColor" stroke-width="1.5" class="dim" opacity="0.8"/>
    <text class="lbl" x="262" y="60" font-size="13" fill="currentColor" font-weight="600">cos θ =</text>
    <text class="pos" x="262" y="84" font-size="22" font-weight="bold" fill="currentColor">1.00
      <animate attributeName="opacity" values="1;0;0;0;0;0;0;0;0;0;0;0;1" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
    <text class="pos" x="262" y="84" font-size="22" font-weight="bold" fill="currentColor">0.87
      <animate attributeName="opacity" values="0;1;0;0;0;0;0;0;0;0;0;1;0" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
    <text class="pos" x="262" y="84" font-size="22" font-weight="bold" fill="currentColor">0.50
      <animate attributeName="opacity" values="0;0;1;0;0;0;0;0;0;0;1;0;0" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
    <text class="dim" x="262" y="84" font-size="22" font-weight="bold" fill="currentColor">0.00
      <animate attributeName="opacity" values="0;0;0;1;0;0;0;0;0;1;0;0;0" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
    <text class="neg" x="262" y="84" font-size="22" font-weight="bold" fill="currentColor">−0.50
      <animate attributeName="opacity" values="0;0;0;0;1;0;0;0;1;0;0;0;0" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
    <text class="neg" x="262" y="84" font-size="22" font-weight="bold" fill="currentColor">−0.87
      <animate attributeName="opacity" values="0;0;0;0;0;1;0;1;0;0;0;0;0" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
    <text class="neg" x="262" y="84" font-size="22" font-weight="bold" fill="currentColor">−1.00
      <animate attributeName="opacity" values="0;0;0;0;0;0;1;0;0;0;0;0;0" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
  </g>

  <g id="interpretation">
    <text class="pos" x="34" y="255" font-size="12" fill="currentColor">θ ≈ 0° → 方向相同
      <animate attributeName="opacity" values="1;1;0;0;0;0;0;0;0;0;0;1;1" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
    <text class="dim" x="150" y="255" font-size="12" fill="currentColor">θ ≈ 90° → 正交
      <animate attributeName="opacity" values="0;0;1;1;1;0;0;0;1;1;1;0;0" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
    <text class="neg" x="250" y="255" font-size="12" fill="currentColor">θ ≈ 180° → 方向相反
      <animate attributeName="opacity" values="0;0;0;0;0;1;1;1;0;0;0;0;0" keyTimes="0;0.0833;0.1667;0.25;0.3333;0.4167;0.5;0.5833;0.6667;0.75;0.8333;0.9167;1" dur="7s" repeatCount="indefinite"/></text>
  </g>
</svg>
```

- [ ] **Step 2: Verify XML well-formedness and constant ‖y‖**

```bash
node -e "const s=require('fs').readFileSync('docs/book/part-1-math/figs/ch01-cosine-similarity-geometry_anim.svg','utf8');const x2=s.match(/attributeName=\"x2\"[^>]*values=\"([^\"]+)\"/)[1].split(';').map(Number);const y2=s.match(/attributeName=\"y2\"[^>]*values=\"([^\"]+)\"/)[1].split(';').map(Number);const norms=x2.map((x,i)=>Math.hypot(x-180,y2[i]-210));console.log('‖y‖ at each keyframe:',norms.map(n=>+n.toFixed(2)));const ok=norms.every(n=>Math.abs(n-140)<0.5);console.log(ok?'PASS constant ‖y‖=140':'FAIL ‖y‖ not constant');process.exit(ok?0:1);"
```
Expected: prints `PASS constant ‖y‖=140` (all values ≈ 140.00), exit 0.

- [ ] **Step 3: Verify readout ↔ angle correspondence (numeric)**

```bash
node -e "const s=require('fs').readFileSync('docs/book/part-1-math/figs/ch01-cosine-similarity-geometry_anim.svg','utf8');const cosvals=[1,0.87,0.5,0,0.5,0.87,1];const readouts=['1.00','0.87','0.50','0.00','−0.50','−0.87','−1.00'];const x2=s.match(/attributeName=\"x2\"[^>]*values=\"([^\"]+)\"/)[1].split(';').map(Number);const y2=s.match(/attributeName=\"y2\"[^>]*values=\"([^\"]+)\"/)[1].split(';').map(Number);[0,3,6].forEach(i=>{const ang=Math.atan2(-(y2[i]-210),x2[i]-180)*180/Math.PI;console.log('keyframe',i,'angle=',+ang.toFixed(1),'° drawn; expects 0/90/180');});console.log('readouts present:',readouts.every(r=>s.includes(r)));"
```
Expected: keyframe 0 angle ≈ 0°, keyframe 3 (θ=90°) angle ≈ 90°, keyframe 6 (θ=180°) angle ≈ 180°; `readouts present: true`.

- [ ] **Step 4: Commit**

```bash
git add docs/book/part-1-math/figs/ch01-cosine-similarity-geometry_anim.svg
git commit -m "Rewrite cosine geometry SVG with truthful live readout and constant y length"
```

> **Visual sanity note for the human reviewer (open in browser):** at θ=90° the orange **y** must point straight up and the readout must show `0.00`; at θ=180° it points left along the axis and shows `−1.00`. If the θ-arc renders *below* the axis instead of above, flip the arc sweep flag from `0 0,0` to `0 0,1` in every arc path value (one-digit change). The numeric checks above do not depend on arc direction.

---

### Task 3: Rewrite calc SVG (non-parallel example x=(3,4), y=(−3,4))

**Files:**
- Modify (full overwrite): `docs/book/part-1-math/figs/ch01-cosine-similarity-calc_anim.svg`

**Interfaces:**
- Produces: corrected animation referenced by §1.3 markdown as `figs/ch01-cosine-similarity-calc_anim.svg`.

**Arithmetic to encode** (every displayed number comes from these; all integer except the final 0.28):
- y column: `−3`, `4`
- Phase 1: `3 × (−3) = −9`; `4 × 4 = 16`; `−9 + 16 = 7`
- Phase 2: `‖x‖ = √(9 + 16) = 5`; `‖y‖ = √(9 + 16) = 5`
- Phase 3: `cos θ = 7 / (5 × 5) = 0.28`
- Conclusion: `x 与 y 方向相近但不一致（θ ≈ 74°）`

- [ ] **Step 1: Overwrite the SVG file**

Replace the entire contents of `docs/book/part-1-math/figs/ch01-cosine-similarity-calc_anim.svg` with:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 320" width="100%" height="100%" font-family="system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
  <title>余弦相似度计算过程（动画）</title>
  <desc>逐步展示 cos θ = (x·y) / (‖x‖‖y‖) 的计算过程，示例 x=(3,4), y=(−3,4)。</desc>
  <style>
    .x{color:#2563eb}.y{color:#ea580c}.calc{color:#059669}.gray{color:#9ca3af}.lbl{color:#374151}.dim{color:#6b7280}.bracket{color:#64748b}.sum{color:#7c3aed}.op{color:#374151}.norm{color:#dc2626}
    @media (prefers-color-scheme: dark){
      .x{color:#60a5fa}.y{color:#fb923c}.calc{color:#34d399}.gray{color:#64748b}.lbl{color:#e5e7eb}.dim{color:#9ca3af}.bracket{color:#94a3b8}.sum{color:#a78bfa}.op{color:#e5e7eb}.norm{color:#f87171}
    }
  </style>

  <defs>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" stroke-width="0.3" class="gray" opacity="0.3"/>
    </pattern>
  </defs>

  <rect width="100%" height="100%" fill="url(#grid)" class="gray"/>

  <line class="gray" x1="140" y1="30" x2="140" y2="290" stroke="currentColor" stroke-width="0.5" stroke-dasharray="4 4" opacity="0.4"/>
  <line class="gray" x1="260" y1="30" x2="260" y2="290" stroke="currentColor" stroke-width="0.5" stroke-dasharray="4 4" opacity="0.4"/>

  <text class="lbl" x="70" y="35" font-size="14" font-weight="600" fill="currentColor" text-anchor="middle">x (行向量)</text>
  <text class="lbl" x="200" y="35" font-size="14" font-weight="600" fill="currentColor" text-anchor="middle">y (列向量)</text>
  <text class="lbl" x="410" y="35" font-size="14" font-weight="600" fill="currentColor" text-anchor="middle">计算过程</text>

  <!-- 左列：行向量 xᵀ = [3, 4] -->
  <g id="row-vector-x">
    <path d="M 27 75 L 21 75 L 21 135 L 27 135" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>
    <path d="M 113 75 L 119 75 L 119 135 L 113 135" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>

    <circle class="x" cx="45" cy="105" r="14" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0">
      <animate attributeName="opacity" values="0;1;0;0;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
      <animate attributeName="fill-opacity" values="0;0.2;0;0;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </circle>
    <text class="x" x="45" y="111" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">3</text>

    <circle class="x" cx="95" cy="105" r="14" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0">
      <animate attributeName="opacity" values="0;0;0;1;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
      <animate attributeName="fill-opacity" values="0;0;0;0.2;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </circle>
    <text class="x" x="95" y="111" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">4</text>

    <text class="x" x="70" y="165" font-size="14" font-style="italic" fill="currentColor" text-anchor="middle">x<tspan dy="-5" font-size="10" font-style="normal">T</tspan></text>
  </g>

  <!-- 中列：列向量 y = [-3, 4]ᵀ -->
  <g id="col-vector-y">
    <path d="M 183 52 L 177 52 L 177 158 L 183 158" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>
    <path d="M 217 52 L 223 52 L 223 158 L 217 158" fill="none" stroke="currentColor" stroke-width="2" class="bracket"/>

    <circle class="y" cx="200" cy="70" r="14" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0">
      <animate attributeName="opacity" values="0;1;0;0;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
      <animate attributeName="fill-opacity" values="0;0.2;0;0;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </circle>
    <text class="y" x="200" y="76" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">−3</text>

    <circle class="y" cx="200" cy="140" r="14" fill="currentColor" fill-opacity="0" stroke="currentColor" stroke-width="2.5" opacity="0">
      <animate attributeName="opacity" values="0;0;0;1;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
      <animate attributeName="fill-opacity" values="0;0;0;0.2;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </circle>
    <text class="y" x="200" y="146" font-size="18" fill="currentColor" text-anchor="middle" font-weight="500">4</text>

    <text class="y" x="200" y="185" font-size="14" font-style="italic" fill="currentColor" text-anchor="middle">y</text>
  </g>

  <!-- 右列：计算过程 -->
  <g id="calculation">
    <text class="dim" x="280" y="55" font-size="12" fill="currentColor" font-weight="600" opacity="0">
      点积 x·y
      <animate attributeName="opacity" values="0;1;1;1;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </text>

    <text x="290" y="80" font-size="18" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
      <tspan class="x">3</tspan><tspan class="op"> × (</tspan><tspan class="y">−3</tspan><tspan class="op">) = </tspan><tspan class="calc" font-weight="bold">−9</tspan>
      <animate attributeName="opacity" values="0;1;1;1;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </text>

    <text x="290" y="110" font-size="18" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
      <tspan class="x">4</tspan><tspan class="op"> × </tspan><tspan class="y">4</tspan><tspan class="op"> = </tspan><tspan class="calc" font-weight="bold">16</tspan>
      <animate attributeName="opacity" values="0;0;0;1;0;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </text>

    <line x1="290" y1="120" x2="420" y2="120" stroke="currentColor" stroke-width="1.5" class="dim" opacity="0">
      <animate attributeName="opacity" values="0;0;0;0;1;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </line>

    <text x="290" y="140" font-size="18" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
      <tspan class="calc">−9</tspan><tspan class="op"> + </tspan><tspan class="calc">16</tspan><tspan class="op"> = </tspan><tspan class="sum" font-weight="bold">7</tspan>
      <animate attributeName="opacity" values="0;0;0;0;1;0;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </text>

    <text class="dim" x="280" y="170" font-size="12" fill="currentColor" font-weight="600" opacity="0">
      范数
      <animate attributeName="opacity" values="0;0;0;0;0;1;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </text>

    <text x="290" y="195" font-size="16" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
      <tspan class="op">‖</tspan><tspan class="x">x</tspan><tspan class="op">‖ = √(9 + 16) = </tspan><tspan class="norm" font-weight="bold">5</tspan>
      <animate attributeName="opacity" values="0;0;0;0;0;1;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </text>

    <text x="290" y="220" font-size="16" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
      <tspan class="op">‖</tspan><tspan class="y">y</tspan><tspan class="op">‖ = √(9 + 16) = </tspan><tspan class="norm" font-weight="bold">5</tspan>
      <animate attributeName="opacity" values="0;0;0;0;0;1;0;0;0;0;0" dur="10s" repeatCount="indefinite"/>
    </text>

    <text class="dim" x="280" y="248" font-size="12" fill="currentColor" font-weight="600" opacity="0">
      余弦相似度
      <animate attributeName="opacity" values="0;0;0;0;0;0;1;1;1;0;0" dur="10s" repeatCount="indefinite"/>
    </text>

    <text x="290" y="273" font-size="16" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
      <tspan class="op">cos θ = 7 / (5 × 5) = </tspan><tspan class="sum" font-weight="bold">0.28</tspan>
      <animate attributeName="opacity" values="0;0;0;0;0;0;0;1;1;0;0" dur="10s" repeatCount="indefinite"/>
    </text>

    <text x="290" y="305" font-size="15" font-family="'Times New Roman', Cambria, serif" fill="currentColor" opacity="0">
      <tspan class="op">∴ </tspan><tspan class="x">x</tspan><tspan class="op"> 与 </tspan><tspan class="y">y</tspan><tspan class="op"> 方向相近但不一致（θ ≈ 74°）</tspan>
      <animate attributeName="opacity" values="0;0;0;0;0;0;0;0;0;1;0" dur="10s" repeatCount="indefinite"/>
    </text>
  </g>

  <text class="dim" x="280" y="315" font-size="12" fill="currentColor" text-anchor="start" font-style="italic">
    余弦相似度 = 点积 / (‖x‖‖y‖) ∈ [-1, 1]
  </text>
</svg>
```

- [ ] **Step 2: Verify arithmetic encoded correctly**

The numbers are split across colored `<tspan>` elements, so the check must strip tags first (matching true SVG rendering, where adjacent tspans are flush):

```bash
node -e "const s=require('fs').readFileSync('docs/book/part-1-math/figs/ch01-cosine-similarity-calc_anim.svg','utf8');const t=s.replace(/<[^>]+>/g,'');const must=['−3','3 × (−3) = −9','4 × 4 = 16','−9 + 16 = 7','√(9 + 16) = 5','7 / (5 × 5) = 0.28','θ ≈ 74°'];const miss=must.filter(m=>!t.includes(m));console.log(miss.length?('MISSING: '+miss.join(' | ')):'PASS all arithmetic present (tag-stripped)');process.exit(miss.length?1:0);"
```
Expected: `PASS all arithmetic present (tag-stripped)`, exit 0.

- [ ] **Step 3: Cross-check the math itself**

```bash
node -e "const dot=3*-3+4*4, nx=Math.hypot(3,4), ny=Math.hypot(-3,4), cos=dot/(nx*ny); console.log('dot',dot,'nx',nx,'ny',ny,'cos',+cos.toFixed(4),'angle°',+(Math.acos(cos)*180/Math.PI).toFixed(1));"
```
Expected: `dot 7 nx 5 ny 5 cos 0.28 angle° 73.7` — matches the SVG (`7`, `5`, `5`, `0.28`, `θ ≈ 74°`).

- [ ] **Step 4: Commit**

```bash
git add docs/book/part-1-math/figs/ch01-cosine-similarity-calc_anim.svg
git commit -m "Rewrite cosine calc SVG with non-parallel example x=(3,4) y=(-3,4)"
```

---

### Task 4: Rewrite §1.3 markdown (intuition-first order, reference all three SVGs)

**Files:**
- Modify: `docs/book/part-1-math/ch01-linear-algebra-vectors.md` — replace the entire `### 向量夹角与余弦相似度` subsection (current lines **184–234**) with the new content below. Do NOT touch line 183 (the preceding `### 范数（norm）` content) or line 235 onward (the `## 1.4 推导与几何` heading).

**Interfaces:**
- Consumes: the three SVGs produced by Tasks 1–3, referenced by relative path `figs/...`.
- Produces: the rewritten §1.3 subsection of the chapter.

- [ ] **Step 1: Locate the exact region to replace**

Run:
```bash
node -e "const fs=require('fs');const lines=fs.readFileSync('docs/book/part-1-math/ch01-linear-algebra-vectors.md','utf8').split('\n');const start=lines.findIndex(l=>l.startsWith('### 向量夹角与余弦相似度'));const end=lines.findIndex((l,i)=>i>start&&l.startsWith('## 1.4'));console.log('subsection header at line',start+1,' next section (## 1.4) at line',end+1);"
```
Expected: `subsection header at line 184` and `next section (## 1.4) at line 236` (i.e. replace lines 184 through 235 inclusive — the blank line before `## 1.4`).

- [ ] **Step 2: Replace the subsection with the new prose**

Replace lines 184–235 (from `### 向量夹角与余弦相似度` through the blank line just before `## 1.4 推导与几何`) with exactly:

```markdown
### 向量夹角与余弦相似度

两个向量之间的「方向接近程度」用什么度量？几何直觉告诉我们：**看夹角**。夹角越小，方向越一致；夹角越大，方向越偏离。按夹角大小可以分成三种情况：

- **夹角小于 90°**（锐角）：两向量方向「朝同一边」，认为**相近 / 相似**；
- **夹角等于 90°**（直角）：两向量**正交**，方向毫无关联；
- **夹角大于 90°**（钝角）：两向量方向「背道而驰」，认为**偏离 / 相反**。

![余弦相似度三种情形](figs/ch01-cosine-similarity-three-cases.svg)

但直接拿夹角 $\theta$ 当相似度有个问题：它从 $0^\circ$ 增长到 $180^\circ$ ，「越大越不相似」这个直觉和「相似度越大越接近」的习惯正好相反。更自然的做法是用 $\cos\theta$ ——它把夹角映射到 $[-1, 1]$ 区间，而且方向一致时为 $1$ 、正交时为 $0$ 、相反时为 $-1$ ，完美符合「相似度」的直觉。这三个标志值正好对应上面三种情形：

- $\cos\theta = 1$ ：夹角 $0^\circ$ ，两向量**方向完全相同**（同向平行）；
- $\cos\theta = 0$ ：夹角 $90^\circ$ ，两向量**正交**（毫不相关）；
- $\cos\theta = -1$ ：夹角 $180^\circ$ ，两向量**方向完全相反**（反向平行）。

现在的关键问题是：我们手头只有向量 $\mathbf{x}, \mathbf{y}$ 的坐标，怎么算出 $\cos\theta$ ？

**推导借助余弦定理**。设两个向量 $\mathbf{x}, \mathbf{y}$ 的夹角为 $\theta$ ，它们差向量的长度满足：

$$
\Vert\mathbf{x} - \mathbf{y}\Vert^2 = \Vert\mathbf{x}\Vert^2 + \Vert\mathbf{y}\Vert^2 - 2 \Vert\mathbf{x}\Vert \Vert\mathbf{y}\Vert\cos\theta
$$

另一方面，直接展开左边（逐元素计算距离平方）：

$$
\Vert\mathbf{x} - \mathbf{y}\Vert^2 = \sum_i (x_i - y_i)^2 = \Vert\mathbf{x}\Vert^2 + \Vert\mathbf{y}\Vert^2 - 2 \sum_i x_i y_i = \Vert\mathbf{x}\Vert^2 + \Vert\mathbf{y}\Vert^2 - 2(\mathbf{x}\cdot\mathbf{y})
$$

> 比较这两个展开式：右边前两项 $\Vert\mathbf{x}\Vert^2 + \Vert\mathbf{y}\Vert^2$ 完全相同，可以消掉；那么含 $\cos\theta$ 的那一项就必须等于含 $\sum_i x_i y_i$ 的那一项，即 $-2\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert\cos\theta = -2(\mathbf{x}\cdot\mathbf{y})$ 。

于是立刻得到点积与夹角的关系：

$$
\mathbf{x} \cdot \mathbf{y} = \Vert\mathbf{x}\Vert \Vert\mathbf{y}\Vert\cos\theta
$$

两边除以 $\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert$ ，就得到 **余弦相似度** 的定义：

$$
\boxed{ \cos\theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\Vert\mathbf{x}\Vert_2 \Vert\mathbf{y}\Vert_2} }
$$

因为 $|\cos\theta| \leq 1$ （余弦的基本性质），这个值天然落在 $[-1, 1]$ 区间——这正是它适合当「相似度」的原因。

**为什么不用点积直接当相似度？** 因为点积同时依赖「长度」和「方向」：两个长向量即使方向偏离，点积也可能很大；两个短向量即使方向一致，点积也可能很小。余弦相似度通过除以范数，把长度的影响消掉，只保留方向信息——这正是语义检索想要的：比较「国王」和「女王」的语义距离时，不应该因为某个词的嵌入向量更长而被误导。

下面这张动画展示了余弦相似度的**计算过程**（以 2 维向量为例）：

![余弦相似度计算过程](figs/ch01-cosine-similarity-calc_anim.svg)

可以看到，余弦相似度的计算分为三步：先算点积 $\mathbf{x}\cdot\mathbf{y}$ 、再算两个向量的范数 $\Vert\mathbf{x}\Vert$ 和 $\Vert\mathbf{y}\Vert$ 、最后相除。示例中 $\mathbf{x}=(3,4)^\top$ 与 $\mathbf{y}=(-3,4)^\top$ 夹角约 $74^\circ$ ，得到 $\cos\theta=0.28$ ——方向相近但不一致。

下面这张动画展示了夹角 $\theta$ 与 $\cos\theta$ 的**几何关系**：

![余弦相似度几何意义](figs/ch01-cosine-similarity-geometry_anim.svg)

可以看到，随着向量 $\mathbf{y}$ 绕原点旋转，夹角 $\theta$ 从 $0^\circ$ 变化到 $180^\circ$ ，余弦相似度从 $1$ 递减到 $-1$ 。这直观展示了「余弦相似度只看方向」的特性——只要夹角相同，相似度就相同，与向量长度无关。

```

- [ ] **Step 3: Verify the new ordering (intuition before derivation before SVGs)**

```bash
node -e "const s=require('fs').readFileSync('docs/book/part-1-math/ch01-linear-algebra-vectors.md','utf8');const i1=s.indexOf('按夹角大小可以分成三种情况');const i2=s.indexOf('**推导借助余弦定理**');const i3=s.indexOf('余弦相似度的**计算过程**');const i4=s.indexOf('**几何关系**');const ok=i1>-1&&i1<i2&&i2<i3&&i3<i4;console.log({threeRegimes:i1,derivation:i2,calcSVG:i3,geomSVG:i4});console.log(ok?'PASS order correct':'FAIL order wrong');process.exit(ok?0:1);"
```
Expected: all indices present and strictly increasing; `PASS order correct`.

- [ ] **Step 4: Verify all three SVG references resolve to existing files**

```bash
node -e "const fs=require('fs'),p=require('path'),d='docs/book/part-1-math';const s=fs.readFileSync(d+'/ch01-linear-algebra-vectors.md','utf8');const refs=[...s.matchAll(/!\[[^\]]*\]\((figs\/[^)]+)\)/g)].map(m=>m[1]);const uniq=[...new Set(refs)];console.log('referenced figs:',uniq);const missing=uniq.filter(r=>!fs.existsSync(p.join(d,r)));console.log(missing.length?('MISSING: '+missing.join(', ')):'PASS all referenced figs exist');process.exit(missing.length?1:0);"
```
Expected: `referenced figs` includes `figs/ch01-cosine-similarity-three-cases.svg`, `...-calc_anim.svg`, `...-geometry_anim.svg`; `PASS all referenced figs exist`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add docs/book/part-1-math/ch01-linear-algebra-vectors.md
git commit -m "Rewrite cosine similarity subsection: intuition-first order with three-regime lead"
```

---

## Final Human Verification (after all 4 tasks)

Open each of these in a browser and confirm:
1. `figs/ch01-cosine-similarity-three-cases.svg` — three panels render, cos signs colored (green/gray/red), labels legible in both light and dark mode.
2. `figs/ch01-cosine-similarity-geometry_anim.svg` — **y** stays the same length throughout; at θ=90° readout reads `0.00` and **y** is vertical; at θ=180° readout reads `−1.00` and **y** points left; the three interpretation labels light up at the right phases; θ-arc is above the axis (if not, flip arc sweep flag per Task 2 note).
3. `figs/ch01-cosine-similarity-calc_anim.svg` — shows `−3, 4` in the y column, steps through `−9, 16, 7, 5, 5, 0.28`, ends with "方向相近但不一致（θ ≈ 74°）".
4. The chapter's §1.3 reads: three regimes → why cos → derivation → range → why-not-dot-product → calc SVG → geometry SVG.

## Self-Review (completed by plan author)

- **Spec coverage:** Spec Part A (narrative order) → Task 4. Part B (3-panel SVG) → Task 1. Part C (geometry fix: truthful readout, constant ‖y‖, fixed label timings, 7 poses) → Task 2. Part D (calc fix, non-parallel example) → Task 3. All sections covered.
- **Placeholders:** none — every step contains complete file content or an exact command with expected output.
- **Consistency:** Calc example numbers (`7, 5, 5, 0.28, 74°`) are identical across Task 3 SVG, Task 3 cross-check, and Task 4 prose. Geometry keyframe coordinates in Task 2 match the node-computed values. SVG class names and palette match the Global Constraints.
