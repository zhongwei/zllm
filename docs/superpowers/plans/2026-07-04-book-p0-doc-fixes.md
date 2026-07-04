# P0 文档修正 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `docs/book/` 中 16 处 P0 级文档错误（9 个数学/事实错误 + 7 个自相矛盾），使文档与源码、与自身一致。

**Architecture:** 纯文档定点编辑，13 个独立任务，每任务改 1 个文件（C6 跨 3 个同构文件）。不改任何 `.py` 源码、README、SUMMARY；不补 P1/P2。

**Tech Stack:** Markdown（含 LaTeX 数学块、mermaid、表格）。

## Global Constraints

- **只动 `docs/book/` 下文件**，源码/README/SUMMARY 零改动。
- **范围严格 = B1–B9 + C1–C7**。不顺手补 P1（如 RoPE attn_factor、padding 故事、术语表补词、Appendix D 补 Scaling Law）和 P2（行号微调、拼写、死代码注释）。
- **每处编辑用 `edit` 工具的精确 `oldString`/`newString`**，不做大段重写。
- **中文为主、保留英文术语**，与原书风格一致。
- **B5 文件订正**：审计/spec 误写为 `ch06-optimization.md`，实际 GradScaler 归因段在 **`ch08-autograd.md:297`**。本计划按 ch08 执行。
- **提交策略**：计划含每任务一次提交；执行时若用户未要求提交，则改为"全部完成后统一 review、按用户指示提交"。

---

## File Structure

13 个任务对应 14 个文件（C6 跨 ch37/ch38/ch40 三个文件）：

| 任务 | 文件 | 编号 |
|------|------|------|
| 1 | `part-4-architecture/ch20-rmsnorm.md` | B1 |
| 2 | `part-2-dl-transformer/ch09-neural-network-basics.md` | B3 |
| 3 | `part-1-math/ch08-autograd.md` | B5 |
| 4 | `part-5-training/ch29-training-infrastructure.md` | B9 |
| 5 | `part-5-training/ch32-pretraining-practice.md` | B8 |
| 6 | `part-3-tokenizer/ch16-project-setup.md` | C7 |
| 7 | `part-3-tokenizer/ch19-production-tokenizer.md` | C1, C2 |
| 8 | `part-7-deployment/ch41-decoding-algorithm.md` | B2, B4 |
| 9 | `part-6-alignment/ch35-rlhf-framework.md` | C3, C4 |
| 10 | `part-6-alignment/ch40-agent-rl-tools.md` | C5 |
| 11 | `part-6-alignment/ch37-ppo-gae-critic.md`, `ch38-grpo-cispo.md`, `ch40-agent-rl-tools.md` | C6 |
| 12 | `appendices/appendix-d-references.md` | B6 |
| 13 | `appendices/appendix-b-hyperparameters.md` | B7 |

---

### Task 1: B1 — RMSNorm 公式（ch20，3 处）

**Files:**
- Modify: `docs/book/part-4-architecture/ch20-rmsnorm.md`（行 46、101、159）

**依据**：`zllm/model/norms.py:18` 用 `x * torch.rsqrt(x.pow(2).mean(-1) + self.eps)`，即 `x/√(mean(x²)+ε)`。ε 在根号内，不是加在 RMS 上。`ch20:65` 自己也写对了——本任务让全章一致。

- [ ] **Step 1: 修正 boxed 公式（行 46）**

`oldString`:
```
\boxed{\;\text{RMSNorm}(x) \;=\; \frac{x}{\text{RMS}(x) + \epsilon}\odot\gamma, \qquad \text{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2}\;}
```
`newString`:
```
\boxed{\;\text{RMSNorm}(x) \;=\; \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}\odot\gamma, \qquad \text{mean}(x^2)=\frac{1}{d}\sum_i x_i^2\;}
```

- [ ] **Step 2: 修正 norm(x) 说明（行 101）**

`oldString`:
```
- **`norm(x)`**（`:17-18`）：只做归一化（**不乘 weight**）。对应公式里的 $\frac{x}{\text{RMS}(x)+\epsilon}=x\cdot\text{rsqrt}(\text{mean}(x^2)+\epsilon)$。`mean(-1, keepdim=True)` 在最后一维（隐藏维度）上求均方，保持维度以便广播。
```
`newString`:
```
- **`norm(x)`**（`:17-18`）：只做归一化（**不乘 weight**）。对应公式里的 $\frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}=x\cdot\text{rsqrt}(\text{mean}(x^2)+\epsilon)$。`mean(-1, keepdim=True)` 在最后一维（隐藏维度）上求均方，保持维度以便广播。
```

- [ ] **Step 3: 修正小结要点（行 159）**

`oldString`:
```
2. **RMSNorm** = $\frac{x}{\text{RMS}(x)+\epsilon}\odot\gamma$，比 LayerNorm 省「减均值」和偏置，更快、效果相当。
```
`newString`:
```
2. **RMSNorm** = $\frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\odot\gamma$，比 LayerNorm 省「减均值」和偏置，更快、效果相当。
```

- [ ] **Step 4: 验证无残留**

Run: `rg "RMS\(x\)\+\\epsilon|RMS\(x\) \+ \\epsilon" docs/book/part-4-architecture/ch20-rmsnorm.md`
Expected: 无输出（旧错误公式已全部清除）。

- [ ] **Step 5: 提交**

```bash
git add docs/book/part-4-architecture/ch20-rmsnorm.md
git commit -m "docs(ch20): 修正 RMSNorm 公式（ε 在根号内，与 norms.py 一致）"
```

---

### Task 2: B3 — intermediate_size 3072→2432（ch09，2 处）

**Files:**
- Modify: `docs/book/part-2-dl-transformer/ch09-neural-network-basics.md`（行 191、294）

**依据**：`config.py:48-52` 与 `ch16:261`、`appendix-b:23` 都是 `ceil(768π/64)·64 = 2432`。Ch09 写成 3072（=4d，原始 Transformer 约定）与三者冲突。

- [ ] **Step 1: 修正行 191（明确区分通用 4d 与 zllm π-缩放）**

`oldString`:
```
MLP 的「层间宽度」 $d_1,d_2,\dots,d_L$ 是超参数。一个常见配置是「先扩张再收缩」：比如 $d_{\text{in}}\to 4d\to d_{\text{out}}$ ——这正是 Transformer FFN（Ch 13、Ch 23）的形状，zllm 里 `intermediate_size=3072` 就是把 $d=768$ 扩张 4 倍。
```
`newString`:
```
MLP 的「层间宽度」 $d_1,d_2,\dots,d_L$ 是超参数。一个常见配置是「先扩张再收缩」：比如 $d_{\text{in}}\to 4d\to d_{\text{out}}$ ——这正是 Transformer FFN（Ch 13、Ch 23）的形状。注意「$4d$」是原始 Transformer 的约定，**zllm 实际采用 π-缩放**：`intermediate_size=2432`（$\lceil 768\pi/64\rceil\times 64$，约 3.17 倍，对齐 64 倍数以提升 Tensor Core 利用率，详见 Ch 16/23）。
```

- [ ] **Step 2: 修正行 294（把 $4d$ 改为 $d_{\text{ff}}$，避免与上文冲突）**

`oldString`:
```
其中 $\mathrm{up}(\mathbf{x})=W_{\text{up}}\mathbf{x}$ 、 $\mathrm{gate}(\mathbf{x})=W_{\text{gate}}\mathbf{x}$ 是两个并行的线性扩张（把 $d$ 维扩到 $4d$ 维）， $\mathrm{SiLU}(\cdot)$ 是本章定义的 $x\sigma(x)$ ， $\odot$ 是逐元素乘， $\mathrm{down}(\cdot)=W_{\text{down}}(\cdot)$ 是线性收缩（把 $4d$ 维压回 $d$ 维）。
```
`newString`:
```
其中 $\mathrm{up}(\mathbf{x})=W_{\text{up}}\mathbf{x}$ 、 $\mathrm{gate}(\mathbf{x})=W_{\text{gate}}\mathbf{x}$ 是两个并行的线性扩张（把 $d$ 维扩到 $d_{\text{ff}}$ 维，zllm 用 π-缩放取 2432，见上文）， $\mathrm{SiLU}(\cdot)$ 是本章定义的 $x\sigma(x)$ ， $\odot$ 是逐元素乘， $\mathrm{down}(\cdot)=W_{\text{down}}(\cdot)$ 是线性收缩（把 $d_{\text{ff}}$ 维压回 $d$ 维）。
```

- [ ] **Step 3: 验证**

Run: `rg "intermediate_size=3072|扩张 4 倍" docs/book/part-2-dl-transformer/ch09-neural-network-basics.md`
Expected: 无输出。

- [ ] **Step 4: 提交**

```bash
git add docs/book/part-2-dl-transformer/ch09-neural-network-basics.md
git commit -m "docs(ch09): intermediate_size 修正为 2432（π-缩放），与 config.py/附录B 一致"
```

---

### Task 3: B5 — GradScaler 归因 bf16→fp16（ch08，1 处）

**Files:**
- Modify: `docs/book/part-1-math/ch08-autograd.md`（行 297）

**依据**：fp16 有 5 位指数（下溢真实风险）；bf16 有 8 位指数（同 fp32），几乎不下溢。`amp.py` 保留 `GradScalerManager` 调用以兼容 fp16 路径，但 bf16 下形同虚设。原文把下溢归因于 bf16，技术上不准。

- [ ] **Step 1: 修正行 297**

`oldString`:
```
- **`GradScaler`**：`bfloat16` 的梯度可能下溢（太小而变成 0），所以先把 loss 放大一个因子再 `backward()`，让梯度落在可表示范围内，更新前再缩回来。
```
`newString`:
```
- **`GradScaler`**：梯度下溢是 **fp16**（5 位指数）的真实风险——梯度太小而变成 0，所以先把 loss 放大一个因子再 `backward()`，让梯度落在可表示范围内，更新前再缩回来。**bf16 有 8 位指数（与 fp32 相同），几乎不下溢**，故 bf16 下 GradScaler 形同虚设；zllm 仍保留该调用以兼容 fp16 路径。
```

- [ ] **Step 2: 验证**

Run: `rg "bfloat16.*的梯度可能下溢" docs/book/part-1-math/ch08-autograd.md`
Expected: 无输出。

- [ ] **Step 3: 提交**

```bash
git add docs/book/part-1-math/ch08-autograd.md
git commit -m "docs(ch08): GradScaler 下溢归因修正（fp16 而非 bf16）"
```

---

### Task 4: B9 — cosine lr Mermaid 中点 η₀/2→0.55·η₀（ch29，1 处）

**Files:**
- Modify: `docs/book/part-5-training/ch29-training-infrastructure.md`（行 57）

**依据**：`get_lr`（`utils.py:46-47`）在 s=T/2 时返回 `base·(0.1+0.45·(1+cos(π/2))) = base·0.55`。`ch29:52` 表格已正确写 `η₀×0.55`，仅 Mermaid 节点标错 `η₀/2`。

- [ ] **Step 1: 修正 Mermaid 节点（行 57）**

`oldString`:
```
    A["η₀ (起点)<br/>大步快学"] --> B["η₀/2 (中点)<br/>稳步下降"]
```
`newString`:
```
    A["η₀ (起点)<br/>大步快学"] --> B["0.55·η₀ (中点)<br/>稳步下降"]
```

- [ ] **Step 2: 验证**

Run: `rg "η₀/2 \(中点\)" docs/book/part-5-training/ch29-training-infrastructure.md`
Expected: 无输出。

- [ ] **Step 3: 提交**

```bash
git add docs/book/part-5-training/ch29-training-infrastructure.md
git commit -m "docs(ch29): cosine lr Mermaid 中点修正为 0.55·η₀，与表格/get_lr 一致"
```

---

### Task 5: B8 — 默认 batch_size 32→64（ch32，1 处）

**Files:**
- Modify: `docs/book/part-5-training/ch32-pretraining-practice.md`（行 148）

**依据**：`pretrain.py:20` `batch_size: int = 64`；`ch31:82` 与 `ch32:150` 都已写 64。仅 `ch32:148` 残留旧值 32（等效 128）。

- [ ] **Step 1: 修正行 148**

`oldString`:
```
zllm 默认 batch=32、accum=4（等效 128），在 24GB 卡上 `max_seq_len=340` 可以跑。显存不够时优先调 batch/accum，其次 seq_len。
```
`newString`:
```
zllm 默认 batch=64、accum=4（等效 256），在 24GB 卡上 `max_seq_len=340` 可以跑。显存不够时优先调 batch/accum，其次 seq_len。
```

- [ ] **Step 2: 验证**

Run: `rg "默认 batch=32|等效 128" docs/book/part-5-training/ch32-pretraining-practice.md`
Expected: 无输出（行 148 已改；行 144 表格的 `8×16=128` 是独立的调参示例，不在本任务范围）。

- [ ] **Step 3: 提交**

```bash
git add docs/book/part-5-training/ch32-pretraining-practice.md
git commit -m "docs(ch32): 默认 batch_size 修正为 64（等效 256），与 pretrain.py 一致"
```

---

### Task 6: C7 — M1 测试数 4→8（ch16，3 处）

**Files:**
- Modify: `docs/book/part-3-tokenizer/ch16-project-setup.md`（行 154、239、253）

**依据**：`pytest tests/m01_foundations/ --collect-only` 实测 8 个测试（test_002 两个 + test_003 六个）。`ch16:250` 代码块打印 `8 passed`，`ch16:267` 也写 8，仅 154/239 残留旧值 4，253 行是试图圆场的括注。

- [ ] **Step 1: 修正行 154**

`oldString`:
```
M1 里程碑只有两个测试文件、4 个测试，但它们是全书的根。先看 `tests/conftest.py` 里三个**所有后续测试都会用到的共享 fixture**。
```
`newString`:
```
M1 里程碑只有两个测试文件、共 8 个测试，但它们是全书的根。先看 `tests/conftest.py` 里三个**所有后续测试都会用到的共享 fixture**。
```

- [ ] **Step 2: 修正行 239**

`oldString`:
```
预期输出（4 个测试全绿）：
```
`newString`:
```
预期输出（8 个测试全绿）：
```

- [ ] **Step 3: 精简行 253 的圆场括注**

`oldString`:
```
> 注意：`pytest tests/m01_foundations/ -v` 实际打印 8 行（含 fixture 校验测试），但它们都属于 M1 的「地基」。如果你想确认整本书的代码都还活着，可以直接跑 `pytest`（全 428 个测试）。
```
`newString`:
```
> 这 8 个测试都属于 M1 的「地基」。想确认整本书的代码都活着，可直接跑 `pytest`（全 428 个测试）。
```

- [ ] **Step 4: 验证**

Run: `rg "4 个测试" docs/book/part-3-tokenizer/ch16-project-setup.md`
Expected: 无输出。

- [ ] **Step 5: 提交**

```bash
git add docs/book/part-3-tokenizer/ch16-project-setup.md
git commit -m "docs(ch16): M1 测试数修正为 8（与实际 collect 数一致）"
```

---

### Task 7: C1+C2 — 特殊 token 分类数 + Jinja2 模板能力（ch19，2 处）

**Files:**
- Modify: `docs/book/part-3-tokenizer/ch19-production-tokenizer.md`（行 111、146）

**依据**：
- C1：下文列 5 个 bullet（对话边界/多模态预留/工具调用/思考链/buffer），`ch19:23` 学习目标与 `special_tokens.py:3-9` 都列 5 类，仅行 111 写"四类"。
- C2：`render_messages`（`chat_template.py:25-72`）支持 tools 注入（`:46-55`）；Jinja2 `CHAT_TEMPLATE`（`:76-94`）无任何 tools 处理。原文"逻辑相同"会误导 Agent RL 使用者。

- [ ] **Step 1: C1 — 分类数 4→5（行 111）**

`oldString`:
```
zllm 的特殊 token（`special_tokens.py:12-34`）分四类，对齐 Qwen3 / minimind-3：
```
`newString`:
```
zllm 的特殊 token（`special_tokens.py:12-34`）分五类，对齐 Qwen3 / minimind-3：
```

- [ ] **Step 2: C2 — Jinja2 模板能力说明（行 146）**

`oldString`:
```
另外还有一个 Jinja2 版本 `CHAT_TEMPLATE`（`:76-94`），逻辑相同，供 `PreTrainedTokenizerFast.chat_template` 用，兼容 Transformers 生态。
```
`newString`:
```
另外还有一个 Jinja2 版本 `CHAT_TEMPLATE`（`:76-94`），**逻辑相近但不支持 `tools` 注入**（只渲染 messages + `open_thinking` + `add_generation_prompt`）；Agent RL（Ch 40）必须用 `render_messages` 才能把工具描述注入 system 块。它供 `PreTrainedTokenizerFast.chat_template` 用，兼容 Transformers 生态。
```

- [ ] **Step 3: 验证**

Run: `rg "分四类|逻辑相同" docs/book/part-3-tokenizer/ch19-production-tokenizer.md`
Expected: 无输出。

- [ ] **Step 4: 提交**

```bash
git add docs/book/part-3-tokenizer/ch19-production-tokenizer.md
git commit -m "docs(ch19): 特殊 token 分五类；标注 Jinja2 模板不支持 tools 注入"
```

---

### Task 8: B2+B4 — Top-P 定义 + softmax 回引（ch41，4 处）

**Files:**
- Modify: `docs/book/part-7-deployment/ch41-decoding-algorithm.md`（行 54、59、61、224）

**依据**：
- B2：`generate.py:63` 用 `cumulative_probs - self >= top_p`（保留累计达到 P 的最小集合）；`ch41:70` 公式 `min{S:Σp_i≥P}` 自洽。表/小结写"≤P"是错的。
- B4：Ch06 是最优化（无 softmax）；softmax 在 `ch03:256`，温度在 `ch05:331-345`。

- [ ] **Step 1: B2 — 表格 Top-P（行 54）**

`oldString`:
```
| **Top-P** | `top_p>0` | 只从累计概率 ≤ P 的最小集合里采样 |
```
`newString`:
```
| **Top-P** | `top_p>0` | 只从累计概率首次达到/超过 P 的最小集合里采样 |
```

- [ ] **Step 2: B4 — 小节标题回引（行 59）**

`oldString`:
```
### 41.2.3 Temperature（回引 Ch 06）
```
`newString`:
```
### 41.2.3 Temperature（回引 Ch 03 / Ch 05）
```

- [ ] **Step 3: B4 — 正文回引（行 61）**

`oldString`:
```
Ch 06 讲过 softmax 温度。解码时 `logits / T` 再 softmax：T < 1 → 分布更尖锐（倾向高概率）；T > 1 → 分布更平坦（更随机）。T=0 退化成 greedy（argmax）。
```
`newString`:
```
Ch 03 讲过 softmax、Ch 05 讲过温度缩放（蒸馏里用 $T^2$ 加权 KL）。解码时 `logits / T` 再 softmax：T < 1 → 分布更尖锐（倾向高概率）；T > 1 → 分布更平坦（更随机）。T=0 退化成 greedy（argmax）。
```

- [ ] **Step 4: B2 — 小结 Top-P（行 224）**

`oldString`:
```
4. **Top-K**：固定 K 个候选；**Top-P**：动态集合（累计概率 ≤ P），自适应。
```
`newString`:
```
4. **Top-K**：固定 K 个候选；**Top-P**：动态集合（累计概率首次达到 P），自适应。
```

- [ ] **Step 5: 验证**

Run: `rg "累计概率 ≤ P|回引 Ch 06|Ch 06 讲过 softmax" docs/book/part-7-deployment/ch41-decoding-algorithm.md`
Expected: 无输出。

- [ ] **Step 6: 提交**

```bash
git add docs/book/part-7-deployment/ch41-decoding-algorithm.md
git commit -m "docs(ch41): Top-P 定义修正为 ≥P；softmax/温度回引修正为 Ch03/Ch05"
```

---

### Task 9: C3+C4 — KL 方向 + RLHF 模型数（ch35，2 处）

**Files:**
- Modify: `docs/book/part-6-alignment/ch35-rlhf-framework.md`（行 60 公式 + 行 63 段末补注 + 行 111 表格）

**依据**：
- C3：`grpo.py:36` 的 k3 估计 `e^(ref-policy)-(ref-policy)-1` 是**反向** `KL(π_ref‖π_θ)` 的无偏估计；`ch38:65` 也写反向。Ch35 的 boxed 目标写前向 `KL(π_θ‖π_ref)` 与两者冲突。本任务把 boxed 改为反向（与代码/Ch38 一致），并补注与标准文献前向 KL 的区别。
- C4：PPO 推理时 RM 已离线训练、不驻留显存，实际 3 模型（policy+ref+critic）。原文"3 模型（policy+ref+critic+RM）"括号里 4 项标 3，自相矛盾。

- [ ] **Step 1: C3 — boxed 目标改反向 KL（行 60）**

`oldString`:
```
\max_{\pi_\theta}\;\mathbb{E}_{y\sim\pi_\theta}\bigl[r_\phi(x,y)\bigr] \;-\; \beta\,\text{KL}\bigl(\pi_\theta(\cdot|x)\,\big\|\,\pi_{\text{ref}}(\cdot|x)\bigr)
```
`newString`:
```
\max_{\pi_\theta}\;\mathbb{E}_{y\sim\pi_\theta}\bigl[r_\phi(x,y)\bigr] \;-\; \beta\,\text{KL}\bigl(\pi_{\text{ref}}(\cdot|x)\,\big\|\,\pi_\theta(\cdot|x)\bigr)
```

- [ ] **Step 2: C3 — 段末补方向说明（行 63 之后追加）**

`oldString`:
```
$\pi_{\text{ref}}$ 是 SFT 后的模型（冻结）。KL 项惩罚 $\pi_\theta$ 偏离 $\pi_{\text{ref}}$ 太远——「你可以变好，但不能变成另一个人」。$\beta$ 控制约束强度。
```
`newString`:
```
$\pi_{\text{ref}}$ 是 SFT 后的模型（冻结）。KL 项惩罚 $\pi_\theta$ 偏离 $\pi_{\text{ref}}$ 太远——「你可以变好，但不能变成另一个人」。$\beta$ 控制约束强度。

> **KL 方向说明**：标准 RLHF 文献常写成前向 $\text{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$；zllm 的实现（Ch 38 GRPO，`grpo.py:36` 的 k3 估计 $e^{(\text{ref}-\text{policy})}-(\text{ref}-\text{policy})-1$）采用**反向** $\text{KL}(\pi_{\text{ref}}\,\|\,\pi_\theta)$。两者特性不同：反向 KL 是 **mode-seeking**（逼策略聚到 ref 的峰），前向是 **mean-matching**。本书公式与代码对齐，统一用反向。
```

- [ ] **Step 3: C4 — RLHF 显存行模型数（行 111）**

`oldString`:
```
| **显存** | 3 模型（policy+ref+critic+RM） | 2 模型（policy+ref） | 2 模型（policy+ref） |
```
`newString`:
```
| **显存** | 3 模型（policy+ref+critic；RM 离线训练不驻留） | 2 模型（policy+ref） | 2 模型（policy+ref） |
```

- [ ] **Step 4: 验证**

Run: `rg "pi_\\\\theta\(\\\\cdot\|x\)\\\\,\\\\big\\\\\|\\\\,\\\\,\\\\pi_\{\\\\text\{ref\}\}" docs/book/part-6-alignment/ch35-rlhf-framework.md`
Expected: 无输出（前向 KL 已改为反向）。

另运行简单检查：`rg "policy\+ref\+critic\+RM" docs/book/part-6-alignment/ch35-rlhf-framework.md`
Expected: 无输出。

- [ ] **Step 5: 提交**

```bash
git add docs/book/part-6-alignment/ch35-rlhf-framework.md
git commit -m "docs(ch35): KL 方向改为反向（与 grpo.py/ch38 一致）；RLHF 模型数修正为 3"
```

---

### Task 10: C5 — Agent 奖励匹配方式描述（ch40，2 处）

**Files:**
- Modify: `docs/book/part-6-alignment/ch40-agent-rl-tools.md`（行 25 学习目标 + 行 140 正文）

**依据**：`agent_rl.py:98` 是 `gt_text.lower().strip() in response.lower()`——大小写不敏感的**精确子串包含**，非语义模糊匹配。原文"模糊匹配"+"气温28度"示例会误导（该示例实际不命中）。

- [ ] **Step 1: 修正学习目标（行 25）**

`oldString`:
```
- 说清 `validate_gt_in_text` 为什么用模糊匹配而非精确匹配。
```
`newString`:
```
- 说清 `validate_gt_in_text` 用「大小写不敏感的精确子串包含」及其局限（不识别同义改写）。
```

- [ ] **Step 2: 修正正文（行 140）**

`oldString`:
```
`validate_gt_in_text`（`:88-98`）：检查 gt 是否**作为子串**出现在 response 中（大小写不敏感）。用模糊匹配而非精确匹配——因为模型可能用不同表述包含正确答案（如「28°C」可能写成「气温28度」... 但这里用精确子串匹配 `28°C in response`）。
```
`newString`:
```
`validate_gt_in_text`（`:88-98`）：检查 gt 是否**作为子串**出现在 response 中——大小写不敏感的精确子串包含（`gt.strip().lower() in response.lower()`）。注意它只匹配**字面**子串，**不识别同义改写**：例如 gt 是「28°C」时，回答写成「气温28度」不会被命中。
```

- [ ] **Step 3: 验证**

Run: `rg "模糊匹配|气温28度" docs/book/part-6-alignment/ch40-agent-rl-tools.md`
Expected: 无输出。

- [ ] **Step 4: 提交**

```bash
git add docs/book/part-6-alignment/ch40-agent-rl-tools.md
git commit -m "docs(ch40): validate_gt_in_text 改述为精确子串包含（非模糊匹配）"
```

---

### Task 11: C6 — PPO/GRPO/Agent 实现范围声明（ch37/ch38/ch40，3 处插入）

**Files:**
- Modify: `docs/book/part-6-alignment/ch37-ppo-gae-critic.md`（学习目标之后，`## 37.2` 之前）
- Modify: `docs/book/part-6-alignment/ch38-grpo-cispo.md`（学习目标之后，`## 38.2` 之前）
- Modify: `docs/book/part-6-alignment/ch40-agent-rl-tools.md`（学习目标之后，`## 40.2` 之前；注意本文件 Task 10 已先改过行 25 附近，插入点在新行 25 之后）

**依据**：`ppo.py`/`grpo.py`/`agent_rl.py` 只含损失/奖励原语 + Config，**无 `train_epoch`/rollout/多轮循环**。三章（含模块 docstring）把原语描述成完整训练器，读者会期待能跑的训练器。统一插入相同格式的"实现范围说明"框，便于识别。

**Interfaces:** 三处插入文案结构相同（`> **实现范围说明**：本章实现 X 的核心组件……外层训练循环不在 zllm 当前代码内……`），仅 X 不同。执行者照抄下方对应 X。

- [ ] **Step 1: ch37 插入声明**

`oldString`（行 26–28，定位插入点：学习目标最后一条 + 空行 + 下一节标题）:
```
- 看懂 CriticModel 如何共享 backbone 但替换 lm_head。

## 37.2 原理回顾：Actor-Critic + GAE
```
`newString`:
```
- 看懂 CriticModel 如何共享 backbone 但替换 lm_head。

> **实现范围说明**：本章实现 PPO 的核心组件——`CriticModel`、`compute_gae`、`ppo_policy_loss`、`ppo_value_loss` 与 `PPOConfig`。**采样—打分—mini-batch 更新的外层训练循环（rollout、approx_kl 早停）不在 zllm 当前代码内**，需读者自行组装或参考社区实现。本章聚焦损失原语与 GAE 的数学。

## 37.2 原理回顾：Actor-Critic + GAE
```

- [ ] **Step 2: ch38 插入声明**

`oldString`:
```
- 解释 CISPO 单边裁剪与 GRPO 双边 clip 的区别及其意义。

## 38.2 原理回顾：群体相对优势
```
`newString`:
```
- 解释 CISPO 单边裁剪与 GRPO 双边 clip 的区别及其意义。

> **实现范围说明**：本章实现 GRPO/CISPO 的核心组件——`compute_group_advantages`、`per_token_kl`、`grpo_loss`、`cispo_loss` 与 `GRPOConfig`。**对同一 prompt 采样 $N$ 个回答、打分、更新的外层训练循环不在 zllm 当前代码内**，需读者自行组装或参考社区实现。本章聚焦损失原语与群体优势的数学。

## 38.2 原理回顾：群体相对优势
```

- [ ] **Step 3: ch40 插入声明**

注意：Task 10 已修改 ch40 行 25 的学习目标措辞。此处 `oldString` 用 Task 10 完成后的新行 25 作为锚点。

`oldString`:
```
- 说清 `validate_gt_in_text` 用「大小写不敏感的精确子串包含」及其局限（不识别同义改写）。

## 40.2 原理回顾：Agent 多轮交互
```
`newString`:
```
- 说清 `validate_gt_in_text` 用「大小写不敏感的精确子串包含」及其局限（不识别同义改写）。

> **实现范围说明**：本章实现 Agent RL 的核心组件——6 个模拟工具、`parse_tool_calls`、`execute_tool`、`calculate_agent_reward` 与 `AgentConfig`。**生成→解析→执行→反馈→再生成的多轮控制循环不在 zllm 当前代码内**，需读者自行组装或参考社区实现。本章聚焦工具环境与多维奖励的设计。

## 40.2 原理回顾：Agent 多轮交互
```

- [ ] **Step 4: 验证三处都已插入**

Run: `rg "实现范围说明" docs/book/part-6-alignment/`
Expected: 3 行匹配（ch37、ch38、ch40 各一）。

- [ ] **Step 5: 提交**

```bash
git add docs/book/part-6-alignment/ch37-ppo-gae-critic.md docs/book/part-6-alignment/ch38-grpo-cispo.md docs/book/part-6-alignment/ch40-agent-rl-tools.md
git commit -m "docs(ch37/38/40): 补「实现范围说明」，如实标注 PPO/GRPO/Agent 仅有损失原语"
```

---

### Task 12: B6 — Adam/AdamW 文献拆分（appendix-d，1 处）

**Files:**
- Modify: `docs/book/appendices/appendix-d-references.md`（行 33）

**依据**：Adam = Kingma & Ba 2014；AdamW = Loshchilov & Huter 2019 *Decoupled Weight Decay Regularization*，是两篇不同论文。原文把 2014 那篇标成"AdamW"。Ch06 强调"zllm 用 AdamW"。同时原行章节列"Ch 11"（序列建模），Adam/AdamW 实际在 Ch 06（最优化），一并修正。

- [ ] **Step 1: 拆分为两行（行 33）**

`oldString`:
```
| Adam: A Method for Stochastic Optimization | Kingma & Ba | 2014 | AdamW 优化器 | Ch 11 |
```
`newString`:
```
| Adam: A Method for Stochastic Optimization | Kingma & Ba | 2014 | Adam 优化器 | Ch 06 |
| Decoupled Weight Decay Regularization | Loshchilov & Huter | 2019 | AdamW 优化器（zllm 实际用） | Ch 06, 29 |
```

- [ ] **Step 2: 验证**

Run: `rg "Kingma & Ba \| 2014 \| AdamW" docs/book/appendices/appendix-d-references.md`
Expected: 无输出。

Run: `rg "Decoupled Weight Decay Regularization" docs/book/appendices/appendix-d-references.md`
Expected: 1 行匹配。

- [ ] **Step 3: 提交**

```bash
git add docs/book/appendices/appendix-d-references.md
git commit -m "docs(appx-d): 拆分 Adam(2014)/AdamW(2019) 文献，修正章节指向 Ch06"
```

---

### Task 13: B7 — 每层 attention 参数量（appendix-b，2 处）

**Files:**
- Modify: `docs/book/appendices/appendix-b-hyperparameters.md`（行 47、49）

**依据**：GQA 下 q(768·768)+k(768·384)+v(768·384)+o(768·768) = 589824+294912+294912+589824 = 1,769,472 ≈ 1.77M/层（非 4.7M）。每层 FFN ≈ 5.6M 正确。8 层（1.77+5.6）×8 ≈ 58.96M，加 embedding 4.9M ≈ 63.86M ≈ 64M（标题对）。原文"82M"与标题"64M"自相矛盾。

- [ ] **Step 1: 修正每层 attention（行 47）**

`oldString`:
```
- 每层 attention：约 4.7M
```
`newString`:
```
- 每层 attention：约 1.77M（GQA：q=768²，k/v=768·384 各一，o=768²）
```

- [ ] **Step 2: 修正 8 层合计（行 49）**

`oldString`:
```
- 8 层合计：约 82M（含 embedding），可训练约 64M
```
`newString`:
```
- 8 层合计（attention + FFN）：约 59M；加 embedding 约 4.9M，总计约 64M
```

- [ ] **Step 3: 验证**

Run: `rg "约 4.7M|约 82M" docs/book/appendices/appendix-b-hyperparameters.md`
Expected: 无输出。

- [ ] **Step 4: 提交**

```bash
git add docs/book/appendices/appendix-b-hyperparameters.md
git commit -m "docs(appx-b): 修正每层 attention 参数量 1.77M（GQA），8 层合计与 64M 自洽"
```

---

## 全局收尾（所有任务完成后）

- [ ] **G1: 跨文件旧表述终检**

Run:
```bash
rg "RMS\(x\)\+\\epsilon|intermediate_size=3072|默认 batch=32|η₀/2 \(中点\)|分四类|累计概率 ≤ P|回引 Ch 06$|Ch 06 讲过 softmax|policy\+ref\+critic\+RM|模糊匹配|Kingma & Ba \| 2014 \| AdamW|约 4.7M|约 82M|bfloat16.*的梯度可能下溢" docs/book/
```
Expected: 无输出（或仅来自本计划未覆盖的 P1/P2 章节，需逐条确认不在 B1–B9/C1–C7 范围）。

- [ ] **G2: 改动文件清单 review**

Run: `git diff --stat docs/book/`
Expected: 14 个文件，总增删行数 < 80。

- [ ] **G3: 人工抽查 3 处数学渲染**

目测确认渲染正确（非必装 mdbook，纯文本核对即可）：
1. `ch20:46` boxed 公式含 `√(mean(x²)+ε)`
2. `ch35:60` boxed 目标 KL 参数顺序为 `π_ref ‖ π_θ`
3. `ch29:57` Mermaid 节点为 `0.55·η₀`

---

## Self-Review（计划作者自查，已完成）

**1. Spec 覆盖**：B1–B9、C1–C7 共 16 项逐一对应 Task 1–13。✓ 全覆盖。
- B1→T1, B2→T8, B3→T2, B4→T8, B5→T3, B6→T12, B7→T13, B8→T5, B9→T4
- C1→T7, C2→T7, C3→T9, C4→T9, C5→T10, C6→T11, C7→T6

**2. 占位符扫描**：无 TBD/TODO/「类似 Task N」。每个 oldString/newString 均为可直接粘贴的完整文本。✓

**3. 类型/命名一致**：C6 三处声明结构一致（仅 X 不同），锚点用各自学习目标最后一条；T11 Step 3 已标注依赖 T10 先完成 ch40 行 25 的修改，执行顺序为 T10 → T11。✓

**4. Spec 订正**：B5 文件 ch06 → ch08（spec 笔误），已在 Global Constraints 与 Task 3 标注。✓

**5. 依赖顺序**：T10 必须在 T11 之前（T11 Step 3 的 oldString 锚点是 T10 的产出）。其余任务相互独立，可任意顺序或并行。建议执行顺序：T1→T13（除 T10 早于 T11）。✓
