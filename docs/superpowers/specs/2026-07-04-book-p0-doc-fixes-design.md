# P0 文档修正设计

**日期**: 2026-07-04
**范围**: `docs/book/` 下 14 个章节/附录文件的定点修正
**目标**: 修复审计发现的 16 处高危文档问题（9 个数学/事实错误 + 7 个自相矛盾）
**不动**: 源码、README、SUMMARY、P1/P2 问题

---

## 背景

对 `docs/book`（47 个 md 文件）与 `zllm/` 源码（40 个 `.py`）+ 50 个测试文件做了逐章交叉比对，共发现 ~90 处问题。本设计仅处理其中 **P0 级文档错误**：即文档自身的事实错误与自相矛盾，不涉及源码 bug 的传播问题（A 类）与结构缺口（P1）。

完整审计发现见本会话记录；本设计聚焦可机械执行的 16 处定点修正。

## 总体策略

每处修正按两类处理：

- **值替换型**（B1, B2, B3, B4, B7, B8, B9, C1, C7）：直接改错值/错词，零歧义。
- **值替换 + 简短说明型**（B5, B6, C2, C3, C4, C5, C6）：除改错外补 1–3 句上下文，避免读者看完修正反而更困惑。

所有修正**只动 `docs/book/` 下的文件**。

## 涉及文件清单（14 个）

| 文件 | 涉及编号 |
|------|---------|
| `part-1-math/ch06-optimization.md` | B5 |
| `part-2-dl-transformer/ch09-neural-network-basics.md` | B3 |
| `part-3-tokenizer/ch16-project-setup.md` | C7 |
| `part-3-tokenizer/ch19-production-tokenizer.md` | C1, C2 |
| `part-4-architecture/ch20-rmsnorm.md` | B1 |
| `part-5-training/ch29-training-infrastructure.md` | B9 |
| `part-5-training/ch32-pretraining-practice.md` | B8 |
| `part-6-alignment/ch35-rlhf-framework.md` | C3, C4 |
| `part-6-alignment/ch37-ppo-gae-critic.md` | C6 |
| `part-6-alignment/ch38-grpo-cispo.md` | C6 |
| `part-6-alignment/ch40-agent-rl-tools.md` | C5, C6 |
| `part-7-deployment/ch41-decoding-algorithm.md` | B2, B4 |
| `appendices/appendix-b-hyperparameters.md` | B7 |
| `appendices/appendix-d-references.md` | B6 |

## 详细修正方案

### B 类 — 数学/事实错误

#### B1 — RMSNorm 公式（`ch20-rmsnorm.md:46,101,159`）
- **现状**: `x / (RMS(x)+ε) ⊙ γ`（ε 加在 RMS 上）
- **修正**: `x / √(mean(x²)+ε) ⊙ γ`（ε 在根号内）
- **依据**: `norms.py:18` 用 `x * torch.rsqrt(x.pow(2).mean(-1) + self.eps)`；`ch20:65` 自己也写对了。3 处统一。
- **类型**: 值替换

#### B2 — Top-P 定义（`ch41-decoding-algorithm.md:54,224`）
- **现状**: "累计概率 **≤ P** 的最小集合"
- **修正**: "累计概率**首次达到/超过 P** 的最小集合"
- **依据**: `generate.py:63` 用 `cumulative_probs - self >= top_p` 过滤；`ch41:70` 公式 `min{S : Σ p_i ≥ P}` 自洽。2 处统一。
- **类型**: 值替换

#### B3 — intermediate_size（`ch09-neural-network-basics.md:191,294`）
- **现状**: "intermediate_size=3072 / 把 d=768 扩张 4 倍"
- **修正**: "intermediate_size=2432 / π-缩放 `ceil(768·π/64)·64`（≈3.17×）"
- **依据**: `config.py:48-52`；附录 B:23 已正确列 2432。Ch09 当前与两者冲突。
- **类型**: 值替换

#### B4 — softmax 温度回引（`ch41-decoding-algorithm.md:59`）
- **现状**: "Ch 06 讲过 softmax 温度"
- **修正**: "Ch 03 讲过 softmax、Ch 05 讲过温度"
- **依据**: Ch06 是最优化（无 softmax）；softmax 在 `ch03-probability.md:256`；温度在 `ch05-information-theory.md:331-345`。
- **类型**: 值替换

#### B5 — GradScaler 归因（`ch06-optimization.md:336` 附近 GradScaler 段）
- **现状**: "bfloat16 的梯度可能下溢（太小而变成 0）"
- **修正**: "fp16 的梯度可能下溢；bf16 指数位与 fp32 相同（8 位）几乎不下溢，故 bf16 下 GradScaler 形同虚设。zllm 仍保留 scaler 调用以兼容 fp16 路径。"
- **依据**: fp16 有 5 位指数（下溢真实风险），bf16 有 8 位指数（同 fp32）。代码确实保留 `GradScalerManager(enabled=True)` 但在 bf16 下无实际作用。
- **类型**: 值替换 + 简短说明

#### B6 — AdamW 文献归属（`appendix-d-references.md:33`）
- **现状**: 一行 "Kingma & Ba 2014 → AdamW 优化器"
- **修正**: 拆为两行：
  1. Kingma & Ba 2014 *Adam: A Method for Stochastic Optimization* → Adam 优化器
  2. Loshchilov & Huter 2019 *Decoupled Weight Decay Regularization* → AdamW 优化器（zllm 实际用）
- **依据**: 这是两篇不同论文；Ch06 强调"zllm 用 AdamW"。
- **类型**: 值替换 + 简短说明

#### B7 — 每层 attention 参数量（`appendix-b-hyperparameters.md:47-49`）
- **现状**: "每层 attention 约 4.7M；8 层合计约 82M"
- **修正**: "每层 attention 约 1.77M（GQA: q=768², k/v=768·384 各一, o=768²）；8 层 attention 合计约 14M；加 8 层 FFN（各 ~5.6M）+ 嵌入 ~4.9M，总计约 64M"
- **依据**: GQA 下 q(768·768)+k(768·384)+v(768·384)+o(768·768) = 589824+294912+294912+589824 = 1,769,472 ≈ 1.77M。原标题"约 64M"是对的，正文 4.7M/82M 错。
- **类型**: 值替换 + 简短说明

#### B8 — 默认 batch_size（`ch32-pretraining-practice.md:148`）
- **现状**: "zllm 默认 batch=32、accum=4（等效 128）"
- **修正**: "zllm 默认 batch=64、accum=4（等效 256）"
- **依据**: `pretrain.py:20` `batch_size: int = 64`；`ch31:82` 与 `ch32:150` 都已是 64。仅 `ch32:148` 残留旧值。
- **类型**: 值替换

#### B9 — cosine lr Mermaid 中点（`ch29-training-infrastructure.md:57`）
- **现状**: Mermaid 节点标 "η₀/2 (中点)"
- **修正**: "0.55·η₀ (中点)"
- **依据**: `get_lr`（`utils.py:46-47`）在 s=S/2 时返回 `base·(0.1+0.45·(1+cos(π/2))) = base·(0.1+0.45) = 0.55·base`。`ch29:52` 表格已对，仅图错。
- **类型**: 值替换

### C 类 — 自相矛盾

#### C1 — 特殊 token 分类数（`ch19-production-tokenizer.md:111`）
- **现状**: "分**四类**"（下文却列 5 个 bullet）
- **修正**: "分**五类**"
- **依据**: 下文 5 个 bullet（对话边界/多模态预留/工具调用/思考链/buffer）；`ch19:23` 学习目标写 5 类；`special_tokens.py:3-9` 模块 docstring 也列 5 类。
- **类型**: 值替换

#### C2 — Jinja2 模板能力（`ch19-production-tokenizer.md:146`）
- **现状**: "另外一个 Jinja2 版本 `CHAT_TEMPLATE`（`:76-94`），**逻辑相同**"
- **修正**: "另外一个 Jinja2 版本 `CHAT_TEMPLATE`（`:76-94`），**逻辑相近但不支持 tools 注入**（仅渲染 messages + open_thinking + add_generation_prompt）；Agent RL（Ch 40）必须用 `render_messages` 才能注入工具描述"
- **依据**: `render_messages`（`chat_template.py:25-72`）实现了 tools 注入（`:46-55`）；Jinja2 串（`:76-94`）无任何 tools 处理。读者若把 Jinja2 串接到 `PreTrainedTokenizerFast.chat_template` 做 Agent RL 会静默丢工具。
- **类型**: 值替换 + 简短说明

#### C3 — KL 方向（`ch35-rlhf-framework.md:60`）
- **现状**: RLHF 目标写作前向 `KL(π_θ ‖ π_ref)`
- **修正**: 改为反向 `KL(π_ref ‖ π_θ)`；段末加一句："zllm 实现采用反向 KL（`grpo.py:36` 的 k3 估计 `e^(ref-policy)-(ref-policy)-1`），与前向 KL 在收敛特性上略有差异：反向 KL 的最优解是 mode-seeking，前向是 mean-matching。"
- **依据**: `ch38:65` 已写反向；`grpo.py:36` 的 k3 是反向 KL 的无偏估计。Ch35 与两者冲突。
- **类型**: 值替换 + 简短说明

#### C4 — RLHF 模型数（`ch35-rlhf-framework.md:111` 对比表显存行）
- **现状**: "**3 模型**（policy+ref+critic+RM）"——括号里 4 项标 3
- **修正**: "**3 模型**（policy+ref+critic；RM 通常离线训练，推理时不驻留显存）"，删去括号中的 "+RM"
- **依据**: PPO 推理时 RM 已训好不驻留，实际 3 个。当前文字自相矛盾。
- **类型**: 值替换 + 简短说明

#### C5 — Agent 奖励匹配方式（`ch40-agent-rl-tools.md:140`）
- **现状**: "用**模糊匹配**而非精确匹配——因为模型可能用不同表述包含正确答案（如「28°C」可能写成「气温28度」... 但这里用精确子串匹配 `28°C in response`）"
- **修正**: "用**大小写不敏感的精确子串包含**（`gt.strip().lower() in response.lower()`）。注意它只匹配字面子串，不识别同义改写——例如「气温28度」不会被算作命中「28°C」。"
- **依据**: `agent_rl.py:98` 是 `gt_text.lower().strip() in response.lower()`，纯子串包含。"模糊匹配"措辞与"气温28度"示例都会误导读者以为有语义匹配。
- **类型**: 值替换 + 简短说明

#### C6 — PPO/GRPO/Agent 实现范围声明（`ch37`/`ch38`/`ch40` 章首）
- **现状**: 三章（含各模块 docstring）把 loss/奖励原语描述成完整训练器，但源码只有 `CriticModel/compute_gae/ppo_policy_loss/...`、`grpo_loss/cispo_loss/...`、`TOOLS/execute_tool/parse_tool_calls/calculate_agent_reward/...` + Config，**无任何 `train_epoch`/rollout/多轮循环**。`grpo.py` 的 `loss_type="cispo"`（Ch38 卖点）全代码库无消费者。
- **修正**: 在 `ch37`、`ch38`、`ch40` 每章开头同一位置（学习目标之后、正文之前）插入统一格式的"实现范围"提示框：
  ```
  > **实现范围说明**：本章实现 X 的核心损失/奖励原语与 Config。
  > 采样—打分—更新的外层训练循环（rollout / mini-batch / 多轮工具调用）
  > 不在 zllm 当前代码内，需读者自行组装或参考社区实现。
  ```
  - Ch37 X = "PPO（CriticModel、GAE、clipped surrogate/value loss）"
  - Ch38 X = "GRPO/CISPO（组标准化优势、per-token KL、单/双边裁剪损失）"
  - Ch40 X = "Agent RL（6 个模拟工具、工具调用解析、多维奖励）"
- **依据**: 现状让读者期待能跑的训练器，实际不存在。统一措辞便于读者识别。
- **类型**: 值替换 + 简短说明（插入式）

#### C7 — M1 测试数（`ch16-project-setup.md:154,239`）
- **现状**: "M1 里程碑只有两个测试文件、**4 个测试**" / "预期输出（**4 个测试**全绿）"——紧接的代码块却打印 `===== 8 passed =====`
- **修正**: 两处 "4 个测试" → "8 个测试"；删掉 `:253` 附近试图圆场的尴尬括注
- **依据**: `pytest tests/m01_foundations/ --collect-only` 实测 8 个（test_002 两个 + test_003 六个）。`ch16:253,267` 本身也写 8，仅 154/239 残留旧值。
- **类型**: 值替换

## 明确排除项

- ❌ 不改任何 `.py` 源码（A1–A8 全留作后续）
- ❌ 不动 `README.md` / `SUMMARY.md`（本设计无依赖）
- ❌ 不补 P1 缺口（RoPE attn_factor、padding/attention_mask 故事、DDP、术语表补词、Appendix D 补 Scaling Law/Chinchilla 等）
- ❌ 不做 P2 打磨（行号微调、死代码注释、拼写、文档片段省略标注等）
- ❌ 不重写章节，只做定点精准编辑

## 验证方式

每处改完后：
1. `grep` 旧错误表述，确认目标文件内无残留（如 B1 的 3 处 `RMS+ε`）。
2. 跨文件交叉确认一致性（如 B3 改 Ch09 后，确认与 `config.py`/附录 B 不再冲突）。
3. 全部完成后对 14 个文件做一次 `git diff --stat` 人工 review，确认无意外改动。

## 成功标准

- 16 处问题对应的旧表述在 `docs/book/` 内均不可 grep 到。
- 14 个文件改动行数总和预期 < 80 行（多数为单行替换 + 少量插入）。
- `docs/book/` 内无新增的内部矛盾（每个修正都显式标注了与哪条原文/源码对齐）。
