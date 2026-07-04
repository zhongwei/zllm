---
part: 6
chapter: 36
title: DPO 直接偏好优化
milestone: M10-a
source: zllm/training/dpo.py
tests: tests/m10_alignment/test_240_dpo.py
status: draft
---

# 第 36 章 DPO 直接偏好优化

Ch 35 讲了 DPO 的理论：RLHF 目标有闭式解，可以直接从偏好数据推导出策略，绕过奖励模型。本章实现这个推导——`dpo_loss` 函数只有几行，但背后的数学很深刻。

DPO 的架构是**双模型**：policy model（训练中）和 reference model（冻结的 SFT 模型）。loss 衡量的是「policy 相对 reference，更偏好 chosen 还是 rejected」。训练让 policy 越来越倾向 chosen、远离 rejected，同时 reference 锚定防止漂移。

## 36.1 学习目标

读完本章，你应该能够：

- 默写出 DPO loss：$-\log\sigma(\beta(\Delta_\theta(y_w) - \Delta_\theta(y_l)))$，其中 $\Delta_\theta = \log\frac{\pi_\theta}{\pi_{\text{ref}}}$；
- 解释 reference model 的作用（锚定防漂移）；
- 说清 DPO 学习率（4e-8）为什么极低（防灾难遗忘）；
- 看懂 `logits_to_log_probs` 如何从 logits 提取 token 级 log 概率；
- 理解 chosen/rejected 拼接后前后半分割的技巧。

## 36.2 原理回顾：DPO loss

### 36.2.1 从 RLHF 闭式解到 DPO（回引 Ch 35）

Ch 35 推导了 DPO loss：

$$
\mathcal{L}_{\text{DPO}} \;=\; -\log\sigma\!\left(\beta\left[\underbrace{\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}}_{\text{chosen 的 log-ratio}} - \underbrace{\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}}_{\text{rejected 的 log-ratio}}\right]\right)
$$

直觉：如果 policy 比 reference **更偏好** chosen（$\Delta_\theta(y_w) > \Delta_\theta(y_l)$），sigmoid 输入为正，$-\log\sigma$ 接近 0（低 loss）。反之 loss 高，梯度推动 policy 向 chosen 靠拢。

### 36.2.2 token 级 log 概率

$\pi_\theta(y|x)$ 是整个回答 $y$ 的概率，等于各 token 概率的乘积。取 log 后变成求和：

$$
\log\pi_\theta(y|x) \;=\; \sum_{t} \log P_\theta(y_t | y_{<t}, x)
$$

实现上：先 forward 得到 logits → log_softmax → 用 `gather` 取出**真实 label 位置**的 log 概率 → 沿序列维求和（只对 assistant 区域，用 mask）。

### 36.2.3 β 温度参数

$\beta$ 控制「偏离 reference 的强度」。zllm 默认 $\beta=0.15$：$\beta$ 大 → 对齐激进（可能崩坏）；$\beta$ 小 → 对齐保守（变化慢）。配合极低学习率 4e-8，DPO 的调整非常轻柔。

## 36.3 代码实现

完整实现见 `zllm/training/dpo.py`（162 行）。

### 36.3.1 logits_to_log_probs：提取 token log 概率

> 完整实现见 `zllm/training/dpo.py:20`

```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)                    # (B, T, V) → log 概率
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    #                          ↑ 取出 labels 对应位置的 log 概率 → (B, T)
```

`logits_to_log_probs`（`:20-31`）：log_softmax 把 logits 变成 log 概率，`gather` 按 `labels`（真实 token id）取出对应位置的值。结果 `(B, T)` 是每个位置真实 token 的 log 概率。

> 对应测试 `tests/m10_alignment/test_240_dpo.py:26`（shape `(2,10)`）、`:32`（值等于手动 log_softmax+gather）、`:40`（log 概率之和 < 0，因为每个 ≤ 0）。

### 36.3.2 dpo_loss：偏好损失核心

> 完整实现见 `zllm/training/dpo.py:34`

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.15):
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)       # assistant 区域求和
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    batch_size = ref_log_probs.shape[0]
    chosen_ref = ref_log_probs[: batch_size // 2]           # 前半 chosen
    reject_ref = ref_log_probs[batch_size // 2 :]           # 后半 rejected
    chosen_policy = policy_log_probs[: batch_size // 2]
    reject_policy = policy_log_probs[batch_size // 2 :]

    pi_logratios = chosen_policy - reject_policy            # Δ_policy = log π(c) - log π(r)
    ref_logratios = chosen_ref - reject_ref                 # Δ_ref
    logits = pi_logratios - ref_logratios                   # β 内部的 logits
    loss = -F.logsigmoid(beta * logits)                     # -log σ(β · logits)
    return loss.mean()
```

`dpo_loss`（`:34-61`）三步：

1. **mask + 求和**（`:48-49`）：`(log_probs * mask).sum(dim=1)` 只对 assistant 区域（mask=1）求和，得到整条回答的 log 概率。
2. **前后半分割**（`:52-55`）：输入 batch 是 `[chosen, rejected]` 拼接的，前半是 chosen、后半是 rejected。
3. **DPO 公式**（`:57-60`）：$\Delta_\theta - \Delta_{\text{ref}}$，再 $-\log\sigma(\beta \cdot \cdot)$。

> 对应测试 `test_240_dpo.py:48`（chosen 被偏好时 loss > 0）、`:56`（偏好 chosen 的 loss < 无偏好）、`:67`（梯度能流回 policy）。

### 36.3.3 train_epoch：双模型训练循环

> 完整实现见 `zllm/training/dpo.py:86`

`train_epoch`（`:86-162`）与 SFT 的循环结构相似，但多了 reference model：

```python
with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
    with torch.no_grad():
        ref_logits = ref_model(x).logits                    # reference 不训
    ref_log_probs = logits_to_log_probs(ref_logits, y)

    outputs = model(x)                                      # policy 训练
    policy_log_probs = logits_to_log_probs(outputs.logits, y)

    dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=cfg.beta)
    loss = dpo_loss_val + outputs.aux_loss
```

关键：`ref_model` 在 `torch.no_grad()` 下跑（`:121-122`，不产生梯度），只提供 reference 的 log 概率作为锚点。policy 和 reference 共享同一份输入 `x`（chosen+rejected 拼接，`:112-114`）。

### 36.3.4 DPOConfig：极低学习率

> 完整实现见 `zllm/training/dpo.py:64`

`DPOConfig`（`:64-83`）：`learning_rate=4e-8`（`:68`）——比 SFT（1e-5）还低 250 倍！因为 DPO 在已经很聪明的 SFT 模型上做极精细的偏好调整，学习率稍大就会灾难遗忘。`beta=0.15`（`:69`）、`from_weight="full_sft"`（`:81`，DFT 接力 SFT）。

> 对应测试 `test_240_dpo.py:77`（lr=4e-8、beta=0.15）、`:84`（DPO lr < SFT lr）。

## 36.4 对应单元测试

> 对应测试 `tests/m10_alignment/test_240_dpo.py`（155 行）

- **TestLogitsToLogProbs**（`:25-44`）：shape、值正确、和 < 0。
- **TestDPOLoss**（`:47-73`）：chosen 偏好时 loss 更低 `:56`、梯度流 `:67`。
- **TestDPOConfig**（`:76-88`）：默认值、lr < SFT。
- **TestDPOTrain**（`:91-155`）：train_epoch 可运行 `:129`、**loss 下降** `:140`（5 epoch 后 avg loss 降）。

## 36.5 动手验证

```bash
pytest tests/m10_alignment/test_240_dpo.py -v
```

预期：全部 PASSED。验证 DPO loss 的方向性：

```bash
python -c "
import torch
from zllm.training.dpo import dpo_loss
ref = torch.tensor([[-1.0], [-1.0]])
mask = torch.ones(2, 1)
# 策略偏好 chosen（logp=-0.5）而非 rejected（logp=-2.0）
policy_pref = torch.tensor([[-0.5], [-2.0]])
loss_pref = dpo_loss(ref, policy_pref, mask, beta=0.1)
# 无偏好
policy_none = torch.tensor([[-1.0], [-1.0]])
loss_none = dpo_loss(ref, policy_none, mask, beta=0.1)
print(f'偏好 chosen: loss={loss_pref:.4f}  <  无偏好: loss={loss_none:.4f}')
"
```

## 36.6 本章小结 + 下章预告

本章要点：

1. **DPO loss** = $-\log\sigma(\beta(\Delta_\theta(y_w) - \Delta_\theta(y_l)))$，$\Delta_\theta = \log\frac{\pi_\theta}{\pi_{\text{ref}}}$。
2. **双模型**：policy 训练 + reference 冻结（no_grad），reference 锚定防漂移。
3. **logits_to_log_probs**：log_softmax + gather 取 token 级 log 概率。
4. **极低学习率**（4e-8）：SFT 的 1/250，极精细的偏好调整。
5. **chosen/rejected 拼接**：batch 前半 chosen、后半 rejected，一次 forward 处理两个。

> **一句话带走**：DPO 用双模型 + logsigmoid loss 直接从偏好数据训练，绕过 RM——reference 锚定 + 极低 lr 让偏好调整既有效又安全。

**下章预告**：DPO 是离线的（用现成偏好数据）。如果想在线采样 + 强化学习呢？Ch 37《PPO + GAE + Critic》——经典 RLHF 的 PPO 路线，Actor-Critic 双模型、GAE 优势估计、clipped surrogate loss。
