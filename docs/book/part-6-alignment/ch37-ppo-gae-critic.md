---
part: 6
chapter: 37
title: PPO + GAE + Critic
milestone: M10-b
source: zllm/training/ppo.py
tests: tests/m10_alignment/test_246_ppo.py
status: draft
---

# 第 37 章 PPO + GAE + Critic

Ch 36 的 DPO 是**离线**方法——用预先标注好的 chosen/rejected 数据。Ch 35 介绍的 RLHF 经典路线是**在线**的：模型自己生成回答 → 打分 → 强化学习优化。**PPO（Proximal Policy Optimization）** 就是这条路线的核心算法。

PPO 是最复杂的对齐方法，涉及四个组件：**Actor**（策略模型，生成回答）、**Critic**（价值模型，估计每个状态的价值）、**GAE**（广义优势估计，平衡偏差与方差）、**Clipped Surrogate Loss**（防止策略更新步子太大）。本章逐个拆解。

## 37.1 学习目标

读完本章，你应该能够：

- 解释 Actor-Critic 架构：Actor 生成、Critic 估值；
- 默写出 GAE 的递推公式 $\delta_t = r_t + \gamma V_{t+1} - V_t$，$A_t = \delta_t + \gamma\lambda\,A_{t+1}$；
- 说清 GAE 的 $\lambda$ 如何平衡偏差（Critic 估计不准）与方差（蒙特卡洛噪声大）；
- 理解 PPO clipped surrogate loss 为何用 $\min(ratio \cdot A,\; clip(ratio) \cdot A)$；
- 解释 value loss 也做 clip 的原因；
- 看懂 CriticModel 如何共享 backbone 但替换 lm_head。

> **实现范围说明**：本章实现 PPO 的核心组件——`CriticModel`、`compute_gae`、`ppo_policy_loss`、`ppo_value_loss` 与 `PPOConfig`。**采样—打分—mini-batch 更新的外层训练循环（rollout、approx_kl 早停）不在 zllm 当前代码内**，需读者自行组装或参考社区实现。本章聚焦损失原语与 GAE 的数学。

## 37.2 原理回顾：Actor-Critic + GAE

### 37.2.1 优势函数与 Critic

强化学习里，**优势（Advantage）** 衡量「某个动作比平均水平好多少」：$A(s,a) = Q(s,a) - V(s)$。$V(s)$ 是状态 $s$ 的**价值**——从 $s$ 出发的期望累计奖励。

PPO 需要一个 **Critic 模型**来估计 $V(s)$。Critic 和 Actor 共享 backbone（Transformer），但 Actor 的 lm_head 输出词表 logits（选词），Critic 的 value_head 输出标量（估值）。

### 37.2.2 GAE：广义优势估计

直接用 $A_t = r_t + \gamma V_{t+1} - V_t$（TD 误差 $\delta_t$）方差小但偏差大（依赖 Critic 准不准）；用蒙特卡洛 $A_t = \sum_l \gamma^l r_{t+l} - V_t$ 无偏但方差大。**GAE（Generalized Advantage Estimation）** 用 $\lambda$ 在两者间插值：

$$
\delta_t = r_t + \gamma V_{t+1} - V_t
$$

$$
\boxed{\;A_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} \;=\; \delta_t + \gamma\lambda\,A_{t+1}\;}
$$

$\lambda=0$：纯 TD（只用一步 $\delta_t$），偏差大方差小。$\lambda=1$：纯蒙特卡洛，无偏方差大。$\lambda=0.95$（zllm 默认）是折中。递推实现从后往前算（`reversed`），高效。

### 37.2.3 PPO Clipped Surrogate Loss

策略梯度 $\nabla\log\pi \cdot A$ 直接用方差太大。PPO 用**重要性采样**的比值 $ratio = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$，并**裁剪**防止步子太大：

$$
L^{\text{CLIP}} = -\min\bigl(ratio \cdot A,\;\; \text{clip}(ratio,\;1{-}\epsilon,\;1{+}\epsilon)\cdot A\bigr)
$$

$\epsilon=0.2$（zllm 默认）限制 ratio 在 $[0.8, 1.2]$ 之间。取 `min` 保证采用**更保守**的那个——策略不会一次更新太多导致崩坏。

## 37.3 代码实现

完整实现见 `zllm/training/ppo.py`（157 行）。

### 37.3.1 CriticModel：共享 backbone + value_head

> 完整实现见 `zllm/training/ppo.py:29`

```python
class CriticModel(ZLLMForCausalLM):
    """价值网络：共享 backbone，替换 lm_head 为 value_head。"""
    def __init__(self, config=None):
        self.config = config or ZLLMConfig()
        super().__init__(self.config)
        self.value_head = nn.Linear(self.config.hidden_size, 1)   # hidden → 标量

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)       # (B, T) 每位置一个价值
        return values
```

`CriticModel`（`:29-44`）：继承 `ZLLMForCausalLM`（复用整个 backbone），但不用 `lm_head`（词表 logits），而是加一个 `value_head`（`Linear(hidden, 1)`，`:38`）。forward 返回 `(B, T)`——每个 token 位置一个价值估计。

> 对应测试 `tests/m10_alignment/test_246_ppo.py:27`（输出 shape `(2,10)`）、`:32`（value_head 是 `Linear(64,1)`）、`:37`（能加载 Actor 的 backbone 权重 `strict=False`）。

### 37.3.2 compute_gae：广义优势估计

> 完整实现见 `zllm/training/ppo.py:47`

```python
def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    B, T = values.shape
    advantages = torch.zeros_like(values)
    last_gae = torch.zeros(B, device=values.device)

    for t in reversed(range(T)):                                # 从后往前递推
        next_val = values[:, t + 1] if t < T - 1 else torch.zeros(B, device=values.device)
        delta = rewards[:, t] + gamma * next_val - values[:, t]  # δ_t
        last_gae = delta + gamma * lam * last_gae               # A_t = δ_t + γλ·A_{t+1}
        advantages[:, t] = last_gae

    advantages = advantages * mask
    # 标准化（mean 0, std 1）
    mean = (advantages * mask).sum() / mask.sum().clamp(min=1)
    var = ((advantages - mean) ** 2 * mask).sum() / mask.sum().clamp(min=1)
    if var.item() > 1e-6:
        advantages = (advantages - mean) * torch.rsqrt(var + 1e-8) * mask

    returns = advantages + values                               # R = A + V
    return advantages, returns
```

`compute_gae`（`:47-81`）：

- **递推**（`:68-72`）：`reversed(range(T))` 从最后一个 token 往前算。$\delta_t = r_t + \gamma V_{t+1} - V_t$（`:70`），$A_t = \delta_t + \gamma\lambda\,A_{t+1}$（`:71`）。最后一个 token 的 $V_{t+1}=0$（`:69`）。
- **标准化**（`:74-78`）：对有效区域（mask=1）做 mean=0、std=1 标准化，稳定训练。
- **returns**（`:80`）：$R_t = A_t + V_t$，给 value loss 用（Critic 要拟合 $R$）。

> 对应测试 `test_246_ppo.py:50`（零奖励零优势）、`:58`（正奖励正优势）、`:67`（$R = A + V$）、`:75`（标准化后 mean ≈ 0）。

### 37.3.3 ppo_policy_loss：Clipped Surrogate

> 完整实现见 `zllm/training/ppo.py:84`

```python
def ppo_policy_loss(new_logps, old_logps, advantages, mask, clip_epsilon=0.2):
    ratio = torch.exp(new_logps - old_logps)                    # π_new / π_old
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    per_token_loss = -torch.min(surr1, surr2)                    # 取更保守的
    return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
```

`ppo_policy_loss`（`:84-103`）：$ratio = e^{\log\pi_{\text{new}} - \log\pi_{\text{old}}}$（`:99`），两个 surrogate 取 min（`:102`）。最后按 mask 求平均。

### 37.3.4 ppo_value_loss：Critic 的 clipped loss

> 完整实现见 `zllm/training/ppo.py:106`

```python
def ppo_value_loss(values, old_values, returns, mask, cliprange=0.2):
    values_clipped = torch.clamp(values, old_values - cliprange, old_values + cliprange)
    loss1 = (values - returns) ** 2
    loss2 = (values_clipped - returns) ** 2
    per_token_loss = 0.5 * torch.max(loss1, loss2)               # 取更大的（更保守）
    return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
```

`ppo_value_loss`（`:106-125`）：Critic 也做 clip——限制 value 在 `old_values ± cliprange` 内，取 `max`（更大的 loss = 更保守）。`0.5` 是 value loss 的惯例系数。

> 对应测试 `test_246_ppo.py:86`（clip 有效）、`:94`（无变化时 loss = $-A$）、`:104`（完美预测 loss=0）、`:112`（不完美 loss>0）。

### 37.3.5 PPOConfig

> 完整实现见 `zllm/training/ppo.py:128`

`PPOConfig`（`:128-157`）：`clip_epsilon=0.2`（`:134`）、`vf_coef=0.5`（value loss 权重，`:135`）、`kl_coef=0.02`（KL 惩罚，`:136`）、`gamma=1.0`（`:137`）、`lam=0.95`（`:138`）、`ppo_update_iters=2`（每批数据更新几次，`:140`）、`early_stop_kl=0.25`（KL 超阈值早停，`:142`）。

## 37.4 对应单元测试

> 对应测试 `tests/m10_alignment/test_246_ppo.py`（129 行）

| 测试类 | 行号 | 验证 |
|--------|------|------|
| TestCriticModel | `:18` | shape `(B,T)` `:27`、value_head `:32`、加载 Actor 权重 `:37` |
| TestGAE | `:49` | 零奖励→零优势 `:50`、正奖励→正优势 `:58`、$R=A+V$ `:67`、标准化 `:75` |
| TestPPOPolicyLoss | `:85` | clip `:86`、无变化=$-A$ `:94` |
| TestPPOValueLoss | `:103` | 完美=0 `:104`、不完美>0 `:112` |
| TestPPOConfig | `:121` | 默认值 `:122` |

## 37.5 动手验证

```bash
pytest tests/m10_alignment/test_246_ppo.py -v
```

预期：全部 PASSED。验证 GAE 递推：

```bash
python -c "
import torch
from zllm.training.ppo import compute_gae
values = torch.zeros(1, 3)
rewards = torch.zeros(1, 3); rewards[0, -1] = 1.0
mask = torch.ones(1, 3)
adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
print('优势:', adv[0].tolist())
print('（最后一个 token 拿到奖励，GAE 把它往前传播）')
"
```

## 37.6 本章小结 + 下章预告

本章要点：

1. **Actor-Critic**：Actor 生成（lm_head）、Critic 估值（value_head），共享 backbone。
2. **GAE**：$A_t = \delta_t + \gamma\lambda A_{t+1}$，$\lambda$ 平衡偏差与方差，从后往前递推。
3. **PPO policy loss**：$\min(ratio \cdot A,\; clip(ratio) \cdot A)$，取更保守的防步子太大。
4. **PPO value loss**：也做 clip，取 max（更大 loss 更保守），系数 0.5。
5. **标准化优势**：GAE 输出做 mean=0/std=1 标准化，稳定训练。

> **一句话带走**：PPO = Actor-Critic + GAE 优势估计 + 双重 clip（policy clip + value clip），是最完整但也最复杂的对齐方法。

**下章预告**：PPO 要 Critic 模型，显存翻倍。能不能去掉 Critic？Ch 38《GRPO + CISPO》——对同一 prompt 生成 N 个回答，组内标准化作为优势，去掉 Critic；外加 CISPO 单边裁剪变体。
