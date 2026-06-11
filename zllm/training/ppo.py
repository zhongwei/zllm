"""PPO（Proximal Policy Optimization）训练。

PPO vs GRPO 差异：
- PPO 有 Critic 模型估计价值函数 → GAE 优势估计
- GRPO 无 Critic，用群体相对优势

核心组件：
- CriticModel: 继承 ZLLMForCausalLM，替换 lm_head 为 value_head
- compute_gae: 广义优势估计
- ppo_policy_loss: Clipped surrogate loss
- ppo_value_loss: Value function clipped loss

训练流程：
1. Rollout: 生成 response → 计算 reward
2. 计算 GAE 优势和 returns
3. Mini-batch PPO 更新（多次）
4. 早停：approx_kl > threshold
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn

from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.config import ZLLMConfig


class CriticModel(ZLLMForCausalLM):
    """价值网络：共享 backbone，替换 lm_head 为 value_head。

    输入 input_ids → 输出每个位置的价值估计 (batch, seq_len)。
    """

    def __init__(self, config=None):
        self.config = config or ZLLMConfig()
        super().__init__(self.config)
        self.value_head = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """广义优势估计（GAE）。

    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

    Args:
        rewards: (batch, seq_len) token 级奖励（通常只在最后一个 token 非零）
        values: (batch, seq_len) Critic 的价值估计
        mask: (batch, seq_len) response mask
        gamma: 折扣因子
        lam: GAE λ

    Returns:
        advantages: (batch, seq_len) 标准化优势
        returns: (batch, seq_len) = advantages + values
    """
    B, T = values.shape
    advantages = torch.zeros_like(values)
    last_gae = torch.zeros(B, device=values.device)

    for t in reversed(range(T)):
        next_val = values[:, t + 1] if t < T - 1 else torch.zeros(B, device=values.device)
        delta = rewards[:, t] + gamma * next_val - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae

    advantages = advantages * mask
    mean = (advantages * mask).sum() / mask.sum().clamp(min=1)
    var = ((advantages - mean) ** 2 * mask).sum() / mask.sum().clamp(min=1)
    if var.item() > 1e-6:
        advantages = (advantages - mean) * torch.rsqrt(var + 1e-8) * mask

    returns = advantages + values
    return advantages, returns


def ppo_policy_loss(new_logps, old_logps, advantages, mask, clip_epsilon=0.2):
    """PPO Clipped Surrogate Loss。

    ratio = exp(new_logp - old_logp)
    L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    Args:
        new_logps: (batch, seq_len) 当前策略的 log 概率
        old_logps: (batch, seq_len) 旧策略的 log 概率
        advantages: (batch, seq_len) GAE 优势
        mask: (batch, seq_len) response mask

    Returns:
        scalar loss
    """
    ratio = torch.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    per_token_loss = -torch.min(surr1, surr2)
    return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()


def ppo_value_loss(values, old_values, returns, mask, cliprange=0.2):
    """PPO Value Function Clipped Loss。

    L = 0.5 * max((V - R)², (clip(V, V_old±c) - R)²)

    Args:
        values: (batch, seq_len) 当前价值估计
        old_values: (batch, seq_len) 旧价值估计
        returns: (batch, seq_len) GAE returns
        mask: (batch, seq_len) response mask
        cliprange: 裁剪范围

    Returns:
        scalar loss
    """
    values_clipped = torch.clamp(values, old_values - cliprange, old_values + cliprange)
    loss1 = (values - returns) ** 2
    loss2 = (values_clipped - returns) ** 2
    per_token_loss = 0.5 * torch.max(loss1, loss2)
    return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()


@dataclass
class PPOConfig:
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 3e-7
    critic_learning_rate: float = 5e-7
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    kl_coef: float = 0.02
    gamma: float = 1.0
    lam: float = 0.95
    cliprange_value: float = 0.2
    ppo_update_iters: int = 2
    mini_batch_size: int = 2
    early_stop_kl: float = 0.25
    max_gen_len: int = 1024
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    max_seq_len: int = 768
    dtype: str = "bfloat16"
    num_workers: int = 1
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    save_weight: str = "ppo_actor"
    from_weight: str = "full_sft"
    from_resume: bool = False
    device: str = "cuda"