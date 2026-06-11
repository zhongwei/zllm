"""GRPO（Group Relative Policy Optimization）训练。

GRPO vs PPO 核心差异：
- 无 Critic 模型，用群体相对优势代替绝对优势
- 对同一 prompt 生成 num_generations 个 response
- 组内标准化 reward → advantages = (reward - mean) / std
- KL 惩罚防止偏离 reference model

Loss 类型：
- grpo_loss: 标准 PPO clip (ratio ∈ [1-ε, 1+ε])
- cispo_loss: 单边裁剪 (ratio ≤ ε_high，无下界)

函数：
- per_token_kl: 计算 token 级 KL 散度
- compute_group_advantages: 群体相对优势
- grpo_loss / cispo_loss: 两种 loss 实现
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


def per_token_kl(ref_log_probs, policy_log_probs):
    """计算 token 级 KL 散度：KL(ref || policy)。

    per_token_kl = exp(ref - policy) - (ref - policy) - 1

    Args:
        ref_log_probs: (batch, seq_len)
        policy_log_probs: (batch, seq_len)

    Returns:
        kl: (batch, seq_len)
    """
    kl = torch.exp(ref_log_probs - policy_log_probs) - (ref_log_probs - policy_log_probs) - 1
    return kl


def compute_group_advantages(rewards, num_generations):
    """计算群体相对优势。

    对同一 prompt 的多个 response 组内标准化：
    advantages = (reward - group_mean) / (group_std + 1e-4)

    Args:
        rewards: (B * num_gen,) 所有 response 的 reward
        num_generations: 每个 prompt 的生成数

    Returns:
        advantages: (B * num_gen,) 标准化优势
    """
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1).repeat_interleave(num_generations)
    std = grouped.std(dim=1, unbiased=False).repeat_interleave(num_generations)
    return (rewards - mean) / (std + 1e-4)


def grpo_loss(policy_logps, old_logps, advantages, mask, ref_logps, beta=0.1, epsilon=0.2):
    """标准 GRPO loss（PPO clip）。

    ratio = exp(policy_logps - old_logps)
    loss = -(min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv) - β * KL)

    Args:
        policy_logps: (batch, seq_len) 当前策略的 log 概率
        old_logps: (batch, seq_len) rollout 时的 log 概率
        advantages: (batch, 1) 或 (batch, seq_len) 标准化优势
        mask: (batch, seq_len) completion mask
        ref_logps: (batch, seq_len) reference 的 log 概率
        beta: KL 惩罚系数
        epsilon: clip 范围

    Returns:
        scalar loss
    """
    ratio = torch.exp(policy_logps - old_logps)
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    kl = per_token_kl(ref_logps, policy_logps)
    per_token_loss = -(torch.min(surr1, surr2) - beta * kl)
    return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()


def cispo_loss(policy_logps, old_logps, advantages, mask, ref_logps, beta=0.1, epsilon_high=5.0):
    """CISPO loss（单边裁剪）。

    只裁上界：ratio ≤ ε_high，不裁下界。
    允许低概率 token 进一步降低（鼓励修剪低质量内容）。

    Args:
        同 grpo_loss，但 epsilon_high 代替 epsilon

    Returns:
        scalar loss
    """
    ratio = torch.exp(policy_logps - old_logps)
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()
    kl = per_token_kl(ref_logps, policy_logps)
    per_token_loss = -(clamped_ratio * advantages * policy_logps - beta * kl)
    return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()


@dataclass
class GRPOConfig:
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 3e-7
    beta: float = 0.1
    epsilon: float = 0.2
    epsilon_high: float = 5.0
    loss_type: str = "cispo"
    num_generations: int = 6
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
    save_weight: str = "grpo"
    from_weight: str = "full_sft"
    from_resume: bool = False
    device: str = "cuda"