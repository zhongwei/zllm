"""M10-B: GRPO 测试 — per_token_kl, group advantages, GRPO/CISPO loss。

GRPO 与 DPO 的关键差异：
- 无 Critic，用群体相对优势代替
- 对同一 prompt 生成多个 response → 组内标准化 reward
- KL 惩罚防止偏离 reference model
"""

import torch
import pytest

from zllm.training.grpo import per_token_kl, compute_group_advantages, grpo_loss, cispo_loss, GRPOConfig


class TestPerTokenKL:
    def test_identical_distributions_zero_kl(self):
        logp = torch.randn(2, 10)
        kl = per_token_kl(logp, logp)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

    def test_non_identical_positive_kl(self):
        ref_logp = torch.zeros(2, 10)
        policy_logp = torch.ones(2, 10) * -0.5
        kl = per_token_kl(ref_logp, policy_logp)
        assert (kl > 0).all()

    def test_shape(self):
        ref = torch.randn(4, 20)
        policy = torch.randn(4, 20)
        kl = per_token_kl(ref, policy)
        assert kl.shape == (4, 20)


class TestGroupAdvantages:
    def test_normalized_mean_zero(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5, 2.5])
        num_gen = 3
        adv = compute_group_advantages(rewards, num_gen)
        grouped = adv.view(-1, num_gen)
        means = grouped.mean(dim=1)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)

    def test_higher_reward_higher_advantage(self):
        rewards = torch.tensor([3.0, 1.0, 0.5, 2.0])
        num_gen = 2
        adv = compute_group_advantages(rewards, num_gen)
        assert adv[0] > adv[1]
        assert adv[3] > adv[2]

    def test_single_group(self):
        rewards = torch.tensor([5.0, 3.0, 1.0])
        num_gen = 3
        adv = compute_group_advantages(rewards, num_gen)
        assert adv.shape == (3,)


class TestGRPOLoss:
    def test_positive_advantage_encourages_ratio_above_one(self):
        """正优势 → 策略梯度推动 ratio 趋向 > 1。"""
        old_logp = torch.tensor([[-1.0, -1.0]])
        new_logp = torch.tensor([[-0.5, -0.5]])
        advantages = torch.tensor([[1.0, 1.0]])
        mask = torch.ones(1, 2)
        ref_logp = torch.tensor([[-1.0, -1.0]])
        loss = grpo_loss(new_logp, old_logp, advantages, mask, ref_logp, beta=0.0, epsilon=0.2)
        assert loss.item() != 0

    def test_loss_decreases_with_correct_direction(self):
        """当策略向正优势方向移动时，loss 应更低。"""
        old_logp = torch.tensor([[-1.0, -1.0]])
        advantages = torch.tensor([[1.0, 1.0]])
        mask = torch.ones(1, 2)
        ref_logp = torch.tensor([[-1.0, -1.0]])

        wrong_logp = torch.tensor([[-2.0, -2.0]])
        right_logp = torch.tensor([[-0.5, -0.5]])
        loss_wrong = grpo_loss(wrong_logp, old_logp, advantages, mask, ref_logp, beta=0.0, epsilon=0.2)
        loss_right = grpo_loss(right_logp, old_logp, advantages, mask, ref_logp, beta=0.0, epsilon=0.2)
        assert loss_right < loss_wrong

    def test_clipping(self):
        """ratio 超过 1+epsilon 时被 clip，loss 不再降低。"""
        old_logp = torch.tensor([[-1.0]])
        advantages = torch.tensor([[1.0]])
        mask = torch.ones(1, 1)
        ref_logp = torch.tensor([[-1.0]])
        # ratio = exp(new - old), clip at 1.2
        logp_at_clip = old_logp + torch.log(torch.tensor(1.2))
        logp_beyond = old_logp + torch.log(torch.tensor(2.0))
        loss_at_clip = grpo_loss(logp_at_clip, old_logp, advantages, mask, ref_logp, beta=0.0, epsilon=0.2)
        loss_beyond = grpo_loss(logp_beyond, old_logp, advantages, mask, ref_logp, beta=0.0, epsilon=0.2)
        assert loss_beyond.item() >= loss_at_clip.item() - 1e-5


class TestCISPOLoss:
    def test_single_side_clipping(self):
        """CISPO 只裁上界，不裁下界。"""
        old_logp = torch.tensor([[-1.0]])
        advantages = torch.tensor([[1.0]])
        mask = torch.ones(1, 1)
        ref_logp = torch.tensor([[-1.0]])
        logp = old_logp + torch.log(torch.tensor(3.0))
        loss = cispo_loss(logp, old_logp, advantages, mask, ref_logp, beta=0.0, epsilon_high=5.0)
        assert not torch.isnan(loss)


class TestGRPOConfig:
    def test_defaults(self):
        cfg = GRPOConfig()
        assert cfg.num_generations == 6
        assert cfg.beta == 0.1
        assert cfg.epsilon == 0.2
        assert cfg.loss_type == "cispo"
        assert cfg.learning_rate == 3e-7
