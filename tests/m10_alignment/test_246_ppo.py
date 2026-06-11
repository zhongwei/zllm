"""M10-C: PPO 测试 — CriticModel, GAE, clipped loss, value loss, PPOConfig。

PPO 是最复杂的对齐方法：
- 双模型：Actor (ZLLMForCausalLM) + Critic (CriticModel)
- 参考模型 (ref) 用于 KL 惩罚
- GAE 优势估计
- Clipped surrogate loss + Value function loss
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.ppo import CriticModel, compute_gae, ppo_policy_loss, ppo_value_loss, PPOConfig


class TestCriticModel:
    @pytest.fixture
    def critic(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=100, max_position_embeddings=32,
        )
        return CriticModel(config).to(device)

    def test_output_shape(self, critic, device):
        input_ids = torch.randint(0, 100, (2, 10), device=device)
        values = critic(input_ids)
        assert values.shape == (2, 10)

    def test_value_head_is_linear(self, critic):
        assert hasattr(critic, "value_head")
        assert critic.value_head.out_features == 1
        assert critic.value_head.in_features == 64

    def test_load_base_weights(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=100, max_position_embeddings=32,
        )
        model = ZLLMForCausalLM(config).to(device)
        critic = CriticModel(config).to(device)
        critic.load_state_dict(model.state_dict(), strict=False)
        shared = sum(1 for k in model.state_dict() if k in critic.state_dict())
        assert shared > 0


class TestGAE:
    def test_zero_advantage_with_zero_reward(self):
        B, T = 2, 5
        values = torch.zeros(B, T)
        rewards = torch.zeros(B, T)
        mask = torch.ones(B, T)
        advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5)

    def test_positive_reward_positive_advantage(self):
        B, T = 1, 3
        values = torch.zeros(B, T)
        rewards = torch.zeros(B, T)
        rewards[0, -1] = 1.0
        mask = torch.ones(B, T)
        advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
        assert advantages[0, -1] > 0

    def test_returns_equal_advantages_plus_values(self):
        B, T = 2, 8
        values = torch.randn(B, T)
        rewards = torch.randn(B, T)
        mask = torch.ones(B, T)
        advantages, returns = compute_gae(rewards, values, mask, gamma=0.99, lam=0.95)
        assert torch.allclose(returns, advantages + values, atol=1e-5)

    def test_normalized_advantages(self):
        B, T = 4, 10
        values = torch.randn(B, T)
        rewards = torch.randn(B, T)
        mask = torch.ones(B, T)
        advantages, _ = compute_gae(rewards, values, mask, gamma=0.99, lam=0.95)
        mean = (advantages * mask).sum() / mask.sum()
        assert abs(mean.item()) < 0.5


class TestPPOPolicyLoss:
    def test_clipping_works(self):
        old_logp = torch.tensor([[-1.0]])
        new_logp = torch.tensor([[-0.5]])
        advantages = torch.tensor([[1.0]])
        mask = torch.ones(1, 1)
        loss = ppo_policy_loss(new_logp, old_logp, advantages, mask, clip_epsilon=0.2)
        assert not torch.isnan(loss)

    def test_no_change_gives_unclipped_advantage(self):
        old_logp = torch.tensor([[-1.0, -1.0]])
        new_logp = old_logp.clone()
        advantages = torch.tensor([[1.0, 1.0]])
        mask = torch.ones(1, 2)
        loss = ppo_policy_loss(new_logp, old_logp, advantages, mask, clip_epsilon=0.2)
        assert torch.allclose(loss, -advantages.mean(), atol=1e-5)


class TestPPOValueLoss:
    def test_perfect_prediction_zero_loss(self):
        values = torch.tensor([[1.0, 2.0, 3.0]])
        old_values = values.clone()
        returns = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones(1, 3)
        loss = ppo_value_loss(values, old_values, returns, mask, cliprange=0.2)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_imperfect_prediction_positive_loss(self):
        values = torch.tensor([[1.0, 2.0]])
        old_values = values.clone()
        returns = torch.tensor([[3.0, 4.0]])
        mask = torch.ones(1, 2)
        loss = ppo_value_loss(values, old_values, returns, mask, cliprange=0.2)
        assert loss.item() > 0


class TestPPOConfig:
    def test_defaults(self):
        cfg = PPOConfig()
        assert cfg.clip_epsilon == 0.2
        assert cfg.gamma == 1.0
        assert cfg.lam == 0.95
        assert cfg.vf_coef == 0.5
        assert cfg.kl_coef == 0.02
        assert cfg.learning_rate == 3e-7
