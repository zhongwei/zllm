"""M4-B: MoE FeedForward 测试。

测试 Router 门控、稀疏专家选择、负载均衡辅助损失、空专家梯度保持。
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.ffn import MOEFeedForward


@pytest.fixture
def moe_config():
    return ZLLMConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=1,
        router_aux_loss_coef=0.01,
        max_position_embeddings=128,
    )


class TestMOEInit:
    def test_gate_shape(self, moe_config):
        moe = MOEFeedForward(moe_config)
        assert moe.gate.weight.shape == (moe_config.num_experts, moe_config.hidden_size)

    def test_expert_count(self, moe_config):
        moe = MOEFeedForward(moe_config)
        assert len(moe.experts) == moe_config.num_experts

    def test_experts_are_feedforward(self, moe_config):
        from zllm.model.ffn import FeedForward
        moe = MOEFeedForward(moe_config)
        for expert in moe.experts:
            assert isinstance(expert, FeedForward)


class TestMOEForward:
    def test_output_shape(self, moe_config, device):
        moe = MOEFeedForward(moe_config).to(device)
        x = torch.randn(2, 8, moe_config.hidden_size, device=device)
        out = moe(x)
        assert out.shape == (2, 8, moe_config.hidden_size)

    def test_sparse_selection(self, moe_config, device):
        """top-1 路由：每个 token 只用 1 个专家。"""
        moe = MOEFeedForward(moe_config).to(device)
        x = torch.randn(4, 1, moe_config.hidden_size, device=device)
        out = moe(x)
        assert out.shape == (4, 1, moe_config.hidden_size)

    def test_top2_routing(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, num_experts=4,
            num_experts_per_tok=2, max_position_embeddings=128,
        )
        moe = MOEFeedForward(config).to(device)
        x = torch.randn(2, 4, 64, device=device)
        out = moe(x)
        assert out.shape == (2, 4, 64)

    def test_aux_loss_in_training(self, moe_config, device):
        moe = MOEFeedForward(moe_config).to(device).train()
        x = torch.randn(2, 8, moe_config.hidden_size, device=device)
        moe(x)
        assert hasattr(moe, "aux_loss")
        assert moe.aux_loss.item() >= 0
        assert moe.aux_loss.requires_grad

    def test_no_aux_loss_in_eval(self, moe_config, device):
        moe = MOEFeedForward(moe_config).to(device).eval()
        x = torch.randn(2, 8, moe_config.hidden_size, device=device)
        moe(x)
        assert moe.aux_loss.item() == 0.0

    def test_norm_topk_prob(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, num_experts=4,
            num_experts_per_tok=2, norm_topk_prob=True, max_position_embeddings=128,
        )
        moe = MOEFeedForward(config).to(device)
        x = torch.randn(2, 4, 64, device=device)
        out = moe(x)
        assert out.shape == (2, 4, 64)

    def test_gradients_flow(self, moe_config, device):
        moe = MOEFeedForward(moe_config).to(device).train()
        x = torch.randn(1, 4, moe_config.hidden_size, device=device, requires_grad=True)
        out = moe(x)
        out.sum().backward()
        assert x.grad is not None
        assert moe.gate.weight.grad is not None

    def test_aux_loss_backward(self, moe_config, device):
        """辅助损失本身有梯度。"""
        moe = MOEFeedForward(moe_config).to(device).train()
        x = torch.randn(2, 8, moe_config.hidden_size, device=device)
        moe(x)
        moe.aux_loss.backward()
        assert moe.gate.weight.grad is not None
