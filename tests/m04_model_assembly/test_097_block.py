"""M4-C: Transformer Block 测试。

Pre-Norm 架构：input_layernorm → Attention → residual → post_attention_layernorm → FFN/MoE → residual。
参数命名对齐 minimind：self_attn / input_layernorm / post_attention_layernorm / mlp。
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.block import ZLLMBlock


class TestBlockInit:
    def test_has_self_attn(self, small_config):
        block = ZLLMBlock(0, small_config)
        assert hasattr(block, "self_attn")

    def test_has_norms(self, small_config):
        block = ZLLMBlock(0, small_config)
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")

    def test_mlp_is_feedforward(self, small_config):
        from zllm.model.ffn import FeedForward
        block = ZLLMBlock(0, small_config)
        assert isinstance(block.mlp, FeedForward)

    def test_mlp_is_moe_when_configured(self):
        from zllm.model.ffn import MOEFeedForward
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, max_position_embeddings=128,
        )
        block = ZLLMBlock(0, config)
        assert isinstance(block.mlp, MOEFeedForward)


class TestBlockForward:
    def test_output_shape(self, small_config, device):
        block = ZLLMBlock(0, small_config).to(device)
        x = torch.randn(2, 8, small_config.hidden_size, device=device)
        cos = torch.randn(8, small_config.head_dim, device=device)
        sin = torch.randn(8, small_config.head_dim, device=device)
        out, present = block(x, (cos, sin))
        assert out.shape == (2, 8, small_config.hidden_size)

    def test_residual_connection(self, small_config, device):
        block = ZLLMBlock(0, small_config).to(device)
        x = torch.randn(1, 4, small_config.hidden_size, device=device)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        out, _ = block(x, (cos, sin))
        # 输出应与输入不同（经过变换）
        assert not torch.allclose(out, x)

    def test_use_cache(self, small_config, device):
        block = ZLLMBlock(0, small_config).to(device)
        x = torch.randn(1, 4, small_config.hidden_size, device=device)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        _, present = block(x, (cos, sin), use_cache=True)
        assert present is not None

    def test_no_cache_by_default(self, small_config, device):
        block = ZLLMBlock(0, small_config).to(device)
        x = torch.randn(1, 4, small_config.hidden_size, device=device)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        _, present = block(x, (cos, sin))
        assert present is None

    def test_moe_block_forward(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, max_position_embeddings=128,
        )
        block = ZLLMBlock(0, config).to(device)
        x = torch.randn(1, 4, 64, device=device)
        cos = torch.randn(4, config.head_dim, device=device)
        sin = torch.randn(4, config.head_dim, device=device)
        out, _ = block(x, (cos, sin))
        assert out.shape == (1, 4, 64)

    def test_gradients_flow(self, small_config, device):
        block = ZLLMBlock(0, small_config).to(device)
        x = torch.randn(1, 4, small_config.hidden_size, device=device, requires_grad=True)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        out, _ = block(x, (cos, sin))
        out.sum().backward()
        assert x.grad is not None
