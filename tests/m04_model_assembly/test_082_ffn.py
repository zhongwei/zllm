"""M4-A: SwiGLU FeedForward 测试。

SwiGLU = SiLU(gate(x)) * up(x) → down(...)
参数命名对齐 minimind：gate_proj / up_proj / down_proj。
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.ffn import FeedForward


class TestFeedForwardInit:
    def test_linear_shapes(self, small_config):
        ffn = FeedForward(small_config)
        inter = small_config.intermediate_size
        hidden = small_config.hidden_size
        assert ffn.gate_proj.weight.shape == (inter, hidden)
        assert ffn.up_proj.weight.shape == (inter, hidden)
        assert ffn.down_proj.weight.shape == (hidden, inter)

    def test_no_bias(self, small_config):
        ffn = FeedForward(small_config)
        assert ffn.gate_proj.bias is None
        assert ffn.up_proj.bias is None
        assert ffn.down_proj.bias is None

    def test_act_fn_is_silu(self, small_config):
        ffn = FeedForward(small_config)
        assert callable(ffn.act_fn)

    def test_pi_scaled_intermediate(self):
        import math
        config = ZLLMConfig(hidden_size=768)
        expected_inter = math.ceil(768 * math.pi / 64) * 64
        assert config.intermediate_size == expected_inter
        ffn = FeedForward(config)
        assert ffn.gate_proj.weight.shape[0] == expected_inter


class TestFeedForwardForward:
    def test_output_shape(self, small_config, device):
        ffn = FeedForward(small_config).to(device)
        x = torch.randn(2, 8, small_config.hidden_size, device=device)
        out = ffn(x)
        assert out.shape == (2, 8, small_config.hidden_size)

    def test_swiglu_formula(self, device):
        """手动验证 SwiGLU: down(silu(gate(x)) * up(x))。"""
        config = ZLLMConfig(hidden_size=16, intermediate_size=32)
        ffn = FeedForward(config).to(device)
        x = torch.randn(1, 1, 16, device=device)
        gate = torch.nn.functional.silu(ffn.gate_proj(x))
        up = ffn.up_proj(x)
        expected = ffn.down_proj(gate * up)
        out = ffn(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradients_flow(self, small_config, device):
        ffn = FeedForward(small_config).to(device)
        x = torch.randn(1, 4, small_config.hidden_size, device=device, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        assert ffn.gate_proj.weight.grad is not None
        assert ffn.up_proj.weight.grad is not None
        assert ffn.down_proj.weight.grad is not None

    def test_bfloat16(self, small_config, device):
        ffn = FeedForward(small_config).to(device).to(torch.bfloat16)
        x = torch.randn(1, 4, small_config.hidden_size, dtype=torch.bfloat16, device=device)
        out = ffn(x)
        assert out.dtype == torch.bfloat16

    def test_custom_intermediate_size(self, device):
        config = ZLLMConfig(hidden_size=64, intermediate_size=128)
        ffn = FeedForward(config).to(device)
        x = torch.randn(1, 4, 64, device=device)
        out = ffn(x)
        assert out.shape == (1, 4, 64)
        assert ffn.gate_proj.weight.shape[0] == 128
