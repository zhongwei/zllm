"""M3-D: GQA Attention 测试。

测试 repeat_kv + Attention（含 GQA、QK-Norm、Flash Attention、Causal Mask、KV Cache）。
参数命名对齐 minimind：q_proj/k_proj/v_proj/o_proj/q_norm/k_norm。
"""

import math
import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.attention import Attention, repeat_kv


class TestRepeatKV:
    def test_no_repeat(self):
        x = torch.randn(2, 8, 4, 16)
        out = repeat_kv(x, n_rep=1)
        assert torch.equal(out, x)

    def test_repeat_2x(self):
        x = torch.randn(2, 8, 4, 16)
        out = repeat_kv(x, n_rep=2)
        assert out.shape == (2, 8, 8, 16)

    def test_repeat_preserves_values(self):
        x = torch.randn(1, 4, 2, 8)
        out = repeat_kv(x, n_rep=3)
        # 每个 kv head 被复制 3 次
        for i in range(2):
            for j in range(3):
                assert torch.equal(out[:, :, i * 3 + j, :], x[:, :, i, :])


class TestAttentionInit:
    def test_linear_shapes(self, small_config):
        attn = Attention(small_config)
        head_dim = small_config.head_dim
        assert attn.q_proj.weight.shape == (small_config.num_attention_heads * head_dim, small_config.hidden_size)
        assert attn.k_proj.weight.shape == (small_config.num_key_value_heads * head_dim, small_config.hidden_size)
        assert attn.v_proj.weight.shape == (small_config.num_key_value_heads * head_dim, small_config.hidden_size)
        assert attn.o_proj.weight.shape == (small_config.hidden_size, small_config.num_attention_heads * head_dim)

    def test_no_bias(self, small_config):
        attn = Attention(small_config)
        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None
        assert attn.v_proj.bias is None
        assert attn.o_proj.bias is None

    def test_n_rep(self, small_config):
        attn = Attention(small_config)
        assert attn.n_rep == small_config.num_attention_heads // small_config.num_key_value_heads

    def test_qk_norm_exists(self, small_config):
        attn = Attention(small_config)
        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")
        assert attn.q_norm.weight.shape == (small_config.head_dim,)
        assert attn.k_norm.weight.shape == (small_config.head_dim,)


class TestAttentionForward:
    def test_output_shape(self, small_config, device):
        attn = Attention(small_config).to(device)
        x = torch.randn(2, 8, small_config.hidden_size, device=device)
        cos, sin = torch.randn(8, small_config.head_dim, device=device), torch.randn(8, small_config.head_dim, device=device)
        out, past_kv = attn(x, (cos, sin))
        assert out.shape == (2, 8, small_config.hidden_size)

    def test_past_kv_none_by_default(self, small_config, device):
        attn = Attention(small_config).to(device)
        x = torch.randn(1, 4, small_config.hidden_size, device=device)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        _, past_kv = attn(x, (cos, sin))
        assert past_kv is None

    def test_use_cache_returns_kv(self, small_config, device):
        attn = Attention(small_config).to(device)
        x = torch.randn(1, 4, small_config.hidden_size, device=device)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        _, past_kv = attn(x, (cos, sin), use_cache=True)
        assert past_kv is not None
        k, v = past_kv
        assert k.shape == (1, 4, small_config.num_key_value_heads, small_config.head_dim)
        assert v.shape == (1, 4, small_config.num_key_value_heads, small_config.head_dim)

    def test_kv_cache_concatenation(self, small_config, device):
        attn = Attention(small_config).to(device)
        x1 = torch.randn(1, 4, small_config.hidden_size, device=device)
        cos = torch.randn(8, small_config.head_dim, device=device)
        sin = torch.randn(8, small_config.head_dim, device=device)
        _, past_kv = attn(x1, (cos[:4], sin[:4]), use_cache=True)
        x2 = torch.randn(1, 2, small_config.hidden_size, device=device)
        _, past_kv2 = attn(x2, (cos[4:6], sin[4:6]), past_key_value=past_kv, use_cache=True)
        k, v = past_kv2
        assert k.shape == (1, 6, small_config.num_key_value_heads, small_config.head_dim)

    def test_causal_mask_applied(self, small_config, device):
        attn = Attention(small_config).to(device)
        attn.flash = False
        x = torch.randn(1, 4, small_config.hidden_size, device=device)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        out, _ = attn(x, (cos, sin))
        assert out.shape == (1, 4, small_config.hidden_size)

    def test_flash_path(self, small_config, device):
        attn = Attention(small_config).to(device)
        if not attn.flash:
            pytest.skip("Flash attention not available")
        x = torch.randn(1, 4, small_config.hidden_size, device=device)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        out, _ = attn(x, (cos, sin))
        assert out.shape == (1, 4, small_config.hidden_size)

    def test_flash_vs_manual_close(self, small_config, device):
        attn = Attention(small_config).to(device)
        if not attn.flash:
            pytest.skip("Flash attention not available")
        torch.manual_seed(42)
        x = torch.randn(1, 4, small_config.hidden_size, device=device)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        out_flash, _ = attn(x.clone(), (cos, sin))
        attn.flash = False
        out_manual, _ = attn(x.clone(), (cos, sin))
        assert torch.allclose(out_flash.float(), out_manual.float(), atol=1e-3)

    def test_gradients_flow(self, small_config, device):
        attn = Attention(small_config).to(device)
        x = torch.randn(1, 4, small_config.hidden_size, device=device, requires_grad=True)
        cos = torch.randn(4, small_config.head_dim, device=device)
        sin = torch.randn(4, small_config.head_dim, device=device)
        out, _ = attn(x, (cos, sin))
        out.sum().backward()
        assert x.grad is not None
        assert attn.q_proj.weight.grad is not None
