"""M3 集成测试：RMSNorm → RoPE → Attention 端到端。

用真实配置走完整 forward pass：embed → norm → rope → attention。
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.norms import RMSNorm
from zllm.model.rope import precompute_freqs_cis
from zllm.model.attention import Attention


class TestM3Integration:
    def test_full_pipeline(self, small_config, device):
        seq_len = 16
        hidden = small_config.hidden_size
        head_dim = small_config.head_dim

        x = torch.randn(2, seq_len, hidden, device=device)
        norm = RMSNorm(hidden, eps=small_config.rms_norm_eps).to(device)
        attn = Attention(small_config).to(device)

        cos, sin = precompute_freqs_cis(
            dim=head_dim,
            end=seq_len,
            rope_base=small_config.rope_theta,
        )
        cos, sin = cos.to(device), sin.to(device)

        x_normed = norm(x)
        out, _ = attn(x_normed, (cos, sin))

        assert out.shape == (2, seq_len, hidden)
        assert torch.all(torch.isfinite(out))

    def test_pipeline_with_cache(self, small_config, device):
        hidden = small_config.hidden_size
        head_dim = small_config.head_dim

        cos, sin = precompute_freqs_cis(dim=head_dim, end=32, rope_base=small_config.rope_theta)
        cos, sin = cos.to(device), sin.to(device)

        norm = RMSNorm(hidden, eps=small_config.rms_norm_eps).to(device)
        attn = Attention(small_config).to(device)

        x1 = torch.randn(1, 8, hidden, device=device)
        x1 = norm(x1)
        _, past_kv = attn(x1, (cos[:8], sin[:8]), use_cache=True)

        x2 = torch.randn(1, 4, hidden, device=device)
        x2 = norm(x2)
        out2, past_kv2 = attn(x2, (cos[8:12], sin[8:12]), past_key_value=past_kv, use_cache=True)

        assert out2.shape == (1, 4, hidden)
        k, v = past_kv2
        assert k.shape == (1, 12, small_config.num_key_value_heads, head_dim)

    def test_pipeline_with_yarn(self, small_config, device):
        head_dim = small_config.head_dim
        yarn_config = {
            "original_max_position_embeddings": 64,
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.0,
            "type": "yarn",
        }
        cos, sin = precompute_freqs_cis(
            dim=head_dim, end=small_config.max_position_embeddings,
            rope_base=small_config.rope_theta, rope_scaling=yarn_config,
        )
        cos, sin = cos.to(device), sin.to(device)

        norm = RMSNorm(small_config.hidden_size).to(device)
        attn = Attention(small_config).to(device)

        x = torch.randn(1, 8, small_config.hidden_size, device=device)
        out, _ = attn(norm(x), (cos[:8], sin[:8]))
        assert out.shape == (1, 8, small_config.hidden_size)

    def test_default_config_pipeline(self, default_config, device):
        """生产配置也能跑通。"""
        seq_len = 4
        head_dim = default_config.head_dim
        hidden = default_config.hidden_size

        cos, sin = precompute_freqs_cis(dim=head_dim, end=seq_len, rope_base=default_config.rope_theta)
        cos, sin = cos.to(device), sin.to(device)

        norm = RMSNorm(hidden, eps=default_config.rms_norm_eps).to(device)
        attn = Attention(default_config).to(device)

        x = torch.randn(1, seq_len, hidden, device=device)
        out, _ = attn(norm(x), (cos, sin))
        assert out.shape == (1, seq_len, hidden)
