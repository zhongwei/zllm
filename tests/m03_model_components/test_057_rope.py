"""M3-B: RoPE 位置编码测试。

TDD 红灯：测试 precompute_freqs_cis 和 apply_rotary_pos_emb。
"""

import math
import torch
import pytest

from zllm.model.rope import precompute_freqs_cis, apply_rotary_pos_emb, rotate_half


class TestPrecomputeFreqsCis:
    def test_output_shapes(self):
        cos, sin = precompute_freqs_cis(dim=16, end=128)
        assert cos.shape == (128, 16)
        assert sin.shape == (128, 16)

    def test_position_0_cos_is_one(self):
        cos, sin = precompute_freqs_cis(dim=16, end=10)
        assert torch.allclose(cos[0].float(), torch.ones(16), atol=1e-5)

    def test_position_0_sin_is_zero(self):
        cos, sin = precompute_freqs_cis(dim=16, end=10)
        assert torch.allclose(sin[0].float(), torch.zeros(16), atol=1e-5)

    def test_freqs_are_harmonic(self):
        """频率应该是 1/θ^(2k/d)，递减。"""
        cos, sin = precompute_freqs_cis(dim=16, end=100, rope_base=10000.0)
        # 位置 1 的值应该较小
        assert cos[1].abs().sum() < cos[0].abs().sum()

    def test_rope_base_affects_period(self):
        """更大的 rope_base → 更长的周期 → 位置 1 更接近 1。"""
        cos1, _ = precompute_freqs_cis(dim=16, end=10, rope_base=1e6)
        cos2, _ = precompute_freqs_cis(dim=16, end=10, rope_base=1e4)
        # rope_base=1e6 的频率更低，位置 1 更接近 1
        assert cos1[1].float().mean() > cos2[1].float().mean()

    def test_cos_sin_concatenated(self):
        """cos/sin 的后半段应与前半段相同（cat 复制）。"""
        cos, sin = precompute_freqs_cis(dim=8, end=4)
        half = 4
        assert torch.allclose(cos[:, :half], cos[:, half:], atol=1e-6)
        assert torch.allclose(sin[:, :half], sin[:, half:], atol=1e-6)

    def test_default_end_is_32768(self):
        cos, sin = precompute_freqs_cis(dim=8)
        assert cos.shape[0] == 32768


class TestRotateHalf:
    def test_basic(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = rotate_half(x)
        # rotate_half: cat(-x[..., d//2:], x[..., :d//2])
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(result, expected)

    def test_shape_preserved(self):
        x = torch.randn(2, 4, 8, 16)
        assert rotate_half(x).shape == x.shape


class TestApplyRotaryPosEmb:
    def test_output_shapes(self, device):
        bsz, seq_len, heads, head_dim = 2, 8, 4, 16
        q = torch.randn(bsz, seq_len, heads, head_dim, device=device)
        k = torch.randn(bsz, seq_len, heads, head_dim, device=device)
        cos, sin = precompute_freqs_cis(dim=head_dim, end=seq_len, rope_base=1e6)
        cos, sin = cos.to(device), sin.to(device)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_position_0_is_identity(self, device):
        """位置 0 时 cos=1, sin=0 → 输出 = 输入。"""
        bsz, heads, head_dim = 2, 4, 16
        q = torch.randn(bsz, 1, heads, head_dim, device=device)
        k = torch.randn(bsz, 1, heads, head_dim, device=device)
        cos, sin = precompute_freqs_cis(dim=head_dim, end=1)
        cos, sin = cos.to(device), sin.to(device)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert torch.allclose(q_out.float(), q.float(), atol=1e-5)
        assert torch.allclose(k_out.float(), k.float(), atol=1e-5)

    def test_different_positions_give_different_output(self, device):
        bsz, heads, head_dim = 1, 2, 8
        q = torch.randn(bsz, 1, heads, head_dim, device=device)
        k = torch.randn(bsz, 1, heads, head_dim, device=device)
        cos1, sin1 = precompute_freqs_cis(dim=head_dim, end=10)
        q1, k1 = apply_rotary_pos_emb(q, k, cos1[0:1].to(device), sin1[0:1].to(device))
        q2, k2 = apply_rotary_pos_emb(q, k, cos1[5:6].to(device), sin1[5:6].to(device))
        assert not torch.allclose(q1.float(), q2.float())

    def test_unsqueeze_dim(self, device):
        """unsqueeze_dim 控制在哪个维度插入广播维度。"""
        bsz, seq_len, heads, head_dim = 2, 8, 4, 16
        q = torch.randn(bsz, seq_len, heads, head_dim, device=device)
        k = torch.randn(bsz, seq_len, heads, head_dim, device=device)
        cos, sin = precompute_freqs_cis(dim=head_dim, end=seq_len)
        cos, sin = cos.to(device), sin.to(device)
        q1, k1 = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        q2, k2 = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        assert torch.allclose(q1, q2)

    def test_gradient_flows(self, device):
        bsz, seq_len, heads, head_dim = 1, 4, 2, 8
        q = torch.randn(bsz, seq_len, heads, head_dim, device=device, requires_grad=True)
        k = torch.randn(bsz, seq_len, heads, head_dim, device=device, requires_grad=True)
        cos, sin = precompute_freqs_cis(dim=head_dim, end=seq_len)
        cos, sin = cos.to(device), sin.to(device)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        q_out.sum().backward()
        assert q.grad is not None
