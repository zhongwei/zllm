"""M3-A: RMSNorm 测试。

TDD 红灯：测试 RMSNorm 的数学正确性、数值稳定性、参数形状。
"""

import torch
import pytest

from zllm.model.norms import RMSNorm


class TestRMSNormInit:
    def test_weight_shape(self):
        norm = RMSNorm(64)
        assert norm.weight.shape == (64,)

    def test_weight_init_ones(self):
        norm = RMSNorm(64)
        assert torch.allclose(norm.weight, torch.ones(64))

    def test_eps_default(self):
        norm = RMSNorm(64)
        assert norm.eps == 1e-5

    def test_eps_custom(self):
        norm = RMSNorm(64, eps=1e-6)
        assert norm.eps == 1e-6


class TestRMSNormForward:
    def test_output_shape(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 10, 64, device=device)
        out = norm(x)
        assert out.shape == x.shape

    def test_identity_when_weight_ones(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 10, 64, device=device)
        out = norm(x)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
        assert torch.allclose(out.float(), expected.float(), atol=1e-5)

    def test_scaling_with_weight(self, device):
        norm = RMSNorm(64).to(device)
        norm.weight.data.fill_(2.0)
        x = torch.ones(2, 10, 64, device=device)
        out = norm(x)
        assert torch.allclose(out.float(), torch.full_like(x, 2.0), atol=1e-5)

    def test_zero_input(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.zeros(2, 10, 64, device=device)
        out = norm(x)
        assert torch.all(torch.isfinite(out))

    def test_bfloat16_preserved(self, device):
        if device.type == "cpu":
            pytest.skip("bfloat16 on CPU may vary")
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16, device=device)
        out = norm(x)
        assert out.dtype == torch.bfloat16

    def test_float32_internal_compute(self, device):
        """内部计算在 float32 下进行，避免低精度溢出。"""
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 10, 64, dtype=torch.float16, device=device)
        out = norm(x)
        assert out.dtype == torch.float16

    def test_gradients_flow(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 10, 64, device=device, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_norm_method(self, device):
        """norm() 方法只做归一化（不乘 weight）。"""
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 10, 64, device=device)
        normalized = norm.norm(x.float())
        rms = x.float().pow(2).mean(-1, keepdim=True).sqrt()
        expected = x.float() / (rms + norm.eps)
        assert torch.allclose(normalized, expected, atol=1e-5)
