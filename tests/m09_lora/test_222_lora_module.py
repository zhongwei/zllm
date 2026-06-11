"""M9-A: LoRA 类实现测试。

验证：
1. LoRA 模块结构（A 降维, B 升维）
2. 初始化策略（A 高斯, B 零）
3. 前向传播 B(A(x))
4. apply_lora 注入到方阵 Linear
5. 非方阵不被注入
6. monkey-patch forward 正确性
"""

import torch
import torch.nn as nn
import pytest

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.model.lora import LoRA, apply_lora, get_lora_params, freeze_non_lora


class TestLoRAModule:
    def test_structure(self):
        lora = LoRA(768, 768, rank=16)
        assert isinstance(lora.A, nn.Linear)
        assert lora.A.in_features == 768
        assert lora.A.out_features == 16
        assert isinstance(lora.B, nn.Linear)
        assert lora.B.in_features == 16
        assert lora.B.out_features == 768

    def test_a_gaussian_init(self):
        lora = LoRA(768, 768, rank=16)
        assert lora.A.weight.data.std() > 0.01
        assert not torch.all(lora.A.weight.data == 0)

    def test_b_zero_init(self):
        lora = LoRA(768, 768, rank=16)
        assert torch.all(lora.B.weight.data == 0)

    def test_forward_at_init_is_zero(self):
        """B 零初始化 → 初始 LoRA 输出 = 0（不改变原模型行为）。"""
        lora = LoRA(64, 64, rank=8)
        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert torch.all(out == 0)

    def test_forward_shape(self):
        lora = LoRA(64, 64, rank=8)
        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 64)

    def test_forward_ba_x(self):
        """验证 forward = B(A(x))。"""
        lora = LoRA(64, 64, rank=8)
        lora.B.weight.data.normal_()
        x = torch.randn(1, 5, 64)
        expected = lora.B(lora.A(x))
        assert torch.allclose(lora(x), expected)


class TestApplyLoRA:
    @pytest.fixture
    def model(self):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=100,
        )
        return ZLLMForCausalLM(config)

    def test_lora_injected_on_square_linear(self, model):
        apply_lora(model, rank=8)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                assert hasattr(module, "lora"), f"{name} should have lora"

    def test_lora_not_injected_on_non_square(self, model):
        apply_lora(model, rank=8)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.in_features != module.out_features:
                assert not hasattr(module, "lora"), f"{name} should not have lora"

    def test_forward_unchanged_at_init(self, model, device):
        """LoRA 初始化（B=0）后模型输出不变。"""
        model = model.to(device)
        x = torch.randint(0, 100, (1, 10), device=device)
        with torch.no_grad():
            out_before = model(x)
        apply_lora(model, rank=8)
        with torch.no_grad():
            out_after = model(x)
        assert torch.allclose(out_before.logits, out_after.logits, atol=1e-6)

    def test_lora_changes_output_after_training(self, model, device):
        """LoRA 参数更新后输出会改变。"""
        model = model.to(device)
        apply_lora(model, rank=8)
        x = torch.randint(0, 100, (1, 10), device=device)
        with torch.no_grad():
            out_before = model(x)
        for name, module in model.named_modules():
            if hasattr(module, "lora"):
                module.lora.B.weight.data.normal_(std=0.1)
        with torch.no_grad():
            out_after = model(x)
        assert not torch.allclose(out_before.logits, out_after.logits, atol=1e-4)

    def test_get_lora_params(self, model):
        apply_lora(model, rank=8)
        params = get_lora_params(model)
        assert len(params) > 0
        for p in params:
            assert p.requires_grad

    def test_freeze_non_lora(self, model):
        apply_lora(model, rank=8)
        lora_params = freeze_non_lora(model)
        assert len(lora_params) > 0
        for name, p in model.named_parameters():
            if "lora" in name:
                assert p.requires_grad
            else:
                assert not p.requires_grad

    def test_lora_param_count_small(self, model):
        apply_lora(model, rank=8)
        lora_params = get_lora_params(model)
        lora_count = sum(p.numel() for p in lora_params)
        total_count = sum(p.numel() for p in model.parameters())
        assert lora_count < total_count
        assert lora_count / total_count < 0.1  # LoRA 应远小于总参数量
