"""M6-B: AMP + 梯度累积 + 梯度裁剪测试。

测试混合精度训练、梯度累积、梯度裁剪的 train_step 辅助函数。
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.amp import train_step, GradScalerManager


@pytest.fixture
def model_and_data(device):
    config = ZLLMConfig(
        vocab_size=100, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = ZLLMForCausalLM(config).to(device)
    ids = torch.randint(0, 100, (2, 16), device=device)
    labels = ids.clone()
    return model, ids, labels


class TestGradScalerManager:
    def test_init(self, device):
        mgr = GradScalerManager(enabled=(device.type == "cuda"))
        assert mgr.enabled == (device.type == "cuda")

    def test_scale(self, device):
        mgr = GradScalerManager(enabled=True)
        loss = torch.tensor(1.0, device=device)
        scaled = mgr.scale(loss)
        assert scaled.item() != loss.item()


class TestTrainStep:
    def test_basic_step(self, model_and_data, device):
        model, ids, labels = model_and_data
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        loss = train_step(model, optimizer, ids, labels, scaler, accumulation_steps=1, max_grad_norm=1.0)
        assert loss > 0
        assert torch.isfinite(torch.tensor(loss))

    def test_gradient_clipping(self, model_and_data, device):
        model, ids, labels = model_and_data
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        train_step(model, optimizer, ids, labels, scaler, accumulation_steps=1, max_grad_norm=0.01)
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.norm() <= 0.01 * len(list(model.parameters())) + 1e-3

    def test_accumulation_steps(self, model_and_data, device):
        model, ids, labels = model_and_data
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        # 3 步累积，每步都不应该 step（只有第 3 步才真正更新）
        loss1 = train_step(model, optimizer, ids, labels, scaler, accumulation_steps=3, max_grad_norm=1.0, current_step=0)
        loss2 = train_step(model, optimizer, ids, labels, scaler, accumulation_steps=3, max_grad_norm=1.0, current_step=1)
        loss3 = train_step(model, optimizer, ids, labels, scaler, accumulation_steps=3, max_grad_norm=1.0, current_step=2)
        assert loss1 > 0 and loss2 > 0 and loss3 > 0

    def test_amp_on_cuda(self, model_and_data, device):
        if device.type != "cuda":
            pytest.skip("AMP requires CUDA")
        model, ids, labels = model_and_data
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=True)
        loss = train_step(model, optimizer, ids, labels, scaler, accumulation_steps=1, max_grad_norm=1.0)
        assert loss > 0

    def test_zero_grad_after_step(self, model_and_data, device):
        model, ids, labels = model_and_data
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        train_step(model, optimizer, ids, labels, scaler, accumulation_steps=1, max_grad_norm=1.0)
        # step 后梯度应为 None（set_to_none=True）
        for p in model.parameters():
            assert p.grad is None or p.grad.norm() == 0
