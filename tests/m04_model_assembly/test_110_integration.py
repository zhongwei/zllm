"""M4 集成测试：完整 ZLLMForCausalLM 端到端。

创建模型 → 前向 → 计算 loss → 反向 → 更新参数 → save/load。
同时测试 Dense 和 MoE 两种配置。
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM


class TestM4Integration:
    def test_dense_train_step(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, small_config.vocab_size, (2, 8), device=device)
        labels = ids.clone()
        out = model(ids, labels=labels)
        loss = out.loss + out.aux_loss
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)

    def test_moe_train_step(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, max_position_embeddings=128,
        )
        model = ZLLMForCausalLM(config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
        labels = ids.clone()
        out = model(ids, labels=labels)
        loss = out.loss + out.aux_loss
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)

    def test_save_load_state_dict(self, small_config, device, tmp_path):
        model = ZLLMForCausalLM(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        out1 = model(ids)
        logits1 = out1.logits.clone()
        torch.save(model.state_dict(), tmp_path / "model.pth")
        model2 = ZLLMForCausalLM(small_config).to(device)
        model2.load_state_dict(torch.load(tmp_path / "model.pth", weights_only=True))
        out2 = model2(ids)
        assert torch.allclose(logits1, out2.logits, atol=1e-5)

    def test_loss_decreases(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, small_config.vocab_size, (4, 16), device=device)
        labels = ids.clone()
        losses = []
        for _ in range(5):
            out = model(ids, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_inference_no_grad(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device).eval()
        ids = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (1, 4, small_config.vocab_size)

    def test_default_config_forward(self, default_config, device):
        model = ZLLMForCausalLM(default_config).to(device)
        ids = torch.randint(0, default_config.vocab_size, (1, 4), device=device)
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (1, 4, default_config.vocab_size)
