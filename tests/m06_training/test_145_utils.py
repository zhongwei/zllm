"""M6-A: 训练工具函数测试。

测试 setup_seed, get_lr, init_model, lm_checkpoint, Logger, is_main_process。
"""

import os
import math
import torch
import pytest

from zllm.training.utils import (
    setup_seed,
    get_lr,
    init_model,
    lm_checkpoint,
    is_main_process,
    Logger,
    get_model_params,
)
from zllm.config import ZLLMConfig


class TestSetupSeed:
    def test_reproducibility(self):
        setup_seed(42)
        a = torch.randn(3, 3)
        setup_seed(42)
        b = torch.randn(3, 3)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        setup_seed(42)
        a = torch.randn(3, 3)
        setup_seed(123)
        b = torch.randn(3, 3)
        assert not torch.allclose(a, b)


class TestGetLR:
    def test_start_lr(self):
        lr = get_lr(0, 100, base_lr=5e-4)
        # 初始 lr = base * (0.1 + 0.45 * (1 + cos(0))) = base * (0.1 + 0.45 * 2) = base * 1.0
        assert abs(lr - 5e-4) < 1e-6

    def test_end_lr(self):
        lr = get_lr(100, 100, base_lr=5e-4)
        # 终点 lr = base * (0.1 + 0.45 * (1 + cos(pi))) = base * (0.1 + 0) = base * 0.1
        assert abs(lr - 5e-5) < 1e-6

    def test_monotonic_decrease(self):
        lrs = [get_lr(s, 100, base_lr=5e-4) for s in range(0, 101, 10)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]

    def test_midpoint(self):
        lr = get_lr(50, 100, base_lr=5e-4)
        # 中点 = base * (0.1 + 0.45 * (1 + cos(pi/2))) = base * (0.1 + 0.45) = base * 0.55
        assert abs(lr - 5e-4 * 0.55) < 1e-6


class TestIsMainProcess:
    def test_true_without_dist(self):
        assert is_main_process() is True


class TestLogger:
    def test_logger_prints(self, capsys):
        Logger("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out


class TestInitModel:
    def test_from_none_creates_random(self, tmp_path, device):
        config = ZLLMConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128,
        )
        model = init_model(config, from_weight="none", save_dir=str(tmp_path), device=str(device))
        assert model is not None
        assert next(model.parameters()).device.type == device.type

    def test_from_weight_loads(self, tmp_path, device):
        config = ZLLMConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128,
        )
        from zllm.model.causal_lm import ZLLMForCausalLM
        model = ZLLMForCausalLM(config).to(device)
        sd = {k: v.half().cpu() for k, v in model.state_dict().items()}
        path = os.path.join(str(tmp_path), f"pretrain_64.pth")
        torch.save(sd, path)
        loaded = init_model(config, from_weight="pretrain", save_dir=str(tmp_path), device=str(device))
        assert loaded is not None


class TestLMCheckpoint:
    def test_save_and_load(self, tmp_path, device):
        config = ZLLMConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128,
        )
        from zllm.model.causal_lm import ZLLMForCausalLM
        model = ZLLMForCausalLM(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        lm_checkpoint(
            config, weight="pretrain", model=model, optimizer=optimizer,
            epoch=1, step=42, save_dir=str(tmp_path),
        )

        assert os.path.exists(os.path.join(str(tmp_path), "pretrain_64.pth"))
        assert os.path.exists(os.path.join(str(tmp_path), "pretrain_64_resume.pth"))

    def test_load_returns_none_when_missing(self, tmp_path):
        config = ZLLMConfig(hidden_size=64)
        result = lm_checkpoint(config, save_dir=str(tmp_path))
        assert result is None

    def test_load_returns_resume_data(self, tmp_path, device):
        config = ZLLMConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128,
        )
        from zllm.model.causal_lm import ZLLMForCausalLM
        model = ZLLMForCausalLM(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lm_checkpoint(
            config, weight="pretrain", model=model, optimizer=optimizer,
            epoch=2, step=100, save_dir=str(tmp_path),
        )
        data = lm_checkpoint(config, weight="pretrain", save_dir=str(tmp_path))
        assert data is not None
        assert data["epoch"] == 2
        assert data["step"] == 100

    def test_moe_suffix(self, tmp_path, device):
        config = ZLLMConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, use_moe=True,
            max_position_embeddings=128,
        )
        from zllm.model.causal_lm import ZLLMForCausalLM
        model = ZLLMForCausalLM(config).to(device)
        lm_checkpoint(config, weight="pretrain", model=model,
                       optimizer=torch.optim.AdamW(model.parameters()),
                       save_dir=str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), "pretrain_64_moe.pth"))


class TestGetModelParams:
    def test_dense_params(self, capsys):
        config = ZLLMConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128,
        )
        from zllm.model.causal_lm import ZLLMForCausalLM
        model = ZLLMForCausalLM(config)
        get_model_params(model, config)
        captured = capsys.readouterr()
        assert "M" in captured.out
