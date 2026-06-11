"""M11-A: 蒸馏测试 — distillation_loss, Temperature, teacher/student, DistillConfig。

验证：
1. distillation_loss: KL(teacher_soft || student_soft) * T²
2. Temperature > 1 使软标签更平滑
3. alpha 权衡 CE 和 Distill loss
4. train_epoch 可运行
"""

import json

import torch
import torch.nn.functional as F
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.distillation import distillation_loss, DistillConfig, train_epoch
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed


class TestDistillationLoss:
    def test_identical_logits_zero_loss(self):
        logits = torch.randn(2, 10, 50)
        loss = distillation_loss(logits, logits, temperature=1.0)
        assert loss.item() < 1e-4

    def test_different_logits_positive_loss(self):
        teacher = torch.randn(2, 10, 50)
        student = torch.randn(2, 10, 50)
        loss = distillation_loss(student, teacher, temperature=1.0)
        assert loss.item() > 0

    def test_temperature_scaling(self):
        teacher = torch.randn(1, 5, 20)
        student = torch.randn(1, 5, 20)
        loss_t1 = distillation_loss(student, teacher, temperature=1.0)
        loss_t2 = distillation_loss(student, teacher, temperature=2.0)
        assert loss_t2.item() != loss_t1.item()

    def test_higher_temperature_smoother(self):
        logits = torch.tensor([[[10.0, 0.0, 0.0]]])
        probs_t1 = F.softmax(logits / 1.0, dim=-1)
        probs_t3 = F.softmax(logits / 3.0, dim=-1)
        assert probs_t3.max() < probs_t1.max()
        assert probs_t3.min() > probs_t1.min()

    def test_gradient_flows(self):
        teacher = torch.randn(1, 5, 20)
        student = torch.randn(1, 5, 20, requires_grad=True)
        loss = distillation_loss(student, teacher, temperature=1.0)
        loss.backward()
        assert student.grad is not None


class TestDistillConfig:
    def test_defaults(self):
        cfg = DistillConfig()
        assert cfg.alpha == 0.5
        assert cfg.temperature == 1.5
        assert cfg.learning_rate == 5e-6
        assert cfg.save_weight == "full_dist"

    def test_alpha_balances_losses(self):
        cfg = DistillConfig(alpha=0.0)
        assert cfg.alpha == 0.0
        cfg = DistillConfig(alpha=1.0)
        assert cfg.alpha == 1.0


class TestDistillTrain:
    @pytest.fixture
    def distill_setup(self, tmp_path, device):
        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer
        from zllm.dataset.sft import SFTDataset

        corpus = ["蒸馏测试数据对话回复内容语言模型"] * 20
        tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
        data_path = str(tmp_path / "sft.jsonl")
        convs = [{"conversations": [
            {"role": "user", "content": "测试"},
            {"role": "assistant", "content": "蒸馏回复"},
        ]}] * 16
        with open(data_path, "w", encoding="utf-8") as f:
            for c in convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        config = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
        student = ZLLMForCausalLM(config).to(device)
        teacher = ZLLMForCausalLM(config).to(device)
        ds = SFTDataset(data_path, tok, max_length=64)
        return student, teacher, ds, config, device

    def test_distill_train_epoch_runs(self, distill_setup):
        student, teacher, ds, config, device = distill_setup
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = DistillConfig(epochs=1, learning_rate=1e-3, accumulation_steps=1, log_interval=999)
        losses = train_epoch(student, teacher, loader, optimizer, scaler, cfg, 0, device)
        assert len(losses) > 0

    def test_distill_loss_decreases(self, distill_setup):
        student, teacher, ds, config, device = distill_setup
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(student.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = DistillConfig(epochs=5, learning_rate=5e-3, accumulation_steps=1, log_interval=999)
        first_losses, last_losses = [], []
        for epoch in range(5):
            losses = train_epoch(student, teacher, loader, optimizer, scaler, cfg, epoch, device)
            if epoch == 0:
                first_losses = losses
            last_losses = losses
        assert sum(last_losses) / len(last_losses) < sum(first_losses) / len(first_losses)

    def test_no_teacher_pure_ce(self, distill_setup):
        """alpha=1.0 + teacher=None → 纯 CE loss。"""
        student, teacher, ds, config, device = distill_setup
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = DistillConfig(epochs=1, alpha=1.0, learning_rate=1e-3, accumulation_steps=1, log_interval=999)
        losses = train_epoch(student, None, loader, optimizer, scaler, cfg, 0, device)
        assert len(losses) > 0
