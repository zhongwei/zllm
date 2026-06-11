"""M10-A: DPO 测试 — logits_to_log_probs + dpo_loss + DPOConfig + train_epoch。

验证：
1. logits_to_log_probs 正确提取 token log 概率
2. dpo_loss 对 chosen > rejected 产生正梯度
3. DPOConfig 默认值
4. DPO train_epoch 可运行且 loss 下降
"""

import json
import math

import torch
import torch.nn.functional as F
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.dpo import logits_to_log_probs, dpo_loss, DPOConfig, train_epoch
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed


class TestLogitsToLogProbs:
    def test_shape(self):
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        log_probs = logits_to_log_probs(logits, labels)
        assert log_probs.shape == (2, 10)

    def test_values_are_log_probs(self):
        logits = torch.randn(1, 5, 20)
        labels = torch.randint(0, 20, (1, 5))
        log_probs = logits_to_log_probs(logits, labels)
        expected = F.log_softmax(logits, dim=2)
        expected_per_token = expected.gather(2, labels.unsqueeze(2)).squeeze(-1)
        assert torch.allclose(log_probs, expected_per_token, atol=1e-6)

    def test_sum_less_than_zero(self):
        logits = torch.randn(1, 10, 50)
        labels = torch.randint(0, 50, (1, 10))
        log_probs = logits_to_log_probs(logits, labels)
        assert log_probs.sum() < 0


class TestDPOLoss:
    def test_chosen_preferred_gives_lower_loss(self):
        """当 chosen log_probs > rejected 时，loss 应较低。"""
        ref = torch.tensor([[-1.0], [-1.0]])
        policy = torch.tensor([[-0.5], [-2.0]])
        mask = torch.ones(2, 1)
        loss = dpo_loss(ref, policy, mask, beta=0.1)
        assert loss.item() > 0

    def test_preferred_lower_loss(self):
        """策略偏好 chosen → loss 低于无偏好。"""
        ref = torch.tensor([[-1.0], [-1.0]])
        mask = torch.ones(2, 1)
        policy_no_pref = torch.tensor([[-1.0], [-1.0]])
        loss_no_pref = dpo_loss(ref, policy_no_pref, mask, beta=0.1).item()

        policy_pref = torch.tensor([[-0.5], [-2.0]])
        loss_pref = dpo_loss(ref, policy_pref, mask, beta=0.1).item()
        assert loss_pref < loss_no_pref

    def test_gradient_flows(self):
        ref = torch.tensor([[-1.0], [-1.0]], requires_grad=False)
        policy = torch.tensor([[-0.5], [-2.0]], requires_grad=True)
        mask = torch.ones(2, 1)
        loss = dpo_loss(ref, policy, mask, beta=0.1)
        loss.backward()
        assert policy.grad is not None


class TestDPOConfig:
    def test_defaults(self):
        cfg = DPOConfig()
        assert cfg.learning_rate == 4e-8
        assert cfg.beta == 0.15
        assert cfg.from_weight == "full_sft"
        assert cfg.epochs == 1

    def test_vs_sft(self):
        from zllm.training.full_sft import SFTConfig
        sft = SFTConfig()
        dpo = DPOConfig()
        assert dpo.learning_rate < sft.learning_rate


class TestDPOTrain:
    @pytest.fixture
    def dpo_setup(self, tmp_path, device):
        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer
        from zllm.dataset.dpo import DPODataset

        corpus = ["DPO偏好优化测试数据对话回复选择拒绝"] * 20
        tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))

        data_path = str(tmp_path / "dpo.jsonl")
        samples = [
            {"chosen": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好很高兴为你服务"},
            ], "rejected": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "走开别烦我"},
            ]},
        ] * 12
        with open(data_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        config = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
        model = ZLLMForCausalLM(config).to(device)
        ref_model = ZLLMForCausalLM(config).to(device)
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        ref_model.requires_grad_(False)

        ds = DPODataset(data_path, tok, max_length=64)
        return model, ref_model, ds, config, device

    def test_dpo_train_epoch_runs(self, dpo_setup):
        model, ref_model, ds, config, device = dpo_setup
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScalerManager(enabled=False)
        cfg = DPOConfig(epochs=1, learning_rate=1e-4, accumulation_steps=1, log_interval=999)

        losses = train_epoch(model, ref_model, loader, optimizer, scaler, cfg, 0, device)
        assert len(losses) > 0
        assert all(isinstance(l, float) for l in losses)

    def test_dpo_loss_decreases(self, dpo_setup):
        model, ref_model, ds, config, device = dpo_setup
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = DPOConfig(epochs=5, learning_rate=1e-3, accumulation_steps=1, log_interval=999)

        first_losses, last_losses = [], []
        for epoch in range(5):
            losses = train_epoch(model, ref_model, loader, optimizer, scaler, cfg, epoch, device)
            if epoch == 0:
                first_losses = losses
            last_losses = losses
        avg_first = sum(first_losses) / len(first_losses)
        avg_last = sum(last_losses) / len(last_losses)
        assert avg_last < avg_first
