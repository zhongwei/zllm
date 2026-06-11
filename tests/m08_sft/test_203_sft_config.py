"""M8-A: SFTConfig + train_epoch 基础测试。

SFT 与 Pretrain 的关键差异：
- learning_rate=1e-5（预训练的 1/50）
- max_seq_len=768（更长，因为对话数据）
- from_weight='pretrain'（加载预训练权重）
- 使用 SFTDataset（只对 assistant 回复计算 loss）
- save_weight='full_sft'
"""

import json
import math

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.training.full_sft import SFTConfig, train_epoch
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed


class TestSFTConfig:
    def test_defaults(self):
        cfg = SFTConfig()
        assert cfg.learning_rate == 1e-5
        assert cfg.max_seq_len == 768
        assert cfg.from_weight == "pretrain"
        assert cfg.save_weight == "full_sft"
        assert cfg.epochs == 2
        assert cfg.batch_size == 16

    def test_custom(self):
        cfg = SFTConfig(epochs=5, learning_rate=2e-5, max_seq_len=512)
        assert cfg.epochs == 5
        assert cfg.learning_rate == 2e-5
        assert cfg.max_seq_len == 512

    def test_sft_vs_pretrain_differs(self):
        from zllm.training.pretrain import PretrainConfig
        pre = PretrainConfig()
        sft = SFTConfig()
        assert sft.learning_rate < pre.learning_rate
        assert sft.max_seq_len > pre.max_seq_len


class TestSFTEpochRuns:
    @pytest.fixture
    def sft_setup(self, tmp_path, device):
        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer
        from zllm.dataset.sft import SFTDataset

        corpus = ["你好世界测试对话数据语言模型分词器训练"] * 30
        tok = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path / "tok"))

        data_path = str(tmp_path / "sft.jsonl")
        conversations = [
            {"conversations": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好很高兴认识你"},
            ]},
            {"conversations": [
                {"role": "user", "content": "测试对话"},
                {"role": "assistant", "content": "这是测试回复内容"},
            ]},
        ] * 16

        with open(data_path, "w", encoding="utf-8") as f:
            for c in conversations:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        config = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
        from zllm.model.causal_lm import ZLLMForCausalLM
        model = ZLLMForCausalLM(config).to(device)
        ds = SFTDataset(data_path, tok, max_length=64)
        return model, ds, config, device

    def test_train_epoch_returns_losses(self, sft_setup):
        model, ds, config, device = sft_setup
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=1, learning_rate=1e-4, accumulation_steps=1, log_interval=999)

        losses = train_epoch(model, loader, optimizer, scaler, cfg, 0, device)
        assert len(losses) > 0
        assert all(isinstance(l, float) for l in losses)

    def test_loss_decreases(self, sft_setup):
        model, ds, config, device = sft_setup
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=4, learning_rate=1e-3, accumulation_steps=1, log_interval=999)

        first_losses, last_losses = [], []
        for epoch in range(4):
            losses = train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)
            if epoch == 0:
                first_losses = losses
            last_losses = losses
        avg_first = sum(first_losses) / len(first_losses)
        avg_last = sum(last_losses) / len(last_losses)
        assert avg_last < avg_first
