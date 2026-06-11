"""M7 集成测试：端到端预训练 pipeline。

setup → train → save → resume → verify。
"""

import json
import os

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.dataset.pretrain import PretrainDataset
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.pretrain import PretrainConfig, train_epoch
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed, lm_checkpoint


@pytest.fixture
def e2e_setup(tmp_path, device):
    setup_seed(42)
    from zllm.tokenizer.trainer import train_tokenizer
    corpus = ["预训练端到端验证测试数据语言模型"] * 30
    tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
    data_path = str(tmp_path / "data.jsonl")
    with open(data_path, "w", encoding="utf-8") as f:
        for i in range(24):
            f.write(json.dumps({"text": "预训练端到端验证测试数据"}, ensure_ascii=False) + "\n")
    config = ZLLMConfig(
        vocab_size=tok.get_vocab_size(),
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=32,
    )
    model = ZLLMForCausalLM(config).to(device)
    ds = PretrainDataset(data_path, tok, max_length=32)
    return model, ds, config, tok, tmp_path, device


class TestE2EPretrain:
    def test_train_save_resume(self, e2e_setup):
        model, ds, config, tok, tmp_path, device = e2e_setup
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scaler = GradScalerManager(enabled=False)
        cfg = PretrainConfig(epochs=2, learning_rate=5e-4, accumulation_steps=1, log_interval=999)

        for epoch in range(2):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)

        lm_checkpoint(config, weight="pretrain", model=model, optimizer=optimizer,
                      epoch=2, step=0, save_dir=str(tmp_path / "ckpts"))

        assert os.path.exists(str(tmp_path / "ckpts" / "pretrain_64.pth"))
        assert os.path.exists(str(tmp_path / "ckpts" / "pretrain_64_resume.pth"))

        resume = lm_checkpoint(config, weight="pretrain", save_dir=str(tmp_path / "ckpts"))
        assert resume["epoch"] == 2

    def test_resumed_model_matches(self, e2e_setup):
        model, ds, config, tok, tmp_path, device = e2e_setup
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scaler = GradScalerManager(enabled=False)
        cfg = PretrainConfig(epochs=1, learning_rate=5e-4, accumulation_steps=1, log_interval=999)
        train_epoch(model, loader, optimizer, scaler, cfg, 0, device)

        lm_checkpoint(config, weight="pretrain", model=model, optimizer=optimizer,
                      epoch=1, step=0, save_dir=str(tmp_path / "ckpts"))

        model2 = ZLLMForCausalLM(config).to(device)
        weights = torch.load(str(tmp_path / "ckpts" / "pretrain_64.pth"), map_location=device, weights_only=True)
        model2.load_state_dict(weights, strict=False)

        ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
        with torch.no_grad():
            out1 = model(ids)
            out2 = model2(ids)
        assert torch.allclose(out1.logits, out2.logits, atol=1e-2)

    def test_moe_pretrain(self, tmp_path, device):
        """MoE 模型也能正常预训练。"""
        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer
        corpus = ["MoE预训练测试数据"] * 20
        tok = train_tokenizer(corpus, vocab_size=200, save_dir=str(tmp_path / "tok"))
        data_path = str(tmp_path / "data.jsonl")
        with open(data_path, "w", encoding="utf-8") as f:
            for _ in range(16):
                f.write(json.dumps({"text": "MoE预训练测试数据内容"}, ensure_ascii=False) + "\n")
        config = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, max_position_embeddings=32,
        )
        model = ZLLMForCausalLM(config).to(device)
        ds = PretrainDataset(data_path, tok, max_length=32)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scaler = GradScalerManager(enabled=False)
        cfg = PretrainConfig(epochs=2, learning_rate=5e-4, accumulation_steps=1, log_interval=999)
        losses = []
        for epoch in range(2):
            losses.extend(train_epoch(model, loader, optimizer, scaler, cfg, epoch, device))
        assert losses[-1] < losses[0]
