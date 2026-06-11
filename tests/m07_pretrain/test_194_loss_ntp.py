"""M7-B: 预训练 loss 下降 + next-token prediction 测试。

用小模型 + 重复数据验证 loss 下降和自回归预测能力。
"""

import json

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.dataset.pretrain import PretrainDataset
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.pretrain import PretrainConfig, train_epoch
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed


@pytest.fixture
def mini_pretrain_setup(tmp_path, device):
    setup_seed(42)
    from zllm.tokenizer.trainer import train_tokenizer
    corpus = ["你好世界测试预训练数据语言模型BPE分词"] * 30
    tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
    data_path = str(tmp_path / "pretrain.jsonl")
    texts = ["你好世界这是预训练测试数据内容"] * 32
    with open(data_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    config = ZLLMConfig(
        vocab_size=tok.get_vocab_size(),
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=32,
    )
    model = ZLLMForCausalLM(config).to(device)
    ds = PretrainDataset(data_path, tok, max_length=32)
    return model, ds, config, device


class TestLossDecrease:
    def test_loss_decreases_over_epochs(self, mini_pretrain_setup):
        model, ds, config, device = mini_pretrain_setup
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scaler = GradScalerManager(enabled=False)
        train_cfg = PretrainConfig(epochs=3, accumulation_steps=1, log_interval=999)

        first_losses, last_losses = [], []
        for epoch in range(3):
            losses = train_epoch(model, loader, optimizer, scaler, train_cfg, epoch, device)
            if epoch == 0:
                first_losses = losses
            last_losses = losses
        avg_first = sum(first_losses) / len(first_losses)
        avg_last = sum(last_losses) / len(last_losses)
        assert avg_last < avg_first


class TestNextTokenPrediction:
    def test_model_can_overfit(self, mini_pretrain_setup):
        """小模型 + 重复数据 → 应能过拟合（loss 显著下降）。"""
        model, ds, config, device = mini_pretrain_setup
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        train_cfg = PretrainConfig(epochs=8, learning_rate=5e-3, accumulation_steps=1, log_interval=999)

        all_losses = []
        for epoch in range(8):
            losses = train_epoch(model, loader, optimizer, scaler, train_cfg, epoch, device)
            all_losses.extend(losses)
        assert all_losses[-1] < all_losses[0] * 0.7

    def test_model_predicts_next_token(self, mini_pretrain_setup):
        """训练后模型对训练数据应有合理的预测概率。"""
        model, ds, config, device = mini_pretrain_setup
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        train_cfg = PretrainConfig(epochs=3, accumulation_steps=1, log_interval=999)
        for epoch in range(3):
            train_epoch(model, loader, optimizer, scaler, train_cfg, epoch, device)

        model.eval()
        input_ids, labels = ds[0]
        input_ids = input_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[0]
        # 对非 pad 位置的下一个 token 预测应有一定准确性
        pad_id = ds.tokenizer.pad_token_id
        correct = 0
        total = 0
        for i in range(len(input_ids[0]) - 1):
            if input_ids[0, i + 1].item() == pad_id:
                continue
            pred = logits[i].argmax().item()
            if pred == input_ids[0, i + 1].item():
                correct += 1
            total += 1
        assert total > 0
        assert correct / total > 0.1  # 至少 10% 准确率
