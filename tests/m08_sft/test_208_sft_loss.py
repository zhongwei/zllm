"""M8-B: SFT loss 下降 + label masking 验证。

验证：
1. SFT labels 只在 assistant 区域非 -100
2. SFT loss 下降
3. 模型能过拟合到 SFT 数据
4. SFT 后模型能生成合理的 assistant 回复
"""

import json

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.dataset.sft import SFTDataset
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.full_sft import SFTConfig, train_epoch
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed


@pytest.fixture
def sft_data(tmp_path, device):
    setup_seed(42)
    from zllm.tokenizer.trainer import train_tokenizer

    corpus = [
        "你好世界测试对话数据语言模型分词器训练回复内容",
        "用户助手交互问答系统回复信息",
        "天气今天怎么样明天后天未来",
        "谢谢再见下次见面的时间约定",
    ] * 20
    tok = train_tokenizer(corpus, vocab_size=500, save_dir=str(tmp_path / "tok"))

    data_path = str(tmp_path / "sft.jsonl")
    conversations = [
        {"conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好很高兴为你服务"},
        ]},
        {"conversations": [
            {"role": "user", "content": "今天天气"},
            {"role": "assistant", "content": "今天天气非常晴朗"},
        ]},
        {"conversations": [
            {"role": "user", "content": "再见"},
            {"role": "assistant", "content": "再见期待下次见面"},
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
    model = ZLLMForCausalLM(config).to(device)
    ds = SFTDataset(data_path, tok, max_length=64)
    return model, ds, config, tok, device


class TestSFTLabelMasking:
    def test_labels_only_on_assistant(self, sft_data):
        """验证 labels 只在 assistant 回复区域非 -100。"""
        from zllm.tokenizer.adapter import wrap
        model, ds, config, tok, device = sft_data
        tok = wrap(tok)
        input_ids, labels = ds[0]
        input_ids = input_ids.tolist()
        labels = labels.tolist()

        assert -100 in labels, "应该有被 mask 的 prompt 区域"
        assert any(l != -100 for l in labels), "应该有非 mask 的 assistant 区域"

        non_masked = [(i, l) for i, l in enumerate(labels) if l != -100]
        assert len(non_masked) > 0

        first_non_masked_idx = non_masked[0][0]
        assert input_ids[first_non_masked_idx] != tok.pad_token_id, "非 mask 区域不应是 pad"

    def test_pad_positions_are_masked(self, sft_data):
        """验证 pad 位置 label 为 -100。"""
        from zllm.tokenizer.adapter import wrap
        model, ds, config, tok, device = sft_data
        tok = wrap(tok)
        pad_id = tok.pad_token_id
        input_ids, labels = ds[0]
        for i in range(len(input_ids)):
            if input_ids[i].item() == pad_id:
                assert labels[i].item() == -100


class TestSFTLossDecrease:
    def test_sft_loss_decreases(self, sft_data):
        model, ds, config, tok, device = sft_data
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=6, learning_rate=5e-3, accumulation_steps=1, log_interval=999)

        first_losses, last_losses = [], []
        for epoch in range(6):
            losses = train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)
            if epoch == 0:
                first_losses = losses
            last_losses = losses
        avg_first = sum(first_losses) / len(first_losses)
        avg_last = sum(last_losses) / len(last_losses)
        assert avg_last < avg_first

    def test_sft_overfit(self, sft_data):
        """小模型 + 少量 SFT 数据 → 应能过拟合。"""
        model, ds, config, tok, device = sft_data
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=10, learning_rate=5e-3, accumulation_steps=1, log_interval=999)

        all_losses = []
        for epoch in range(10):
            losses = train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)
            all_losses.extend(losses)
        assert all_losses[-1] < all_losses[0] * 0.6


class TestSFTVsPretrain:
    def test_sft_loss_lower_than_random(self, sft_data):
        """SFT 训练后 loss 应低于随机初始化的 loss。"""
        model, ds, config, tok, device = sft_data
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=1, learning_rate=5e-3, accumulation_steps=1, log_interval=999)

        model.eval()
        with torch.no_grad():
            input_ids, labels = ds[0]
            input_ids = input_ids.unsqueeze(0).to(device)
            labels_t = labels.unsqueeze(0).to(device)
            before_loss = model(input_ids, labels=labels_t).loss.item()

        for epoch in range(5):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)

        model.eval()
        with torch.no_grad():
            after_loss = model(input_ids, labels=labels_t).loss.item()

        assert after_loss < before_loss
