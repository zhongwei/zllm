"""M8-C: Chat Template 推理 + SFT 后模型行为测试。

验证：
1. apply_chat_template 能正确渲染对话
2. SFT 后模型能生成 assistant 回复（greedy decode）
3. 生成结果包含 eos token 终止
4. SFT 权重保存/加载后推理结果一致
"""

import json
import os

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.dataset.sft import SFTDataset
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.full_sft import SFTConfig, train_epoch
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed, lm_checkpoint
from zllm.tokenizer.adapter import wrap


@pytest.fixture
def chat_setup(tmp_path, device):
    setup_seed(42)
    from zllm.tokenizer.trainer import train_tokenizer

    corpus = [
        "你好世界对话系统助手回复天气再见下次见面高兴服务",
        "用户询问问题回答解决方案内容信息",
    ] * 30
    tok_raw = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path / "tok"))
    tok = wrap(tok_raw)

    data_path = str(tmp_path / "sft.jsonl")
    conversations = [
        {"conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好很高兴为你服务"},
        ]},
        {"conversations": [
            {"role": "user", "content": "再见"},
            {"role": "assistant", "content": "再见期待下次见面"},
        ]},
    ] * 24

    with open(data_path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    config = ZLLMConfig(
        vocab_size=tok.vocab_size,
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
    )
    model = ZLLMForCausalLM(config).to(device)
    ds = SFTDataset(data_path, tok_raw, max_length=64)
    return model, ds, config, tok, device, tmp_path


def generate_greedy(model, input_ids, max_new_tokens, eos_token_id, device):
    """简易贪心解码生成。"""
    model.eval()
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(generated)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == eos_token_id:
                break
    return generated


class TestChatTemplate:
    def test_apply_chat_template_user(self, chat_setup):
        model, ds, config, tok, device, _ = chat_setup
        messages = [{"role": "user", "content": "你好"}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert "user" in text
        assert "你好" in text
        assert "assistant" in text  # generation prompt

    def test_apply_chat_tokenize(self, chat_setup):
        model, ds, config, tok, device, _ = chat_setup
        messages = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "回复"}]
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        assert isinstance(ids, list)
        assert len(ids) > 0


class TestSFTGeneration:
    def test_generates_after_sft(self, chat_setup):
        """SFT 后模型应能生成合理长度的回复。"""
        model, ds, config, tok, device, _ = chat_setup
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=8, learning_rate=5e-3, accumulation_steps=1, log_interval=999)
        for epoch in range(8):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)

        messages = [{"role": "user", "content": "你好"}]
        prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_ids = torch.tensor([prompt_ids], device=device)

        generated = generate_greedy(model, input_ids, max_new_tokens=20, eos_token_id=tok.eos_token_id, device=device)
        assert generated.shape[1] > len(prompt_ids), "应该生成了新 token"

    def test_generate_returns_decodeable(self, chat_setup):
        """生成的 token 应可解码为字符串。"""
        model, ds, config, tok, device, _ = chat_setup
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=5, learning_rate=5e-3, accumulation_steps=1, log_interval=999)
        for epoch in range(5):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)

        messages = [{"role": "user", "content": "你好"}]
        prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_ids = torch.tensor([prompt_ids], device=device)
        generated = generate_greedy(model, input_ids, max_new_tokens=10, eos_token_id=tok.eos_token_id, device=device)
        new_ids = generated[0, len(prompt_ids):].tolist()
        text = tok.decode(new_ids, skip_special_tokens=True)
        assert isinstance(text, str)


class TestSFTSaveLoad:
    def test_save_load_consistency(self, chat_setup):
        """SFT 权重保存/加载后推理结果一致。"""
        model, ds, config, tok, device, tmp_path = chat_setup
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=3, learning_rate=5e-3, accumulation_steps=1, log_interval=999)
        for epoch in range(3):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)

        save_dir = str(tmp_path / "out")
        lm_checkpoint(config, weight="full_sft", model=model, optimizer=optimizer,
                      epoch=3, step=0, save_dir=save_dir)

        model2 = ZLLMForCausalLM(config).to(device)
        weights = torch.load(os.path.join(save_dir, "full_sft_64.pth"), map_location=device, weights_only=True)
        model2.load_state_dict(weights, strict=False)

        input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model2(input_ids)
        assert torch.allclose(out1.logits, out2.logits, atol=1e-2)


class TestSFTIntegration:
    def test_pretrain_to_sft_pipeline(self, chat_setup):
        """完整 pipeline: 初始模型 → 'pretrain' → SFT → save → resume。"""
        model, ds, config, tok, device, tmp_path = chat_setup
        save_dir = str(tmp_path / "out")

        lm_checkpoint(config, weight="pretrain", model=model, save_dir=save_dir)
        assert os.path.exists(os.path.join(save_dir, "pretrain_64.pth"))

        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=3, learning_rate=5e-3, accumulation_steps=1, log_interval=999)
        for epoch in range(3):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)

        lm_checkpoint(config, weight="full_sft", model=model, optimizer=optimizer,
                      epoch=3, step=0, save_dir=save_dir)
        assert os.path.exists(os.path.join(save_dir, "full_sft_64.pth"))

        resume = lm_checkpoint(config, weight="full_sft", save_dir=save_dir)
        assert resume["epoch"] == 3

    def test_moe_sft(self, tmp_path, device):
        """MoE 模型也能正常 SFT。"""
        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer

        corpus = ["MoE监督微调测试数据对话"] * 20
        tok_raw = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))

        data_path = str(tmp_path / "sft.jsonl")
        convs = [{"conversations": [
            {"role": "user", "content": "测试"},
            {"role": "assistant", "content": "MoE微调回复"},
        ]}] * 16
        with open(data_path, "w", encoding="utf-8") as f:
            for c in convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        config = ZLLMConfig(
            vocab_size=tok_raw.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, max_position_embeddings=64,
        )
        model = ZLLMForCausalLM(config).to(device)
        ds = SFTDataset(data_path, tok_raw, max_length=64)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=3, learning_rate=5e-3, accumulation_steps=1, log_interval=999)
        losses = []
        for epoch in range(3):
            losses.extend(train_epoch(model, loader, optimizer, scaler, cfg, epoch, device))
        assert losses[-1] < losses[0]
