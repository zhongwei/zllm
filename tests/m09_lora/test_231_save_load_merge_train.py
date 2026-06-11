"""M9-B: LoRA save/load/merge 测试 + LoRA 训练。

验证：
1. save_lora 只保存 LoRA 权重
2. load_lora 正确恢复
3. merge_lora 合并 W + B@A 后推理一致
4. LoRA 训练 loss 下降
5. LoRAConfig 默认值
"""

import json
import os

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.model.lora import apply_lora, save_lora, load_lora, merge_lora, freeze_non_lora
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed


@pytest.fixture
def model_with_lora(device):
    config = ZLLMConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=100,
    )
    model = ZLLMForCausalLM(config).to(device)
    apply_lora(model, rank=8)
    return model, config


class TestSaveLoad:
    def test_save_lora_only_has_lora_keys(self, model_with_lora, tmp_path):
        model, config = model_with_lora
        path = str(tmp_path / "lora.pth")
        save_lora(model, path)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        assert len(sd) > 0
        for k in sd:
            assert "lora" in k
            assert "A" in k or "B" in k

    def test_load_lora_restores_output(self, model_with_lora, tmp_path, device):
        model, config = model_with_lora
        path = str(tmp_path / "lora.pth")
        for name, module in model.named_modules():
            if hasattr(module, "lora"):
                module.lora.B.weight.data.normal_(std=0.1)
        save_lora(model, path)

        x = torch.randint(0, 100, (1, 10), device=device)
        with torch.no_grad():
            out_trained = model(x).logits.clone()

        model2 = ZLLMForCausalLM(config).to(device)
        model2.load_state_dict(model.state_dict(), strict=False)
        apply_lora(model2, rank=8)
        load_lora(model2, path)

        with torch.no_grad():
            out_loaded = model2(x).logits
        assert torch.allclose(out_trained, out_loaded, atol=1e-2)


class TestMergeLora:
    def test_merge_produces_standard_weights(self, model_with_lora, tmp_path, device):
        model, config = model_with_lora
        lora_path = str(tmp_path / "lora.pth")
        merged_path = str(tmp_path / "merged.pth")

        for name, module in model.named_modules():
            if hasattr(module, "lora"):
                module.lora.B.weight.data.normal_(std=0.1)
        save_lora(model, lora_path)
        merge_lora(model, lora_path, merged_path)

        sd = torch.load(merged_path, map_location="cpu", weights_only=True)
        for k in sd:
            assert "lora" not in k

    def test_merged_inference_matches_lora(self, model_with_lora, tmp_path, device):
        model, config = model_with_lora
        lora_path = str(tmp_path / "lora.pth")
        merged_path = str(tmp_path / "merged.pth")

        for name, module in model.named_modules():
            if hasattr(module, "lora"):
                module.lora.B.weight.data.normal_(std=0.1)
        save_lora(model, lora_path)

        x = torch.randint(0, 100, (1, 10), device=device)
        with torch.no_grad():
            out_lora = model(x).logits.clone()

        merge_lora(model, lora_path, merged_path)

        model_merged = ZLLMForCausalLM(config).to(device)
        sd = torch.load(merged_path, map_location=device, weights_only=True)
        model_merged.load_state_dict(sd, strict=False)
        with torch.no_grad():
            out_merged = model_merged(x).logits
        assert torch.allclose(out_lora, out_merged, atol=1e-2)


class TestLoRAConfig:
    def test_defaults(self):
        from zllm.training.lora_sft import LoRAConfig
        cfg = LoRAConfig()
        assert cfg.learning_rate == 1e-4
        assert cfg.rank == 16
        assert cfg.from_weight == "full_sft"
        assert cfg.epochs == 10

    def test_custom(self):
        from zllm.training.lora_sft import LoRAConfig
        cfg = LoRAConfig(rank=32, epochs=5)
        assert cfg.rank == 32
        assert cfg.epochs == 5


class TestLoRATrain:
    def test_lora_loss_decreases(self, tmp_path, device):
        from zllm.training.lora_sft import LoRAConfig, train_epoch
        from zllm.dataset.sft import SFTDataset

        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer

        corpus = ["LoRA训练测试数据对话回复内容"] * 20
        tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
        data_path = str(tmp_path / "sft.jsonl")
        convs = [{"conversations": [
            {"role": "user", "content": "测试"},
            {"role": "assistant", "content": "这是LoRA回复"},
        ]}] * 16
        with open(data_path, "w", encoding="utf-8") as f:
            for c in convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        config = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
        model = ZLLMForCausalLM(config).to(device)
        apply_lora(model, rank=8)
        lora_params = freeze_non_lora(model)

        ds = SFTDataset(data_path, tok, max_length=64)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(lora_params, lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = LoRAConfig(epochs=5, learning_rate=5e-3, rank=8, accumulation_steps=1, log_interval=999)

        all_losses = []
        for epoch in range(5):
            losses = train_epoch(model, loader, optimizer, scaler, cfg, epoch, device, lora_params)
            all_losses.extend(losses)
        assert all_losses[-1] < all_losses[0]

    def test_only_lora_params_have_grad(self, tmp_path, device):
        from zllm.training.lora_sft import LoRAConfig, train_epoch
        from zllm.dataset.sft import SFTDataset

        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer

        corpus = ["LoRA梯度测试"] * 10
        tok = train_tokenizer(corpus, vocab_size=200, save_dir=str(tmp_path / "tok"))
        data_path = str(tmp_path / "sft.jsonl")
        convs = [{"conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好"},
        ]}] * 8
        with open(data_path, "w", encoding="utf-8") as f:
            for c in convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        config = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
        model = ZLLMForCausalLM(config).to(device)
        apply_lora(model, rank=8)
        lora_params = freeze_non_lora(model)

        ds = SFTDataset(data_path, tok, max_length=64)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        optimizer = torch.optim.AdamW(lora_params, lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = LoRAConfig(epochs=1, learning_rate=1e-3, rank=8, accumulation_steps=1, log_interval=999)

        train_epoch(model, loader, optimizer, scaler, cfg, 0, device, lora_params)

        lora_updated = any(
            not torch.equal(p, torch.zeros_like(p))
            for n, p in model.named_parameters() if "lora.B" in n
        )
        assert lora_updated

        non_lora_unchanged = all(
            p.grad is None
            for n, p in model.named_parameters() if "lora" not in n
        )
        assert non_lora_unchanged
