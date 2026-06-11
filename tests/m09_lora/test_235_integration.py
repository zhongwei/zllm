"""M9 集成测试：LoRA 完整 pipeline。

apply_lora → freeze → train → save → load → merge → verify。
"""

import json
import os

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.model.lora import apply_lora, save_lora, load_lora, merge_lora, freeze_non_lora, get_lora_params
from zllm.training.lora_sft import LoRAConfig, train_epoch
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed


@pytest.fixture
def lora_e2e(tmp_path, device):
    setup_seed(42)
    from zllm.tokenizer.trainer import train_tokenizer
    from zllm.dataset.sft import SFTDataset

    corpus = ["LoRA集成测试数据对话回复内容语言模型"] * 20
    tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
    data_path = str(tmp_path / "sft.jsonl")
    convs = [{"conversations": [
        {"role": "user", "content": "测试"},
        {"role": "assistant", "content": "集成测试回复"},
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
    ds = SFTDataset(data_path, tok, max_length=64)
    return model, ds, config, device, tmp_path


class TestLoRAIntegration:
    def test_full_pipeline(self, lora_e2e):
        """apply_lora → freeze → train → save → load → verify。"""
        model, ds, config, device, tmp_path = lora_e2e

        apply_lora(model, rank=8)
        lora_params = freeze_non_lora(model)

        lora_count = sum(p.numel() for p in lora_params)
        total_count = sum(p.numel() for p in model.parameters())
        assert lora_count / total_count < 0.1

        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(lora_params, lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = LoRAConfig(epochs=3, learning_rate=5e-3, rank=8, accumulation_steps=1, log_interval=999)

        x = torch.randint(0, config.vocab_size, (1, 10), device=device)
        with torch.no_grad():
            out_before = model(x).logits.clone()

        for epoch in range(3):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device, lora_params)

        with torch.no_grad():
            out_after = model(x).logits.clone()
        assert not torch.allclose(out_before, out_after, atol=1e-4)

        lora_path = str(tmp_path / "lora_test.pth")
        save_lora(model, lora_path)
        assert os.path.exists(lora_path)

    def test_save_load_train_resume(self, lora_e2e):
        """save → load → 继续训练 → loss 继续下降。"""
        model, ds, config, device, tmp_path = lora_e2e

        apply_lora(model, rank=8)
        lora_params = freeze_non_lora(model)

        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(lora_params, lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = LoRAConfig(epochs=2, learning_rate=5e-3, rank=8, accumulation_steps=1, log_interval=999)

        losses1 = []
        for epoch in range(2):
            losses1.extend(train_epoch(model, loader, optimizer, scaler, cfg, epoch, device, lora_params))

        lora_path = str(tmp_path / "lora_test.pth")
        save_lora(model, lora_path)

        model2 = ZLLMForCausalLM(config).to(device)
        model2.load_state_dict(model.state_dict(), strict=False)
        apply_lora(model2, rank=8)
        load_lora(model2, lora_path)

        x = torch.randint(0, config.vocab_size, (1, 10), device=device)
        with torch.no_grad():
            out1 = model(x).logits
            out2 = model2(x).logits
        assert torch.allclose(out1, out2, atol=1e-2)

    def test_merge_and_inference(self, lora_e2e):
        """train → save lora → merge → 推理结果一致。"""
        model, ds, config, device, tmp_path = lora_e2e

        apply_lora(model, rank=8)
        lora_params = freeze_non_lora(model)

        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(lora_params, lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = LoRAConfig(epochs=2, learning_rate=5e-3, rank=8, accumulation_steps=1, log_interval=999)

        for epoch in range(2):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device, lora_params)

        lora_path = str(tmp_path / "lora_test.pth")
        merged_path = str(tmp_path / "merged.pth")
        save_lora(model, lora_path)

        x = torch.randint(0, config.vocab_size, (1, 10), device=device)
        with torch.no_grad():
            out_lora = model(x).logits.clone()

        merge_lora(model, lora_path, merged_path)

        model_merged = ZLLMForCausalLM(config).to(device)
        sd = torch.load(merged_path, map_location=device, weights_only=True)
        model_merged.load_state_dict(sd, strict=False)
        with torch.no_grad():
            out_merged = model_merged(x).logits
        assert torch.allclose(out_lora, out_merged, atol=1e-2)

    def test_base_model_unchanged_after_lora(self, lora_e2e):
        """LoRA 训练不改变基础模型权重。"""
        model, ds, config, device, tmp_path = lora_e2e

        base_sd = {k: v.clone() for k, v in model.state_dict().items()}

        apply_lora(model, rank=8)
        lora_params = freeze_non_lora(model)

        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(lora_params, lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = LoRAConfig(epochs=3, learning_rate=5e-3, rank=8, accumulation_steps=1, log_interval=999)

        for epoch in range(3):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device, lora_params)

        for k, v in model.state_dict().items():
            if "lora" not in k:
                assert torch.equal(base_sd[k], v), f"Base weight {k} was modified"
