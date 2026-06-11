"""M12 集成测试：端到端推理 + API + 模型转换。

验证：
1. SFT 模型可生成对话
2. API server 可启动
3. KV Cache 与无 cache 结果一致
4. 完整 pipeline: train → save → load → generate
"""

import json
import os
import time

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.serving.generate import generate, generate_with_cache
from zllm.serving.api_server import create_app
from zllm.serving.cli import CLIConfig
from zllm.training.full_sft import SFTConfig, train_epoch
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed, lm_checkpoint
from zllm.tokenizer.adapter import wrap


class TestE2EGeneration:
    @pytest.fixture
    def trained_model(self, tmp_path, device):
        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer
        from zllm.dataset.sft import SFTDataset

        corpus = ["端到端推理测试对话数据回复内容语言模型"] * 20
        tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
        data_path = str(tmp_path / "sft.jsonl")
        convs = [{"conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好很高兴为你服务"},
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
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = SFTConfig(epochs=5, learning_rate=5e-3, accumulation_steps=1, log_interval=999)
        for epoch in range(5):
            train_epoch(model, loader, optimizer, scaler, cfg, epoch, device)
        tok = wrap(tok)
        return model, config, tok, device

    def test_generate_after_sft(self, trained_model):
        model, config, tok, device = trained_model
        messages = [{"role": "user", "content": "你好"}]
        prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_ids = torch.tensor([prompt_ids], device=device)
        out = generate(model, input_ids, max_new_tokens=20, temperature=0.0)
        assert out.shape[1] > len(prompt_ids)

    def test_cache_matches_no_cache(self, trained_model):
        model, config, tok, device = trained_model
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)
        out1 = generate(model, input_ids, max_new_tokens=10, temperature=0.0)
        out2 = generate_with_cache(model, input_ids, max_new_tokens=10, temperature=0.0)
        assert torch.equal(out1, out2)

    def test_save_load_generate(self, trained_model, tmp_path):
        model, config, tok, device = trained_model
        save_dir = str(tmp_path / "out")
        lm_checkpoint(config, weight="full_sft", model=model, save_dir=save_dir)

        model2 = ZLLMForCausalLM(config).to(device)
        weights = torch.load(os.path.join(save_dir, "full_sft_64.pth"), map_location=device, weights_only=True)
        model2.load_state_dict(weights, strict=False)

        input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)
        with torch.no_grad():
            out1 = model(input_ids).logits
            out2 = model2(input_ids).logits
        assert torch.allclose(out1, out2, atol=1e-2)


class TestAPIServer:
    def test_app_starts(self):
        app = create_app()
        assert app.title == "ZLLM API"

    def test_models_endpoint(self):
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0

    def test_chat_completions_endpoint(self):
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={
            "model": "zllm",
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 0.85,
            "max_tokens": 50,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert data["object"] == "chat.completion"


class TestSpeedMeasurement:
    def test_tokens_per_second(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=100, max_position_embeddings=128,
        )
        model = ZLLMForCausalLM(config).to(device)
        model.eval()
        input_ids = torch.randint(0, 100, (1, 8), device=device)

        start = time.time()
        out = generate(model, input_ids, max_new_tokens=20, temperature=0.0)
        elapsed = time.time() - start

        new_tokens = out.shape[1] - 8
        speed = new_tokens / max(elapsed, 1e-6)
        assert speed > 0
