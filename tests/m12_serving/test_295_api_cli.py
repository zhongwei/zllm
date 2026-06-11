"""M12-B: OpenAI-compatible API server + CLI 测试。

验证：
1. ChatCompletionRequest 结构
2. API endpoint 路径正确
3. CLI config defaults
4. 模型格式转换
"""

import torch
import pytest

from zllm.serving.api_server import ChatCompletionRequest, create_app
from zllm.serving.cli import CLIConfig


class TestChatCompletionRequest:
    def test_defaults(self):
        req = ChatCompletionRequest(
            model="zllm",
            messages=[{"role": "user", "content": "你好"}],
        )
        assert req.model == "zllm"
        assert req.temperature == 0.85
        assert req.top_p == 0.95
        assert req.max_tokens == 512

    def test_custom_params(self):
        req = ChatCompletionRequest(
            model="zllm",
            messages=[{"role": "user", "content": "你好"}],
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
        )
        assert req.temperature == 0.5
        assert req.top_p == 0.9
        assert req.max_tokens == 100

    def test_messages_format(self):
        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ]
        req = ChatCompletionRequest(model="zllm", messages=messages)
        assert len(req.messages) == 2
        assert req.messages[0]["role"] == "system"


class TestCreateApp:
    def test_app_creates(self):
        app = create_app()
        assert app is not None

    def test_routes_exist(self):
        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/v1/chat/completions" in routes
        assert "/v1/models" in routes


class TestCLIConfig:
    def test_defaults(self):
        cfg = CLIConfig()
        assert cfg.load_from == "model"
        assert cfg.weight == "full_sft"
        assert cfg.temperature == 0.85
        assert cfg.top_p == 0.95
        assert cfg.max_new_tokens == 8192
        assert cfg.device == "cuda"

    def test_custom(self):
        cfg = CLIConfig(
            load_from="minimind-3",
            weight="pretrain",
            temperature=0.5,
        )
        assert cfg.load_from == "minimind-3"
        assert cfg.weight == "pretrain"
        assert cfg.temperature == 0.5


class TestModelConversion:
    def test_pytorch_to_dict_roundtrip(self, device):
        from zllm.config import ZLLMConfig
        from zllm.model.causal_lm import ZLLMForCausalLM

        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=100, max_position_embeddings=32,
        )
        model = ZLLMForCausalLM(config).to(device)
        sd = model.state_dict()
        assert len(sd) > 0
        assert "model.embed_tokens.weight" in sd
        assert "lm_head.weight" in sd

        model2 = ZLLMForCausalLM(config).to(device)
        model2.load_state_dict(sd)
        x = torch.randint(0, 100, (1, 8), device=device)
        with torch.no_grad():
            out1 = model(x).logits
            out2 = model2(x).logits
        assert torch.allclose(out1, out2, atol=1e-5)