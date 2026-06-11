"""M4-D: ZLLMModel backbone 测试。

ZLLMModel = embed_tokens + N × ZLLMBlock + final_norm + RoPE buffers。
返回 (hidden_states, past_key_values, aux_loss)。
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.backbone import ZLLMModel


class TestZLLMModelInit:
    def test_embed_tokens(self, small_config):
        model = ZLLMModel(small_config)
        assert model.embed_tokens.weight.shape == (small_config.vocab_size, small_config.hidden_size)

    def test_layer_count(self, small_config):
        model = ZLLMModel(small_config)
        assert len(model.layers) == small_config.num_hidden_layers

    def test_final_norm(self, small_config):
        model = ZLLMModel(small_config)
        assert hasattr(model, "norm")

    def test_rope_buffers(self, small_config):
        model = ZLLMModel(small_config)
        assert hasattr(model, "freqs_cos")
        assert hasattr(model, "freqs_sin")
        assert model.freqs_cos.shape == (small_config.max_position_embeddings, small_config.head_dim)
        assert model.freqs_sin.shape == (small_config.max_position_embeddings, small_config.head_dim)


class TestZLLMModelForward:
    def test_output_shapes(self, small_config, device):
        model = ZLLMModel(small_config).to(device)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 8), device=device)
        hidden, presents, aux_loss = model(input_ids)
        assert hidden.shape == (2, 8, small_config.hidden_size)
        assert len(presents) == small_config.num_hidden_layers
        assert aux_loss.item() == 0.0

    def test_no_cache_by_default(self, small_config, device):
        model = ZLLMModel(small_config).to(device)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        _, presents, _ = model(input_ids)
        assert all(p is None for p in presents)

    def test_use_cache(self, small_config, device):
        model = ZLLMModel(small_config).to(device)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        _, presents, _ = model(input_ids, use_cache=True)
        assert all(p is not None for p in presents)
        k, v = presents[0]
        assert k.shape == (1, 4, small_config.num_key_value_heads, small_config.head_dim)

    def test_kv_cache_incremental(self, small_config, device):
        model = ZLLMModel(small_config).to(device)
        ids1 = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        _, presents, _ = model(ids1, use_cache=True)
        ids2 = torch.randint(0, small_config.vocab_size, (1, 2), device=device)
        _, presents2, _ = model(ids2, past_key_values=presents, use_cache=True)
        k, v = presents2[0]
        assert k.shape == (1, 6, small_config.num_key_value_heads, small_config.head_dim)

    def test_moe_aux_loss(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, max_position_embeddings=128,
        )
        model = ZLLMModel(config).to(device).train()
        ids = torch.randint(0, config.vocab_size, (2, 4), device=device)
        _, _, aux_loss = model(ids)
        assert aux_loss.item() > 0

    def test_gradients(self, small_config, device):
        model = ZLLMModel(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        hidden, _, _ = model(ids)
        hidden.sum().backward()
        assert model.embed_tokens.weight.grad is not None

    def test_attention_mask(self, small_config, device):
        model = ZLLMModel(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (2, 8), device=device)
        mask = torch.ones(2, 8, device=device)
        hidden, _, _ = model(ids, attention_mask=mask)
        assert hidden.shape == (2, 8, small_config.hidden_size)
