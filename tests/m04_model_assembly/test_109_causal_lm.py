"""M4-E: ZLLMForCausalLM 测试。

测试 Weight Tying、交叉熵 loss、ignore_index、logits_to_keep 优化。
返回 MoeCausalLMOutputWithPast。
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM


class TestCausalLMInit:
    def test_lm_head_shape(self, small_config):
        model = ZLLMForCausalLM(small_config)
        assert model.lm_head.weight.shape == (small_config.vocab_size, small_config.hidden_size)

    def test_weight_tying(self, small_config):
        model = ZLLMForCausalLM(small_config)
        assert torch.equal(model.lm_head.weight.data, model.model.embed_tokens.weight.data)

    def test_weight_tying_disabled(self):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, tie_word_embeddings=False, max_position_embeddings=128,
        )
        model = ZLLMForCausalLM(config)
        assert not torch.equal(model.lm_head.weight.data, model.model.embed_tokens.weight.data)

    def test_config_class(self, small_config):
        assert ZLLMForCausalLM.config_class == ZLLMConfig


class TestCausalLMForward:
    def test_logits_shape(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (2, 8), device=device)
        out = model(ids)
        assert out.logits.shape == (2, 8, small_config.vocab_size)

    def test_loss_none_without_labels(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        out = model(ids)
        assert out.loss is None

    def test_loss_with_labels(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (1, 8), device=device)
        labels = ids.clone()
        out = model(ids, labels=labels)
        assert out.loss is not None
        assert out.loss.item() > 0

    def test_ignore_index(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (1, 8), device=device)
        labels = ids.clone()
        labels[:, :4] = -100
        out = model(ids, labels=labels)
        assert out.loss is not None
        assert torch.isfinite(out.loss)

    def test_loss_backward(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (1, 8), device=device)
        labels = ids.clone()
        out = model(ids, labels=labels)
        out.loss.backward()
        assert model.model.embed_tokens.weight.grad is not None

    def test_logits_to_keep(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (1, 8), device=device)
        out = model(ids, logits_to_keep=2)
        assert out.logits.shape == (1, 2, small_config.vocab_size)

    def test_aux_loss_zero_dense(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        out = model(ids)
        assert out.aux_loss.item() == 0.0

    def test_aux_loss_nonzero_moe(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, max_position_embeddings=128,
        )
        model = ZLLMForCausalLM(config).to(device).train()
        ids = torch.randint(0, config.vocab_size, (2, 4), device=device)
        out = model(ids)
        assert out.aux_loss.item() > 0

    def test_past_key_values(self, small_config, device):
        model = ZLLMForCausalLM(small_config).to(device)
        ids = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        out = model(ids, use_cache=True)
        assert out.past_key_values is not None
        assert len(out.past_key_values) == small_config.num_hidden_layers
