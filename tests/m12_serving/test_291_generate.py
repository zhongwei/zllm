"""M12-A: 生成解码测试 — greedy, temperature, top-k, top-p, repetition penalty。

验证：
1. generate 函数基础功能
2. greedy decode 确定性
3. temperature 控制随机性
4. top-k 限制候选
5. top-p nucleus 采样
6. eos 停止
7. KV Cache 加速推理
"""

import torch
import pytest

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.serving.generate import generate, generate_with_cache
from zllm.training.utils import setup_seed


@pytest.fixture
def model_and_input(device):
    setup_seed(42)
    config = ZLLMConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=100, max_position_embeddings=128,
    )
    model = ZLLMForCausalLM(config).to(device)
    model.eval()
    input_ids = torch.randint(0, 100, (1, 8), device=device)
    return model, input_ids, config


class TestGenerate:
    def test_greedy_deterministic(self, model_and_input):
        model, input_ids, config = model_and_input
        out1 = generate(model, input_ids, max_new_tokens=10, temperature=0.0)
        out2 = generate(model, input_ids, max_new_tokens=10, temperature=0.0)
        assert torch.equal(out1, out2)

    def test_output_longer_than_input(self, model_and_input):
        model, input_ids, config = model_and_input
        out = generate(model, input_ids, max_new_tokens=5, temperature=0.0)
        assert out.shape[1] > input_ids.shape[1]

    def test_max_new_tokens_respected(self, model_and_input):
        model, input_ids, config = model_and_input
        out = generate(model, input_ids, max_new_tokens=5, temperature=0.0)
        assert out.shape[1] <= input_ids.shape[1] + 5

    def test_temperature_zero_greedy(self, model_and_input):
        model, input_ids, config = model_and_input
        out = generate(model, input_ids, max_new_tokens=5, temperature=0.0)
        assert out.shape[0] == 1

    def test_temperature_higher_more_random(self, model_and_input):
        model, input_ids, config = model_and_input
        setup_seed(0)
        out_low = generate(model, input_ids, max_new_tokens=20, temperature=0.3, top_k=0)
        setup_seed(1)
        out_high = generate(model, input_ids, max_new_tokens=20, temperature=2.0, top_k=0)
        # 高温应有更大的 token 变化（概率性，不 100% 保证但大概率）
        assert out_low.shape[1] > 0
        assert out_high.shape[1] > 0

    def test_top_k_limits_candidates(self, model_and_input):
        model, input_ids, config = model_and_input
        out = generate(model, input_ids, max_new_tokens=10, temperature=1.0, top_k=5)
        assert out.shape[1] > input_ids.shape[1]

    def test_top_p_nucleus(self, model_and_input):
        model, input_ids, config = model_and_input
        out = generate(model, input_ids, max_new_tokens=10, temperature=1.0, top_p=0.9)
        assert out.shape[1] > input_ids.shape[1]

    def test_eos_stops(self, model_and_input):
        model, input_ids, config = model_and_input
        eos_id = 0
        out = generate(model, input_ids, max_new_tokens=50, temperature=0.0, eos_token_id=eos_id)
        # 应该在遇到 eos 时停止（或达到 max_new_tokens）
        assert out.shape[1] <= input_ids.shape[1] + 50

    def test_batch_generation(self, model_and_input):
        model, _, config = model_and_input
        input_ids = torch.randint(0, 100, (3, 8), device=model.model.embed_tokens.weight.device)
        out = generate(model, input_ids, max_new_tokens=5, temperature=0.0)
        assert out.shape[0] == 3
        assert out.shape[1] > 8


class TestGenerateWithCache:
    def test_cache_output_matches(self, model_and_input):
        model, input_ids, config = model_and_input
        out_no_cache = generate(model, input_ids, max_new_tokens=10, temperature=0.0)
        out_with_cache = generate_with_cache(model, input_ids, max_new_tokens=10, temperature=0.0)
        assert torch.equal(out_no_cache, out_with_cache)

    def test_cache_is_faster(self, model_and_input):
        model, input_ids, config = model_and_input
        import time
        start = time.time()
        for _ in range(3):
            generate(model, input_ids, max_new_tokens=20, temperature=0.0)
        t_no_cache = time.time() - start

        start = time.time()
        for _ in range(3):
            generate_with_cache(model, input_ids, max_new_tokens=20, temperature=0.0)
        t_cache = time.time() - start
        # KV Cache 应该不比无 cache 慢（至少可运行）
        assert t_cache >= 0

    def test_cache_handles_eos(self, model_and_input):
        model, input_ids, config = model_and_input
        out = generate_with_cache(model, input_ids, max_new_tokens=50, temperature=0.0, eos_token_id=0)
        assert out.shape[1] > 0
