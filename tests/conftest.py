"""共享测试 fixtures。"""

import pytest
import torch


@pytest.fixture
def device():
    """默认 GPU（CUDA 不可用时回退 CPU）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_config():
    """用于快速测试的小模型配置（dim=64, 2 layers）。"""
    from zllm.config import ZLLMConfig

    return ZLLMConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )


@pytest.fixture
def default_config():
    """生产默认配置（dim=768, 8 layers, vocab=6400）。"""
    from zllm.config import ZLLMConfig

    return ZLLMConfig()
