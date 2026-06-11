"""验证共享 fixtures 可用。"""


def test_device_fixture(device):
    import torch

    assert device is not None
    assert isinstance(device, torch.device)


def test_small_config_fixture(small_config):
    assert small_config.hidden_size == 64
    assert small_config.num_hidden_layers == 2


def test_default_config_fixture(default_config):
    assert default_config.hidden_size == 768
    assert default_config.num_hidden_layers == 8
    assert default_config.vocab_size == 6400


def test_gqa_config(default_config):
    assert default_config.num_attention_heads == 8
    assert default_config.num_key_value_heads == 4


def test_pi_scaled_intermediate(default_config):
    import math

    expected = math.ceil(768 * math.pi / 64) * 64
    assert default_config.intermediate_size == expected


def test_head_dim(default_config):
    assert default_config.head_dim == 96
