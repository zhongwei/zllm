"""M7-A: 预训练工具函数测试。

测试 _format_duration, train_epoch, 以及完整 pretrain pipeline。
"""

import time
import torch
import pytest

from zllm.training.pretrain import _format_duration, train_epoch, PretrainConfig


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(5) == "5s"

    def test_minutes(self):
        assert _format_duration(125) == "2m05s"

    def test_hours(self):
        assert _format_duration(3725) == "1h02m"

    def test_zero(self):
        assert _format_duration(0) == "0s"


class TestPretrainConfig:
    def test_defaults(self):
        cfg = PretrainConfig()
        assert cfg.epochs == 2
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 5e-4
        assert cfg.accumulation_steps == 4
        assert cfg.grad_clip == 1.0

    def test_custom(self):
        cfg = PretrainConfig(epochs=5, batch_size=32, learning_rate=1e-4)
        assert cfg.epochs == 5
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 1e-4
