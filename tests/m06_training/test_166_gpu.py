"""M6-C: 分布式支持 + SkipBatchSampler + GPU 性能测试。

测试单卡模式下的分布式工具函数、SkipBatchSampler、TF32/cudnn 配置。
"""

import torch
import pytest
from torch.utils.data import SequentialSampler

from zllm.training.utils import (
    SkipBatchSampler,
    init_distributed_mode,
    setup_seed,
)
from zllm.training.gpu import enable_tf32, enable_flash_sdpa, setup_gpu_performance


class TestInitDistributed:
    def test_returns_zero_without_env(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        assert init_distributed_mode() == 0


class TestSkipBatchSampler:
    def test_no_skip(self):
        sampler = SequentialSampler(list(range(10)))
        batch_sampler = SkipBatchSampler(sampler, batch_size=2, skip_batches=0)
        batches = list(batch_sampler)
        assert len(batches) == 5
        assert batches[0] == [0, 1]

    def test_skip_2(self):
        sampler = SequentialSampler(list(range(10)))
        batch_sampler = SkipBatchSampler(sampler, batch_size=2, skip_batches=2)
        batches = list(batch_sampler)
        assert len(batches) == 3
        assert batches[0] == [4, 5]

    def test_skip_all(self):
        sampler = SequentialSampler(list(range(10)))
        batch_sampler = SkipBatchSampler(sampler, batch_size=2, skip_batches=5)
        batches = list(batch_sampler)
        assert len(batches) == 0

    def test_len(self):
        sampler = SequentialSampler(list(range(10)))
        batch_sampler = SkipBatchSampler(sampler, batch_size=3, skip_batches=1)
        assert len(batch_sampler) == 3  # ceil(10/3)=4, 4-1=3


class TestGPUPerformance:
    def test_enable_tf32(self):
        enable_tf32()

    def test_enable_flash_sdpa(self):
        enable_flash_sdpa()

    def test_setup_gpu_performance(self):
        setup_gpu_performance()
