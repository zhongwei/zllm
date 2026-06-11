"""zllm training 包。

提供：
- utils: 训练工具（seed/lr/checkpoint/logger/sampler）
- amp: 混合精度 + 梯度累积 + 梯度裁剪
- gpu: GPU 性能优化（TF32/cudnn/flash）
"""

from zllm.training.amp import GradScalerManager, train_step
from zllm.training.gpu import enable_flash_sdpa, enable_tf32, setup_gpu_performance
from zllm.training.utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    get_model_params,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

__all__ = [
    "GradScalerManager",
    "Logger",
    "SkipBatchSampler",
    "enable_flash_sdpa",
    "enable_tf32",
    "get_lr",
    "get_model_params",
    "init_distributed_mode",
    "init_model",
    "is_main_process",
    "lm_checkpoint",
    "setup_gpu_performance",
    "setup_seed",
    "train_step",
]
