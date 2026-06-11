"""GPU 性能优化配置。

TF32、cuDNN benchmark、Flash SDPA 等硬件加速开关。
"""

import torch


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def enable_cudnn_benchmark():
    torch.backends.cudnn.benchmark = True


def enable_flash_sdpa():
    torch.backends.cuda.enable_flash_sdp(True)


def setup_gpu_performance():
    enable_tf32()
    enable_cudnn_benchmark()
    enable_flash_sdpa()
