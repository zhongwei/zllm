"""RMSNorm — Root Mean Square Layer Normalization。

比 LayerNorm 更高效：不需要计算均值，只用 RMS 归一化。
内部 float32 计算保证数值稳定性。
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight * self.norm(x.float())).type_as(x)
