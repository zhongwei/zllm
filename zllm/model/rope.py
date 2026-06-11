"""RoPE — Rotary Position Embedding。

用旋转矩阵编码相对位置信息。
支持 YaRN 长序列外推。

关键函数：
- precompute_freqs_cis: 预计算 cos/sin 频率表
- rotate_half: 旋转一半维度
- apply_rotary_pos_emb: 对 Q/K 施加旋转位置编码
"""

import math

import torch


def precompute_freqs_cis(
    dim: int,
    end: int = 32768,
    rope_base: float = 1e6,
    rope_scaling: dict = None,
):
    freqs = 1.0 / (
        rope_base
        ** (
            torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
        )
    )
    attn_factor = 1.0
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:

            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                    2 * math.log(rope_base)
                )

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]),
        dim=-1,
    )


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed
