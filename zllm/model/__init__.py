"""zllm 模型组件包。

提供：
- norms: RMSNorm 归一化
- rope: RoPE 位置编码 + YaRN 缩放
- attention: GQA 分组查询注意力
- ffn: SwiGLU FeedForward + MoE
- block: Transformer Block
- backbone: ZLLMModel 主体
- causal_lm: ZLLMForCausalLM 完整语言模型
"""

from zllm.model.attention import Attention, repeat_kv
from zllm.model.backbone import ZLLMModel
from zllm.model.block import ZLLMBlock
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.model.ffn import FeedForward, MOEFeedForward
from zllm.model.norms import RMSNorm
from zllm.model.rope import apply_rotary_pos_emb, precompute_freqs_cis, rotate_half

__all__ = [
    "Attention",
    "FeedForward",
    "MOEFeedForward",
    "RMSNorm",
    "ZLLMBlock",
    "ZLLMForCausalLM",
    "ZLLMModel",
    "apply_rotary_pos_emb",
    "precompute_freqs_cis",
    "repeat_kv",
    "rotate_half",
]
