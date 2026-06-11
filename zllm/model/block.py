"""Transformer Block — Pre-Norm 架构。

流程：input_layernorm → Attention → residual → post_attention_layernorm → FFN/MoE → residual

参数命名对齐 minimind：self_attn / input_layernorm / post_attention_layernorm / mlp。
"""

from torch import nn

from zllm.model.attention import Attention
from zllm.model.ffn import FeedForward, MOEFeedForward
from zllm.model.norms import RMSNorm


class ZLLMBlock(nn.Module):
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask,
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
