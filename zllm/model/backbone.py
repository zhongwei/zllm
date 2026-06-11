"""ZLLMModel — 模型主体（backbone）。

组件：embed_tokens + N × ZLLMBlock + final_norm + RoPE buffers
返回：(hidden_states, past_key_values, aux_loss)

参数命名对齐 minimind：embed_tokens / layers / norm / freqs_cos / freqs_sin。
"""

from torch import nn

from zllm.model.block import ZLLMBlock
from zllm.model.ffn import MOEFeedForward
from zllm.model.norms import RMSNorm
from zllm.model.rope import precompute_freqs_cis


class ZLLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.0))
        self.layers = nn.ModuleList(
            [ZLLMBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = (
            past_key_values[0][0].shape[1]
            if past_key_values[0] is not None
            else 0
        )
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=getattr(self.config, "rope_scaling", None),
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss
