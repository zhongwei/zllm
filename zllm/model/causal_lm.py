"""ZLLMForCausalLM — 完整语言模型。

组件：ZLLMModel + lm_head + loss 计算
支持：Weight Tying、ignore_index、logits_to_keep 优化、MoE 辅助损失。

参数命名对齐 minimind：model / lm_head。
_tied_weights_keys 对齐：lm_head.weight → model.embed_tokens.weight。
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from zllm.config import ZLLMConfig
from zllm.model.backbone import ZLLMModel


class ZLLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ZLLMConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config=None):
        self.config = config or ZLLMConfig()
        super().__init__(self.config)
        self.model = ZLLMModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else slice(None)
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
