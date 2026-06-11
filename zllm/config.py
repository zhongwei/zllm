"""ZLLM 模型配置。

继承 PretrainedConfig，兼容 Transformers 生态。
对齐 Qwen3 / minimind-3：GQA、RoPE、SwiGLU、MoE、Weight Tying。
"""

import math

from transformers import PretrainedConfig


class ZLLMConfig(PretrainedConfig):
    model_type = "zllm"

    def __init__(
        self,
        vocab_size=6400,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        intermediate_size=None,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        flash_attn=True,
        # MoE
        use_moe=False,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=None,
        norm_topk_prob=True,
        router_aux_loss_coef=5e-4,
        # YaRN RoPE Scaling
        inference_rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        # π 缩放：ceil(hidden_size * π / 64) * 64，对齐 64 倍数提升 Tensor Core 利用率
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.flash_attn = flash_attn
        # MoE
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = (
            moe_intermediate_size
            if moe_intermediate_size is not None
            else self.intermediate_size
        )
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.inference_rope_scaling = inference_rope_scaling
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads
