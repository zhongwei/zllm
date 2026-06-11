"""zllm tokenizer 模块。

提供：
- bpe: 从零实现的 BPE 算法（教学版）
- trainer: 生产级 tokenizer 训练（tokenizers 库）
- special_tokens: 特殊 token 定义
- chat_template: 对话格式化模板
"""

from zllm.tokenizer.bpe import byte_level_encode, decode, encode, train_bpe
from zllm.tokenizer.chat_template import CHAT_TEMPLATE, render_messages
from zllm.tokenizer.special_tokens import ALL_SPECIAL_TOKENS, IM_END, IM_START
from zllm.tokenizer.trainer import load_tokenizer, train_tokenizer

__all__ = [
    "ALL_SPECIAL_TOKENS",
    "CHAT_TEMPLATE",
    "IM_END",
    "IM_START",
    "byte_level_encode",
    "decode",
    "encode",
    "load_tokenizer",
    "render_messages",
    "train_bpe",
    "train_tokenizer",
]
