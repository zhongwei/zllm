"""TokenizerAdapter — 轻量级适配器。

将 tokenizers.Tokenizer 包装为具有 transformers 风格 API 的对象。
提供 .bos_token_id / .eos_token_id / .pad_token_id / __call__ / apply_chat_template 等。
"""

from zllm.tokenizer.chat_template import render_messages


class TokenizerAdapter:
    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.bos_token = "<|im_start|>"
        self.eos_token = "<|im_end|>"
        self.pad_token = "<|pad|>"
        self.bos_token_id = tokenizer.token_to_id(self.bos_token)
        self.eos_token_id = tokenizer.token_to_id(self.eos_token)
        self.pad_token_id = tokenizer.token_to_id(self.pad_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def get_vocab(self):
        return self._tok.get_vocab()

    def token_to_id(self, token):
        return self._tok.token_to_id(token)

    def id_to_token(self, id):
        return self._tok.id_to_token(id)

    def encode(self, text, add_special_tokens=False, max_length=None, truncation=None):
        ids = self._tok.encode(text).ids
        if max_length is not None and (truncation or truncation is None):
            ids = ids[:max_length]
        return _Encoding(ids)

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, list):
            return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)
        return self._tok.decode([ids], skip_special_tokens=skip_special_tokens)

    def __call__(self, text=None, **kwargs):
        if isinstance(text, str):
            encoding = self._tok.encode(text)
            add_special_tokens = kwargs.get("add_special_tokens", False)
            if not add_special_tokens:
                return {"input_ids": encoding.ids}
            ids = [self.bos_token_id] + encoding.ids + [self.eos_token_id]
            return {"input_ids": ids}
        return {"input_ids": []}

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, tools=None, open_thinking=False, **kwargs
    ):
        text = render_messages(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            open_thinking=open_thinking,
        )
        if tokenize:
            return self._tok.encode(text).ids
        return text


class _Encoding:
    def __init__(self, ids):
        self.ids = ids
        self.input_ids = ids


def wrap(tokenizer):
    """将 raw tokenizers.Tokenizer 包装为 TokenizerAdapter。"""
    if isinstance(tokenizer, TokenizerAdapter):
        return tokenizer
    return TokenizerAdapter(tokenizer)
