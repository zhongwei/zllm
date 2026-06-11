"""M2 集成测试：完整 tokenizer 管线。

train → save → load → render chat → encode → decode 往返。
"""

import pytest

from zllm.tokenizer.chat_template import render_messages
from zllm.tokenizer.trainer import train_tokenizer, load_tokenizer


@pytest.fixture
def tokenizer(tmp_path):
    corpus = [
        "你好世界，这是 zllm 项目的测试语料。",
        "从零训练中文大语言模型。",
        "Transformer 是一种基于自注意力机制的架构。",
        "BPE 分词算法合并高频字节对。",
        "预训练让模型学习语言规律。",
    ] * 20
    return train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path))


class TestFullPipeline:
    def test_train_save_load_encode_decode(self, tmp_path):
        corpus = ["你好世界测试语料"] * 20
        # 训练并保存
        tok = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path))
        # 加载
        loaded = load_tokenizer(str(tmp_path))
        # 编解码一致
        text = "你好世界"
        assert tok.encode(text).ids == loaded.encode(text).ids
        assert loaded.decode(loaded.encode(text).ids) == text

    def test_chat_render_then_tokenize(self, tokenizer):
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
        ]
        rendered = render_messages(messages)
        # 渲染结果可被 tokenizer 正确编码
        ids = tokenizer.encode(rendered).ids
        assert len(ids) > 0
        # 特殊 token 出现在编码结果中
        vocab = tokenizer.get_vocab()
        assert vocab["<|im_start|>"] in ids
        assert vocab["<|im_end|>"] in ids

    def test_generation_prompt_render_structure(self, tokenizer):
        messages = [{"role": "user", "content": "解释 BPE"}]
        rendered = render_messages(messages, add_generation_prompt=True)
        # 渲染文本以 assistant 生成提示结尾（chat_template 的职责）
        assert rendered.endswith("<|im_start|>assistant\n")
        # 可被 tokenizer 编码（不报错）
        ids = tokenizer.encode(rendered).ids
        assert len(ids) > 0

    def test_thinking_mode_render(self, tokenizer):
        messages = [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "思考后回答"},
        ]
        rendered = render_messages(messages, open_thinking=True)
        vocab = tokenizer.get_vocab()
        ids = tokenizer.encode(rendered).ids
        # 思考链 token 应出现
        assert vocab["<reasoningchain_start>"] in ids
        assert vocab["<reasoningchain_end>"] in ids
