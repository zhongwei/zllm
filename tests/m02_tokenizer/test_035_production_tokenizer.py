"""步骤 35-43: 生产级 Tokenizer（使用 HuggingFace tokenizers 库）。

测试 BPE 训练、保存、加载、编解码、特殊 token。
"""

import pytest
from zllm.tokenizer.trainer import train_tokenizer, load_tokenizer


@pytest.fixture
def corpus():
    """小型训练语料（中文为主）。"""
    base = [
        "你好世界，这是 zllm 项目的测试语料。",
        "从零训练中文大语言模型。",
        "Transformer 是一种基于自注意力机制的深度学习架构。",
        "BPE 分词算法通过合并高频字节对来构建词表。",
        "预训练阶段让模型学习语言的统计规律。",
    ]
    return base * 20  # 重复以提供足够统计量


@pytest.fixture
def trained_tokenizer(corpus, tmp_path):
    return train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path))


class TestTrainTokenizer:
    def test_returns_tokenizer(self, corpus, tmp_path):
        tok = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path))
        assert tok is not None

    def test_vocab_size_not_exceeding_target(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocab()
        assert len(vocab) <= 500  # 含特殊 token 可能略超 400

    def test_save_creates_files(self, corpus, tmp_path):
        train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path))
        import os

        assert os.path.exists(os.path.join(str(tmp_path), "tokenizer.json"))


class TestSpecialTokens:
    def test_im_start_is_single_token(self, trained_tokenizer):
        ids = trained_tokenizer.encode("<|im_start|>").ids
        assert len(ids) == 1

    def test_im_end_is_single_token(self, trained_tokenizer):
        ids = trained_tokenizer.encode("<|im_end|>").ids
        assert len(ids) == 1

    def test_reasoning_tokens_single(self, trained_tokenizer):
        assert len(trained_tokenizer.encode("<reasoningchain_start>").ids) == 1

    def test_special_token_ids_at_start(self, trained_tokenizer):
        # 特殊 token 应占据词表前部 ID（训练时 special_tokens 顺序决定）
        vocab = trained_tokenizer.get_vocab()
        assert vocab["<|im_start|>"] < 50


class TestEncodeDecode:
    def test_encode_returns_ids(self, trained_tokenizer):
        ids = trained_tokenizer.encode("你好").ids
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_decode_text(self, trained_tokenizer):
        ids = trained_tokenizer.encode("你好世界").ids
        assert trained_tokenizer.decode(ids) == "你好世界"

    def test_roundtrip_chinese(self, trained_tokenizer):
        text = "中文大语言模型"
        ids = trained_tokenizer.encode(text).ids
        assert trained_tokenizer.decode(ids) == text

    def test_roundtrip_mixed(self, trained_tokenizer):
        text = "Python 是一门编程语言"
        ids = trained_tokenizer.encode(text).ids
        assert trained_tokenizer.decode(ids) == text


class TestSaveLoad:
    def test_load_and_compare(self, corpus, tmp_path):
        tok1 = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path))
        tok2 = load_tokenizer(str(tmp_path))
        text = "你好世界"
        assert tok1.encode(text).ids == tok2.encode(text).ids

    def test_load_from_explicit_path(self, corpus, tmp_path):
        train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path))
        import os

        tok = load_tokenizer(os.path.join(str(tmp_path), "tokenizer.json"))
        assert tok is not None


class TestCompression:
    def test_chinese_compression(self, trained_tokenizer):
        """中文字符约 1.5~1.7 字符/token，编码后 token 数应少于字符数。"""
        text = "从零训练中文大语言模型" * 5
        ids = trained_tokenizer.encode(text).ids
        assert len(ids) < len(text)
