"""步骤 41-42: 用训练好的 merges 进行 BPE 编码与解码。

测试 encode/decode 的正确性与往返一致性。
"""

from zllm.tokenizer.bpe import train_bpe, encode, decode


class TestEncode:
    def test_encode_reduces_sequence_length(self):
        # "abab" 训练后，"ab" 应被合并，编码结果比字节序列短
        merges = train_bpe(["ababab"], vocab_size=258)
        encoded = encode("ababab", merges)
        # 原始 6 字节 -> 合并后应更短
        assert len(encoded) < 6

    def test_encode_applies_merge(self):
        # (97,98)->256，encode("ab") 应得到 [256]
        merges = train_bpe(["ab"], vocab_size=257)
        assert encode("ab", merges) == [256]

    def test_encode_no_merges(self):
        # 空 merges，encode 等同 byte_level_encode
        assert encode("abc", {}) == [97, 98, 99]

    def test_encode_chinese(self):
        merges = train_bpe(["你好世界"], vocab_size=270)
        encoded = encode("你好世界", merges)
        assert len(encoded) < len("你好世界".encode("utf-8"))  # 被压缩


class TestDecode:
    def test_decode_single_merge(self):
        merges = train_bpe(["ab"], vocab_size=257)
        # encode("ab") = [256], decode([256]) = "ab"
        assert decode([256], merges) == "ab"

    def test_decode_mixed(self):
        merges = {(97, 98): 256}
        # [256, 99] -> "abc"
        assert decode([256, 99], merges) == "abc"

    def test_decode_pure_bytes(self):
        # 无合并的 token，直接还原字节
        assert decode([97, 98, 99], {}) == "abc"


class TestRoundtrip:
    def test_roundtrip_ascii(self):
        text = "hello world"
        merges = train_bpe([text] * 10, vocab_size=280)
        assert decode(encode(text, merges), merges) == text

    def test_roundtrip_chinese(self):
        text = "从零训练中文大语言模型"
        merges = train_bpe([text] * 10, vocab_size=280)
        assert decode(encode(text, merges), merges) == text

    def test_roundtrip_mixed(self):
        text = "Python 是最好的语言！Python rocks!"
        merges = train_bpe([text] * 10, vocab_size=290)
        assert decode(encode(text, merges), merges) == text
