"""步骤 26-29: 从零实现 BPE 核心算法。

测试 byte-level 编码、相邻对计数、合并操作、训练循环。
"""

from zllm.tokenizer.bpe import (
    byte_level_encode,
    get_pair_counts,
    merge,
    train_bpe,
)


class TestByteLevelEncode:
    def test_ascii_text(self):
        # 'ab' -> [97, 98]
        assert byte_level_encode("ab") == [97, 98]

    def test_chinese_text(self):
        # 中文字符是多字节 UTF-8
        ids = byte_level_encode("你")
        assert isinstance(ids, list)
        assert all(0 <= b < 256 for b in ids)
        assert len(ids) == 3  # '你' 是 3 字节 UTF-8

    def test_empty_string(self):
        assert byte_level_encode("") == []

    def test_roundtrip_via_bytes(self):
        text = "hello 世界"
        ids = byte_level_encode(text)
        # bytes(ids).decode('utf-8') 应还原原文
        assert bytes(ids).decode("utf-8") == text


class TestGetPairCounts:
    def test_single_sequence(self):
        # [1,2,3,1,2] -> pairs: (1,2):2, (2,3):1, (3,1):1
        counts = get_pair_counts([[1, 2, 3, 1, 2]])
        assert counts[(1, 2)] == 2
        assert counts[(2, 3)] == 1
        assert counts[(3, 1)] == 1

    def test_multiple_sequences(self):
        counts = get_pair_counts([[1, 2], [1, 2], [3, 4]])
        assert counts[(1, 2)] == 2
        assert counts[(3, 4)] == 1

    def test_short_sequence_no_pairs(self):
        assert get_pair_counts([[1]]) == {}
        assert get_pair_counts([[]]) == {}

    def test_most_frequent_pair(self):
        counts = get_pair_counts([[1, 2, 1, 2, 1, 2]])
        assert max(counts, key=counts.get) == (1, 2)


class TestMerge:
    def test_merge_all_occurrences(self):
        assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]

    def test_no_match(self):
        assert merge([1, 2, 3], (9, 9), 99) == [1, 2, 3]

    def test_adjacent_merges_not_double_merged(self):
        # [1,2,1,2] -> merging (1,2) -> [99,99], not [99] twice merged
        assert merge([1, 2, 1, 2], (1, 2), 99) == [99, 99]

    def test_overlapping_pair(self):
        # [1,1,1] merging (1,1) -> [99,1]
        assert merge([1, 1, 1], (1, 1), 99) == [99, 1]


class TestTrainBPE:
    def test_returns_merges_dict(self):
        texts = ["abab", "ab"]
        merges = train_bpe(texts, vocab_size=258)  # 256 + 2 merges
        assert isinstance(merges, dict)
        assert len(merges) == 2

    def test_first_merge_is_most_frequent(self):
        # 'ab' appears 3 times, most frequent byte pair is (97,98)
        texts = ["abab", "ab"]
        merges = train_bpe(texts, vocab_size=257)  # 1 merge
        assert (97, 98) in merges
        assert merges[(97, 98)] == 256

    def test_new_ids_are_sequential(self):
        merges = train_bpe(["aabb"], vocab_size=258)
        ids = sorted(merges.values())
        assert ids == [256, 257]

    def test_no_merges_needed(self):
        # vocab_size == 256 means no merges
        merges = train_bpe(["abc"], vocab_size=256)
        assert merges == {}
