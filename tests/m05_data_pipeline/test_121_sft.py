"""M5-C: SFTDataset 测试。

测试 chat template 渲染、generate_labels 标签构造、prompt 掩码。
"""

import json

import torch
import pytest

from zllm.dataset.sft import SFTDataset


@pytest.fixture
def sft_jsonl(tmp_path):
    path = tmp_path / "sft.jsonl"
    lines = [
        {"conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
        ]},
        {"conversations": [
            {"role": "user", "content": "什么是BPE？"},
            {"role": "assistant", "content": "BPE是字节对编码。"},
        ]},
    ]
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def sft_dataset(sft_jsonl, tmp_path):
    from zllm.tokenizer.trainer import train_tokenizer
    corpus = ["你好世界测试语料BPE字节对编码Transformer注意力"] * 30
    tok = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path / "tok"))
    return SFTDataset(sft_jsonl, tok, max_length=128)


class TestSFTDataset:
    def test_len(self, sft_dataset):
        assert len(sft_dataset) == 2

    def test_item_shapes(self, sft_dataset):
        input_ids, labels = sft_dataset[0]
        assert input_ids.shape == (128,)
        assert labels.shape == (128,)

    def test_dtypes(self, sft_dataset):
        input_ids, labels = sft_dataset[0]
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long

    def test_labels_have_some_valid(self, sft_dataset):
        """assistant 回复区域应有非 -100 的 label。"""
        _, labels = sft_dataset[0]
        assert (labels != -100).any()

    def test_prompt_masked(self, sft_dataset):
        """user 区域应被 mask（-100）。"""
        _, labels = sft_dataset[0]
        assert (labels == -100).any()

    def test_pad_masked(self, sft_dataset):
        input_ids, labels = sft_dataset[0]
        pad_mask = input_ids == sft_dataset.tokenizer.pad_token_id
        assert torch.all(labels[pad_mask] == -100)


class TestGenerateLabels:
    def test_all_masked_when_no_assistant(self, sft_dataset):
        ids = [0] * 10
        labels = sft_dataset.generate_labels(ids)
        assert all(l == -100 for l in labels)

    def test_assistant_region_marked(self, sft_dataset):
        bos = sft_dataset.bos_id
        eos = sft_dataset.eos_id
        # 构造：[bos, a, b, eos, pad]
        ids = bos + [10, 20] + eos + [sft_dataset.tokenizer.pad_token_id]
        labels = sft_dataset.generate_labels(ids)
        # a, b, eos 应有 label
        assert labels[len(bos)] == 10
        assert labels[len(bos) + 1] == 20
