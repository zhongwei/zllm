"""M5-B: PretrainDataset 测试。

测试 JSONL 加载、tokenize、BOS/EOS、padding、labels 构造。
"""

import json

import torch
import pytest

from zllm.dataset.pretrain import PretrainDataset


@pytest.fixture
def pretrain_jsonl(tmp_path):
    path = tmp_path / "pretrain.jsonl"
    lines = [
        {"text": "你好世界，这是测试语料。"},
        {"text": "Transformer 是一种基于自注意力机制的架构。"},
        {"text": "BPE 分词算法合并高频字节对。"},
    ]
    path.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def pretrain_dataset(pretrain_jsonl, tmp_path):
    from zllm.tokenizer.trainer import train_tokenizer
    corpus = ["你好世界测试语料TransformerBPE分词算法"] * 20
    tok = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path / "tok"))
    return PretrainDataset(pretrain_jsonl, tok, max_length=32)


class TestPretrainDataset:
    def test_len(self, pretrain_dataset):
        assert len(pretrain_dataset) == 3

    def test_item_shapes(self, pretrain_dataset):
        input_ids, labels = pretrain_dataset[0]
        assert input_ids.shape == (32,)
        assert labels.shape == (32,)

    def test_input_ids_dtype(self, pretrain_dataset):
        input_ids, _ = pretrain_dataset[0]
        assert input_ids.dtype == torch.long

    def test_labels_dtype(self, pretrain_dataset):
        _, labels = pretrain_dataset[0]
        assert labels.dtype == torch.long

    def test_bos_token_at_start(self, pretrain_dataset):
        input_ids, _ = pretrain_dataset[0]
        assert input_ids[0].item() == pretrain_dataset.tokenizer.bos_token_id

    def test_pad_masked_in_labels(self, pretrain_dataset):
        input_ids, labels = pretrain_dataset[0]
        pad_id = pretrain_dataset.tokenizer.pad_token_id
        pad_mask = input_ids == pad_id
        assert torch.all(labels[pad_mask] == -100)

    def test_non_pad_labels_equal_input(self, pretrain_dataset):
        input_ids, labels = pretrain_dataset[0]
        pad_id = pretrain_dataset.tokenizer.pad_token_id
        non_pad = input_ids != pad_id
        assert torch.all(labels[non_pad] == input_ids[non_pad])
