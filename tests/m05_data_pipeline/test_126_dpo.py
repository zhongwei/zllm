"""M5-D: DPODataset 测试。

测试 chosen/rejected 渲染、loss_mask 构造、(x, y, mask) 三元组。
"""

import json

import torch
import pytest

from zllm.dataset.dpo import DPODataset


@pytest.fixture
def dpo_jsonl(tmp_path):
    path = tmp_path / "dpo.jsonl"
    lines = [
        {
            "chosen": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！很高兴见到你。"},
            ],
            "rejected": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "不知道。"},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def dpo_dataset(dpo_jsonl, tmp_path):
    from zllm.tokenizer.trainer import train_tokenizer
    corpus = ["你好世界测试语料BPE字节对编码不知道很高兴见到你"] * 30
    tok = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path / "tok"))
    return DPODataset(dpo_jsonl, tok, max_length=64)


class TestDPODataset:
    def test_len(self, dpo_dataset):
        assert len(dpo_dataset) == 1

    def test_returns_dict(self, dpo_dataset):
        item = dpo_dataset[0]
        assert isinstance(item, dict)

    def test_keys(self, dpo_dataset):
        item = dpo_dataset[0]
        assert "x_chosen" in item
        assert "y_chosen" in item
        assert "mask_chosen" in item
        assert "x_rejected" in item
        assert "y_rejected" in item
        assert "mask_rejected" in item

    def test_tensor_shapes(self, dpo_dataset):
        item = dpo_dataset[0]
        assert item["x_chosen"].shape == (63,)
        assert item["y_chosen"].shape == (63,)
        assert item["mask_chosen"].shape == (63,)

    def test_tensors_are_long(self, dpo_dataset):
        item = dpo_dataset[0]
        for key in ["x_chosen", "y_chosen", "mask_chosen", "x_rejected", "y_rejected", "mask_rejected"]:
            assert item[key].dtype == torch.long

    def test_mask_has_some_ones(self, dpo_dataset):
        item = dpo_dataset[0]
        assert (item["mask_chosen"] == 1).any()
        assert (item["mask_rejected"] == 1).any()

    def test_chosen_rejected_differ(self, dpo_dataset):
        item = dpo_dataset[0]
        assert not torch.equal(item["x_chosen"], item["x_rejected"])
