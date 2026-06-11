"""M5 集成测试：DataLoader + 模型前向。

验证所有 Dataset 可被 DataLoader 批量加载，且 Pretrain/SFT 输出可直接喂给模型。
"""

import json

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.dataset.pretrain import PretrainDataset
from zllm.dataset.sft import SFTDataset
from zllm.dataset.dpo import DPODataset
from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM


@pytest.fixture
def tokenizer(tmp_path):
    from zllm.tokenizer.trainer import train_tokenizer
    corpus = ["你好世界测试语料BPE分词Transformer注意力机制偏好优化"] * 30
    return train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path / "tok"))


@pytest.fixture
def pretrain_jsonl(tmp_path):
    path = tmp_path / "pretrain.jsonl"
    lines = [{"text": f"测试文本第{i}行包含中文内容"} for i in range(8)]
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def sft_jsonl(tmp_path):
    path = tmp_path / "sft.jsonl"
    lines = [
        {"conversations": [
            {"role": "user", "content": f"问题{i}"},
            {"role": "assistant", "content": f"回答{i}"},
        ]}
        for i in range(8)
    ]
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")
    return str(path)


class TestDataLoaderIntegration:
    def test_pretrain_dataloader(self, pretrain_jsonl, tokenizer):
        ds = PretrainDataset(pretrain_jsonl, tokenizer, max_length=32)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch[0].shape == (4, 32)
        assert batch[1].shape == (4, 32)

    def test_sft_dataloader(self, sft_jsonl, tokenizer):
        ds = SFTDataset(sft_jsonl, tokenizer, max_length=64)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch[0].shape == (4, 64)
        assert batch[1].shape == (4, 64)

    def test_pretrain_into_model(self, pretrain_jsonl, tokenizer, device):
        config = ZLLMConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=32,
        )
        model = ZLLMForCausalLM(config).to(device)
        ds = PretrainDataset(pretrain_jsonl, tokenizer, max_length=32)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        input_ids, labels = next(iter(loader))
        input_ids, labels = input_ids.to(device), labels.to(device)
        out = model(input_ids, labels=labels)
        assert out.loss is not None
        assert torch.isfinite(out.loss)

    def test_sft_into_model(self, sft_jsonl, tokenizer, device):
        config = ZLLMConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
        model = ZLLMForCausalLM(config).to(device)
        ds = SFTDataset(sft_jsonl, tokenizer, max_length=64)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        input_ids, labels = next(iter(loader))
        input_ids, labels = input_ids.to(device), labels.to(device)
        out = model(input_ids, labels=labels)
        assert out.loss is not None
        assert torch.isfinite(out.loss)

    def test_dpo_dataloader(self, sft_jsonl, tokenizer):
        path = sft_jsonl.replace("sft", "dpo")
        lines = []
        for i in range(4):
            lines.append({
                "chosen": [
                    {"role": "user", "content": f"问题{i}"},
                    {"role": "assistant", "content": f"好回答{i}"},
                ],
                "rejected": [
                    {"role": "user", "content": f"问题{i}"},
                    {"role": "assistant", "content": f"差回答{i}"},
                ],
            })
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(l, ensure_ascii=False) for l in lines))
        ds = DPODataset(path, tokenizer, max_length=32)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["x_chosen"].shape == (2, 31)
