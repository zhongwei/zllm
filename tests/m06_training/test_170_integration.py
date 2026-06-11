"""M6 集成测试：完整训练循环（mini 版）。

用小模型 + 小数据验证 setup_seed → init_model → train_step → checkpoint 全流程。
"""

import os
import json

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.dataset.pretrain import PretrainDataset
from zllm.training.utils import setup_seed, init_model, lm_checkpoint, get_lr
from zllm.training.amp import GradScalerManager, train_step
from zllm.training.gpu import setup_gpu_performance


@pytest.fixture
def mini_setup(tmp_path, device):
    setup_gpu_performance()
    setup_seed(42)
    from zllm.tokenizer.trainer import train_tokenizer
    corpus = ["你好世界测试语料训练验证循环"] * 20
    tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
    data_path = str(tmp_path / "data.jsonl")
    with open(data_path, "w", encoding="utf-8") as f:
        for i in range(16):
            f.write(json.dumps({"text": f"训练文本第{i}行内容"}, ensure_ascii=False) + "\n")
    config = ZLLMConfig(
        vocab_size=tok.get_vocab_size(),
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=32,
    )
    model = init_model(config, from_weight="none", save_dir=str(tmp_path), device=str(device))
    ds = PretrainDataset(data_path, tok, max_length=32)
    return model, ds, config, tmp_path


class TestM6Integration:
    def test_full_train_loop(self, mini_setup, device):
        model, ds, config, tmp_path = mini_setup
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scaler = GradScalerManager(enabled=False)

        losses = []
        for step, (input_ids, labels) in enumerate(loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            lr = get_lr(step, 20, base_lr=5e-4)
            for g in optimizer.param_groups:
                g["lr"] = lr
            loss = train_step(model, optimizer, input_ids, labels, scaler,
                              accumulation_steps=1, max_grad_norm=1.0, current_step=step)
            losses.append(loss)
            if step >= 4:
                break
        assert losses[-1] < losses[0]

    def test_save_resume_restore(self, mini_setup, device):
        model, ds, config, tmp_path = mini_setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        labels = ids.clone()
        scaler = GradScalerManager(enabled=False)
        train_step(model, optimizer, ids, labels, scaler)

        lm_checkpoint(config, weight="pretrain", model=model, optimizer=optimizer,
                      epoch=1, step=5, save_dir=str(tmp_path / "ckpts"))

        data = lm_checkpoint(config, weight="pretrain", save_dir=str(tmp_path / "ckpts"))
        assert data["epoch"] == 1
        assert data["step"] == 5

        model2 = init_model(config, from_weight="pretrain", save_dir=str(tmp_path / "ckpts"), device=str(device))
        with torch.no_grad():
            out1 = model(ids)
            out2 = model2(ids)
        assert torch.allclose(out1.logits, out2.logits, atol=1e-2)

    def test_gradient_accumulation_loss_decreases(self, mini_setup, device):
        model, ds, config, tmp_path = mini_setup
        loader = DataLoader(ds, batch_size=2, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scaler = GradScalerManager(enabled=False)

        first_loss = None
        last_loss = None
        step = 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            loss = train_step(model, optimizer, input_ids, labels, scaler,
                              accumulation_steps=2, max_grad_norm=1.0, current_step=step)
            if step == 0:
                first_loss = loss
            if step >= 7:
                last_loss = loss
                break
            step += 1
        assert last_loss < first_loss
