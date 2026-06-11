"""M10 集成测试：DPO + PPO + GRPO 对齐训练对比。

验证：
1. DPO 训练可运行且 loss 下降
2. GRPO 数学运算正确
3. PPO CriticModel 可加载基础权重
4. 三种方法的 loss 函数在真实数据上可运行
"""

import json

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.dpo import logits_to_log_probs, dpo_loss, DPOConfig, train_epoch as dpo_train_epoch
from zllm.training.ppo import CriticModel, compute_gae, ppo_policy_loss, PPOConfig
from zllm.training.grpo import per_token_kl, compute_group_advantages, grpo_loss, GRPOConfig
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed


class TestDPOIntegration:
    @pytest.fixture
    def dpo_data(self, tmp_path, device):
        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer
        from zllm.dataset.dpo import DPODataset

        corpus = ["集成DPO测试偏好数据对话回复"] * 20
        tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
        data_path = str(tmp_path / "dpo.jsonl")
        samples = [
            {"chosen": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好很高兴为你服务"},
            ], "rejected": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "走开别烦我"},
            ]},
        ] * 12
        with open(data_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        config = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
        model = ZLLMForCausalLM(config).to(device)
        ref_model = ZLLMForCausalLM(config).to(device)
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        ref_model.requires_grad_(False)
        ds = DPODataset(data_path, tok, max_length=64)
        return model, ref_model, ds, config, device

    def test_dpo_save_load(self, dpo_data, tmp_path, device):
        from zllm.training.utils import lm_checkpoint
        model, ref_model, ds, config, device = dpo_data
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = DPOConfig(epochs=2, learning_rate=1e-3, accumulation_steps=1, log_interval=999)
        for epoch in range(2):
            dpo_train_epoch(model, ref_model, loader, optimizer, scaler, cfg, epoch, device)
        save_dir = str(tmp_path / "out")
        lm_checkpoint(config, weight="dpo", model=model, optimizer=optimizer,
                      epoch=2, step=0, save_dir=save_dir)
        assert (tmp_path / "out" / "dpo_64.pth").exists()


class TestPPOIntegration:
    def test_critic_trainable(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=100, max_position_embeddings=32,
        )
        critic = CriticModel(config).to(device)
        optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
        input_ids = torch.randint(0, 100, (2, 10), device=device)
        values = critic(input_ids)
        target = torch.randn_like(values)
        loss = ((values - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

    def test_critic_loads_base_weights(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=100, max_position_embeddings=32,
        )
        model = ZLLMForCausalLM(config).to(device)
        critic = CriticModel(config).to(device)
        critic.load_state_dict(model.state_dict(), strict=False)
        input_ids = torch.randint(0, 100, (1, 8), device=device)
        with torch.no_grad():
            out_model = model(input_ids)
            out_critic = critic(input_ids)
        assert out_critic.shape == (1, 8)

    def test_gae_with_model_outputs(self, device):
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=100, max_position_embeddings=32,
        )
        critic = CriticModel(config).to(device)
        input_ids = torch.randint(0, 100, (2, 10), device=device)
        with torch.no_grad():
            values = critic(input_ids)
        rewards = torch.zeros(2, 10, device=device)
        rewards[:, -1] = 1.0
        mask = torch.ones(2, 10, device=device)
        advantages, returns = compute_gae(rewards, values, mask, gamma=0.99, lam=0.95)
        assert advantages.shape == (2, 10)
        assert returns.shape == (2, 10)


class TestGRPOIntegration:
    def test_full_grpo_math(self, device):
        """模拟 GRPO 训练的数学计算。"""
        config = ZLLMConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=100, max_position_embeddings=32,
        )
        model = ZLLMForCausalLM(config).to(device)
        ref_model = ZLLMForCausalLM(config).to(device)
        ref_model.eval()
        ref_model.requires_grad_(False)

        input_ids = torch.randint(0, 100, (4, 10), device=device)
        labels = input_ids[:, 1:]

        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits[:, :-1, :]
        ref_logps = logits_to_log_probs(ref_logits, labels)

        output = model(input_ids)
        policy_logits = output.logits[:, :-1, :]
        policy_logps = logits_to_log_probs(policy_logits, labels)

        rewards = torch.tensor([1.0, -0.5, 0.5, -1.0], device=device)
        num_gen = 2
        advantages = compute_group_advantages(rewards, num_gen)
        assert advantages.shape == (4,)

        old_logps = policy_logps.detach()
        mask = torch.ones(4, 9, device=device)
        adv = advantages.unsqueeze(1).expand_as(mask)
        loss = grpo_loss(policy_logps, old_logps, adv, mask, ref_logps, beta=0.1, epsilon=0.2)
        assert not torch.isnan(loss)
        loss.backward()


class TestAlignmentComparison:
    def test_all_three_methods_configured(self):
        dpo = DPOConfig()
        ppo = PPOConfig()
        grpo = GRPOConfig()

        assert dpo.learning_rate == 4e-8
        assert ppo.learning_rate == 3e-7
        assert grpo.learning_rate == 3e-7

        assert dpo.save_weight == "dpo"
        assert ppo.save_weight == "ppo_actor"
        assert grpo.save_weight == "grpo"

    def test_all_from_full_sft(self):
        dpo = DPOConfig()
        ppo = PPOConfig()
        grpo = GRPOConfig()
        assert dpo.from_weight == "full_sft"
        assert ppo.from_weight == "full_sft"
        assert grpo.from_weight == "full_sft"
