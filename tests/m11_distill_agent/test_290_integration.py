"""M11 集成测试：蒸馏 + Agent RL 端到端验证。

验证：
1. MoE → Dense 蒸馏可运行
2. 蒸馏 loss 下降
3. Agent 工具调用端到端
4. parse → execute → validate 流程
"""

import json

import torch
import pytest
from torch.utils.data import DataLoader

from zllm.config import ZLLMConfig
from zllm.model.causal_lm import ZLLMForCausalLM
from zllm.training.distillation import distillation_loss, DistillConfig, train_epoch as distill_train_epoch
from zllm.training.agent_rl import (
    TOOLS, execute_tool, parse_tool_calls,
    validate_gt_in_text, calculate_agent_reward,
)
from zllm.training.amp import GradScalerManager
from zllm.training.utils import setup_seed


class TestDistillationIntegration:
    @pytest.fixture
    def distill_data(self, tmp_path, device):
        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer
        from zllm.dataset.sft import SFTDataset

        corpus = ["蒸馏集成测试数据对话回复内容语言模型"] * 20
        tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
        data_path = str(tmp_path / "sft.jsonl")
        convs = [{"conversations": [
            {"role": "user", "content": "测试"},
            {"role": "assistant", "content": "集成蒸馏回复"},
        ]}] * 16
        with open(data_path, "w", encoding="utf-8") as f:
            for c in convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        config = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
        student = ZLLMForCausalLM(config).to(device)
        teacher = ZLLMForCausalLM(config).to(device)
        ds = SFTDataset(data_path, tok, max_length=64)
        return student, teacher, ds, config, device

    def test_distill_with_teacher(self, distill_data):
        student, teacher, ds, config, device = distill_data
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(student.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = DistillConfig(epochs=3, alpha=0.5, learning_rate=5e-3, accumulation_steps=1, log_interval=999)

        first_losses, last_losses = [], []
        for epoch in range(3):
            losses = distill_train_epoch(student, teacher, loader, optimizer, scaler, cfg, epoch, device)
            if epoch == 0:
                first_losses = losses
            last_losses = losses
        assert sum(last_losses) / len(last_losses) < sum(first_losses) / len(first_losses)

    def test_moe_teacher_dense_student(self, tmp_path, device):
        """MoE 教师 → Dense 学生蒸馏。"""
        setup_seed(42)
        from zllm.tokenizer.trainer import train_tokenizer
        from zllm.dataset.sft import SFTDataset

        corpus = ["MoE蒸馏测试数据"] * 20
        tok = train_tokenizer(corpus, vocab_size=300, save_dir=str(tmp_path / "tok"))
        data_path = str(tmp_path / "sft.jsonl")
        convs = [{"conversations": [
            {"role": "user", "content": "测试"},
            {"role": "assistant", "content": "MoE蒸馏回复"},
        ]}] * 16
        with open(data_path, "w", encoding="utf-8") as f:
            for c in convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        config_s = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
        config_t = ZLLMConfig(
            vocab_size=tok.get_vocab_size(),
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, use_moe=True, max_position_embeddings=64,
        )
        student = ZLLMForCausalLM(config_s).to(device)
        teacher = ZLLMForCausalLM(config_t).to(device)
        ds = SFTDataset(data_path, tok, max_length=64)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(student.parameters(), lr=5e-3)
        scaler = GradScalerManager(enabled=False)
        cfg = DistillConfig(epochs=2, alpha=0.5, learning_rate=5e-3, accumulation_steps=1, log_interval=999)
        losses = distill_train_epoch(student, teacher, loader, optimizer, scaler, cfg, 0, device)
        assert all(l > 0 for l in losses)


class TestAgentRLIntegration:
    def test_full_tool_calling_pipeline(self):
        """parse → execute → validate 完整流程。"""
        model_output = (
            "让我查询一下北京的天气。\n"
            "```json\n"
            '{"name": "get_current_weather", "arguments": {"location": "北京"}}\n'
            "```"
        )

        calls = parse_tool_calls(model_output)
        assert len(calls) == 1

        result = execute_tool(calls[0]["name"], calls[0]["arguments"])
        assert result is not None
        assert "temperature" in result

        response = f"根据查询结果，北京今天的天气是{result['condition']}，气温{result['temperature']}。"
        assert validate_gt_in_text("28°C", response)

        reward = calculate_agent_reward(response, gt_answer="28°C", tool_calls=calls)
        assert reward > 1.5

    def test_multi_tool_pipeline(self):
        """多工具调用。"""
        text = (
            '```json\n{"name": "calculate_math", "arguments": {"expression": "100*7.21"}}\n```\n'
            '```json\n{"name": "get_exchange_rate", "arguments": {"from_currency": "USD", "to_currency": "CNY"}}\n```'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2

        for call in calls:
            result = execute_tool(call["name"], call["arguments"])
            assert result is not None

    def test_reward_differentiates_quality(self):
        good = "让我查询天气。```json\n{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"北京\"}}\n```北京今天28°C。"
        bad = "我不知道。"
        r_good = calculate_agent_reward(good, tool_calls=parse_tool_calls(good))
        r_bad = calculate_agent_reward(bad)
        assert r_good > r_bad
