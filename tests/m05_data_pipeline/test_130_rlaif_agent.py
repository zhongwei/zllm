"""M5-E: RLAIFDataset + AgentRLDataset 测试。

RLAIFDataset: prompt-only，按 thinking_ratio 开启推理模式。
AgentRLDataset: messages + tools + ground truth。
"""

import json

import pytest

from zllm.dataset.rlaif import RLAIFDataset
from zllm.dataset.agent import AgentRLDataset


@pytest.fixture
def rlaif_jsonl(tmp_path):
    path = tmp_path / "rlaif.jsonl"
    lines = [
        {"conversations": [
            {"role": "user", "content": "解释注意力机制"},
            {"role": "assistant", "content": "注意力机制是..."},
        ]},
        {"conversations": [
            {"role": "user", "content": "什么是梯度下降"},
            {"role": "assistant", "content": "梯度下降是..."},
        ]},
    ]
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def rlaif_dataset(rlaif_jsonl, tmp_path):
    from zllm.tokenizer.trainer import train_tokenizer
    corpus = ["注意力机制梯度下降解释测试"] * 30
    tok = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path / "tok"))
    return RLAIFDataset(rlaif_jsonl, tok, max_length=128, thinking_ratio=0.5)


class TestRLAIFDataset:
    def test_len(self, rlaif_dataset):
        assert len(rlaif_dataset) == 2

    def test_returns_prompt(self, rlaif_dataset):
        item = rlaif_dataset[0]
        assert "prompt" in item
        assert "answer" in item
        assert isinstance(item["prompt"], str)

    def test_prompt_has_generation_prompt(self, rlaif_dataset):
        item = rlaif_dataset[0]
        assert "<|im_start|>assistant" in item["prompt"]


@pytest.fixture
def agent_jsonl(tmp_path):
    path = tmp_path / "agent.jsonl"
    lines = [
        {
            "conversations": [
                {"role": "system", "content": "", "tools": json.dumps([{"name": "calc", "description": "calculator"}])},
                {"role": "user", "content": "计算 1+1"},
                {"role": "assistant", "content": "结果是 2"},
            ],
            "gt": "2",
        },
    ]
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def agent_dataset(agent_jsonl, tmp_path):
    from zllm.tokenizer.trainer import train_tokenizer
    corpus = ["计算工具调用结果测试"] * 30
    tok = train_tokenizer(corpus, vocab_size=400, save_dir=str(tmp_path / "tok"))
    return AgentRLDataset(agent_jsonl, tok, max_length=128)


class TestAgentRLDataset:
    def test_len(self, agent_dataset):
        assert len(agent_dataset) == 1

    def test_returns_dict(self, agent_dataset):
        item = agent_dataset[0]
        assert isinstance(item, dict)

    def test_has_messages(self, agent_dataset):
        item = agent_dataset[0]
        assert "messages" in item
        assert isinstance(item["messages"], list)

    def test_has_tools(self, agent_dataset):
        item = agent_dataset[0]
        assert "tools" in item
        assert item["tools"] is not None

    def test_has_gt(self, agent_dataset):
        item = agent_dataset[0]
        assert "gt" in item
        assert item["gt"] == "2"

    def test_drops_last_message(self, agent_dataset):
        item = agent_dataset[0]
        assert item["messages"][-1]["role"] == "user"
