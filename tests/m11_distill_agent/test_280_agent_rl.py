"""M11-B: Agent RL 测试 — TOOLS, execute_tool, parse_tool_calls, validate_gt, reward。

验证：
1. TOOLS 定义完整（6 个工具）
2. execute_tool 正确执行模拟工具
3. parse_tool_calls 从文本解析 JSON 工具调用
4. validate_gt 验证 ground truth
5. calculate_agent_reward 多维度奖励
"""

import json
import math

import torch
import pytest

from zllm.training.agent_rl import (
    TOOLS, execute_tool, parse_tool_calls,
    validate_gt_in_text, calculate_agent_reward, AgentConfig,
)


class TestTools:
    def test_six_tools_defined(self):
        assert len(TOOLS) == 6

    def test_tool_names(self):
        names = {t["function"]["name"] for t in TOOLS}
        assert "calculate_math" in names
        assert "get_current_weather" in names
        assert "get_current_time" in names
        assert "get_exchange_rate" in names
        assert "translate_text" in names
        assert "unit_converter" in names

    def test_each_tool_has_required_fields(self):
        for tool in TOOLS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]


class TestExecuteTool:
    def test_calculate_math(self):
        result = execute_tool("calculate_math", {"expression": "2+3"})
        assert result is not None
        assert result["result"] == "5"

    def test_get_weather(self):
        result = execute_tool("get_current_weather", {"location": "北京"})
        assert result is not None
        assert "temperature" in result

    def test_get_time(self):
        result = execute_tool("get_current_time", {})
        assert result is not None
        assert "datetime" in result

    def test_exchange_rate(self):
        result = execute_tool("get_exchange_rate", {"from_currency": "USD", "to_currency": "CNY"})
        assert result is not None
        assert result["rate"] == 7.21

    def test_translate(self):
        result = execute_tool("translate_text", {"text": "你好世界", "target_language": "english"})
        assert result is not None
        assert result["translated_text"] == "Hello World"

    def test_unknown_tool_returns_none(self):
        result = execute_tool("nonexistent", {})
        assert result is None


class TestParseToolCalls:
    def test_single_call(self):
        text = '让我来算一下\n```json\n{"name": "calculate_math", "arguments": {"expression": "2+3"}}\n```'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "calculate_math"
        assert calls[0]["arguments"]["expression"] == "2+3"

    def test_multiple_calls(self):
        text = (
            '```json\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n```\n'
            '```json\n{"name": "get_current_time", "arguments": {}}\n```'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2

    def test_no_calls(self):
        text = "今天天气不错"
        calls = parse_tool_calls(text)
        assert len(calls) == 0

    def test_invalid_json_ignored(self):
        text = '```json\n{not valid json}\n```'
        calls = parse_tool_calls(text)
        assert len(calls) == 0

    def test_non_name_dict_ignored(self):
        text = '```json\n{"foo": "bar"}\n```'
        calls = parse_tool_calls(text)
        assert len(calls) == 0


class TestValidateGT:
    def test_exact_match(self):
        assert validate_gt_in_text("28°C", "北京今天气温28°C，天气晴朗")

    def test_case_insensitive(self):
        assert validate_gt_in_text("hello", "Hello World")

    def test_no_match(self):
        assert not validate_gt_in_text("30°C", "北京今天气温28°C")

    def test_partial_match(self):
        assert validate_gt_in_text("7.21", "汇率是7.21")


class TestAgentReward:
    def test_good_response_high_reward(self):
        response = "让我来查询一下北京的天气。根据查询结果，北京今天气温28°C，天气晴。"
        calls = [{"name": "get_current_weather", "arguments": {"location": "北京"}}]
        reward = calculate_agent_reward(response, gt_answer="28°C", tool_calls=calls)
        assert reward > 1.0

    def test_bad_length_low_reward(self):
        reward = calculate_agent_reward("短")
        assert reward < 0

    def test_no_tool_calls_no_gt(self):
        response = "这是一段合理长度的回复内容但是没有调用任何工具也没有正确答案"
        reward = calculate_agent_reward(response)
        assert 0 <= reward <= 1.0

    def test_gt_match_boost(self):
        resp_no_gt = "根据查询，汇率约为七点二一"
        resp_with_gt = "根据查询，USD兑CNY汇率是7.21"
        r1 = calculate_agent_reward(resp_no_gt, gt_answer="7.21")
        r2 = calculate_agent_reward(resp_with_gt, gt_answer="7.21")
        assert r2 > r1


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.max_turns == 3
        assert cfg.max_gen_len == 256
        assert cfg.save_weight == "agent"
        assert cfg.from_weight == "full_sft"
