"""Agent RL — 工具调用强化学习。

组件：
- TOOLS: 6 个模拟工具定义（天气/时间/汇率/翻译/计算/单位换算）
- MOCK_RESULTS: 模拟数据 + 执行函数
- execute_tool: 执行工具调用
- parse_tool_calls: 从文本解析 JSON 工具调用
- validate_gt_in_text: 验证 ground truth 是否在文本中
- calculate_agent_reward: 多维度奖励计算
- AgentConfig: Agent RL 训练配置
"""

import json
import math
import re

import torch
from dataclasses import dataclass


TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592}

MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}


def execute_tool(name, args):
    """执行模拟工具调用。

    Args:
        name: 工具名称
        args: 工具参数 dict

    Returns:
        dict 结果，或 None（工具不存在/执行失败）
    """
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return None
    try:
        return fn(args)
    except Exception:
        return None


def parse_tool_calls(text):
    """从文本中解析工具调用。

    格式：```json\n{"name": "...", "arguments": {...}}\n```

    Args:
        text: 模型生成的文本

    Returns:
        list[dict]: 解析出的工具调用列表
    """
    calls = []
    pattern = r"```json\s*(.*?)\s*```"
    for m in re.findall(pattern, text, re.DOTALL):
        try:
            parsed = json.loads(m.strip())
            if isinstance(parsed, dict) and "name" in parsed:
                calls.append(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    return calls


def validate_gt_in_text(gt_text, response):
    """验证 ground truth 是否出现在响应中。

    Args:
        gt_text: 期望的文本片段
        response: 模型生成的文本

    Returns:
        bool
    """
    return gt_text.lower().strip() in response.lower()


def calculate_agent_reward(response, gt_answer=None, tool_calls=None, rep_penalty_cap=0.5):
    """计算 Agent 多维度奖励。

    维度：
    1. 长度合理性：20-800 字符 → +0.5，否则 -0.5
    2. 工具调用正确性：有有效工具调用 → +1.0
    3. GT 匹配：ground truth 出现在回复中 → +1.0
    4. 重复惩罚：n-gram 重复率

    Args:
        response: 模型生成的文本
        gt_answer: 期望的答案文本（可选）
        tool_calls: 解析出的工具调用列表（可选）
        rep_penalty_cap: 重复惩罚上限

    Returns:
        float: 总奖励
    """
    reward = 0.0
    if 20 <= len(response.strip()) <= 800:
        reward += 0.5
    else:
        reward -= 0.5

    if tool_calls:
        for call in tool_calls:
            name = call.get("name", "")
            if name in MOCK_RESULTS:
                reward += 1.0
                break

    if gt_answer and validate_gt_in_text(gt_answer, response):
        reward += 1.0

    toks = re.findall(r"\w+|[^\w\s]", response.lower())
    n = 3
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if grams:
        rep = (len(grams) - len(set(grams))) * rep_penalty_cap * 2 / len(grams)
        reward -= min(rep_penalty_cap, rep)

    return reward


@dataclass
class AgentConfig:
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 3e-7
    max_turns: int = 3
    max_gen_len: int = 256
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    max_seq_len: int = 768
    dtype: str = "bfloat16"
    num_workers: int = 1
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    save_weight: str = "agent"
    from_weight: str = "full_sft"
    from_resume: bool = False
    device: str = "cuda"