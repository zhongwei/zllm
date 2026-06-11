"""Chat Template — 对话格式化。

将 OpenAI 风格的 messages 列表渲染为模型输入格式。

格式：
    <|im_start|>role\ncontent<|im_end|>\n

支持：
- system / user / assistant / tool 角色
- 工具声明（tools 注入 system 块）
- 思考链模式（open_thinking 包裹 assistant 内容）
- 生成提示（add_generation_prompt 追加 assistant 头）
"""

import json

from zllm.tokenizer.special_tokens import (
    IM_END,
    IM_START,
    REASONINGCHAIN_END,
    REASONINGCHAIN_START,
)


def render_messages(
    messages,
    tools=None,
    open_thinking=False,
    add_generation_prompt=False,
):
    """将 messages 渲染为模型输入文本。

    Args:
        messages: [{"role": str, "content": str}, ...]
        tools: 工具定义列表（注入 system 块）
        open_thinking: 是否用思考链标签包裹 assistant 内容
        add_generation_prompt: 是否追加 <|im_start|>assistant\\n（用于推理）

    Returns:
        渲染后的字符串
    """
    parts = []
    start_idx = 0

    # 处理 system 块（含工具声明）
    if tools:
        tool_str = "".join(
            json.dumps(t, ensure_ascii=False) + "\n" for t in tools
        ).strip()
        if messages and messages[0].get("role") == "system":
            sys_content = messages[0]["content"] + "\n你可以使用以下工具：\n" + tool_str
            start_idx = 1
        else:
            sys_content = "你可以使用以下工具：\n" + tool_str
        parts.append(f"{IM_START}system\n{sys_content}{IM_END}\n")
    elif messages and messages[0].get("role") == "system":
        parts.append(f"{IM_START}system\n{messages[0]['content']}{IM_END}\n")
        start_idx = 1

    # 格式化每条消息
    for msg in messages[start_idx:]:
        role = msg["role"]
        content = msg["content"]
        if open_thinking and role == "assistant":
            content = f"{REASONINGCHAIN_START}{content}{REASONINGCHAIN_END}"
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")

    # 生成提示
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")

    return "".join(parts)


# Jinja2 模板（用于 PreTrainedTokenizerFast.chat_template，兼容 Transformers 生态）
CHAT_TEMPLATE = """\
{%- for message in messages %}
{%- if message['role'] == 'system' %}
{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}
{%- elif message['role'] == 'user' %}
{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}
{%- elif message['role'] == 'assistant' %}
{{- '<|im_start|>assistant\\n' }}
{%- if open_thinking %}{{- '<reasoningchain_start>' }}{%- endif %}
{{- message['content'] }}
{%- if open_thinking %}{{- '<reasoningchain_end>' }}{%- endif %}
{{- '<|im_end|>\\n' }}
{%- elif message['role'] == 'tool' %}
{{- '<|im_start|>tool\\n' + message['content'] + '<|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- endif %}"""
