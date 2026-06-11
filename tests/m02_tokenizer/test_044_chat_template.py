"""步骤 44-48: Chat Template — 对话格式化。

测试角色格式化、system prompt、generation prompt、思考链、工具声明。
"""

from zllm.tokenizer.chat_template import render_messages, CHAT_TEMPLATE


class TestBasicFormatting:
    def test_user_assistant(self):
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        text = render_messages(messages)
        assert "<|im_start|>user\n你好<|im_end|>" in text
        assert "<|im_start|>assistant\n你好！<|im_end|>" in text

    def test_message_order_preserved(self):
        messages = [
            {"role": "user", "content": "第一"},
            {"role": "assistant", "content": "第二"},
            {"role": "user", "content": "第三"},
        ]
        text = render_messages(messages)
        assert text.index("第一") < text.index("第二") < text.index("第三")

    def test_no_system_by_default(self):
        messages = [{"role": "user", "content": "你好"}]
        text = render_messages(messages)
        assert "system" not in text


class TestSystemPrompt:
    def test_system_message_formatted(self):
        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ]
        text = render_messages(messages)
        assert "<|im_start|>system\n你是助手<|im_end|>" in text

    def test_system_before_user(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        text = render_messages(messages)
        assert text.index("sys") < text.index("usr")


class TestGenerationPrompt:
    def test_add_generation_prompt(self):
        messages = [{"role": "user", "content": "你好"}]
        text = render_messages(messages, add_generation_prompt=True)
        assert text.endswith("<|im_start|>assistant\n")

    def test_no_generation_prompt_by_default(self):
        messages = [{"role": "user", "content": "你好"}]
        text = render_messages(messages)
        assert not text.endswith("<|im_start|>assistant\n")


class TestThinkingMode:
    def test_open_thinking_wraps_assistant(self):
        messages = [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ]
        text = render_messages(messages, open_thinking=True)
        assert "<reasoningchain_start>回答<reasoningchain_end>" in text

    def test_thinking_not_applied_to_user(self):
        messages = [{"role": "user", "content": "问题"}]
        text = render_messages(messages, open_thinking=True)
        assert "<reasoningchain_start>" not in text.split("user")[1]

    def test_no_thinking_by_default(self):
        messages = [
            {"role": "assistant", "content": "回答"},
        ]
        text = render_messages(messages)
        assert "<reasoningchain_start>" not in text


class TestTools:
    def test_tools_inject_system(self):
        tools = [{"name": "calculator", "description": "计算器"}]
        messages = [{"role": "user", "content": "算一下"}]
        text = render_messages(messages, tools=tools)
        # 工具声明应出现在 system 块中
        assert "<|im_start|>system" in text
        assert "calculator" in text

    def test_tools_system_before_user(self):
        tools = [{"name": "calc"}]
        messages = [{"role": "user", "content": "你好"}]
        text = render_messages(messages, tools=tools)
        assert text.index("system") < text.index("你好")


class TestChatTemplateString:
    def test_template_is_string(self):
        assert isinstance(CHAT_TEMPLATE, str)
        assert len(CHAT_TEMPLATE) > 0

    def test_template_contains_im_start(self):
        assert "<|im_start|>" in CHAT_TEMPLATE
