"""步骤 30-34: 特殊 Token 定义。

测试系统级、工具调用、思考链、buffer 预留 token 的完整性与唯一性。
"""

from zllm.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    BUFFER_TOKENS,
    IM_START,
    IM_END,
    TOOLCALL_START,
    TOOLCALL_END,
    REASONINGCHAIN_START,
    REASONINGCHAIN_END,
)


class TestSpecialTokensList:
    def test_is_list_of_strings(self):
        assert isinstance(SPECIAL_TOKENS, list)
        assert all(isinstance(t, str) for t in SPECIAL_TOKENS)

    def test_no_duplicates(self):
        assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))

    def test_all_special_no_duplicates(self):
        assert len(ALL_SPECIAL_TOKENS) == len(set(ALL_SPECIAL_TOKENS))

    def test_conversation_boundary_tokens(self):
        assert "<|im_start|>" in SPECIAL_TOKENS
        assert "<|im_end|>" in SPECIAL_TOKENS

    def test_vision_tokens_present(self):
        assert "<|vision_start|>" in SPECIAL_TOKENS
        assert "<|vision_end|>" in SPECIAL_TOKENS
        assert "<|image_pad|>" in SPECIAL_TOKENS

    def test_toolcall_tokens_present(self):
        assert "📞" in SPECIAL_TOKENS
        assert TOOLCALL_START in SPECIAL_TOKENS
        assert TOOLCALL_END in SPECIAL_TOKENS
        assert "<executionsandbox_start>" in SPECIAL_TOKENS
        assert "<executionsandbox_end>" in SPECIAL_TOKENS

    def test_reasoning_tokens_present(self):
        assert REASONINGCHAIN_START in SPECIAL_TOKENS
        assert REASONINGCHAIN_END in SPECIAL_TOKENS


class TestBufferTokens:
    def test_buffer_count(self):
        assert len(BUFFER_TOKENS) == 8

    def test_buffer_format(self):
        assert BUFFER_TOKENS[0] == "<|buffer1|>"
        assert BUFFER_TOKENS[-1] == "<|buffer8|>"

    def test_buffers_in_all_special(self):
        for bt in BUFFER_TOKENS:
            assert bt in ALL_SPECIAL_TOKENS


class TestConstants:
    def test_im_start(self):
        assert IM_START == "<|im_start|>"

    def test_im_end(self):
        assert IM_END == "<|im_end|>"

    def test_toolcall_start(self):
        assert TOOLCALL_START == "<toolcall_start>"

    def test_reasoning_start(self):
        assert REASONINGCHAIN_START == "<reasoningchain_start>"
