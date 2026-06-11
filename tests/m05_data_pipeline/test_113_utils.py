"""M5-A: 数据工具函数测试。

测试 pre_processing_chat 和 post_processing_chat。
"""

import random
import pytest

from zllm.dataset.utils import pre_processing_chat, post_processing_chat


class TestPreProcessingChat:
    def test_no_system_added_when_ratio_zero(self):
        convs = [{"role": "user", "content": "hello"}]
        result = pre_processing_chat(convs, add_system_ratio=0.0)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_system_always_added_when_ratio_one(self):
        convs = [{"role": "user", "content": "hello"}]
        result = pre_processing_chat(convs, add_system_ratio=1.0)
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_existing_system_not_duplicated(self):
        convs = [
            {"role": "system", "content": "custom"},
            {"role": "user", "content": "hello"},
        ]
        result = pre_processing_chat(convs, add_system_ratio=1.0)
        assert len(result) == 2
        assert result[0]["content"] == "custom"

    def test_tools_data_preserved(self):
        convs = [
            {"role": "system", "content": "", "tools": "[{}]"},
            {"role": "user", "content": "hello"},
        ]
        result = pre_processing_chat(convs, add_system_ratio=1.0)
        assert len(result) == 2

    def test_returns_same_object_when_no_change(self):
        convs = [{"role": "user", "content": "hello"}]
        result = pre_processing_chat(convs, add_system_ratio=0.0)
        assert result == convs


class TestPostProcessingChat:
    def test_empty_think_removed(self):
        text = "<reasoningchain_start>\n\n<reasoningchain_end>\n\nanswer"
        result = post_processing_chat(text, empty_think_ratio=0.0)
        assert "<reasoningchain_start>" not in result
        assert "answer" in result

    def test_empty_think_kept_when_ratio_one(self):
        text = "<reasoningchain_start>\n\n<reasoningchain_end>\n\nanswer"
        result = post_processing_chat(text, empty_think_ratio=1.0)
        assert "<reasoningchain_start>" in result

    def test_non_empty_think_preserved(self):
        text = "<reasoningchain_start>actual thinking<reasoningchain_end>\n\nanswer"
        result = post_processing_chat(text, empty_think_ratio=0.0)
        assert "actual thinking" in result

    def test_no_think_tags_unchanged(self):
        text = "just a normal answer"
        result = post_processing_chat(text)
        assert result == text
