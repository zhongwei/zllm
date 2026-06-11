"""数据工具函数。

pre_processing_chat: 概率性添加 system prompt
post_processing_chat: 概率性移除空思考链标签
"""

import random


SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是zllm，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是zllm，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are zllm, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are zllm, a small but useful language model.",
]


def pre_processing_chat(conversations, add_system_ratio=0.2):
    if any(conv.get("tools") for conv in conversations):
        return conversations
    if conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    empty_tag = "<reasoningchain_start>\n\n<reasoningchain_end>\n\n"
    if empty_tag in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace(empty_tag, "")
    return prompt_content
