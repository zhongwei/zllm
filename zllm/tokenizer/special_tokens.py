"""特殊 Token 定义。

zllm 的特殊 token 设计，对齐 Qwen3 / minimind-3：
- 系统级：对话边界（im_start / im_end）
- 多模态预留：视觉/音频
- 工具调用：Agent RL 专用标记
- 思考链：推理模式
- buffer：预留扩展位
"""

# 系统级与功能 token（顺序决定 ID 分配）
SPECIAL_TOKENS = [
    # 对话边界
    "<|im_start|>",
    "<|im_end|>",
    # 多模态预留（当前项目不实现，但保留兼容性）
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|audio_pad|>",
    # 工具调用（Agent RL）
    "📞",
    "<toolcall_start>",
    "<toolcall_end>",
    "<executionsandbox_start>",
    "<executionsandbox_end>",
    "<executionresult_start>",
    "<executionresult_end>",
    # 思考链（推理模式）
    "<reasoningchain_start>",
    "<reasoningchain_end>",
]

# Buffer 预留位（模型可用但当前未使用，为未来功能扩展保留 ID 槽位）
BUFFER_TOKENS = [f"<|buffer{i}|>" for i in range(1, 9)]

# 全部特殊 token（用于 tokenizer 训练）
ALL_SPECIAL_TOKENS = SPECIAL_TOKENS + BUFFER_TOKENS

# 常用 token 常量
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOLCALL_START = "<toolcall_start>"
TOOLCALL_END = "<toolcall_end>"
EXECUTIONSANDBOX_START = "<executionsandbox_start>"
EXECUTIONSANDBOX_END = "<executionsandbox_end>"
EXECUTIONRESULT_START = "<executionresult_start>"
EXECUTIONRESULT_END = "<executionresult_end>"
REASONINGCHAIN_START = "<reasoningchain_start>"
REASONINGCHAIN_END = "<reasoningchain_end>"
