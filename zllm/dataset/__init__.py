"""zllm dataset 包。

提供 5 种 Dataset：
- PretrainDataset: 预训练（纯文本）
- SFTDataset: 监督微调（对话 + prompt 掩码）
- DPODataset: 直接偏好优化（chosen/rejected 对）
- RLAIFDataset: 强化学习 prompt-only
- AgentRLDataset: Agent 工具调用
"""

from zllm.dataset.agent import AgentRLDataset
from zllm.dataset.dpo import DPODataset
from zllm.dataset.pretrain import PretrainDataset
from zllm.dataset.rlaif import RLAIFDataset
from zllm.dataset.sft import SFTDataset
from zllm.dataset.utils import post_processing_chat, pre_processing_chat

__all__ = [
    "AgentRLDataset",
    "DPODataset",
    "PretrainDataset",
    "RLAIFDataset",
    "SFTDataset",
    "post_processing_chat",
    "pre_processing_chat",
]
