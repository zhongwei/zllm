"""RLAIFDataset — 强化学习（PPO/GRPO）prompt-only 数据集。

按 thinking_ratio 概率开启推理模式，只返回 prompt（不含回答）。
"""

import os
import random

from datasets import load_dataset
from torch.utils.data import Dataset

from zllm.dataset.utils import pre_processing_chat
from zllm.tokenizer.adapter import wrap

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = wrap(tokenizer)
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True,
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample["conversations"])
        return {"prompt": prompt, "answer": ""}
