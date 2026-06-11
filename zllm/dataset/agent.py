"""AgentRLDataset — Agent RL 工具调用数据集。

返回 messages（去掉最后一条 assistant）+ tools + ground truth。
"""

import json
import os

from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample["conversations"])
        return {"messages": messages, "tools": tools, "gt": sample["gt"]}
