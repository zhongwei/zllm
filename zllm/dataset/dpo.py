"""DPODataset — 直接偏好优化数据集。

加载 chosen/rejected 对，渲染 chat template，构造 loss_mask。
返回 (x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected)。
"""

import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from zllm.dataset.utils import post_processing_chat
from zllm.tokenizer.adapter import wrap

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = wrap(tokenizer)
        self.max_length = max_length
        self.bos_id = self.tokenizer.encode(f"{self.tokenizer.bos_token}assistant\n").ids
        self.eos_id = self.tokenizer.encode(f"{self.tokenizer.eos_token}\n").ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        chosen_prompt = post_processing_chat(chosen_prompt)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_ids = self.tokenizer.encode(chosen_prompt).ids[: self.max_length]
        chosen_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(chosen_ids))
        rejected_ids = self.tokenizer.encode(rejected_prompt).ids[: self.max_length]
        rejected_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(rejected_ids))
        chosen_mask = self.generate_loss_mask(chosen_ids)
        rejected_mask = self.generate_loss_mask(rejected_ids)
        return {
            "x_chosen": torch.tensor(chosen_ids[:-1], dtype=torch.long),
            "y_chosen": torch.tensor(chosen_ids[1:], dtype=torch.long),
            "mask_chosen": torch.tensor(chosen_mask[1:], dtype=torch.long),
            "x_rejected": torch.tensor(rejected_ids[:-1], dtype=torch.long),
            "y_rejected": torch.tensor(rejected_ids[1:], dtype=torch.long),
            "mask_rejected": torch.tensor(rejected_mask[1:], dtype=torch.long),
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
