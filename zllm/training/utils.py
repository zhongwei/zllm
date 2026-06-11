"""训练工具函数集合。

提供：
- setup_seed: 随机种子设置
- get_lr: 余弦退火学习率
- init_model: 模型初始化 + 权重加载
- lm_checkpoint: 保存/恢复训练状态（原子写入）
- is_main_process / Logger: 分布式日志
- get_model_params: 参数统计
- SkipBatchSampler: 断点续训跳过已训练 batch
"""

import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "num_experts", 0)
    n_active = getattr(config, "num_experts_per_tok", 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    base = total - expert * n_routed
    active = base + expert * n_active
    if active < total:
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        Logger(f"Model Params: {total:.2f}M")


def get_lr(current_step, total_steps, base_lr=5e-4):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_model(lm_config, from_weight="pretrain", save_dir="out", device="cuda"):
    from zllm.model.causal_lm import ZLLMForCausalLM

    model = ZLLMForCausalLM(lm_config)
    if from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = os.path.join(save_dir, f"{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth")
        weights = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(weights, strict=False)
    get_model_params(model, lm_config)
    Logger(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")
    return model.to(device)


def lm_checkpoint(
    lm_config, weight="full_sft", model=None, optimizer=None,
    epoch=0, step=0, wandb=None, save_dir="checkpoints", **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp_path = os.path.join(save_dir, f"{weight}_{lm_config.hidden_size}{moe_path}.pth")
    resume_path = os.path.join(save_dir, f"{weight}_{lm_config.hidden_size}{moe_path}_resume.pth")

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        ckp_tmp = ckp_path + ".tmp"
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, "_orig_mod", raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu", weights_only=True)
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}")
            return ckp_data
        return None


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
