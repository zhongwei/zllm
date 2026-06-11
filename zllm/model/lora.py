"""LoRA（Low-Rank Adaptation）实现。

低秩分解：ΔW = B @ A，其中 A 降维、B 升维。
- A 高斯初始化，B 零初始化 → 训练初始时 ΔW=0，不改变模型行为
- 只注入到方阵 Linear（in_features == out_features），即 q_proj / o_proj
- monkey-patch forward: original_forward(x) + lora(x)

函数：
- apply_lora(model, rank): 注入 LoRA 到所有方阵 Linear
- get_lora_params(model): 收集 LoRA 参数
- freeze_non_lora(model): 冻结非 LoRA 参数，返回 LoRA 参数列表
- save_lora(model, path): 只保存 LoRA 权重
- load_lora(model, path): 加载 LoRA 权重
- merge_lora(model, lora_path, save_path): 合并 W + B@A 后保存
"""

import torch
from torch import nn


class LoRA(nn.Module):
    """低秩适配器：ΔW = B(A(x))。

    Args:
        in_features: 输入维度
        out_features: 输出维度
        rank: 低秩维度（远小于 in/out_features）
    """

    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """将 LoRA 注入到模型中所有方阵 Linear 层。

    通过 monkey-patch forward 实现：
    new_forward(x) = original_forward(x) + lora(x)

    只注入 in_features == out_features 的 Linear（q_proj / o_proj）。
    """
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def get_lora_params(model):
    """收集模型中所有 LoRA 参数。"""
    raw_model = getattr(model, "_orig_mod", model)
    return [p for n, p in raw_model.named_parameters() if "lora" in n]


def freeze_non_lora(model):
    """冻结所有非 LoRA 参数，返回 LoRA 参数列表。"""
    raw_model = getattr(model, "_orig_mod", model)
    lora_params = []
    for name, param in raw_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    return lora_params


def save_lora(model, path):
    """只保存 LoRA 权重（A 和 B 矩阵）。"""
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f"{clean_name}.lora.{k}": v.cpu().half()
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def load_lora(model, path):
    """加载 LoRA 权重到模型。"""
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device, weights_only=True)
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            if lora_state:
                module.lora.load_state_dict(lora_state)


def merge_lora(model, lora_path, save_path):
    """合并 LoRA 权重到基础权重：W_merged = W + B @ A，保存为标准权重。"""
    load_lora(model, lora_path)
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {
        k: v.cpu().half()
        for k, v in raw_model.state_dict().items()
        if ".lora." not in k
    }
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and ".lora." not in name:
            state_dict[f"{name}.weight"] = module.weight.data.clone().cpu().half()
            if hasattr(module, "lora"):
                delta = (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
                state_dict[f"{name}.weight"] += delta
    torch.save(state_dict, save_path)
