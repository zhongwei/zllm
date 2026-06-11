# 第9章 LoRA 微调 — 低秩适配器高效训练

## 学习目标

掌握 LoRA（Low-Rank Adaptation）：用极少参数（<1%）实现高效领域适配。

## LoRA 原理

冻结预训练权重 W，学习低秩增量 ΔW = B @ A：

```
原始:  y = W·x
LoRA:  y = W·x + B·A·x
             ↑_______↑
             低秩适配器

A: [d × r]   降维（d=768, r=16）
B: [r × d]   升维
```

**参数量对比**（dim=768, rank=16, 每层 q_proj + o_proj）：
- 全量微调：768×768×2 = 1,179,648
- LoRA：(768×16 + 16×768)×2 = 49,152（**4.2%**）

## 初始化策略

| 矩阵 | 初始化 | 原因 |
|------|--------|------|
| A | 高斯 N(0, 0.02) | 打破对称性 |
| B | **全零** | 保证 ΔW=0 → 训练初始不改变模型行为 |

## 注入目标

只注入**方阵** Linear（in_features == out_features）：
- `self_attn.q_proj`（768→768）
- `self_attn.o_proj`（768→768）

不注入 k_proj/v_proj（768→384，非方阵）、FFN（768→2432）。

## 核心函数

| 函数 | 作用 |
|------|------|
| `apply_lora(model, rank)` | monkey-patch forward → `original(x) + lora(x)` |
| `freeze_non_lora(model)` | 冻结基础参数，返回 LoRA 参数列表 |
| `save_lora(model, path)` | 只保存 A/B 矩阵（几百 KB） |
| `load_lora(model, path)` | 加载 LoRA 权重 |
| `merge_lora(model, path, save_path)` | W_merged = W + B@A，导出标准权重 |

## LoRAConfig vs SFTConfig

| 参数 | SFT | LoRA |
|------|-----|------|
| learning_rate | 1e-5 | **1e-4**（10x，参数少需更大 lr） |
| epochs | 2 | **10**（LoRA 收敛慢） |
| from_weight | pretrain | **full_sft**（基于已微调模型） |
| rank | — | **16** |

## Merge 优势

合并后推理无额外开销，与全量微调等效：
```
W_final = W_base + B @ A
```

## 验证

```bash
pytest tests/m09_lora/ -v   # 25 个测试全绿
```
