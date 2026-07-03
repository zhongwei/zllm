---
part: 5
chapter: 29
title: 训练基础设施：种子/学习率/checkpoint
milestone: M6-a
source: zllm/training/utils.py
tests: tests/m06_training/test_145_utils.py
status: draft
---

# 第 29 章 训练基础设施：种子/学习率/checkpoint

Ch 27–28 搞定了数据，但训练还需要三样「水电煤」式的基础设施：

1. **随机种子**——让训练可复现。同样的数据、同样的种子，应该得到同样的结果。这是科学实验的基本要求，也是调试「为什么 loss 不对」的前提。
2. **学习率调度**——Ch 11 讲过学习率是训练最重要的超参数。本章实现**余弦退火**：开始大步快学，后面小步精调，平滑降到最低。
3. **Checkpoint**——训练几十小时，中途断电怎么办？checkpoint 把模型权重和优化器状态**原子地**存下来，崩了能从断点续训。

这三样东西在 `zllm/training/utils.py` 里，是所有训练脚本（预训练、SFT、DPO、RL）共享的基础设施。

## 29.1 学习目标

读完本章，你应该能够：

- 用 `setup_seed` 设置四重随机种子（random/numpy/torch/cuda）并理解为什么每一层都要设；
- 默写出余弦退火公式 $\eta_t = \eta_0(0.1 + 0.45(1+\cos(\pi t/T)))$，说清起点=base、终点=0.1·base；
- 解释 `init_model` 的权重命名规则 `f"{from_weight}_{hidden}{_moe}.pth"`；
- 看懂 `lm_checkpoint` 的**原子写入**（`.tmp` → `os.replace`）如何防止训练中途崩溃损坏文件；
- 理解 `SkipBatchSampler` 如何实现断点续训（跳过已训练的 batch）。

## 29.2 原理回顾：种子、学习率与 checkpoint

### 29.2.1 随机种子与可复现性

深度学习里有两个随机源：**数据打乱**（DataLoader 的 shuffle）和**权重初始化**（`nn.Linear` 的随机初始化）。如果不固定种子，每次训练结果都不同——没法判断「loss 改善是因为我的代码改动还是随机抖动」。

Python 的 `random`、`numpy`、`torch`（CPU）、`torch.cuda`（GPU）是**四个独立的随机数生成器**，必须**全部固定**才彻底可复现。少固定任何一个，对应层的随机性就失控。

### 29.2.2 余弦退火（回引 Ch 11）

Ch 11《优化器》讲过学习率调度。**余弦退火**（Cosine Annealing）是 LLM 训练最常用的策略——学习率按余弦曲线从高到低平滑下降：

$$
\eta_t \;=\; \eta_0 \left(0.1 + 0.45\left(1 + \cos\frac{\pi t}{T}\right)\right)
$$

代入几个关键点：

| 时刻 $t$ | $\cos$ 值 | $\eta_t$ |
|----------|----------|----------|
| $0$（起点） | $1$ | $\eta_0 \times 1.0 = \eta_0$ |
| $T/2$（中点） | $0$ | $\eta_0 \times 0.55$ |
| $T$（终点） | $-1$ | $\eta_0 \times 0.1$ |

```mermaid
graph LR
    A["η₀ (起点)<br/>大步快学"] --> B["η₀/2 (中点)<br/>稳步下降"]
    B --> C["0.1·η₀ (终点)<br/>小步精调"]
    style A fill:#ffcdd2
    style C fill:#c8e6c9
```

起点保留满学习率（快学），终点降到 1/10（精调，避免最后震荡）。这个「warmup-free」的简化版没有单独的 warmup 阶段，靠余弦曲线本身的平滑起步替代。

### 29.2.3 Checkpoint 与原子写入

训练中途最怕两件事：**断电**和**OOM**。如果在写 checkpoint 文件的一瞬间崩溃，文件就坏了——白训几十小时。

**原子写入**解决这个：先写到临时文件 `.tmp`，写完后用 `os.replace` **原子地**重命名。`os.replace` 在操作系统层面是原子的——要么改名成功（完整文件），要么没改（旧文件还在），不会出现「半截文件」。

## 29.3 代码实现：utils.py

完整实现见 `zllm/training/utils.py`（165 行）。

### 29.3.1 setup_seed：四重种子

> 完整实现见 `zllm/training/utils.py:59`

```python
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

`setup_seed`（`:59-64`）：五次调用覆盖四层随机源——`random`（Python）、`np`（NumPy）、`torch.manual_seed`（CPU 张量）、`cuda.manual_seed` + `manual_seed_all`（单卡/多卡 GPU）。`manual_seed_all` 是给所有 GPU 设种子，多卡时必须调。

> 对应测试 `tests/m06_training/test_145_utils.py:24`（同种子可复现）、`:31`（不同种子不同结果）。

### 29.3.2 get_lr：余弦退火一行流

> 完整实现见 `zllm/training/utils.py:46`

```python
def get_lr(current_step, total_steps, base_lr=5e-4):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

`get_lr`（`:46-47`）：就是 29.2.2 的公式，一行实现。`current_step` 从 0 到 `total_steps`，学习率从 `base_lr` 平滑降到 `0.1 * base_lr`。每个 step 调一次，更新优化器的 `param_groups["lr"]`。

> 对应测试 `test_145_utils.py:40-43`（起点 = base_lr）、`:45-48`（终点 = 0.1·base_lr）、`:50-53`（单调递减）、`:55-58`（中点 = 0.55·base_lr）。

### 29.3.3 init_model：权重命名与加载

> 完整实现见 `zllm/training/utils.py:67`

```python
def init_model(lm_config, from_weight="pretrain", save_dir="out", device="cuda"):
    from zllm.model.causal_lm import ZLLMForCausalLM
    model = ZLLMForCausalLM(lm_config)
    if from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = os.path.join(save_dir, f"{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth")
        weights = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(weights, strict=False)
    ...
    return model.to(device)
```

`init_model`（`:67-78`）：创建模型，可选加载权重。权重路径命名规则（`:72-73`）：`f"{阶段}_{hidden_size}{_moe?}.pth"`——比如 `pretrain_768.pth`、`full_sft_768_moe.pth`。用 `hidden_size` 区分模型规格，`_moe` 后缀区分密集/专家。`strict=False`（`:75`）允许权重和模型结构有微小出入（比如新增的 buffer 不在旧权重里）。

### 29.3.4 lm_checkpoint：原子写入 + 断点续训

> 完整实现见 `zllm/training/utils.py:81`

`lm_checkpoint`（`:81-139`）是个**双向函数**——传 `model` 时存，不传时读：

**存储分支**（`:90-129`）：

```python
state_dict = {k: v.half().cpu() for k, v in state_dict.items()}  # half 省一半磁盘
ckp_tmp = ckp_path + ".tmp"
torch.save(state_dict, ckp_tmp)
os.replace(ckp_tmp, ckp_path)                                     # 原子重命名
```

- `half().cpu()`（`:94`）：权重转 fp16 存盘，省一半空间；CPU 张量不绑 GPU。
- 原子写（`:96-98`）：先 `.tmp` 再 `os.replace`。
- **resume dict**（`:108-115`）：除了模型权重，还存 `optimizer`（AdamW 动量）、`epoch`、`step`、`world_size`、`wandb_id`——这些是续训必须恢复的状态。

**读取分支**（`:130-139`）：

```python
saved_ws = ckp_data.get("world_size", 1)
current_ws = dist.get_world_size() if dist.is_initialized() else 1
if saved_ws != current_ws:
    ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws  # GPU 数变了，换算 step
```

亮点在 `:133-137`：如果续训时 **GPU 数量变了**（比如之前 4 卡训了 1000 step，现在换 1 卡），step 要按比例换算（`1000 × 4 / 1 = 4000`）。因为之前每个 step 是 4 卡各处理一个 batch，现在 1 卡要跑 4 倍 step 才等效。

> 对应测试 `test_145_utils.py:100-116`（存两个文件 `*.pth` + `*_resume.pth`）、`:118-121`（不存在返回 None）、`:123-139`（读出 epoch/step）、`:141-152`（MoE 后缀 `_moe`）。

### 29.3.5 SkipBatchSampler：断点续训跳 batch

> 完整实现见 `zllm/training/utils.py:142`

```python
class SkipBatchSampler(Sampler):
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:   # 还没跳够，丢弃这个 batch
                    skipped += 1
                    batch = []
                    continue
                yield batch                        # 跳够了，正常产出
                batch = []
```

`SkipBatchSampler`（`:142-165`）：包一层普通 sampler，`skip_batches` 指定要跳过多少个 batch。续训时从 checkpoint 读出 `step`，设 `skip_batches=step`，DataLoader 就直接从第 `step+1` 个 batch 开始——不重复训练。

> 对应测试 `tests/m06_training/test_166_gpu.py:32-37`（skip 2 个 batch 后从第 3 个开始）、`:39-43`（skip 全部则空）。

## 29.4 对应单元测试

> 对应测试 `tests/m06_training/test_145_utils.py`（166 行）

- **TestSetupSeed**（`:23-36`）：可复现 `:24`、不同种子不同 `:31`。
- **TestGetLR**（`:39-58`）：起点/终点/中点/单调性，四个关键点精确验证余弦曲线。
- **TestInitModel**（`:73-96`）：from_none 随机初始化 `:74`、from_weight 加载 `:84`。
- **TestLMCheckpoint**（`:99-152`）：存两文件 `:100`、不存在返回 None `:118`、读出 epoch/step `:123`、MoE 后缀 `:141`。
- **TestGetModelParams**（`:155-166`）：打印参数量。

## 29.5 动手验证

```bash
pytest tests/m06_training/test_145_utils.py tests/m06_training/test_166_gpu.py::TestSkipBatchSampler -v
```

预期：全部 PASSED。亲手验证余弦退火曲线：

```bash
python -c "
from zllm.training.utils import get_lr
for t in [0, 25, 50, 75, 100]:
    print(f'step {t:3d}: lr = {get_lr(t, 100, base_lr=5e-4):.6f}')
"
```

## 29.6 本章小结 + 下章预告

本章要点：

1. **四重种子**（`setup_seed`）：random/np/torch/cuda 全固定，才彻底可复现。
2. **余弦退火**（`get_lr`）：$\eta_0(0.1+0.45(1+\cos(\pi t/T)))$，起点满、终点 1/10，平滑过渡。
3. **权重命名**（`init_model`）：`{阶段}_{hidden}{_moe?}.pth`，`strict=False` 容错加载。
4. **原子写入**（`lm_checkpoint`）：`.tmp` → `os.replace`，崩了不坏文件；存 optimizer/epoch/step 支持续训。
5. **SkipBatchSampler**：跳过已训 batch，断点续训不重复。

> **一句话带走**：种子保证可复现，余弦退火调度学习率，原子 checkpoint 让训练不怕崩溃。

**下章预告**：基础设施有了，但训练慢、显存不够怎么办？Ch 30《混合精度 AMP + 梯度累积 + GPU 优化》——用 bf16 混合精度省一半显存、用梯度累积模拟大 batch、用 TF32/Flash Attention 加速。这是让训练**跑得动、跑得快**的关键。
