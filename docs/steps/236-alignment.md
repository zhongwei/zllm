# 第10章 对齐训练 — DPO + PPO + GRPO

## 学习目标

掌握三种 RLHF 对齐方法，理解各自适用场景。

## RLHF 三步流程

```
1. Pretrain  → 学会语言
2. SFT       → 学会指令跟随
3. 对齐训练  → 学会人类偏好
    ├─ DPO  (直接偏好优化，最简单)
    ├─ PPO  (近端策略优化，最经典)
    └─ GRPO (群体相对策略优化，最新)
```

## 三种方法对比

| 维度 | DPO | PPO | GRPO |
|------|-----|-----|------|
| 需要 Reward Model | 否 | 是 | 是 |
| 需要 Critic | 否 | 是 | 否 |
| 需要在线采样 | 否 | 是 | 是 |
| 训练稳定性 | 高 | 中 | 中高 |
| 数据格式 | chosen/rejected | prompt-only | prompt-only |
| 学习率 | 4e-8 | 3e-7 | 3e-7 |

## DPO（Direct Preference Optimization）

直接用偏好数据训练，无需 RL：

```python
loss = -log_sigmoid(β * (log π_θ(chosen)/π_ref(chosen) - log π_θ(rejected)/π_ref(rejected)))
```

- 双模型：policy（训练）+ reference（冻结）
- 数据：`{"chosen": [...], "rejected": [...]}`
- β=0.15 控制偏好强度

## PPO（Proximal Policy Optimization）

最经典 RL 方法，Actor-Critic 架构：

- **CriticModel**: 共享 backbone + value_head → V(s)
- **GAE**: δ_t = r_t + γV(s_{t+1}) - V(s_t), A_t = Σ(γλ)^l δ_{t+l}
- **Clipped Loss**: -min(ratio·A, clip(ratio, 1±ε)·A)
- **Value Loss**: 0.5·max((V-R)², (clip(V, V_old±c)-R)²)

关键超参数：clip_epsilon=0.2, gamma=1.0, lam=0.95, vf_coef=0.5

## GRPO（Group Relative Policy Optimization）

PPO 的简化版，无需 Critic：

- 对同一 prompt 生成 N 个 response
- **群体相对优势**: A = (r - mean) / std（组内标准化）
- **KL 惩罚**: per_token_kl = exp(ref - policy) - (ref - policy) - 1
- 两种 Loss：
  - `grpo_loss`: 标准 PPO clip (ratio ∈ [1-ε, 1+ε])
  - `cispo_loss`: 单边裁剪 (ratio ≤ ε_high)

## 验证

```bash
pytest tests/m10_alignment/ -v   # 35 个测试全绿
```
