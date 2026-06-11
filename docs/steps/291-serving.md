# 第12章 推理与部署 — 从模型到服务

## 学习目标

掌握模型推理、解码策略、API 服务部署，完成完整训练管线闭环。

## 解码策略

### Greedy（贪心）

```python
next_token = argmax(logits)  # 每步选概率最高的
```

确定性输出，适合评测。但可能重复、缺乏多样性。

### Temperature（温度）

```python
logits = logits / T
probs = softmax(logits)
next_token = sample(probs)
```

- T=0: 等价 greedy
- T<1: 更确定性（尖锐分布）
- T>1: 更随机（平滑分布）
- 推荐：0.85

### Top-K 采样

```python
top_k_logits = logits.topk(k)  # 只保留最高 k 个
```

- k=1: greedy
- k=50: 限制候选，防止低概率 token
- 推荐：k=50 或 0（不限制）

### Top-P（Nucleus）采样

```python
sorted_probs → cumsum → 取累计概率 ≤ p 的最小集合
```

- p=0.95: 覆盖 95% 概率质量的 token
- 比 Top-K 更自适应（概率集中时少取，分散时多取）
- 推荐：0.95

## KV Cache 加速

标准推理每步重新计算所有 token 的注意力：

```
Step 1: [t1, t2, t3, t4] → predict t5
Step 2: [t1, t2, t3, t4, t5] → predict t6  ← 重复计算 t1-t4!
```

KV Cache 缓存已计算的 Key/Value：

```
Step 1: [t1, t2, t3, t4] → predict t5, cache K1-K4, V1-V4
Step 2: [t5] + cache → predict t6  ← 只计算 t5!
```

推理复杂度从 O(n²) 降到 O(n)。

## API 服务

OpenAI 兼容接口：

```
POST /v1/chat/completions
GET  /v1/models
```

请求格式：
```json
{
  "model": "zllm",
  "messages": [{"role": "user", "content": "你好"}],
  "temperature": 0.85,
  "top_p": 0.95,
  "max_tokens": 512
}
```

## 完整训练管线

```
Tokenizer → Pretrain → SFT → [LoRA/DPO/PPO/GRPO/Distillation/Agent]
                                                        ↓
                                              save → load → generate → API
```

12 个里程碑完成！

## 验证

```bash
pytest tests/m12_serving/ -v   # 20 个测试全绿
pytest tests/                  # 全部 421 个测试全绿
```
