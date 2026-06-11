# 第4章 模型组件 II + 组装 — 从零件到完整模型

## 学习目标

组装 SwiGLU FFN、MoE、Transformer Block，最终构建可训练的 ZLLMForCausalLM。

## 本章概览

| 组件 | 文件 | 职责 |
|------|------|------|
| FeedForward | `model/ffn.py` | SwiGLU 前馈网络 |
| MOEFeedForward | `model/ffn.py` | MoE 稀疏专家（Router + top-k + 辅助损失） |
| ZLLMBlock | `model/block.py` | Pre-Norm Transformer Block |
| ZLLMModel | `model/backbone.py` | 模型主体（embed + N layers + norm + RoPE） |
| ZLLMForCausalLM | `model/causal_lm.py` | 完整语言模型（lm_head + loss） |

## 4.1 SwiGLU

```
FFN(x) = down(silu(gate(x)) * up(x))
```

三路投影：gate_proj（门控）、up_proj（上投影）、down_proj（下投影）。
SiLU 激活 = x * σ(x)，比 ReLU 更平滑。

intermediate_size 使用 π 缩放：`ceil(hidden_size * π / 64) * 64`，对齐 64 倍数以优化 Tensor Core。

## 4.2 MoE（Mixture of Experts）

- **Router**: `gate(x) → softmax → top-k`，选择 k 个专家
- **稀疏计算**: 只激活被选中的专家，节省计算量
- **负载均衡辅助损失**: `aux_loss = (load * router_prob).sum() * num_experts * coef`，防止路由坍塌
- **空专家梯度保持**: 无 token 命中的专家通过 `0 * sum(params)` 保持梯度流通

默认：4 专家 / top-1 路由，~198M 总参 / ~64M 活跃参。

## 4.3 Transformer Block

Pre-Norm 残差结构：
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

命名对齐 minimind：`self_attn` / `input_layernorm` / `post_attention_layernorm` / `mlp`。

## 4.4 ZLLMModel（Backbone）

- `embed_tokens`: 词嵌入
- `layers`: N 个 Transformer Block
- `norm`: 最终 RMSNorm
- `freqs_cos/freqs_sin`: RoPE 预计算缓存（register_buffer）
- 返回 `(hidden_states, past_key_values, aux_loss)`

## 4.5 ZLLMForCausalLM

- **Weight Tying**: `lm_head.weight = embed_tokens.weight`（节省参数量）
- **Loss**: `CrossEntropy(logits[..., :-1], labels[..., 1:], ignore_index=-100)`
- **logits_to_keep**: 只计算最后 N 个位置的 logits（推理优化）
- **返回**: `MoeCausalLMOutputWithPast`（兼容 transformers 生态）

## 验证

```bash
pytest tests/m04_model_assembly/ -v   # 57 个测试全绿
```
