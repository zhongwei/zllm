# 第3章 模型组件 I — 归一化、位置编码、注意力

## 学习目标

理解 Transformer 的三个核心子组件：RMSNorm、RoPE、GQA Attention。

## 本章概览

| 组件 | 文件 | 职责 |
|------|------|------|
| RMSNorm | `zllm/model/norms.py` | 高效归一化，float32 内部计算 |
| RoPE | `zllm/model/rope.py` | 旋转位置编码 + YaRN 长序列外推 |
| Attention | `zllm/model/attention.py` | GQA 分组查询注意力 + KV Cache |

## 3.1 RMSNorm

比 LayerNorm 更高效：不计算均值，只用 RMS（Root Mean Square）归一化。

```
RMSNorm(x) = weight * (x / RMS(x))
RMS(x) = sqrt(mean(x²))
```

关键实现细节：
- 内部 float32 计算避免低精度溢出
- `norm()` 方法暴露纯归一化（不乘 weight），供外部复用

## 3.2 RoPE（Rotary Position Embedding）

用旋转矩阵在 Q/K 上编码**相对位置**信息：
- `precompute_freqs_cis`: 预计算 cos/sin 频率表（shape: `[seq_len, head_dim]`）
- `apply_rotary_pos_emb`: 对 Q/K 施加旋转（复数乘法等价）
- 频率公式：`f_i = 1 / θ^(2i/d)`，θ = rope_theta（默认 1e6）

### YaRN Scaling

当序列长度超过 `original_max_position_embeddings` 时，YaRN 通过线性 ramp 混合原始/缩放频率，实现长序列外推。

## 3.3 GQA Attention

Grouped Query Attention 是 MHA 和 MQA 的折中：
- MHA: 每个 Q head 有独立 KV head → 精度高但内存大
- MQA: 所有 Q head 共享 1 个 KV head → 内存小但精度低
- **GQA**: Q heads 分组，每组共享 1 个 KV head → 平衡精度和效率

zllm 默认：8 Q heads / 4 KV heads（n_rep=2）。

### 双路径注意力

- **Flash Attention**: 使用 `F.scaled_dot_product_attention`（硬件优化，需 seq_len > 1）
- **手动路径**: scores → causal mask → softmax → dropout → @ V（用于 KV cache 推理）

### QK-Norm

在 RoPE 之前对 Q/K 施加 RMSNorm，稳定训练。这是 Qwen3 的改进。

### KV Cache

推理时缓存历史 K/V，每步只计算新 token，将 O(n²) 降为 O(n)。

## 验证

```bash
pytest tests/m03_model_components/ -v   # 47 个测试全绿
```
