# 步骤 001：项目初始化

## 目标

搭建 zllm 项目骨架：包配置、目录结构、测试基础设施。

## 完成内容

- `pyproject.toml` — Python 3.14+，最新依赖（torch/transformers/datasets/tokenizers 等）
- `zllm/` 主包，含 `config.py`（ZLLMConfig）
- `tests/` 测试目录，按 milestone 分组（m01-m12）
- `conftest.py` 共享 fixtures（device, small_config, default_config）

## ZLLMConfig 默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 768 | 隐藏层维度 |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `vocab_size` | 6400 | 词表大小 |
| `num_attention_heads` | 8 | Q head 数（GQA） |
| `num_key_value_heads` | 4 | KV head 数（GQA，2:1 分组） |
| `rope_theta` | 1e6 | RoPE 基础频率 |
| `max_position_embeddings` | 32768 | 最大序列长度 |
| `intermediate_size` | 384 | FFN 中间维度（π 缩放） |
| `tie_word_embeddings` | True | Weight Tying |

## 验证

```bash
pip install -e ".[dev]"
pytest tests/m01_foundations/ -v
```
