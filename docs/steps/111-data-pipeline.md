# 第5章 数据流水线 — 文本如何变成训练数据

## 学习目标

掌握 5 种 Dataset 类的设计，理解标签构造、prompt 掩码、DataLoader 集成。

## 本章概览

| Dataset | 用途 | 数据格式 | 返回 |
|---------|------|----------|------|
| PretrainDataset | 预训练 | `{"text": "..."}` | `(input_ids, labels)` |
| SFTDataset | 监督微调 | `{"conversations": [...]}` | `(input_ids, labels)` |
| DPODataset | 偏好优化 | `{"chosen": [...], "rejected": [...]}` | `{x,y,mask} × 2` |
| RLAIFDataset | 强化学习 | `{"conversations": [...]}` | `{"prompt", "answer"}` |
| AgentRLDataset | 工具调用 | `{"conversations", "gt"}` | `{"messages", "tools", "gt"}` |

## 5.1 TokenizerAdapter

将 `tokenizers.Tokenizer`（C++ 库对象）包装为具有 transformers 风格 API 的适配器：
- `.bos_token_id` / `.eos_token_id` / `.pad_token_id`
- `.__call__()` / `.encode()` / `.decode()`
- `.apply_chat_template()`

## 5.2 PretrainDataset

- 加载 JSONL `{"text": "..."}`
- tokenize → 加 BOS/EOS → pad 到 max_length
- labels = input_ids.clone()，pad 位置标 -100

## 5.3 SFTDataset

核心：`generate_labels()` — 只标记 assistant 回复区域：
1. 搜索 `bos_id`（`<|im_start|>assistant\n`）定位 assistant 起点
2. 搜索 `eos_id`（`<|im_end|>\n`）定位 assistant 终点
3. 该区间内 labels = input_ids，区间外 = -100

## 5.4 DPODataset

返回 chosen/rejected 两个序列的 `(x, y, mask)` 三元组：
- x = input_ids[:-1]，y = input_ids[1:]（自回归移位）
- mask = loss_mask[1:]（只对 assistant 区域计算 loss）

## 5.5 RLAIFDataset

Prompt-only：只返回 prompt 文本（带 generation prompt），不含回答。
按 `thinking_ratio` 概率开启推理模式。

## 5.6 AgentRLDataset

返回 `messages`（去掉最后一条 assistant）+ `tools` + `gt`（ground truth）。
用于 Agent RL 训练中工具调用评估。

## 验证

```bash
pytest tests/m05_data_pipeline/ -v   # 40 个测试全绿
```
