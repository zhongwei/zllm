# 第2章 Tokenizer — 文本如何变成数字

## 学习目标

理解 BPE (Byte Pair Encoding) 分词原理，掌握 tokenizer 的训练、使用和 chat template 机制。

## 本章概览

| 步骤 | 内容 | 文件 |
|------|------|------|
| 26-29 | BPE 核心算法（从零实现） | `zllm/tokenizer/bpe.py` |
| 30-34 | 特殊 Token 定义 | `zllm/tokenizer/special_tokens.py` |
| 35-43 | 生产级 Tokenizer（tokenizers 库） | `zllm/tokenizer/trainer.py` |
| 44-48 | Chat Template | `zllm/tokenizer/chat_template.py` |
| 49-50 | 集成测试 | `tests/m02_tokenizer/` |

## 2.1 BPE 算法原理

BPE (Byte Pair Encoding) 的核心思想：
1. 将所有文本转换为 UTF-8 字节序列（初始词表 = 256 字节）
2. 统计相邻字节对频率，贪心合并最高频对
3. 重复直到达到目标词表大小

**为什么用字节级？** 保证能表示任意文本（无 OOV），中文字符自然成为多字节序列。

zllm 提供两个实现：
- `bpe.py` — 教学版，从零实现 `byte_level_encode` / `get_pair_counts` / `merge` / `train_bpe` / `encode` / `decode`
- `trainer.py` — 生产版，使用 HuggingFace `tokenizers` 库的 C++ 优化实现

## 2.2 特殊 Token

| 类别 | Token | 用途 |
|------|-------|------|
| 对话边界 | `<\|im_start\|>` `<\|im_end\|>` | 标记每轮对话开始/结束 |
| 多模态预留 | `<\|vision_start\|>` `<\|image_pad\|>` 等 | 预留视觉/音频扩展 |
| 工具调用 | `📞` `<toolcall_start>` 等 | Agent RL 专用 |
| 思考链 | `<reasoningchain_start>` `<reasoningchain_end>` | 推理模式 |
| Buffer | `<\|buffer1\|>` ~ `<\|buffer8\|>` | 预留扩展位 |

## 2.3 Chat Template

对话格式化为模型输入：

```
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮你？<|im_end|>
```

`render_messages()` 支持：
- `tools` — 注入工具声明到 system 块
- `open_thinking` — 用思考链标签包裹 assistant 内容
- `add_generation_prompt` — 追加 `<|im_start|>assistant\n`（用于推理）

## 验证

```bash
pytest tests/m02_tokenizer/ -v   # 80 个测试（含 M1）全绿
```
