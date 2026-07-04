---
part: appendix
appendix: D
title: 参考文献
status: draft
---

# 附录 D 参考文献

全书引用的核心论文和资源，按主题分组。

## D.1 基础架构

| 论文 | 作者 | 年份 | 贡献 | 相关章节 |
|------|------|------|------|---------|
| Attention Is All You Need | Vaswani et al. | 2017 | 提出 Transformer 架构，自注意力机制 | Ch 12–13, 22 |
| Layer Normalization | Ba et al. | 2016 | LayerNorm，RMSNorm 的基础 | Ch 20 |
| Root Mean Square Normalization | Zhang & Sennrich | 2019 | RMSNorm，去掉均值中心化，更快 | Ch 20 |
| GLU Variants Improve Transformer | Shazeer | 2020 | SwiGLU 激活函数 | Ch 23 |
| GQA: Training Generalized Multi-Query | Ainslie et al. | 2023 | 分组查询注意力，省 KV 显存 | Ch 22 |

## D.2 位置编码

| 论文 | 作者 | 年份 | 贡献 | 相关章节 |
|------|------|------|------|---------|
| RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al. | 2021 | RoPE 旋转位置编码 | Ch 21 |
| YaRN: Efficient Context Window Extension | Peng et al. | 2023 | YaRN RoPE 缩放，长上下文外推 | Ch 21 |

## D.3 训练与优化

| 论文 | 作者 | 年份 | 贡献 | 相关章节 |
|------|------|------|------|---------|
| Adam: A Method for Stochastic Optimization | Kingma & Ba | 2014 | Adam 优化器 | Ch 06 |
| Decoupled Weight Decay Regularization | Loshchilov & Huter | 2019 | AdamW 优化器（zllm 实际用） | Ch 06, 29 |
| On the Difficulty of Training Recurrent Neural Networks | Pascanu et al. | 2013 | 梯度裁剪 | Ch 10, 30 |
| MixPrecision Training | Micikevicius et al. | 2017 | 混合精度训练（AMP） | Ch 30 |
| FlashAttention | Dao et al. | 2022 | Flash Attention，减少 HBM 读写 | Ch 22 |

## D.4 微调与参数效率

| 论文 | 作者 | 年份 | 贡献 | 相关章节 |
|------|------|------|------|---------|
| LoRA: Low-Rank Adaptation | Hu et al. | 2021 | 低秩适配，参数高效微调 | Ch 34 |
| Using the Output Embedding to Improve Language Models | Press & Wolf | 2016 | Weight Tying | Ch 15, 26 |

## D.5 对齐与强化学习

| 论文 | 作者 | 年份 | 贡献 | 相关章节 |
|------|------|------|------|---------|
| Training language models to follow instructions (InstructGPT) | Ouyang et al. | 2022 | RLHF 三阶段框架 | Ch 35 |
| Direct Preference Optimization | Rafailov et al. | 2023 | DPO，绕过 RM 的偏好优化 | Ch 36 |
| Proximal Policy Optimization Algorithms | Schulman et al. | 2017 | PPO，clipped surrogate | Ch 37 |
| DeepSeekMath / GRPO | Shao et al. | 2024 | GRPO，群体相对优势去 Critic | Ch 38 |
| DeepSeek-R1 | DeepSeek-AI | 2025 | CISPO 单边裁剪 + 推理训练 | Ch 38 |
| High-Dimensional Continuous Control Using GAE | Schulman et al. | 2015 | GAE 广义优势估计 | Ch 37 |

## D.6 知识蒸馏

| 论文 | 作者 | 年份 | 贡献 | 相关章节 |
|------|------|------|------|---------|
| Distilling the Knowledge in a Neural Network | Hinton et al. | 2015 | 知识蒸馏，软标签 + 温度 | Ch 39 |

## D.7 分词

| 论文 | 作者 | 年份 | 贡献 | 相关章节 |
|------|------|------|------|---------|
| Neural Machine Translation of Rare Words with Subword Units | Sennrich et al. | 2016 | BPE 字节对编码 | Ch 18 |
| Byte-Pair Encoding tokenization | — | — | BPE 分词实现 | Ch 18–19 |

## D.8 数学基础

| 资源 | 作者 | 年份 | 贡献 | 相关章节 |
|------|------|------|------|---------|
| The Matrix Cookbook | Petersen & Pedersen | 2012 | 矩阵求导速查 | Ch 07 |
| Pattern Recognition and Machine Learning | Bishop | 2006 | 概率论、信息论基础 | Ch 03, 05 |
| Deep Learning | Goodfellow et al. | 2016 | 深度学习教材 | Ch 08–11 |

## D.9 开源项目

| 项目 | 说明 | 相关章节 |
|------|------|---------|
| [minimind](https://github.com/jingyaogong/minimind) | zllm 的主要参考项目，小尺寸 LLM 全流程 | 全书 |
| [LLaMA](https://github.com/meta-llama/llama) | Meta 开源 LLM，RMSNorm/RoPE/SwiGLU 架构 | Ch 20–23 |
| [Qwen](https://github.com/QwenLM/Qwen2) | 阿里通义千问，GQA + QK-Norm | Ch 22 |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | 模型生态框架 | Ch 26, 43 |
| [PyTorch](https://pytorch.org/) | 深度学习框架 | 全书 |

## D.10 工具与库

| 库 | 用途 | 相关章节 |
|----|------|---------|
| `tokenizers` | BPE 分词器训练 | Ch 18–19 |
| `datasets` | JSONL 数据加载 | Ch 28 |
| `fastapi` | API 服务器 | Ch 43 |
| `torch.amp` | 混合精度训练 | Ch 30 |
| `torch.compile` | 图编译加速 | Ch 30 |
