# Summary

本书导航索引。各章文件将在后续章节任务中陆续创建；此处先登记全部相对路径。

## Part 0 序言

- [Ch 00 关于本书](part-0-preface/ch00-about.md) ✅

## Part I 数学基础

- [Ch 01 线性代数：向量与矩阵](part-1-math/ch01-linear-algebra-vectors.md)
- [Ch 02 线性代数：分解与几何](part-1-math/ch02-linear-algebra-decomposition.md)
- [Ch 03 概率论基础](part-1-math/ch03-probability.md)
- [Ch 04 统计推断](part-1-math/ch04-statistical-inference.md)
- [Ch 05 信息论](part-1-math/ch05-information-theory.md)
- [Ch 06 最优化基础](part-1-math/ch06-optimization.md)
- [Ch 07 微积分与链式法则](part-1-math/ch07-calculus-chain-rule.md)
- [Ch 08 张量计算与 PyTorch 自动微分](part-1-math/ch08-tensor-autograd.md)

## Part II 深度学习与 Transformer 理论

- [Ch 09 神经网络基础](part-2-transformer/ch09-nn-basics.md)
- [Ch 10 反向传播与训练动力学](part-2-transformer/ch10-backprop-training-dynamics.md)
- [Ch 11 序列建模：从 RNN/LSTM 到瓶颈](part-2-transformer/ch11-sequence-rnn-lstm.md)
- [Ch 12 注意力机制](part-2-transformer/ch12-attention.md)
- [Ch 13 Transformer 架构详解](part-2-transformer/ch13-transformer.md)
- [Ch 14 解码策略理论](part-2-transformer/ch14-decoding-strategy.md)
- [Ch 15 现代语言模型全景](part-2-transformer/ch15-modern-llm-landscape.md)

## Part III 基石与分词（M1 + M2）

- [Ch 16 项目初始化与开发环境](part-3-tokenizer/ch16-project-setup.md)
- [Ch 17 分词理论：BPE/WordPiece/SentencePiece](part-3-tokenizer/ch17-tokenizer-theory.md)
- [Ch 18 教学版 BPE 实现](part-3-tokenizer/ch18-bpe-implementation.md)
- [Ch 19 生产版 Tokenizer + 特殊 Token + Chat Template](part-3-tokenizer/ch19-production-tokenizer.md)

## Part IV 模型架构（M3 + M4）

- [Ch 20 RMSNorm 归一化](part-4-architecture/ch20-rmsnorm.md)
- [Ch 21 RoPE 旋转位置编码 + YaRN](part-4-architecture/ch21-rope-yarn.md)
- [Ch 22 GQA 注意力 + QK-Norm + KV Cache](part-4-architecture/ch22-gqa-qknorm-kv-cache.md)
- [Ch 23 SwiGLU 前馈网络](part-4-architecture/ch23-swiglu.md)
- [Ch 24 MoE 混合专家](part-4-architecture/ch24-moe.md)
- [Ch 25 Block + Backbone 组装](part-4-architecture/ch25-block-backbone.md)
- [Ch 26 CausalLM 头 + Weight Tying + Loss](part-4-architecture/ch26-causal-lm-head.md)

## Part V 数据与训练（M5 + M6 + M7）

- [Ch 27 数据流水线总览与 TokenizerAdapter](part-5-training/ch27-data-pipeline-adapter.md)
- [Ch 28 五种 Dataset 实现](part-5-training/ch28-dataset-implementations.md)
- [Ch 29 训练基础设施：种子/学习率/checkpoint](part-5-training/ch29-training-infrastructure.md)
- [Ch 30 混合精度 AMP + 梯度累积 + GPU 优化](part-5-training/ch30-amp-grad-accumulation.md)
- [Ch 31 预训练：NTP 与训练循环](part-5-training/ch31-pretraining-ntp.md)
- [Ch 32 预训练实战：数据准备/训练/loss 监控](part-5-training/ch32-pretraining-practice.md)

## Part VI 微调与对齐（M8–M11）

- [Ch 33 监督微调 SFT + Label Masking](part-6-alignment/ch33-sft-label-masking.md)
- [Ch 34 LoRA 低秩适配](part-6-alignment/ch34-lora.md)
- [Ch 35 RLHF 框架与对齐总论](part-6-alignment/ch35-rlhf-framework.md)
- [Ch 36 DPO 直接偏好优化](part-6-alignment/ch36-dpo.md)
- [Ch 37 PPO + GAE + Critic](part-6-alignment/ch37-ppo-gae-critic.md)
- [Ch 38 GRPO + CISPO](part-6-alignment/ch38-grpo-cispo.md)
- [Ch 39 知识蒸馏](part-6-alignment/ch39-distillation.md)
- [Ch 40 Agent RL 工具调用](part-6-alignment/ch40-agent-rl-tools.md)

## Part VII 推理与部署（M12）

- [Ch 41 解码算法实现](part-7-deployment/ch41-decoding-algorithm.md)
- [Ch 42 KV Cache 加速推理](part-7-deployment/ch42-kv-cache-inference.md)
- [Ch 43 OpenAI 兼容 API + CLI 部署](part-7-deployment/ch43-openai-api-cli.md)

## 附录

- [附录 A 命令速查](appendix-a-commands.md)
- [附录 B 超参数表](appendix-b-hyperparameters.md)
- [附录 C 术语表](appendix-c-glossary.md)
- [附录 D 参考文献](appendix-d-references.md)
