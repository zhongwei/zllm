---
part: appendix
appendix: C
title: 术语表
status: draft
---

# 附录 C 术语表

全书术语中英对照，按拼音/字母排序。

| 中文 | 英文 | 章节 | 释义 |
|------|------|------|------|
| 半精度 | BF16 / FP16 | Ch 08/30 | 16 位浮点格式，省显存；BF16 动态范围大 |
| 采样 | Sampling | Ch 03/41 | 按概率分布随机抽取 token |
| 残差连接 | Residual Connection | Ch 10/25 | $y = x + F(x)$，让深网络梯度可回流 |
| 暗知识 | Dark Knowledge | Ch 39 | 软标签中的类间相似度信息 |
| 优势函数 | Advantage Function | Ch 37 | $A = Q - V$，动作比平均好多少 |
| 标签掩码 | Label Masking | Ch 28/33 | 只对 assistant 回复算 loss |
| 裁剪 | Clipping | Ch 37/38 | 限制 ratio 范围防策略崩坏 |
| 策略梯度 | Policy Gradient | Ch 35/37 | 沿梯度方向优化策略 |
| 词嵌入 | Token Embedding | Ch 15/26 | token id → 向量 |
| 词表 | Vocabulary | Ch 17/19 | 所有 token 的集合 |
| 等价变换/位置编码 | RoPE | Ch 21 | 旋转位置编码，编码相对位置 |
| 动量 | Momentum | Ch 11/29 | 优化器累积历史梯度方向 |
| 对齐 | Alignment | Ch 35 | 让模型有用/无害/诚实 |
| 对齐税 | Alignment Tax | Ch 35 | 对齐训练降低其他能力 |
| 多头注意力 | Multi-Head Attention | Ch 22 | 多个注意力头并行 |
| 反向传播 | Backpropagation | Ch 10 | 链式法则计算梯度 |
| 范数 | Norm | Ch 20 | 向量的「长度」 |
| 分词 | Tokenization | Ch 17/18 | 文本 → token 序列 |
| 分组查询注意力 | GQA | Ch 22 | KV 头少于 Q 头，省显存 |
| 广义优势估计 | GAE | Ch 37 | $\lambda$ 平衡偏差与方差 |
| 荒野梯度 | Reward Hacking | Ch 35 | 模型钻奖励函数漏洞 |
| 交叉熵 | Cross-Entropy | Ch 05/26 | 真实分布与预测分布的差异 |
| 梯度累积 | Gradient Accumulation | Ch 30 | 多个小 batch 梯度累加等效大 batch |
| 梯度裁剪 | Gradient Clipping | Ch 30 | 限制梯度范数防爆炸 |
| 梯度下降 | Gradient Descent | Ch 09/11 | 沿梯度反方向更新参数 |
| 检查点 | Checkpoint | Ch 29 | 保存训练状态用于续训 |
| 监督微调 | SFT | Ch 33 | 用对话数据微调预训练模型 |
| 矩阵乘法 | Matrix Multiplication | Ch 07 | 线性代数核心运算 |
| 均方误差 | MSE | Ch 05 | 回归损失 $(y-\hat{y})^2$ |
| 可逆性 | Reversibility | Ch 15 | Weight Tying 的 embedding/lm_head 互逆 |
| 累积梯度 | Accumulated Gradient | Ch 30 | 见梯度累积 |
| 离散概率 | Discrete Probability | Ch 03 | 有限样本空间的概率分布 |
| 量词 | Quantizer | Ch 18 | BPE 合并的 token 对 |
| 链式法则 | Chain Rule | Ch 07/10 | 复合函数求导法则 |
| 临界点 | Critical Point | Ch 09 | 梯度为零的点（极小/极大/鞍点） |
| 流水线 | Pipeline | Ch 27 | 数据处理流程 |
| 蒙特卡洛 | Monte Carlo | Ch 37 | 随机采样估计期望 |
| 偏好优化 | Preference Optimization | Ch 36 | DPO，直接用偏好数据训练 |
| 软最大 | Softmax | Ch 03/06 | 把 logits 变成概率分布 |
| 数学期望 | Expectation | Ch 03 | 随机变量的平均值 |
| 损失函数 | Loss Function | Ch 05 | 衡量预测与真实差距 |
| 随机种子 | Random Seed | Ch 29 | 固定随机性保证可复现 |
| 梯度消失 | Vanishing Gradient | Ch 10 | 深网络梯度逐层衰减 |
| 提示工程 | Prompt Engineering | Ch 27/43 | 设计输入 prompt 引导模型 |
| 温度 | Temperature | Ch 06/39/41 | 控制 softmax 平滑度 |
| 微调 | Fine-tuning | Ch 33/34 | 在预训练模型上继续训练 |
| 无害性 | Harmlessness | Ch 35 | 对齐目标之一 |
| 下一个 token 预测 | NTP | Ch 05/15/31 | 预测序列的下一个 token |
| 线性层 | Linear Layer | Ch 15 | $y = Wx + b$ 仿射变换 |
| 效用 | Helpfulness | Ch 35 | 对齐目标之一 |
| 协方差 | Covariance | Ch 07 | 衡量两变量线性关系 |
| 信息熵 | Entropy | Ch 03 | 分布的不确定度 |
| 学习率 | Learning Rate | Ch 09/11/29 | 参数更新步长 |
| 学习率调度 | LR Schedule | Ch 29 | 训练中动态调整学习率 |
| 原子写入 | Atomic Write | Ch 29 | 先写临时文件再 rename，防崩溃损坏 |
| 余弦退火 | Cosine Annealing | Ch 29 | 余弦曲线降学习率 |
| 预训练 | Pretraining | Ch 31/32 | 用海量文本做 NTP |
| 知识蒸馏 | Knowledge Distillation | Ch 39 | 教师→学生传递暗知识 |
| 直接偏好优化 | DPO | Ch 36 | 绕过 RM 直接偏好训练 |
| 群体相对策略优化 | GRPO | Ch 38 | 去 Critic 用群体基线 |
| 近端策略优化 | PPO | Ch 37 | Actor-Critic + clip + GAE |
| 正则化 | Regularization | Ch 11 | 防止过拟合 |
| 知识 | Knowledge | Ch 39 | 教师模型传递的信息 |
| 自注意力 | Self-Attention | Ch 22 | 序列内每个位置关注所有位置 |
| 专家混合 | MoE | Ch 24 | 多个 FFN 专家稀疏激活 |
| 重复惩罚 | Repetition Penalty | Ch 41 | 降低已出现 token 概率 |
| 噪声 | Noise | Ch 03/30 | 随机扰动 |
| 遮蔽语言模型 | Masked LM | Ch 05 | BERT 式双向预测 |
| 指数移动平均 | EMA | Ch 11 | 梯度的滑动平均 |
| 指南 | Guidance | Ch 43 | API 服务的使用说明 |
| 智能体 | Agent | Ch 40 | 能调用工具的模型 |
| 钟形曲线 | Bell Curve | Ch 03 | 正态分布 |
| 灾难性遗忘 | Catastrophic Forgetting | Ch 33 | 微调破坏预训练知识 |
| 蒸馏温度 | Distillation Temperature | Ch 39 | 软化教师输出的温度 |
