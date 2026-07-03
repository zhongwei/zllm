---
part: 3
chapter: 17
title: 分词理论：BPE/WordPiece/SentencePiece
milestone: M2
source: null
tests: null
status: draft
---

# 第 17 章 分词理论：BPE / WordPiece / SentencePiece

第 16 章我们把环境装好、把 `ZLLMConfig` 读懂。但模型还不认识任何文本——它只认识**整数 id**。在模型能学「预测下一个 token」之前，必须先把人类写的字符串变成一串整数。这一步叫**分词（tokenization）**，干这活的组件叫**分词器（tokenizer）**。

本章是 Part III 里唯一的一章纯理论章（Ch 18、19 才动手实现）。我们先讲清「为什么要分词」「分词的几个层级」「子词为什么胜出」，再比较三种主流子词算法——**BPE、WordPiece、Unigram/SentencePiece**——的数学判据，最后说清 zllm 为什么选 BPE。读完这一章，你就能带着完整的算法地图走进 Ch 18 的代码。

## 17.1 学习目标

读完本章，你应该能够：

- 说清文本、token、token id 三者的关系，以及为什么 LLM 必须分词；
- 解释字符级 / 词级 / 子词级三种分词的取舍，以及 OOV（Out-Of-Vocabulary）问题；
- 默写出 BPE、WordPiece、Unigram 三种算法的**合并/删除判据**（用频率、用似然增益、用似然损失）；
- 讲清 SentencePiece「语言无关」的含义，以及它和 BPE/Unigram 的关系；
- 解释 zllm 选择 BPE 的理由。

## 17.2 直觉与动机

### 17.2.1 文本怎么变成 id

神经网络的输入输出都是**向量**。对 LLM 而言，最基本的离散单元是 **token**：一个整数 id，取值范围是 $[0, V)$，$V$ 是词表大小（zllm 默认 $V=6400$）。分词器就是把字符串 $\to$ token id 序列的双射桥梁：

```mermaid
graph LR
    T["文本<br/>'你好世界'"] --> TOK["分词器 Tokenizer"]
    TOK --> IDS["token id 序列<br/>[1521, 89, 3404, ...]"]
    IDS --> EMB["Embedding 查表<br/>→ 浮点向量序列"]
    EMB --> M["Transformer 模型"]
    M --> OUT["输出 id 概率分布"]
    OUT --> TOK2["分词器（反向）"]
    TOK2 --> TXT["还原为文本"]
```

这张图的两端是字符串，中间全是整数和向量。分词器是**模型与人类语言之间的唯一接口**——它的质量直接决定模型能「看见」什么。注意一个关键事实：**分词是在预训练之前一次性训练好的**（zllm 在 M2 阶段训练，见 Ch 19），之后训练和推理都用同一份，不再变化。

### 17.2.2 三个层级的分词

历史上分词有三种粒度，各有硬伤：

| 粒度 | 例子（"unhappiness"） | 词表大小 | 主要问题 |
|------|----------------------|----------|----------|
| **词级**（word） | `[unhappiness]` 一个 | 极大（百万级） | **OOV**：没见过的词（新词、拼写错误、屈折变化）无法表示；中文无空格更难 |
| **字符级**（char） | `[u,n,h,a,p,p,i,n,e,s,s]` | 极小（~100） | 序列太长（一个词十几个 token），模型要建模超长依赖；字符本身无语义 |
| **子词级**（subword） | `[un, happ, iness]` | 中等（数千~数万） | **兼得两者优点**：高频整体成词、低频拆成有意义片段，几乎无 OOV |

**子词（subword）** 是现代 LLM 的标准选择。它的核心思想是：**学会一组「片段」**，常见词整体作为一个 token（如 `the`、`的`），罕见词拆成有意义的片段（如 `unhappiness → un + happ + iness`）。这同时解决了 OOV 和序列长度问题。

> **OOV 为什么致命**：词级分词遇到没收录的词，只能映射成一个特殊的 `<unk>` token，模型对这个 token 学不到任何语义。子词分词因为任何字符都在词表里，永远不会产生 `<unk>`——这是 zllm 用字节级编码的根本原因（Ch 18 会看到 `byte_level_encode`）。

### 17.2.3 为什么需要「训练」分词器

子词词表不是人手写的，而是**从语料里学出来的**：统计哪些字符组合经常一起出现，把它们合并成一个 token。「学习词表」就是分词器的训练过程。不同的「合并/删除判据」就产生了不同的算法——这正是本章后半部分的主题。

## 17.3 数学定义：三种子词算法

三种算法都从「初始词表 = 所有基本单元（字符或字节）」出发，区别在于**如何扩充/裁剪词表**。

### 17.3.1 BPE：贪心合并最高频对

**字节对编码（Byte Pair Encoding, BPE）** 是最经典的子词算法。它的规则极其简单：每一步找**出现次数最多的相邻 token 对**，把它们合并成一个新 token，加进词表。重复直到词表达到目标大小。

设当前所有序列的相邻对集合为 $\mathcal{P}$，每对 $(a,b)$ 的出现次数为 $\text{count}(a,b)$，则每一步的合并判据是：

$$
\boxed{\;(a^*, b^*) \;=\; \arg\max_{(a,b)\,\in\,\mathcal{P}} \;\text{count}(a,b)\;}
$$

合并后分配一个新 id（zllm 里从 256 开始递增，见 Ch 18 的 `train_bpe`），下一轮重新统计。BPE 是**完全确定性**的：给定相同语料和目标词表大小，结果唯一。这正是它适合教学和复现的原因。

### 17.3.2 WordPiece：合并似然增益最大的对

**WordPiece**（BERT 用）和 BPE 长得很像——也是「每步合并一个对」——但**判据不同**。它不选频率最高的对，而选「合并后让语言模型似然提升最大」的对。打分函数是：

$$
\boxed{\;s(A, B) \;=\; \frac{\text{count}(AB)}{\text{count}(A)\,\cdot\,\text{count}(B)}\;}
$$

选 $(a^*, b^*)=\arg\max s(A,B)$ 合并。这个分数其实是 **PMI（点互信息）的变体**：$\text{count}(A)\cdot\text{count}(B)$ 是「如果 A、B 独立」时的期望共现，分母把它归一化。直觉是——**两个片段如果一起出现得远比「偶然撞上」多，就值得合并**，哪怕绝对频率不高。

对比 BPE：BPE 只看绝对频率 $\text{count}(AB)$，WordPiece 看相对增益 $\frac{\text{count}(AB)}{\text{count}(A)\text{count}(B)}$。后果是 WordPiece 更愿意合并「低频但强相关」的对（比如 `play` + `ing`），而 BPE 偏爱绝对高频对。

### 17.3.3 Unigram / SentencePiece：反向删除

**Unigram** 走完全相反的路：先建一个**超大词表**（所有可能的子串），再逐步**删除**让总似然下降最小的 token。它假设一个 **Unigram 语言模型**：每个 token $t$ 有概率 $P(t)$，一段文本 $x$（切分为 $t_1 t_2 \dots t_k$）的概率是各 token 概率之积：

$$
P(x) \;=\; \prod_{i=1}^{k} P(t_i), \qquad \mathcal{L} \;=\; \sum_{x\,\in\,\text{corpus}} \log P(x)
$$

每一步，删掉那个「删除后 $\mathcal{L}$ 下降最少」的 token（即对总似然贡献最小的），直到词表缩到目标大小。Unigram 是**概率性**的——同一语料可能训出不同词表，编码时也用最大似然寻找最优切分（动态规划）。

**SentencePiece** 不是第四种算法，而是一个**实现框架**：它把 BPE 或 Unigram 做成**语言无关**——直接吃原始字节/字符，**不预切空格**。这对中文、日文（无词间空格）至关重要：英语分词器习惯先按空格切词再子词化，SentencePiece 跳过这一步，把空格也当成普通字符处理（用特殊符号 `▁` 表示词首）。HuggingFace 的 `tokenizers` 库在 `ByteLevel` 模式下也实现了类似的「语言无关」思想——zllm 用的就是它（Ch 19）。

### 17.3.4 一张对比表

```
              BPE              WordPiece          Unigram
训练方向      自底向上合并      自底向上合并        自顶向下删除
判据          count(a,b) 最大   count(ab)/(count(a)·count(b)) 最大   删除后似然损失最小
确定性        完全确定          完全确定            概率性（可能有多解）
编码方式      按学习顺序套用    按学习顺序套用       动态规划求最大似然切分
代表模型      GPT / LLaMA / Qwen BERT              T5 / XLNet
```

举个具体例子，同一组词 `low / lowest / newest` 在三种算法下可能切出不同的子词（高频片段会因判据不同而优先合并）：

```
词级:      low | lowest | newest          （3 个独立词，但 new/est 没复用）
BPE:       low | low + est | new + est    （est 复用，因 low/est 高频）
WordPiece: low | low + est | new + est    （类似，但 un+happy 这类更易合并）
Unigram:   low | low + est | new + est    （切分可能随概率略有不同）
```

三种算法最后都能把 `est`、`new` 这种有意义的片段抠出来复用，差异主要在**边界情形**和**词表的统计稳定性**上。

## 17.4 推导与几何：为什么「子词 = 更短的编码」

分词本质上是个**压缩**问题（Ch 05 信息论）。一段固定文本，切得越短（token 数越少），平均每个 token 承载的「信息」就越多，模型要预测的目标序列就越短，交叉熵负担就越低。

形式上，设文本长度为 $L$ 个字符，分词后变成 $N$ 个 token，则压缩比 $\rho = N/L$。好的分词器让 $\rho$ 尽量小（中文理想约 $0.5\sim0.7$，即每个 token 覆盖 $1.5\sim2$ 个字符）。BPE 合并最高频对，正是在**贪心地最大化压缩比**——把最常见的组合固化成单 token，整体序列就变短了。

$$
\text{压缩比} \;\rho \;=\; \frac{N_{\text{tokens}}}{L_{\text{chars}}} \;\downarrow \quad\Longleftrightarrow\quad \text{平均每个 token 承载的信息} \;\uparrow
$$

这也是为什么 Ch 19 的测试 `test_chinese_compression`（`tests/m02_tokenizer/test_035_production_tokenizer.py:99`）会断言「编码后 token 数少于字符数」——一个好的 BPE 分词器对中文必然有压缩效果。我们在 Ch 05 推导过「更短编码 = 更接近熵下界」，这里正是它的工程落地。

## 17.5 与本项目联系

本章是理论，但每一节都为接下来的两章铺路。

**前向钩子一：Ch 18 从零实现 BPE。** zllm 在 `zllm/tokenizer/bpe.py` 里用纯 Python 实现了 BPE 的全部 6 个函数——`byte_level_encode`（字节级编码，保证无 OOV）、`get_pair_counts`（统计相邻对，对应 17.3.1 的 $\text{count}$）、`merge`（合并）、`train_bpe`（训练循环，对应 $\arg\max$ 判据）、`encode`/`decode`。Ch 18 会逐函数拆解，你会看到 17.3.1 的公式怎么变成几行代码。

**前向钩子二：Ch 19 生产级实现。** 教学版 `bpe.py` 清晰但慢，跑不动真实语料。Ch 19 用 HuggingFace 的 `tokenizers` 库（C++ 实现）训练生产级 BPE（`zllm/tokenizer/trainer.py`），并加上特殊 token（`special_tokens.py`，对话边界 `<|im_start|>`/`<|im_end|>`、工具调用 `📞`、思考链标记）和对话模板（`chat_template.py`，把 OpenAI 风格的 messages 渲染成模型输入）。

**为什么 zllm 选 BPE 而非 WordPiece / Unigram**：三个理由——① **确定性**，便于教学复现（同一语料 → 同一词表）；② **生态一致**，GPT / LLaMA / Qwen3 都用 BPE，zllm 对齐 Qwen3 自然选它；③ **工程简单**，HuggingFace `tokenizers` 原生支持 `BPE` 模型（Ch 19 的 `Tokenizer(BPE())`）。WordPiece 多用于 BERT 系（理解任务），Unigram 多用于 T5 系，都不适合 decoder-only 的生成式 LLM。

> **一句话总结**：分词是模型与语言的接口，子词是最佳粒度，BPE 是 zllm 的选择。下一章我们就把 BPE 的公式变成代码。

## 17.6 本章小结 + 思考题

本章要点：

1. **分词**把字符串变成 token id 序列，是 LLM 与人类语言的唯一接口，在预训练前一次性训好。
2. **子词级**胜出：高频整体成词、低频拆成片段，几乎无 OOV。
3. **三种算法的判据**：BPE 合并最高频对 $\arg\max\,\text{count}$；WordPiece 合并似然增益最大的对 $\arg\max\,\frac{\text{count}(ab)}{\text{count}(a)\text{count}(b)}$；Unigram 反向删除似然损失最小的 token。
4. **SentencePiece** 是语言无关的实现框架（不预切空格），HuggingFace 的 `ByteLevel` 借鉴了这一思想。
5. **zllm 选 BPE**：确定性 + 生态一致（Qwen3）+ 工程简单。

### 思考题

> 这些题帮你把三种算法的判据刻进脑子。

1. **判据对比**：假设语料里 `the` 出现 1000 次（其中 `th`+`e` 相邻 1000 次），`xyz` 出现 10 次（但 `xy`+`z` 相邻 10 次，而 `xy`、`xz` 全语料各只出现这 10 次）。BPE 会先合并哪个？WordPiece 呢？（提示：算一算 $s(A,B)$，体会「绝对频率 vs 相对增益」的差别。）
2. **OOV 免疫**：为什么字节级 BPE（把文本先编码成 0–255 的字节）能保证**任何**文本都不会产生 `<unk>`？哪怕是一段乱码或 emoji？
3. **SentencePiece 与空格**：英文分词器常先按空格切词，为什么这对中文不适用？SentencePiece 用 `▁` 代替空格的做法，如何让同一个分词器同时服务中英文？

**下章预告**：Ch 18《教学版 BPE 实现》——我们把 17.3.1 的 $\arg\max\,\text{count}$ 判据变成 `zllm/tokenizer/bpe.py` 里 99 行纯 Python 代码，亲手训练一个能压缩中文的小 BPE。
