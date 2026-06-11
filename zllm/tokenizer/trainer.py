"""生产级 Tokenizer 训练器（基于 HuggingFace tokenizers 库）。

使用 tokenizers 库的优化 C++ 实现，支持大规模语料高效训练。
与 bpe.py 的教学版形成对比：原理相同，工程实现不同。
"""

import os

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from zllm.tokenizer.special_tokens import ALL_SPECIAL_TOKENS


def train_tokenizer(texts, vocab_size=6400, save_dir=None):
    """训练 BPE tokenizer。

    Args:
        texts: 训练语料（可迭代字符串）
        vocab_size: 目标词表大小
        save_dir: 保存目录（None 则不保存）

    Returns:
        tokenizers.Tokenizer 对象
    """
    tokenizer = Tokenizer(BPE())
    # ByteLevel pre-tokenizer：先转字节序列再分词，保证无 OOV
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    # 配套 decoder：将 ByteLevel 字符映射还原为原始字节文本
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=ALL_SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    return tokenizer


def load_tokenizer(path):
    """加载已保存的 tokenizer。

    Args:
        path: 目录路径（含 tokenizer.json）或 tokenizer.json 文件路径

    Returns:
        tokenizers.Tokenizer 对象
    """
    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.json")
    return Tokenizer.from_file(path)
