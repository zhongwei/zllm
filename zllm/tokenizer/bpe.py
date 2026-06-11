"""BPE (Byte Pair Encoding) 核心算法 — 从零实现。

教学版：清晰展示 byte-level 编码、相邻对统计、贪心合并。
生产环境使用 HuggingFace tokenizers 库（见 trainer.py）。
"""


def byte_level_encode(text):
    """将文本转换为 UTF-8 字节级 ID 序列（0-255）。

    BPE 的第一步：所有文本统一编码为字节序列，
    初始词表就是 256 个字节值，保证能表示任意文本（无 OOV）。
    """
    return list(text.encode("utf-8"))


def get_pair_counts(sequences):
    """统计所有序列中相邻 token 对的出现次数。

    返回 dict: {(token_a, token_b): count}
    """
    counts = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(seq, pair, new_id):
    """将序列中所有相邻的 pair 合并为 new_id。

    遵循 BPE 标准：从左到右扫描，已合并的位置不重复参与。
    """
    result = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(seq[i])
            i += 1
    return result


def encode(text, merges):
    """用训练好的合并规则编码文本。

    按合并规则的学习顺序（new_id 升序）逐条应用，
    每条规则一次扫描合并所有匹配位置。这与 BPE 的贪心编码等价。
    """
    ids = byte_level_encode(text)
    for pair, new_id in sorted(merges.items(), key=lambda x: x[1]):
        ids = merge(ids, pair, new_id)
    return ids


def decode(ids, merges):
    """将 BPE token 序列解码还原为文本。

    递归展开每个合并 token 为其原始字节对，最后按 UTF-8 解码。
    """
    rev = {v: k for k, v in merges.items()}

    def expand(token):
        if token in rev:
            a, b = rev[token]
            return expand(a) + expand(b)
        return [token]

    all_bytes = []
    for token in ids:
        all_bytes.extend(expand(token))
    return bytes(all_bytes).decode("utf-8")


def train_bpe(texts, vocab_size):
    """从零训练 BPE，返回合并规则。

    Args:
        texts: 训练文本列表
        vocab_size: 目标词表大小（必须 >= 256）

    Returns:
        merges: dict {(id_a, id_b): new_id}，new_id 从 256 开始递增
    """
    sequences = [byte_level_encode(text) for text in texts]
    num_merges = vocab_size - 256
    merges = {}
    for i in range(num_merges):
        counts = get_pair_counts(sequences)
        if not counts:
            break
        pair = max(counts, key=counts.get)
        new_id = 256 + i
        merges[pair] = new_id
        sequences = [merge(seq, pair, new_id) for seq in sequences]
    return merges
