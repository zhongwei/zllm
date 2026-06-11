"""生成解码 — 支持 greedy/temperature/top-k/top-p/repetition penalty。

核心函数：
- generate: 基础自回归生成（逐步解码）
- generate_with_cache: KV Cache 加速推理

采样策略：
- temperature=0: greedy（argmax）
- temperature>0: 采样（越高越随机）
- top_k>0: 只从概率最高的 k 个 token 采样
- top_p>0: nucleus 采样（累计概率 ≤ p 的最小集合）
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model, input_ids, max_new_tokens=128, temperature=1.0,
    top_k=0, top_p=0.0, repetition_penalty=1.0, eos_token_id=None,
):
    """自回归生成（无 KV Cache）。

    Args:
        model: ZLLMForCausalLM
        input_ids: (batch, seq_len)
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度（0=greedy）
        top_k: Top-K 采样（0=不限制）
        top_p: Nucleus 采样（0=不限制）
        repetition_penalty: 重复惩罚（1.0=不惩罚）
        eos_token_id: 停止 token（None=不停止）

    Returns:
        generated_ids: (batch, seq_len + new_tokens)
    """
    model.eval()
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        outputs = model(generated)
        logits = outputs.logits[:, -1, :]

        if repetition_penalty != 1.0:
            for token_id in generated[0].unique():
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float("-inf")
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return generated


@torch.no_grad()
def generate_with_cache(
    model, input_ids, max_new_tokens=128, temperature=1.0,
    top_k=0, top_p=0.0, repetition_penalty=1.0, eos_token_id=None,
):
    """KV Cache 加速自回归生成。

    首次前向传播处理整个 prompt，后续只处理新 token。
    """
    model.eval()
    generated = input_ids.clone()
    batch_size = input_ids.shape[0]
    past_key_values = None

    for step in range(max_new_tokens):
        if step == 0:
            outputs = model(generated, use_cache=True, past_key_values=past_key_values)
        else:
            outputs = model(next_token, use_cache=True, past_key_values=past_key_values)

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        if repetition_penalty != 1.0:
            for token_id in generated[0].unique():
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float("-inf")
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return generated
