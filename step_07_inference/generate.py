"""
Text Generation — Sampling Strategies
=======================================
Given a trained model and a prompt, generate new tokens one at a time.

The autoregressive generation loop:
    1. Encode prompt to token IDs
    2. Run model forward pass → logits over vocab
    3. Sample next token from logits (using a strategy)
    4. Append new token to sequence
    5. Repeat from step 2 until EOS or max_new_tokens reached

Sampling strategies (from conservative → creative):
    Greedy:     always pick the highest-probability token.
                Deterministic. Often repetitive.

    Top-K:      sample from the K most likely tokens only.
                Balances diversity and quality. K=50 is a good default.

    Top-P       sample from the smallest set of tokens whose cumulative
    (nucleus):  probability ≥ p. Adaptive: more tokens on flat distributions.
                p=0.9 is a good default. Often better than top-K.

    Temperature: divide logits by T before softmax.
                T < 1.0 → sharper, more conservative
                T > 1.0 → flatter, more random / creative
                T = 1.0 → unmodified distribution
"""

import torch
import torch.nn.functional as F
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate new tokens autoregressively from a trained model.

    Args:
        model:          trained RunningGPT model (in eval mode)
        token_ids:      [1, T] starting context (encoded prompt)
        max_new_tokens: maximum tokens to generate
        temperature:    softmax temperature (1.0 = no change)
        top_k:          if set, only sample from top K tokens
        top_p:          if set, nucleus sampling threshold
        eos_id:         stop if this token ID is generated
        device:         torch device

    Returns:
        [1, T + new_tokens] tensor of all token IDs (prompt + generated)
    """
    model.eval()
    if device is not None:
        token_ids = token_ids.to(device)

    max_seq_len = model.max_seq_len

    for _ in range(max_new_tokens):
        # Truncate context to model's max length
        ctx = token_ids[:, -max_seq_len:]

        # Forward pass
        logits, _ = model(ctx)                    # [1, T, vocab_size]
        logits = logits[:, -1, :] / temperature   # [1, vocab_size] — last position only

        # Top-K filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, top_k).values[:, -1, None]
            logits = logits.masked_fill(logits < kth_val, float("-inf"))

        # Top-P (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above p
            sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_remove] = float("-inf")
            # Scatter back to original ordering
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)           # [1, vocab_size]
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        token_ids = torch.cat([token_ids, next_token], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    return token_ids


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    max_new_tokens: int = 100,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Deterministic greedy generation — always picks the most likely next token.
    Useful for reproducible outputs or evaluation.
    """
    model.eval()
    max_seq_len = model.max_seq_len

    for _ in range(max_new_tokens):
        ctx = token_ids[:, -max_seq_len:]
        logits, _ = model(ctx)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_ids = torch.cat([token_ids, next_token], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    return token_ids
