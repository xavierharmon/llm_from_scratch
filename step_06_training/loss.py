"""
Loss Functions and Metrics
===========================
Cross-entropy loss for language model training, plus perplexity calculation.

The Language Modeling Objective:
    Given tokens [t1, t2, ..., t_n], maximize the probability of each token
    given all previous tokens: P(t_i | t_1, ..., t_{i-1}).

    Equivalently, minimize the negative log likelihood:
    Loss = -1/N * Σ log P(t_i | context)

    This is exactly cross-entropy loss between the model's predicted
    distribution and the one-hot true next token.

Perplexity:
    PP = exp(Loss)
    
    Interpretation: perplexity ≈ "how many tokens was the model choosing between?"
    - PP = 1.0  → perfect prediction (impossible in practice)
    - PP = 8000 → random guessing over our vocab
    - PP = 50   → good small model on domain-specific text
    - PP = 20   → excellent for our running domain

    Data engineering analogy: perplexity is like selectivity in a query plan.
    Low perplexity = the model narrows down candidates quickly.
"""

import torch
import torch.nn.functional as F
import math


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        logits:       [B, T, vocab_size] raw model output
        targets:      [B, T] true next-token IDs
        ignore_index: token ID to ignore in loss (e.g., padding)

    Returns:
        scalar loss tensor (mean over non-ignored positions)
    """
    B, T, V = logits.shape
    # Flatten batch and time dims for F.cross_entropy
    loss = F.cross_entropy(
        logits.view(B * T, V),   # [B*T, vocab_size]
        targets.view(B * T),     # [B*T]
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


def perplexity(loss: torch.Tensor) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss: scalar cross-entropy loss (nats, not bits)

    Returns:
        perplexity value (float)
    """
    return math.exp(loss.item())


def bits_per_byte(loss: torch.Tensor) -> float:
    """
    Convert nats-per-token loss to bits-per-byte.
    A common alternative metric, especially for comparing across tokenizers.

    bits/byte = loss (nats/token) * log2(e) / avg_bytes_per_token
    Assumes ~4 bytes per token for BPE tokenizers.
    """
    nats_per_token = loss.item()
    bits_per_token = nats_per_token / math.log(2)
    avg_bytes_per_token = 4.0   # rough average for BPE on English text
    return bits_per_token / avg_bytes_per_token
