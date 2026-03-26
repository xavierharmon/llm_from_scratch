"""
Scaled Dot-Product Attention — The Core Operation
==================================================
This is the fundamental operation that makes transformers work.
Every other attention variant (multi-head, cross, grouped-query) 
is built on top of this function.

Formula:
    Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

Intuition for data engineers:
    Think of it as a fuzzy lookup / soft JOIN:
    - Q (Query)  = "what I'm looking for"        — like a SELECT clause
    - K (Key)    = "what each row advertises"     — like an index
    - V (Value)  = "what each row actually gives" — like the SELECT columns
    
    The dot product Q·Kᵀ measures query-key compatibility — like a similarity score.
    Softmax turns raw scores into a probability distribution (weights sum to 1).
    The final multiply by V is a weighted average of all values.
    
    The √d_k scaling prevents the dot products from getting too large in
    high dimensions (which would push softmax into its saturating region,
    producing near-zero gradients).

The causal mask:
    For autoregressive generation, token at position t must NOT be able
    to attend to positions t+1, t+2, ... (it can't see the future).
    We enforce this by setting those attention scores to -infinity
    before the softmax — they become exactly 0 after softmax.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        Q:         query tensor,  shape [..., T_q, d_k]
        K:         key tensor,    shape [..., T_k, d_k]
        V:         value tensor,  shape [..., T_k, d_v]
        mask:      boolean mask,  shape [..., T_q, T_k]
                   True  = keep this position
                   False = mask this position (set to -inf)
        dropout_p: dropout probability on attention weights
        training:  whether model is in training mode

    Returns:
        output:  weighted sum of values, shape [..., T_q, d_v]
        weights: attention weight matrix, shape [..., T_q, T_k]
    """
    d_k = Q.size(-1)

    # Step 1: Compute raw attention scores via dot product
    # Shape: [..., T_q, T_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: Apply causal mask (if provided)
    # Future positions are set to -inf so they become 0 after softmax
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 3: Softmax — convert scores to a probability distribution
    # Each row (query position) now sums to 1.0
    weights = F.softmax(scores, dim=-1)

    # Handle NaN from all-masked rows (edge case at sequence boundaries)
    weights = torch.nan_to_num(weights, nan=0.0)

    # Step 4: Dropout on attention weights (regularization)
    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    # Step 5: Weighted sum of values
    # Shape: [..., T_q, d_v]
    output = torch.matmul(weights, V)

    return output, weights


def generate_causal_mask(seq_len: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generate an upper-triangular causal mask.
    Position i can only attend to positions 0..i (not the future).

    Returns:
        Boolean tensor of shape [1, 1, seq_len, seq_len]
        True  = can attend
        False = blocked (future tokens)

    Example (seq_len=4):
        [[True,  False, False, False],
         [True,  True,  False, False],
         [True,  True,  True,  False],
         [True,  True,  True,  True ]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T] for broadcasting over batch + heads
