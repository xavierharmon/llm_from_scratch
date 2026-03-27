"""
Multi-Head Attention
====================
Runs scaled dot-product attention h times in parallel on different
linear projections of Q, K, V, then concatenates and projects the results.

Why multiple heads?
    Different heads can specialize in different *types* of relationships:
    - Head 1 might learn syntactic dependencies (verb → subject)
    - Head 2 might learn semantic similarity (marathon ↔ race ↔ 26.2)
    - Head 3 might learn positional proximity (adjacent words)
    
    Running h=4 heads at d_k=32 each has the same compute cost as
    h=1 head at d_k=128 — but captures richer structure.

Data engineering analogy:
    Like running h parallel GROUP BY queries on different feature subsets,
    then combining all the aggregations into one wide output row.

Shapes throughout this module (using our conventions):
    B  = batch size
    T  = sequence length
    C  = d_model (total embedding dim)
    h  = num_heads
    d_k = C / h (per-head dimension)
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from .scaled_dot_product import scaled_dot_product_attention, generate_causal_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer.

    Args:
        d_model:   total embedding dimension (must be divisible by num_heads)
        num_heads: number of parallel attention heads
        dropout:   dropout on attention weights
        bias:      whether to include bias in projection layers
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # per-head dimension
        self.dropout = dropout

        # Single combined projection for Q, K, V — more efficient than 3 separate layers
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2))  # output scaled init

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:              [B, T, d_model] input embeddings
            mask:           [1, 1, T, T] causal mask (optional, generated if None)
            return_weights: if True, also return the attention weight tensor

        Returns:
            output:   [B, T, d_model] attended representations
            weights:  [B, h, T, T] attention weights (only if return_weights=True)
        """
        B, T, C = x.shape

        # Generate causal mask if not provided
        if mask is None:
            mask = generate_causal_mask(T, device=x.device)

        # --- Project to Q, K, V in one shot ---
        # qkv shape: [B, T, 3 * d_model]
        qkv = self.qkv_proj(x)

        # Split along last dim into three equal pieces
        Q, K, V = qkv.split(self.d_model, dim=-1)  # each: [B, T, d_model]

        # --- Reshape for multi-head: [B, h, T, d_k] ---
        def reshape_for_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q = reshape_for_heads(Q)   # [B, h, T, d_k]
        K = reshape_for_heads(K)
        V = reshape_for_heads(V)

        # --- Scaled dot-product attention (runs in parallel across all heads) ---
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V,
            mask=mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        # attn_output: [B, h, T, d_k]

        # --- Concatenate heads and project back to d_model ---
        # Transpose: [B, T, h, d_k] → contiguous → [B, T, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)  # [B, T, d_model]

        if return_weights:
            return output, attn_weights
        return output, None
