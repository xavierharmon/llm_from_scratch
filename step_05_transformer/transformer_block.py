"""
Transformer Block
=================
One complete layer of the transformer: attention + feed-forward,
each wrapped with a residual connection and layer norm.

Architecture (Pre-Norm / GPT-2 style):
    ┌──────────────┐
    │   Input x    │
    └──────┬───────┘
           │ ◄─────────────────── residual (skip connection)
           ▼
    LayerNorm(x)
           │
           ▼
    MultiHeadAttention
           │
           ▼
      Dropout
           │
           ▼
      x = x + output   ◄── Add (residual)
           │
           │ ◄─────────────────── residual (skip connection)
           ▼
    LayerNorm(x)
           │
           ▼
    FeedForward
           │
           ▼
      Dropout
           │
           ▼
      x = x + output   ◄── Add (residual)
           │
           ▼
        Output

Why residual connections?
    Without them, gradients vanish as they backpropagate through many layers.
    The skip connection gives gradients a "highway" back to early layers.
    This is what allows transformers to be stacked 96+ layers deep (GPT-4).
    
    Mathematically: x_{l+1} = x_l + F(x_l)
    The gradient: ∂x_{l+1}/∂x_l = I + ∂F/∂x_l (always at least identity)
"""

import torch
import torch.nn as nn
from typing import Optional

from step_05_transformer.layer_norm import LayerNorm
from step_05_transformer.feed_forward import FeedForward

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from step_04_attention.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block (GPT-style, pre-norm).

    Args:
        d_model:    embedding dimension
        num_heads:  number of attention heads
        d_ff:       feed-forward hidden dimension
        dropout:    dropout rate applied after attention and FFN
        activation: FFN activation function ('gelu', 'relu', 'swiglu')
        bias:       use bias in linear projections
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = False,
    ):
        super().__init__()

        self.ln1 = LayerNorm(d_model, bias=bias)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, bias=bias)
        self.ln2 = LayerNorm(d_model, bias=bias)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout, activation=activation, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:              [B, T, d_model]
            mask:           [1, 1, T, T] causal mask
            return_weights: whether to return attention weights

        Returns:
            output:   [B, T, d_model]
            weights:  [B, num_heads, T, T] or None
        """
        # --- Sub-layer 1: Self-attention with residual ---
        residual = x
        x_norm = self.ln1(x)
        attn_out, attn_weights = self.attn(x_norm, mask=mask, return_weights=return_weights)
        x = residual + self.dropout(attn_out)

        # --- Sub-layer 2: Feed-forward with residual ---
        residual = x
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout(ffn_out)

        return x, attn_weights
