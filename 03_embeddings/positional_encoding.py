"""
Positional Encoding
===================
Adds position information to token embeddings so the model knows
*where* in the sequence each token appears.

Why is this needed?
    Self-attention is permutation-invariant — "the cat sat" and "sat cat the"
    produce the same attention scores without positional encoding.
    We must inject order information explicitly.

Two variants implemented here:
    1. Sinusoidal (original Transformer paper) — fixed, not learned
    2. Learned positional embedding — GPT-2 style, trainable

Sinusoidal formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Intuition: each dimension oscillates at a different frequency, like
    a binary counter but in continuous space. The model can learn to
    extract relative positions by attending to differences in these values.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).
    Not learned — computed once and reused.

    Args:
        d_model:     embedding dimension
        max_seq_len: maximum sequence length the model will ever see
        dropout:     applied after adding positional encoding
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the sinusoidal table once
        pe = torch.zeros(max_seq_len, d_model)  # [T, d_model]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # [T, 1]

        # Frequency denominators: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)   # even dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims

        # Register as a buffer (not a parameter — won't be updated by optimizer)
        self.register_buffer("pe", pe.unsqueeze(0))    # [1, T, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] token embeddings

        Returns:
            [B, T, d_model] embeddings + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embedding (GPT-2 style).
    Treats positions like tokens — each position index has its own
    trainable embedding vector.

    Args:
        max_seq_len: maximum context length
        d_model:     embedding dimension
        dropout:     applied after adding
    """

    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] token embeddings

        Returns:
            [B, T, d_model] embeddings + learned positional vectors
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        pos_emb = self.embedding(positions)                          # [1, T, d_model]
        return self.dropout(x + pos_emb)
