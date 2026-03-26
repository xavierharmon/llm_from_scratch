"""
Token Embeddings
================
Maps integer token IDs to dense floating-point vectors.

Data engineering analogy:
    nn.Embedding is a lookup table — exactly like a dimension table in a star schema.
    Token ID is the foreign key. The embedding row is the feature vector.
    The lookup table is learned (its values are updated via backprop).

Why do we scale by sqrt(d_model)?
    With random initialization, embedding vectors have variance ~1.
    The positional encoding also has values in [-1, 1].
    Scaling keeps their magnitudes comparable so neither dominates the sum.

Shape convention used throughout this project:
    B = batch size
    T = sequence length (number of tokens)
    C = d_model (embedding dimension / "channels")
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding table.

    Args:
        vocab_size: number of unique tokens in the vocabulary
        d_model:    embedding dimension (must equal d_model throughout the model)

    Input:  token_ids — shape [B, T], dtype long
    Output: embeddings — shape [B, T, d_model], dtype float
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with small random values (standard practice)."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] long tensor of token IDs

        Returns:
            [B, T, d_model] float tensor of embedding vectors
        """
        # Scale by sqrt(d_model) to keep magnitudes in a useful range
        return self.embedding(token_ids) * math.sqrt(self.d_model)

    @property
    def weight(self) -> torch.Tensor:
        """Direct access to the embedding table (used for weight tying)."""
        return self.embedding.weight
