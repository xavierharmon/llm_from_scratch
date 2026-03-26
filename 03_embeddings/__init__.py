"""
Phase 3: Embeddings
===================
Convert token IDs into dense vectors that capture semantic meaning.

Key classes:
    TokenEmbedding            — learnable lookup table (vocab_size × d_model)
    SinusoidalPositionalEncoding — fixed position encoding (original paper)
    LearnedPositionalEmbedding   — trainable position encoding (GPT-2 style)
"""

from .token_embedding import TokenEmbedding
from .positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEmbedding

__all__ = ["TokenEmbedding", "SinusoidalPositionalEncoding", "LearnedPositionalEmbedding"]
