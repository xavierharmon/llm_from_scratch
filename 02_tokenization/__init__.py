"""
Phase 2: Tokenization
=====================
Convert raw text into sequences of integer token IDs.

Key class:
    BPETokenizer — Byte-Pair Encoding tokenizer built from scratch

Learning objective:
    Understand that tokenization is a lossy compression step.
    The vocabulary is a codebook. Encoding is compression.
    Decoding is decompression. The merge rules are the codec.
"""

from .bpe_tokenizer import BPETokenizer

__all__ = ["BPETokenizer"]
