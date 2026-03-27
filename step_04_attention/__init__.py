"""
Phase 4: Attention
==================
The mechanism that allows every token to gather information
from every other token in the sequence.

Key functions/classes:
    scaled_dot_product_attention — the core math operation
    generate_causal_mask         — prevents attending to future tokens
    MultiHeadAttention           — h parallel attention heads
"""

from .scaled_dot_product import scaled_dot_product_attention, generate_causal_mask
from .multi_head_attention import MultiHeadAttention

__all__ = ["scaled_dot_product_attention", "generate_causal_mask", "MultiHeadAttention"]
