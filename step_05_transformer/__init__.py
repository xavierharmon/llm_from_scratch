"""
Phase 5: Transformer
====================
Assemble all components into a complete GPT-style language model.

Key classes:
    LayerNorm        — per-token feature normalization
    FeedForward      — position-wise MLP (the "think" layer)
    TransformerBlock — attention + FFN + residuals (one full layer)
    RunningGPT       — complete stacked model with embedding + LM head
"""

from .layer_norm import LayerNorm
from .feed_forward import FeedForward
from .transformer_block import TransformerBlock
from .gpt_model import RunningGPT

__all__ = ["LayerNorm", "FeedForward", "TransformerBlock", "RunningGPT"]
