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

from step_05_transformer.layer_norm import LayerNorm
from step_05_transformer.feed_forward import FeedForward
from step_05_transformer.transformer_block import TransformerBlock
from step_05_transformer.gpt_model import RunningGPT

__all__ = ["LayerNorm", "FeedForward", "TransformerBlock", "RunningGPT"]
