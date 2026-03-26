"""
Phase 7: Inference
==================
Generate text from a trained model using various decoding strategies.

Key functions:
    generate        — stochastic generation (top-k, top-p, temperature)
    greedy_generate — deterministic argmax generation
    beam_search     — beam search decoder (beam_search.py)
"""

from .generate import generate, greedy_generate

__all__ = ["generate", "greedy_generate"]
