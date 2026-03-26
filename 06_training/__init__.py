"""
Phase 6: Training
=================
The gradient descent loop that adjusts all model parameters
to minimize next-token prediction loss.

Key classes:
    Trainer                — full training orchestrator
    CosineWarmupScheduler  — LR warmup + cosine decay

Key functions:
    build_optimizer        — AdamW with param group separation
    cross_entropy_loss     — language modeling loss
    perplexity             — human-readable loss metric
"""

from .trainer import Trainer
from .optimizer import build_optimizer, CosineWarmupScheduler
from .loss import cross_entropy_loss, perplexity

__all__ = ["Trainer", "build_optimizer", "CosineWarmupScheduler",
           "cross_entropy_loss", "perplexity"]
