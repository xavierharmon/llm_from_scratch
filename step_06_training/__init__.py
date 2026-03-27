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

from step_06_training.trainer import Trainer
from step_06_training.optimizer import build_optimizer, CosineWarmupScheduler
from step_06_training.loss import cross_entropy_loss, perplexity

__all__ = ["Trainer", "build_optimizer", "CosineWarmupScheduler",
           "cross_entropy_loss", "perplexity"]
