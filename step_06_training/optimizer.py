"""
Optimizer and Learning Rate Schedule
======================================
AdamW optimizer + cosine decay with linear warmup.

Why AdamW?
    Adam: adaptive per-parameter learning rates using gradient moment estimates.
    AdamW: fixes weight decay in Adam (decouples it from the gradient update).
    Standard choice for transformer training since GPT-2.

    Adam hyperparameters:
        beta1=0.9:   exponential decay for first moment (gradient mean)
        beta2=0.95:  exponential decay for second moment (gradient variance)
                     (Note: 0.95 not 0.999 — Karpathy's recommendation for LLMs)
        eps=1e-8:    numerical stability
        weight_decay=0.1: L2 regularization on weights (not biases/norms)

Why cosine decay with warmup?
    Warmup: start with tiny LR, ramp up over ~200 steps.
            Without warmup, large initial gradients destabilize early training.
    Cosine decay: smoothly reduce LR from peak to ~10% of peak.
            Outperforms step decay and linear decay empirically.

Why decay only weights (not biases, norms, embeddings)?
    Weight decay is a regularizer — it penalizes large weights.
    Biases and layer norm params are 1-D vectors that don't benefit from this.
    Not decaying them is standard practice (GPT-2, NanoGPT, etc.)
"""

import torch
import torch.nn as nn
import math


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.1) -> torch.optim.AdamW:
    """
    Construct AdamW with separate param groups:
        - decay group:    weight matrices (2-D tensors)
        - no-decay group: biases, layer norms, embeddings (1-D or special)

    Args:
        model:        the transformer model
        lr:           peak learning rate
        weight_decay: L2 penalty for the decay group

    Returns:
        Configured AdamW optimizer
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Decay 2-D weight matrices; skip everything else
        if param.dim() >= 2 and "embedding" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    print(f"Optimizer param groups:")
    print(f"  Decay:    {sum(p.numel() for p in decay_params):,} params")
    print(f"  No-decay: {sum(p.numel() for p in no_decay_params):,} params")

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup → cosine decay to min_lr.

    Schedule:
        step < warmup_steps:  lr = peak_lr * (step / warmup_steps)
        step >= warmup_steps: lr = min_lr + 0.5 * (peak_lr - min_lr)
                                   * (1 + cos(π * progress))
        where progress = (step - warmup_steps) / (max_steps - warmup_steps)

    Args:
        optimizer:     the optimizer to wrap
        warmup_steps:  number of linear warmup steps
        max_steps:     total training steps
        min_lr_ratio:  LR at end of training as a fraction of peak_lr (default 0.1)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        super().__init__(optimizer, lr_lambda)
