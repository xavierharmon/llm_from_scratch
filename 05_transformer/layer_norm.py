"""
Layer Normalization
====================
Normalizes the activations within each token's feature vector independently.

Why normalize?
    Deep networks suffer from "internal covariate shift" — the distribution of
    activations changes as weights update, making training unstable.
    Normalization keeps activations in a stable range throughout training.

Layer Norm vs Batch Norm:
    Batch Norm: normalize across the batch dimension (B) for each feature.
        Problem: statistics depend on batch size. Breaks at batch_size=1.
    Layer Norm: normalize across the feature dimension (C) for each token.
        Each token is normalized independently — no batch dependency.
        This is why transformers use Layer Norm, not Batch Norm.

Formula:
    y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

    gamma (scale) and beta (shift) are learned parameters that let the
    network undo the normalization if that's what the gradient descent wants.

Pre-norm vs Post-norm:
    Original paper: Post-norm (normalize after attention/FFN)
    Modern practice: Pre-norm (normalize before — more stable training)
    This implementation uses Pre-norm (GPT-2 style).
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalization with learnable scale (gamma) and shift (beta).

    Args:
        d_model:  feature dimension to normalize over
        eps:      small constant for numerical stability in the denominator
        bias:     whether to include the beta (shift) parameter

    Input/output shape: [..., d_model] — same as input
    """

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        # Gamma (scale) initialized to 1 — identity transform at start
        self.weight = nn.Parameter(torch.ones(d_model))
        # Beta (shift) initialized to 0 — identity transform at start
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_model] tensor (any number of leading dims)

        Returns:
            Normalized tensor, same shape as x
        """
        # Compute mean and variance along the last (feature) dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift (learned)
        out = self.weight * x_norm
        if self.bias is not None:
            out = out + self.bias
        return out
