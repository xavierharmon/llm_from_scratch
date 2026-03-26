"""
LoRA — Low-Rank Adaptation
============================
Fine-tune a large pretrained model by training only a small number
of additional parameters, keeping the original weights frozen.

Key insight:
    Weight updates during fine-tuning have low "intrinsic dimensionality."
    Instead of updating the full W (d × d), we decompose the update as:
        ΔW = B · A    where B is (d × r), A is (r × d), rank r << d

    For r=4, d=128: ΔW has 128×128=16,384 params.
                    B+A have 128×4 + 4×128 = 1,024 params.
                    That's 16× fewer parameters to train.

    For GPT-3 (d=12288, r=4):
        ΔW: 150M params per weight matrix
        B+A: ~100K params — 1500× reduction

Why does this work?
    The pretrained weight matrix W captures general language understanding.
    The fine-tuning update ΔW = BA only needs to capture the *delta* —
    the small domain-specific shift. Low-rank is sufficient for that delta.

In practice:
    - Apply LoRA to Q, V projections in attention (standard choice)
    - Merge ΔW back into W at inference time (zero overhead)
    - alpha/rank ratio controls effective learning rate of LoRA params

Paper: Hu et al., 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALinear(nn.Module):
    """
    A Linear layer with a parallel low-rank adapter.

    Forward pass:
        output = x @ W.T + (x @ A.T @ B.T) * (alpha / rank)
        where W is the frozen pretrained weight, A and B are the trained adapters.

    Args:
        in_features:  input dimension
        out_features: output dimension
        rank:         LoRA rank r (try 4, 8, or 16)
        alpha:        LoRA scaling factor (often set equal to rank)
        dropout:      dropout on the LoRA path
        bias:         whether the base linear has a bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Pretrained weight (frozen after initialization)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # LoRA adapters (trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))   # initialized to 0
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))  # random init
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        # Note: lora_A=0 means output is pure W at step 0, so training starts
        # from the same point as the pretrained model.

        self.lora_dropout = nn.Dropout(dropout)
        self.weight.requires_grad_(False)   # freeze base weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear (frozen)
        base_out = x @ self.weight.T
        if self.bias_param is not None:
            base_out = base_out + self.bias_param

        # LoRA path (trainable)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA delta into base weights for zero-overhead inference.
        Returns a standard nn.Linear with merged weights.
        """
        merged_weight = self.weight + (self.lora_B @ self.lora_A) * self.scaling
        layer = nn.Linear(self.weight.shape[1], self.weight.shape[0],
                          bias=self.bias_param is not None)
        layer.weight = nn.Parameter(merged_weight)
        if self.bias_param is not None:
            layer.bias = nn.Parameter(self.bias_param.clone())
        return layer

    def trainable_parameters(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora(model: nn.Module, rank: int = 4, alpha: float = 4.0) -> nn.Module:
    """
    Replace all attention Q and V projections in a model with LoRA versions.
    Freezes all other parameters.

    Args:
        model: RunningGPT model with pretrained weights
        rank:  LoRA rank
        alpha: LoRA scaling

    Returns:
        Model with LoRA adapters applied, all non-LoRA params frozen
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad_(False)

    replaced = 0
    for name, module in model.named_modules():
        # Target Q and V projections in attention blocks
        if isinstance(module, nn.Linear) and any(k in name for k in ("W_q", "W_v", "qkv_proj")):
            parent_name, attr = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
            lora_layer = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha,
            )
            # Copy pretrained weights
            lora_layer.weight = nn.Parameter(module.weight.data.clone())
            lora_layer.weight.requires_grad_(False)
            setattr(parent, attr, lora_layer)
            replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied to {replaced} layers | "
          f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model
