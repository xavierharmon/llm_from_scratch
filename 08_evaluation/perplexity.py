"""
Evaluation: Perplexity and Benchmarks
=======================================
Compute model quality metrics on held-out test data.
"""

import torch
import math
from torch.utils.data import DataLoader
from typing import Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from 06_training.loss import cross_entropy_loss


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute perplexity on an entire dataset split.

    Perplexity = exp(average cross-entropy loss)

    Lower is better:
        ~8000 = random (vocab size)
        ~200  = bad model
        ~50   = decent small model
        ~20   = good domain-specific model
        ~5    = excellent (near human on domain text)

    Args:
        model:       trained model in eval mode
        data_loader: DataLoader over tokenized text
        device:      compute device

    Returns:
        perplexity value (float)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = cross_entropy_loss(logits, y)
        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return math.exp(avg_loss)


@torch.no_grad()
def compute_token_accuracy(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute top-1 token prediction accuracy.
    Simple sanity check: what fraction of next tokens does the model get right?

    Note: this is a weaker metric than perplexity for LM evaluation,
    but useful for detecting training bugs early.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        preds = logits.argmax(dim=-1)   # [B, T]
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)
