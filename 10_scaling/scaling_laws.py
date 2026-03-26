"""
Phase 10: Scaling Laws
=======================
How do loss, compute, and model size relate to each other?

Key papers:
    Kaplan et al. (2020) — "Scaling Laws for Neural Language Models" (OpenAI)
    Hoffmann et al. (2022) — "Training Compute-Optimal LLMs" (Chinchilla, DeepMind)

Core Chinchilla finding:
    For a given compute budget C (in FLOPs):
        Optimal model size:   N* ≈ sqrt(C / 6)
        Optimal training tokens: D* ≈ 20 × N*

    In plain English: you should train a model with ~20 tokens per parameter.
    GPT-3 (175B params) was undertrained. A 70B model trained on 1.4T tokens
    (Chinchilla) outperforms it at the same compute cost.

    Data engineering analogy:
        Scaling laws = query cost estimation in a query planner.
        Given a compute budget, predict the optimal resource allocation.

FLOPs per training step:
    Roughly 6 × N × D
    where N = number of parameters, D = number of training tokens.
    (Factor of 6: forward pass ≈ 2N FLOPs, backward ≈ 4N FLOPs)
"""

import math
import numpy as np


def chinchilla_optimal(compute_budget_flops: float) -> dict:
    """
    Given a FLOPs budget, return Chinchilla-optimal model size and token count.

    Args:
        compute_budget_flops: total training FLOPs (e.g. 1e21 for ~GPT-3 scale)

    Returns:
        dict with optimal_params, optimal_tokens, and ratio
    """
    # From Chinchilla paper: N* = (C / 6 / 20)^0.5  (approximately)
    optimal_params = math.sqrt(compute_budget_flops / (6 * 20))
    optimal_tokens = 20 * optimal_params
    return {
        "compute_flops": compute_budget_flops,
        "optimal_params": optimal_params,
        "optimal_tokens": optimal_tokens,
        "tokens_per_param": optimal_tokens / optimal_params,
    }


def estimate_flops(num_params: int, num_tokens: int) -> float:
    """Estimate total training FLOPs: ~6 × N × D."""
    return 6 * num_params * num_tokens


def kaplan_loss_curve(
    num_params: float,
    num_tokens: float,
    irreducible_loss: float = 1.69,
) -> float:
    """
    Approximate the Kaplan scaling law loss curve.
    L(N, D) ≈ irreducible_loss + A/N^alpha + B/D^beta

    Default constants from Kaplan et al. (2020).
    Returns estimated cross-entropy loss (nats).
    """
    A, alpha = 406.4, 0.34
    B, beta = 410.7, 0.28
    return irreducible_loss + (A / num_params**alpha) + (B / num_tokens**beta)


def print_scaling_table() -> None:
    """Print a table of model sizes vs Chinchilla-optimal training tokens."""
    configs = [
        ("Tiny (ours)",     2e6,   "laptop demo"),
        ("Small",           117e6, "GPT-2 small"),
        ("Medium",          345e6, "GPT-2 medium"),
        ("Large",           774e6, "GPT-2 large"),
        ("XL",              1.5e9, "GPT-2 XL"),
        ("GPT-3",           175e9, "GPT-3"),
        ("Chinchilla-7B",   7e9,   "LLaMA-7B territory"),
        ("Chinchilla-70B",  70e9,  "LLaMA-70B territory"),
    ]
    print(f"\n{'Model':<22} {'Params':>12} {'Chinchilla tokens':>20} {'Est. FLOPs':>18}")
    print("─" * 74)
    for name, params, note in configs:
        tokens = 20 * params
        flops = estimate_flops(int(params), int(tokens))
        print(f"{name:<22} {params/1e6:>10.1f}M {tokens/1e9:>18.1f}B {flops:.2e}  ({note})")


if __name__ == "__main__":
    print_scaling_table()

    print("\nChinchilla-optimal for a 1e21 FLOP budget:")
    result = chinchilla_optimal(1e21)
    print(f"  Optimal params:  {result['optimal_params']/1e9:.1f}B")
    print(f"  Optimal tokens:  {result['optimal_tokens']/1e12:.1f}T")
