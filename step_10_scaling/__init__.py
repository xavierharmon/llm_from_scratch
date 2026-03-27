"""Phase 10: Scaling Laws — compute-optimal training, Chinchilla, parameter budgets."""
from .scaling_laws import chinchilla_optimal, estimate_flops, kaplan_loss_curve
__all__ = ["chinchilla_optimal", "estimate_flops", "kaplan_loss_curve"]
