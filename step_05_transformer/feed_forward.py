"""
Feed-Forward Network (FFN)
==========================
The second sub-layer in each transformer block. Applied independently
to each token position after the attention layer.

Architecture:
    Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)

Why is d_ff typically 4 × d_model?
    This is an empirical finding from the original paper. The expansion gives
    the network a higher-capacity "working memory" to perform per-token
    computations, before projecting back to d_model.
    GPT-2 small: d_model=768, d_ff=3072 (exactly 4×)
    GPT-3: d_model=12288, d_ff=49152 (exactly 4×)

What does the FFN actually do?
    Attention = "gather information from other tokens"
    FFN = "process and transform the gathered information"

    Research suggests FFN layers act as key-value memory stores:
    each row of W1 is a "key" and corresponding row of W2 is a "value".
    When an activation pattern matches a key, the value is retrieved.
    (Geva et al., 2021 — "Transformer Feed-Forward Layers Are Key-Value Memories")

Activation functions:
    ReLU:  original paper — max(0, x)
    GELU:  GPT-2/BERT — x * Φ(x), smoother than ReLU, generally performs better
    SwiGLU: LLaMA — gated variant, SOTA for large models
    We default to GELU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Args:
        d_model:    input/output dimension
        d_ff:       hidden dimension (typically 4 × d_model)
        dropout:    dropout rate after activation
        activation: 'gelu' (default), 'relu', or 'swiglu'
        bias:       whether to include bias in linear layers
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        self.activation_name = activation

        if activation == "swiglu":
            # SwiGLU: uses a gating mechanism — requires two parallel projections
            # Effective d_ff is typically 2/3 × 4 × d_model for parameter parity
            self.w1 = nn.Linear(d_model, d_ff, bias=bias)
            self.w2 = nn.Linear(d_model, d_ff, bias=bias)   # gate projection
            self.w3 = nn.Linear(d_ff, d_model, bias=bias)
        else:
            self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
            self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        if self.activation_name == "swiglu":
            for layer in [self.w1, self.w2, self.w3]:
                nn.init.normal_(layer.weight, std=0.02)
        else:
            nn.init.normal_(self.fc1.weight, std=0.02)
            nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        if self.activation_name == "swiglu":
            # SwiGLU: silu(W1·x) ⊙ (W2·x), then project back
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

        # Standard path: expand → activate → dropout → contract
        hidden = self.fc1(x)                    # [B, T, d_ff]

        if self.activation_name == "gelu":
            hidden = F.gelu(hidden)
        elif self.activation_name == "relu":
            hidden = F.relu(hidden)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

        hidden = self.dropout(hidden)
        return self.fc2(hidden)                 # [B, T, d_model]
