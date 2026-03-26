"""
RunningGPT — Full GPT-Style Language Model
============================================
Assembles all components into a complete autoregressive language model
trained to predict the next token in running log text.

Architecture overview:
    Token IDs
        │
        ▼
    TokenEmbedding (vocab_size → d_model)
        +
    PositionalEmbedding (position → d_model)
        │
        ▼
    Dropout
        │
        ▼ ×num_layers
    TransformerBlock (attention + FFN + residuals + norms)
        │
        ▼
    Final LayerNorm
        │
        ▼
    LM Head: Linear(d_model → vocab_size)  ← weight-tied to TokenEmbedding
        │
        ▼
    Logits [B, T, vocab_size]

Weight tying:
    The LM head (output projection) shares weights with the token embedding.
    This halves the parameter count and typically improves performance.
    Intuition: the embedding maps token → vector, and the LM head maps
    vector → token scores. They're inverses of each other, so sharing
    weights makes geometric sense.

Parameter count formula:
    params ≈ vocab_size * d_model            (embeddings, shared with LM head)
            + num_layers * (
                4 * d_model^2                 (attention QKV + output projections)
              + 2 * d_model * d_ff            (FFN up + down projections)
              + 4 * d_model                   (layer norm params)
              )

For our small config (d=128, h=4, L=4, d_ff=512, vocab=8000):
    ≈ 8000*128 + 4*(4*128² + 2*128*512) ≈ 1.0M + 0.8M ≈ 1.8M params
"""

import torch
import torch.nn as nn
from typing import Optional

from .transformer_block import TransformerBlock
from .layer_norm import LayerNorm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from 03_embeddings.token_embedding import TokenEmbedding
from 03_embeddings.positional_encoding import LearnedPositionalEmbedding


class RunningGPT(nn.Module):
    """
    GPT-style decoder-only transformer for running log language modeling.

    Args:
        vocab_size:  size of the token vocabulary
        d_model:     embedding / hidden dimension
        num_heads:   number of attention heads per block
        num_layers:  number of transformer blocks
        d_ff:        feed-forward hidden dim (default: 4 × d_model)
        max_seq_len: maximum context window length
        dropout:     dropout probability
        activation:  FFN activation ('gelu', 'relu', 'swiglu')
        bias:        use bias in linear layers (False = cleaner, slight perf boost)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: Optional[int] = None,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        d_ff = d_ff or (4 * d_model)

        # --- Embedding layers ---
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = LearnedPositionalEmbedding(max_seq_len, d_model, dropout=dropout)
        self.emb_dropout = nn.Dropout(dropout)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                bias=bias,
            )
            for _ in range(num_layers)
        ])

        # --- Output head ---
        self.final_norm = LayerNorm(d_model, bias=bias)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: lm_head and token_embedding share the same weight matrix
        self.lm_head.weight = self.token_embedding.weight

        # Log parameter count on construction
        n_params = sum(p.numel() for p in self.parameters())
        print(f"RunningGPT | {num_layers}L × {num_heads}H × d{d_model} "
              f"| {n_params / 1e6:.2f}M parameters")

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            token_ids:          [B, T] integer token IDs
            mask:               optional [1,1,T,T] causal mask
            return_all_weights: if True, collect attention weights from all layers

        Returns:
            logits:   [B, T, vocab_size] — raw scores over vocabulary
            weights:  list of [B, h, T, T] per-layer attention weights, or None
        """
        B, T = token_ids.shape
        assert T <= self.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # --- Embedding ---
        x = self.token_embedding(token_ids)   # [B, T, d_model]
        x = self.pos_embedding(x)             # adds positional vectors

        # --- Transformer blocks ---
        all_weights = [] if return_all_weights else None
        for block in self.blocks:
            x, w = block(x, mask=mask, return_weights=return_all_weights)
            if return_all_weights and w is not None:
                all_weights.append(w)

        # --- Output ---
        x = self.final_norm(x)                # [B, T, d_model]
        logits = self.lm_head(x)              # [B, T, vocab_size]

        return logits, all_weights

    @torch.no_grad()
    def get_next_token_probs(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: run the model and return probability distribution
        over the next token (last position only).

        Args:
            token_ids: [B, T] or [T] token IDs

        Returns:
            [B, vocab_size] probability distribution
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        logits, _ = self(token_ids)
        return torch.softmax(logits[:, -1, :], dim=-1)

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts broken down by component."""
        return {
            "embeddings": sum(p.numel() for p in self.token_embedding.parameters())
                        + sum(p.numel() for p in self.pos_embedding.parameters()),
            "transformer_blocks": sum(p.numel() for p in self.blocks.parameters()),
            "final_norm": sum(p.numel() for p in self.final_norm.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
