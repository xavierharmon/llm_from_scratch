"""
Tests: Attention Shapes and Properties
========================================
Verifies that tensor shapes are correct throughout the attention pipeline
and that mathematical properties hold (weights sum to 1, mask works, etc.).

Run with: pytest tests/test_attention_shapes.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from 04_attention.scaled_dot_product import scaled_dot_product_attention, generate_causal_mask
from 04_attention.multi_head_attention import MultiHeadAttention


class TestScaledDotProductAttention:

    def test_output_shape(self):
        """Output should match [B, h, T_q, d_v]."""
        B, h, T, d_k, d_v = 2, 4, 10, 32, 32
        Q = torch.randn(B, h, T, d_k)
        K = torch.randn(B, h, T, d_k)
        V = torch.randn(B, h, T, d_v)
        out, weights = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (B, h, T, d_v)
        assert weights.shape == (B, h, T, T)

    def test_attention_weights_sum_to_one(self):
        """Attention weights must form a valid probability distribution (sum = 1)."""
        B, h, T, d_k = 2, 4, 8, 16
        Q = torch.randn(B, h, T, d_k)
        K = torch.randn(B, h, T, d_k)
        V = torch.randn(B, h, T, d_k)
        _, weights = scaled_dot_product_attention(Q, K, V)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_causal_mask_blocks_future(self):
        """With causal mask, position i should not attend to positions > i."""
        B, h, T, d_k = 1, 1, 6, 16
        Q = torch.randn(B, h, T, d_k)
        K = torch.randn(B, h, T, d_k)
        V = torch.randn(B, h, T, d_k)
        mask = generate_causal_mask(T)
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        weights_2d = weights[0, 0]  # [T, T]
        # Upper triangle (future positions) must be zero
        upper = torch.triu(weights_2d, diagonal=1)
        assert upper.abs().max().item() < 1e-6, "Causal mask failed: future tokens attended to"

    def test_no_nan_in_output(self):
        """Output should never contain NaN, even with all-masked rows."""
        B, h, T, d_k = 2, 2, 5, 8
        Q = torch.randn(B, h, T, d_k)
        K = torch.randn(B, h, T, d_k)
        V = torch.randn(B, h, T, d_k)
        mask = generate_causal_mask(T)
        out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert not torch.isnan(out).any(), "NaN detected in attention output"
        assert not torch.isnan(weights).any(), "NaN detected in attention weights"

    def test_causal_mask_shape(self):
        """Causal mask should be [1, 1, T, T] for broadcasting."""
        mask = generate_causal_mask(8)
        assert mask.shape == (1, 1, 8, 8)

    def test_causal_mask_is_lower_triangular(self):
        """The True values in the causal mask should form a lower triangle."""
        T = 5
        mask = generate_causal_mask(T)[0, 0]  # [T, T]
        expected = torch.tril(torch.ones(T, T)).bool()
        assert torch.equal(mask, expected)


class TestMultiHeadAttention:

    @pytest.fixture
    def mha(self):
        return MultiHeadAttention(d_model=64, num_heads=4, dropout=0.0)

    def test_output_shape(self, mha):
        """Output shape must match input shape [B, T, d_model]."""
        B, T, d_model = 2, 10, 64
        x = torch.randn(B, T, d_model)
        out, _ = mha(x)
        assert out.shape == (B, T, d_model)

    def test_d_model_divisibility(self):
        """Should raise AssertionError when d_model % num_heads != 0."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=4)

    def test_returns_weights_when_requested(self, mha):
        """return_weights=True should return a weight tensor."""
        x = torch.randn(2, 8, 64)
        _, weights = mha(x, return_weights=True)
        assert weights is not None
        assert weights.shape == (2, 4, 8, 8)  # [B, h, T, T]

    def test_returns_none_weights_by_default(self, mha):
        """By default, weights should be None (save memory)."""
        x = torch.randn(2, 8, 64)
        _, weights = mha(x)
        assert weights is None

    def test_gradient_flows(self, mha):
        """Gradients should flow back through the attention layer."""
        x = torch.randn(1, 5, 64, requires_grad=True)
        out, _ = mha(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_sequence_lengths(self, mha):
        """Model should handle varying sequence lengths up to max."""
        for T in [1, 4, 16, 32]:
            x = torch.randn(1, T, 64)
            out, _ = mha(x)
            assert out.shape == (1, T, 64)
