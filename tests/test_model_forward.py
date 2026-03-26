"""
Tests: Full Model Forward Pass
================================
Verifies that the complete RunningGPT model produces correct shapes,
doesn't produce NaN, and that all components interact correctly.

Run with: pytest tests/test_model_forward.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from 05_transformer.gpt_model import RunningGPT
from 05_transformer.layer_norm import LayerNorm
from 05_transformer.feed_forward import FeedForward
from 05_transformer.transformer_block import TransformerBlock


# Small config for fast tests
SMALL_CONFIG = dict(
    vocab_size=512, d_model=32, num_heads=2,
    num_layers=2, d_ff=64, max_seq_len=64, dropout=0.0
)


@pytest.fixture(scope="module")
def model():
    m = RunningGPT(**SMALL_CONFIG)
    m.eval()
    return m


class TestLayerNorm:

    def test_output_shape(self):
        ln = LayerNorm(d_model=32)
        x = torch.randn(2, 10, 32)
        out = ln(x)
        assert out.shape == x.shape

    def test_zero_mean_unit_variance(self):
        """After layer norm, each token's features should have ~0 mean and ~1 var."""
        ln = LayerNorm(d_model=64)
        x = torch.randn(4, 8, 64) * 5 + 3   # deliberately skewed
        out = ln(x)
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)
        assert mean.abs().max().item() < 1e-4, "Layer norm mean not near zero"
        assert (var - 1.0).abs().max().item() < 0.1, "Layer norm variance not near 1"


class TestFeedForward:

    @pytest.mark.parametrize("activation", ["gelu", "relu", "swiglu"])
    def test_output_shape(self, activation):
        ffn = FeedForward(d_model=32, d_ff=64, dropout=0.0, activation=activation)
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        assert out.shape == (2, 10, 32)

    def test_no_nan(self):
        ffn = FeedForward(d_model=32, d_ff=64, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        assert not torch.isnan(out).any()


class TestTransformerBlock:

    @pytest.fixture
    def block(self):
        return TransformerBlock(d_model=32, num_heads=2, d_ff=64, dropout=0.0)

    def test_output_shape(self, block):
        x = torch.randn(2, 10, 32)
        out, _ = block(x)
        assert out.shape == (2, 10, 32)

    def test_residual_preserves_magnitude(self, block):
        """Residual connections keep output in a reasonable magnitude range."""
        x = torch.randn(1, 8, 32)
        out, _ = block(x)
        # Output should not explode (rough sanity check)
        assert out.abs().max().item() < 100.0


class TestRunningGPT:

    def test_forward_output_shape(self, model):
        """Logits should be [B, T, vocab_size]."""
        B, T = 2, 20
        x = torch.randint(0, SMALL_CONFIG["vocab_size"], (B, T))
        logits, _ = model(x)
        assert logits.shape == (B, T, SMALL_CONFIG["vocab_size"])

    def test_no_nan_in_logits(self, model):
        x = torch.randint(0, SMALL_CONFIG["vocab_size"], (2, 16))
        logits, _ = model(x)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_sequence_length_one(self, model):
        """Model must handle single-token input (autoregressive generation step)."""
        x = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 1))
        logits, _ = model(x)
        assert logits.shape == (1, 1, SMALL_CONFIG["vocab_size"])

    def test_exceeds_max_seq_len_raises(self, model):
        """Sequences longer than max_seq_len should raise an AssertionError."""
        too_long = SMALL_CONFIG["max_seq_len"] + 1
        x = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, too_long))
        with pytest.raises(AssertionError):
            model(x)

    def test_weight_tying(self, model):
        """LM head weight should be the same object as token embedding weight."""
        assert model.lm_head.weight is model.token_embedding.weight

    def test_return_all_weights(self, model):
        """return_all_weights should return one weight tensor per layer."""
        x = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 10))
        _, weights = model(x, return_all_weights=True)
        assert weights is not None
        assert len(weights) == SMALL_CONFIG["num_layers"]
        for w in weights:
            # [B, num_heads, T, T]
            assert w.shape == (1, SMALL_CONFIG["num_heads"], 10, 10)

    def test_parameter_count_is_positive(self, model):
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert counts["transformer_blocks"] > counts["embeddings"]

    def test_gradient_flows_to_all_params(self):
        """A backward pass should compute gradients for all trainable params."""
        m = RunningGPT(**SMALL_CONFIG)
        m.train()
        x = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 8))
        y = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 8))
        logits, _ = m(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, SMALL_CONFIG["vocab_size"]), y.view(-1)
        )
        loss.backward()
        for name, param in m.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_vs_train_mode_dropout(self):
        """In eval mode, repeated forward passes should give identical results."""
        m = RunningGPT(**SMALL_CONFIG)
        m.eval()
        x = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 10))
        with torch.no_grad():
            out1, _ = m(x)
            out2, _ = m(x)
        assert torch.allclose(out1, out2), "Eval mode output is non-deterministic"
