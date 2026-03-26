# Phase 5: Transformer — Learning Notes

## What you're building
The full GPT-style model: a stack of identical transformer blocks,
each refining the token representations before a final prediction layer.

## The transformer block — two sub-layers

```
Input x
  │
  ├──────────────────┐  (residual connection)
  │                  │
LayerNorm(x)         │
  │                  │
MultiHeadAttention   │  ← "gather information from context"
  │                  │
Dropout              │
  │                  │
  x = x +────────────┘
  │
  ├──────────────────┐  (residual connection)
  │                  │
LayerNorm(x)         │
  │                  │
FeedForward          │  ← "process and transform the gathered info"
  │                  │
Dropout              │
  │                  │
  x = x +────────────┘
  │
Output x
```

## Why residual connections?

Without skip connections, gradients must flow through every layer to reach
early parameters. In a 96-layer model, the gradient is multiplied 96 times.
If each multiplication is slightly less than 1, the gradient vanishes to zero.

The residual connection provides a direct gradient highway:
```
x_{l+1} = x_l + F(x_l)
∂x_{l+1}/∂x_l = I + ∂F/∂x_l    ← always at least identity
```

Even if ∂F/∂x_l → 0 (dead layer), the gradient still flows through the I term.
This is why ResNets (2015) and Transformers can be hundreds of layers deep.

## Why Pre-Norm instead of Post-Norm?

**Post-Norm** (original 2017 paper):
```
x = LayerNorm(x + Attention(x))
```

**Pre-Norm** (GPT-2, modern practice):
```
x = x + Attention(LayerNorm(x))
```

Pre-Norm: the residual stream stays un-normalized, which gives gradients
a cleaner path back. Trains more stably, especially at large scales.
GPT-2 switched to Pre-Norm and most modern models follow.

## What does the FFN actually do?

The attention layer "gathers information from other tokens."
The FFN layer "processes that information for the current token."

Research (Geva et al., 2021) shows FFN layers act as key-value memories:
- W1 rows = "keys" (patterns that activate certain computations)
- W2 rows = "values" (what to output when that pattern fires)

For our running model, the FFN might learn:
- "If I see a pace token after a distance token → this is a running performance"
- "If I see 'Boston' near 'marathon' → this is about the Boston Marathon"

## Weight tying

The LM head (vocab projection) and the token embedding share the same matrix:
```python
self.lm_head.weight = self.token_embedding.weight
```

Intuition: the embedding maps token → vector.
The LM head maps vector → token scores.
They're approximate inverses, so sharing weights makes sense.
Also halves the parameter count for the largest single matrix in the model.

## Parameter count breakdown (our small model: d=128, L=4, h=4, V=8000)

| Component | Formula | Count |
|---|---|---|
| Token embedding | V × d | 8000 × 128 = 1,024,000 |
| Positional embedding | T × d | 256 × 128 = 32,768 |
| Per-layer attention | 4 × d² | 4 × 128² × 4 = 262,144 |
| Per-layer FFN | 2 × d × d_ff | 2 × 128 × 512 × 4 = 524,288 |
| Layer norms | ~4 × d × 4 | ~2,048 |
| **Total** | | **~1.85M** |

LM head is free (weight tied to embedding).

## Exercises

1. Count the parameters in our model with `model.count_parameters()`.
   Which component dominates at small scale? At large scale (GPT-3)?

2. Remove the residual connections from `TransformerBlock.forward()`.
   Train for 200 steps. What happens to the loss curve?

3. Try `bias=True` in the model config. How many extra parameters does this add?
   Does validation loss improve?

4. Increase `num_layers` from 4 to 8, keeping total parameters similar by
   reducing `d_model`. Which config gets lower perplexity?
