# Phase 4: Attention — Learning Notes

## What you're building
A mechanism that allows each token to gather relevant information
from all other tokens in the sequence, weighted by relevance.

## The core formula

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

Unpacked:
- **Q (Query)**: "What information am I looking for?"
- **K (Key)**: "What information do I advertise having?"
- **V (Value)**: "What information do I actually share when queried?"

## Database analogy (the best one)

Attention is a **soft, differentiable lookup**:

```sql
-- Hard lookup (standard DB query):
SELECT value FROM table WHERE key = query   -- returns 0 or 1 rows

-- Soft lookup (attention):
SELECT SUM(value * similarity(query, key))  -- weighted sum over ALL rows
FROM table
```

The difference: in a database, a key either matches or doesn't.
In attention, every key matches with a different *strength*.
The model learns what constitutes a "match."

For the sentence "My marathon pace was 3:45":
- When processing "pace", the attention might strongly attend to "3:45"
  (because pace and a time are closely related)
- The "3:45" value vector gets high weight in the output for "pace"

## Why scale by √d_k?

In high dimensions (d_k=32), the dot product Q·K can get very large.
Large values push the softmax into its saturating region:
```
softmax([30, 1, 1]) ≈ [1.0, 0.0, 0.0]   # one-hot, no learning signal
softmax([3, 1, 1])  ≈ [0.73, 0.13, 0.13]  # smooth, gradient flows
```

Dividing by √d_k keeps the variance of the dot product near 1.0.

## Multi-head: why run attention multiple times?

One attention head = one "type of relationship" the model can capture.

With 4 heads (d_k=32 each from d_model=128):
- Head 1 might learn: "attend to the subject of the verb"
- Head 2 might learn: "attend to numerically similar tokens"  
- Head 3 might learn: "attend to tokens in similar positions"
- Head 4 might learn: "attend to the previous token"

Each head sees a different projection of Q, K, V — they specialize.
The outputs are concatenated: [head1, head2, head3, head4] → d_model.

## The causal mask

For **autoregressive** generation, token at position t must not see t+1, t+2, ...

We enforce this by setting attention scores for future positions to -∞:
```
softmax([-∞, -∞, 3.2, 1.4, 0.8])  # positions 0,1 are future (masked)
→ [0.0, 0.0, 0.59, 0.27, 0.14]     # -∞ → 0 after softmax
```

The mask matrix looks like:
```
t=0: [T, F, F, F, F]   token 0 only sees itself
t=1: [T, T, F, F, F]   token 1 sees tokens 0 and 1
t=2: [T, T, T, F, F]   token 2 sees tokens 0, 1, 2
...
```

## Attention complexity

Attention is **O(T²)** in sequence length — the attention matrix is T×T.
This is the main bottleneck for long sequences.

For T=256 (our model): 256² = 65,536 values per head — fine.
For T=100,000 (Claude): 10¹⁰ values — not fine. Requires sparse/linear attention.

## Exercises

1. Run `visualize_attention.py` after training. Which tokens does "pace" attend to most?
   What about "marathon"? What about numbers like "3:45"?

2. In `scaled_dot_product.py`, temporarily remove the `/ math.sqrt(d_k)` scaling.
   Train for 100 steps and compare the loss. What happens?

3. Change `num_heads` from 4 to 1 (single-head attention). Does quality drop?
   Why or why not for this small model?

4. The attention weight matrix is [B, h, T, T]. At inference time for T=100,
   how many floats are stored per layer? How does this scale with context length?
