# Phase 7: Inference — Learning Notes

## What you're building
The autoregressive generation loop that uses a trained model to produce new text
one token at a time.

## The generation loop

```
Input: "My marathon pace was"
Tokenize: [318, 2096, 4521, 8, 12]

Loop:
  Step 1: model([318, 2096, 4521, 8, 12]) → logits → sample → 374 ("8")
  Step 2: model([318, 2096, 4521, 8, 12, 374]) → logits → sample → 31 (":")
  Step 3: model([...374, 31]) → logits → sample → 1892 ("30")
  ...until EOS or max_new_tokens
```

Each step: one full forward pass through the entire model.

## Decoding strategies

### Greedy (deterministic)
```python
next_token = logits.argmax()
```
Always picks the highest-probability token.
Problem: often repetitive ("the pace was good. The pace was good. The pace...")

### Temperature sampling
```python
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)
```
`T < 1.0`: sharper distribution → more conservative, predictable
`T = 1.0`: unchanged distribution
`T > 1.0`: flatter distribution → more creative, sometimes incoherent

### Top-K sampling
```python
# Zero out all logits except the top K
logits[logits < kth_value] = -inf
next_token = sample(softmax(logits))
```
K=1: greedy. K=50: good balance. K=vocab_size: pure sampling.

### Nucleus (Top-P) sampling
```python
# Sort by probability, keep smallest set summing to >= p
# Zero out the rest
```
Adaptive: on confident predictions (peaked distribution), K is small.
On uncertain predictions (flat distribution), K is large.
p=0.9 is the most common choice. Generally better than fixed Top-K.

### Beam search
Maintains B candidate sequences simultaneously, expanding each by all vocab tokens,
keeping only the top B by cumulative log-probability.

B=1: greedy. B=4: standard beam search. B=10+: diminishing returns.

Pro: higher average quality (finds higher-probability sequences)
Con: 3–5× slower than greedy, often less diverse

## Comparison for running log generation

| Strategy | Speed | Diversity | Coherence | Best for |
|---|---|---|---|---|
| Greedy | Fastest | None | High | Evaluation/benchmarking |
| Top-K (k=50) | Fast | Medium | Good | General use |
| Nucleus (p=0.9) | Fast | High | Good | Creative generation |
| Beam (b=4) | Slow | Low | Highest | Structured outputs |

**Recommendation**: Use nucleus sampling (p=0.9, T=0.8) for interactive demo.
Use greedy for reproducible evaluation.

## The KV cache (not implemented here, but important to know)

In production inference, the key and value matrices for all previous tokens
are cached. This means each step only requires computing Q, K, V for the new token,
then appending to the cache. This gives ~T× speedup for long sequences.

Our implementation recomputes from scratch each step (educational clarity > speed).
See `torch.nn.MultiheadAttention` with `need_weights=False` for a starting point.

## Exercises

1. Generate 50 completions for "marathon pace" with T=0.5, 1.0, 1.5.
   Compute the average token length per completion. How does temperature affect length?

2. Implement a simple repetition penalty:
   `logits[previously_seen_tokens] *= penalty_factor`
   Does it reduce repetitive output?

3. Measure tokens/second for greedy vs nucleus vs beam (b=4).
   How does generation speed scale with beam width?

4. For beam search, add a `length_penalty` parameter. Set it to 0.5 (favors shorter)
   vs 1.5 (favors longer). How do outputs change?
