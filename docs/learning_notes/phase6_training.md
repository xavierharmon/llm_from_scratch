# Phase 6: Training — Learning Notes

## What you're building
The gradient descent loop that adjusts all ~2M model parameters to minimize
next-token prediction loss on the running corpus.

## The training loop — one step

```
1. Sample a batch of (input, target) pairs: x=[t0..t_{n-1}], y=[t1..t_n]
2. Forward pass: model(x) → logits [B, T, vocab_size]
3. Compute loss: cross_entropy(logits, y)  ← scalar
4. Backward pass: loss.backward() → fills .grad on every parameter
5. Clip gradients: prevents exploding gradients in early training
6. Optimizer step: update each parameter using its gradient
7. LR scheduler step: adjust learning rate
8. Zero gradients: reset .grad for the next step
```

## Cross-entropy loss — unpacked

For a single position in the sequence:
```
true next token: "pace"  →  one-hot: [0, 0, ..., 1, ..., 0]  (index 2096)
model prediction: softmax(logits) → probability distribution over vocab

loss = -log(P(true_token))
     = -log(model_output[2096])
```

If the model assigns 90% probability to the correct token: loss = -log(0.9) = 0.105
If the model assigns 1% (random): loss = -log(0.01) = 4.6

**The model learns by being "punished" for low probability on the true token.**

## AdamW — why not plain SGD?

Plain SGD: `param -= lr * grad`
- Same learning rate for every parameter
- Sensitive to gradient scale differences across layers

Adam: `param -= lr * grad_mean / sqrt(grad_variance)`
- Adapts learning rate per-parameter based on gradient history
- Much less sensitive to hyperparameters than SGD
- Standard choice for transformers since 2018

AdamW adds proper weight decay:
- Penalizes large weights (L2 regularization)
- Decoupled from the adaptive learning rate (fixes a bug in original Adam)

## The learning rate schedule

```
Steps 0-200 (warmup):  LR ramps from 0 → 3e-4 linearly
Steps 200-5000 (decay): LR decays from 3e-4 → 3e-5 following cosine curve
```

**Why warmup?**
Early in training, gradients are large and noisy. A high initial LR causes
destructive parameter updates. Warmup lets the gradient estimates stabilize
before using the full LR.

**Why cosine decay instead of step decay?**
Cosine is smooth — no sudden LR drops that can destabilize training.
Empirically outperforms step decay on most LLM benchmarks.

## Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the total gradient norm exceeds 1.0, all gradients are scaled down proportionally.
This prevents "gradient explosions" — rare but catastrophic events where a single
bad batch causes the model to jump to a very bad region of parameter space.

## Mixed precision training (on GPU)

```python
with torch.cuda.amp.autocast():
    logits = model(x)           # computed in float16 (2× faster, 2× less memory)
    loss = cross_entropy(...)   # in float16

scaler.scale(loss).backward()   # gradients scaled to prevent float16 underflow
scaler.step(optimizer)          # unscale before optimizer step
```

Float16 has less precision than float32, which causes small gradients to underflow to zero.
The GradScaler multiplies the loss by a large factor before backward, then divides
by the same factor before the optimizer step — keeping gradients in the float16 range.

## Perplexity milestones

| Steps | Expected Perplexity | What it means |
|---|---|---|
| 0 (random) | ~8000 (vocab size) | Model knows nothing |
| 100 | ~500-1000 | Learning basic token patterns |
| 500 | ~100-200 | Getting domain structure |
| 2000 | ~50-80 | Reasonable running-domain model |
| 5000 | ~30-50 | Good small model |

## Exercises

1. Plot your training loss curve. Does it follow a smooth power-law decline?
   Any spikes? (Spikes = large batches or LR too high.)

2. Try removing gradient clipping (set `grad_clip=100`). Does training become
   unstable? Check `torch.nn.utils.clip_grad_norm_` return value.

3. Change `weight_decay` from 0.1 to 0.0. Does validation loss improve or worsen
   after 5000 steps? This tests whether regularization is helping.

4. Set `warmup_steps=0`. Does training become unstable in the first 50 steps?
   Look at the loss curve closely.
