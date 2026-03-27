# Phase 9: Fine-Tuning with LoRA

## What is fine-tuning?

Pretraining = teach the model general language patterns from a huge corpus.
Fine-tuning = adapt that pretrained model to a specific task or domain with a small dataset.

The challenge: fine-tuning all parameters on a small dataset leads to **catastrophic forgetting** —
the model forgets what it learned during pretraining.

LoRA solves this by keeping base weights frozen and only training a tiny set of adapter parameters.

## LoRA math

For a weight matrix `W` of shape `(d_out, d_in)`, LoRA adds:

```
W_effective = W + ΔW
ΔW = B · A    where B is (d_out × r), A is (r × d_in), rank r << d_in
```

The forward pass becomes:
```python
output = x @ W.T + (x @ A.T @ B.T) * (alpha / rank)
```

Only `A` and `B` are trained. `W` stays frozen.

## Parameter savings example

| Config | Base params | LoRA params (r=8) | Savings |
|---|---|---|---|
| Our model (d=128) | 1.85M | ~8K | 230× |
| GPT-2 small (d=768) | 117M | ~300K | 390× |
| LLaMA-7B (d=4096) | 7B | ~4M | 1750× |

## Practical guidance

**Rank selection:**
- `r=4`:  very few params, good for style transfer
- `r=8`:  standard choice for domain adaptation
- `r=16`: higher capacity, useful for task-specific fine-tuning
- `r=64`: approaching full fine-tuning capacity

**Which layers to apply LoRA to:**
- Standard: Q and V projections in attention
- More capacity: add K and output projections too
- Maximum: add FFN layers as well

**Learning rate:**
- Use ~10× lower than pretraining LR
- For our model: pretraining LR=3e-4, fine-tuning LR=2e-4 to 3e-5

**When to stop:**
- Watch validation loss — stop when it starts rising (overfitting)
- With small fine-tuning sets (<1000 docs), 200-500 steps is often enough

## Merging LoRA back into base weights

For zero-overhead inference, merge `ΔW` back into `W`:

```python
# In LoRALinear.merge_weights():
W_merged = W + (B @ A) * (alpha / rank)
```

The resulting model is identical in architecture to the base model
but with updated weights — no LoRA overhead at inference time.

## Running fine-tuning

```bash
# Fine-tune on your own run logs
python 09_fine_tuning/fine_tune_runner.py \
    --base-checkpoint experiments/baseline/best.pt \
    --data my_run_logs.txt \
    --lora-rank 8 \
    --max-steps 500

# Generate from fine-tuned model
python 07_inference/demo.py \
    --checkpoint experiments/finetuned/lora_finetuned.pt \
    --prompt "My 50K trail race went"
```

## Exercises

1. Fine-tune on only trail running logs. Does the model start generating more trail-specific vocabulary?
2. Compare perplexity before and after fine-tuning on a held-out trail running test set.
3. Try `lora_rank=4` vs `lora_rank=16`. How does generation quality differ?
4. After fine-tuning, does the model "forget" easy road running patterns? Test both.
