# Phase 10: Scaling Laws

## The core question

Given a fixed compute budget (measured in FLOPs), what is the optimal combination of:
- Model size (number of parameters N)
- Training data size (number of tokens D)

## Kaplan et al. (2020) — OpenAI Scaling Laws

Key finding: loss scales as a power law with compute, model size, and data size independently:

```
L(N) ≈ (N_c / N)^(α_N)     α_N ≈ 0.076
L(D) ≈ (D_c / D)^(α_D)     α_D ≈ 0.095
```

Implication: **larger models are more compute-efficient**. Given a fixed budget,
train a very large model on somewhat less data.

GPT-3 (175B parameters) followed this advice — very large model, ~300B tokens.

## Chinchilla (2022) — DeepMind corrects the record

Hoffmann et al. re-ran the scaling experiments more carefully and found:

> **For compute-optimal training: train with ~20 tokens per parameter.**

```
Optimal N* ≈ (C / 6 / 20)^0.5
Optimal D* ≈ 20 × N*
```

Where C is the total FLOPs budget.

**Chinchilla result**: A 70B model trained on 1.4T tokens (20 tokens/param)
outperforms GPT-3 (175B, ~1.7 tokens/param) at the same compute cost.

GPT-3 was massively **undertrained** — it had too many parameters for its token count.

## What this means for our project

```
Our model: ~2M params
Chinchilla-optimal tokens: 20 × 2M = 40M tokens
Our corpus: ~400K tokens (10k runs × 40 tokens/run)

We are training at 1/100th the Chinchilla-optimal token count.
```

Options to improve:
1. Generate more synthetic data (`--n 100000` in download_data.py)
2. Scrape real running blogs, race reports, r/running posts
3. Reduce model size to match our data budget (d_model=32, 2 layers)

## FLOPs calculator

```python
from 10_scaling.scaling_laws import estimate_flops, chinchilla_optimal

# How many FLOPs is our training run?
flops = estimate_flops(num_params=2_000_000, num_tokens=400_000)
print(f"~{flops:.2e} FLOPs")   # ~4.8e12

# What's Chinchilla-optimal for that budget?
result = chinchilla_optimal(flops)
print(result)  # much smaller model, same token count
```

## Scaling beyond this project

| Model | Params | Tokens | FLOPs | Compute time (A100) |
|---|---|---|---|---|
| Our model | 2M | 400K | ~5e12 | seconds |
| GPT-2 small | 117M | 40B | ~2.8e19 | hours |
| LLaMA-7B | 7B | 1T | ~4.2e22 | weeks |
| GPT-4 (est.) | ~1T | ~13T | ~2.4e25 | years |

## The bitter lesson (Rich Sutton, 2019)

> "The biggest lesson that can be read from 70 years of AI research is that general
> methods that leverage computation are ultimately the most effective."

Scaling (more compute, more data, more parameters) has consistently beaten
hand-crafted inductive biases. This is why the transformer architecture — which
has very few hard-coded assumptions — has dominated since 2017.

## Exercises

1. Run `python 10_scaling/scaling_laws.py` to see the full compute table.
2. What is the Chinchilla-optimal model size for our 400K token corpus?
   Train that model and compare perplexity to our larger baseline.
3. Double the training data to 800K tokens. Does perplexity improve by the
   expected amount according to Kaplan's scaling law?
4. Plot loss vs compute for 3 different model sizes trained to the same step count.
   Do you see the predicted power-law relationship?
