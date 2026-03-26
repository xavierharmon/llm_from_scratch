# Phase 3: Embeddings — Learning Notes

## What you're building
A lookup table that maps token IDs (integers) to dense floating-point vectors.
After training, semantically similar tokens will have geometrically similar vectors.

## The core idea

Token ID 2096 ("marathon") starts as a random vector:
```
[0.023, -0.412, 0.891, -0.034, ..., 0.227]   # 128 dimensions
```

After training on running logs, it becomes something like:
```
[-0.821, 0.334, -0.102, 0.651, ..., -0.445]   # learned
```

And crucially:
```
distance("marathon", "race")    < distance("marathon", "shoe")
distance("easy", "recovery")    < distance("easy", "tempo")
distance("5K", "10K")           < distance("5K", "marathon")
```

The model learns these relationships purely from co-occurrence statistics.
Words that appear in similar contexts get similar vectors.

## Data engineering analogy

`nn.Embedding` is literally a lookup table — identical to a SQL dimension table:

```sql
-- This is what nn.Embedding is doing:
SELECT feature_vector
FROM token_embeddings
WHERE token_id = 2096;
```

The difference: the `feature_vector` values are not fixed — they're learned
parameters updated by gradient descent during training.

## Why d_model = 128 for our model?

Each token gets a 128-dimensional vector. Think of it as 128 features:
- Some dimensions might encode "this is a pace value"
- Some might encode "this is a race-related term"  
- Some might encode "this is an effort level"

The model learns which dimensions encode what. We don't assign them manually.

GPT-3 uses d_model=12288 (12,288 dimensions per token).

## Positional encoding — why we need it

Self-attention is **permutation-invariant**:
```
"ran a strong marathon" → same attention scores as "marathon a strong ran"
```

Without positional information, the model can't distinguish word order.
We solve this by adding a positional vector to each token's embedding.

**Sinusoidal encoding** (original paper):
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each position gets a unique pattern of sines and cosines across dimensions.
The model can learn to decode relative position from the differences.

**Learned encoding** (GPT-2 style):
Just another embedding table, indexed by position (0, 1, 2, ..., T-1).
Simpler and often performs slightly better in practice.
We use this approach in RunningGPT.

## The sqrt(d_model) scaling

```python
return self.embedding(token_ids) * math.sqrt(self.d_model)
```

Why? Random initialization gives embeddings with variance ~1/d_model.
The positional encoding has values in [-1, 1] (from sine/cosine).
The scaling makes them the same order of magnitude.
Without it, positional encoding dominates at the start of training.

## Exercises

1. After training, run `visualize_embeddings.py`. Do running terms cluster?
   Does "easy" cluster near "recovery" and "jog"? Does "5K" cluster near "10K"?

2. Find the embedding for "pace" and "speed" after training.
   Compute their cosine similarity. What about "pace" and "shoes"?

3. The "king - man + woman = queen" analogy is famous in word2vec.
   Try: embedding("marathon") - embedding("race") + embedding("sprint") = ?
   Does the nearest neighbor make sense?

4. Change d_model from 128 to 32. How does this affect model quality?
   What about 256? This is a scaling experiment.
