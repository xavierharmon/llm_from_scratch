# Phase 1: Data — Learning Notes

## What you're building
A corpus of running log text that a language model will learn to predict.
The corpus is the "database" that contains all of the world knowledge
the model will ever have access to.

## Key concept: The LM training objective
A language model learns a single deceptively simple task:

> **Given the previous N tokens, predict the next token.**

That's it. No labels. No task-specific supervision. Just next-token prediction.
This is called **self-supervised learning** — the labels are generated
automatically from the data itself (the next token is always the label).

## Data engineering mapping

| Your world | LLM world |
|---|---|
| Raw source tables (CSV) | Raw text corpus |
| ETL / transformation | Text cleaning + serialization |
| Fact table | Tokenized corpus (flat sequence of IDs) |
| Train/val/test split | Same concept, applied to token sequences |
| Feature engineering | Tokenization |
| Batch size | Number of sequences per gradient step |
| Epoch | One pass through the full corpus |

## What makes good LLM training data?

1. **Diversity** — expose the model to many writing styles, phrasings, domains
2. **Quality** — clean text > noisy text. The model memorizes its training data.
3. **Scale** — more tokens = better model (up to compute budget limits)
4. **Domain relevance** — for our small model, running-specific text lets it learn
   the vocabulary and patterns of running logs with limited compute

## The running corpus

Our corpus has two components:

**Structured → natural language (activities.csv → sentences)**
```
"Easy run on 2021-03-14: 6.2 miles at 9:30/mi, avg HR 142 bpm..."
```
This teaches the model: pace formats, distance vocabulary, HR ranges, dates.

**Free-text logs (run_notes.txt)**
```
"Long run today. 18 miles at the lake path. Legs felt heavy but
 pushed through miles 14-16. Finished strong. Good confidence
 builder for Boston prep."
```
This teaches narrative patterns, runner vocabulary, training concepts.

## Exercises

1. Run `python 01_data/download_data.py --n 1000` and inspect `raw/run_notes.txt`.
   Can you spot patterns the model will learn?

2. Open `01_data/preprocessing.py` and modify `serialize_activity()` to
   add a new field (e.g., calories). How does this change the corpus?

3. Check the corpus stats: total characters, average document length,
   unique words. What vocabulary size does this suggest for the tokenizer?

## Common questions

**Q: Why serialize structured data to text instead of using the CSV directly?**
A: The transformer operates on token sequences. Structured tables would need
   a different architecture (like a table encoder). By serializing to text,
   we can use the same architecture for all data types — which is how
   real foundation models are trained on web + code + tables + math.

**Q: How much data do we actually need?**
A: Chinchilla's rule: ~20 tokens per parameter. For our 2M param model,
   that's 40M tokens. Our 10k runs × ~40 tokens/run = ~400K tokens —
   enough to learn the domain vocabulary but the model will be small.
   See `10_scaling/` for a detailed treatment.
