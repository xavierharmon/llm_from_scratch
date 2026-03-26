# Phase 2: Tokenization — Learning Notes

## What you're building
A compression algorithm that converts raw text into compact integer sequences.

## Why not just use characters or words?

**Character-level**: "marathon" → [m, a, r, a, t, h, o, n] = 8 tokens
- Pro: tiny vocabulary (256 chars), handles any text
- Con: very long sequences, model must learn spelling from scratch

**Word-level**: "marathon" → [marathon] = 1 token
- Pro: short sequences, semantically meaningful units
- Con: huge vocabulary (100k+ words), can't handle "marathoner", "marathons"

**BPE (Byte-Pair Encoding)**: "marathon" → [mar, athon] = 2 tokens (example)
- Balances vocabulary size vs sequence length
- Handles morphology: "run", "runner", "running" share a "run" prefix token
- This is what GPT-2, GPT-4, LLaMA all use

## The BPE algorithm — step by step

Starting corpus: `"run runner running"`

**Step 0 — character split:**
```
r u n | r u n n e r | r u n n i n g
```
Vocabulary: {r, u, n, e, i, g, space}

**Step 1 — count pairs, merge most frequent:**
Most frequent pair: (r, u) → appears 3 times
Merge: `[ru] n | [ru] n n e r | [ru] n n i n g`
Add "ru" to vocabulary.

**Step 2:**
Most frequent pair: (ru, n) → 3 times
Merge: `[run] | [run] n e r | [run] n i n g`
Add "run" to vocabulary.

**Continue until vocab_size is reached.**

After training, the merge rules are stored as an ordered list.
To encode new text: apply the same merges in the same order.

## Data engineering analogy

BPE is fundamentally a **compression algorithm**:
- Training = learning a codebook by counting pair frequencies (like LZ77)
- Encoding = applying the codebook to compress new text
- The vocabulary = the codebook
- Token IDs = compressed representation

The merge rules are like a lookup table of derived columns.
Each merge says: "whenever you see pattern X, replace it with shorthand Y."

## Vocabulary size tradeoffs

| Vocab size | Token count for "marathon training today" | Notes |
|---|---|---|
| 256 (char) | ~30 tokens | Too long, slow training |
| 1,000 | ~15 tokens | Better, but OOV on numbers |
| 8,000 | ~8 tokens | Good for domain-specific small model |
| 50,257 | ~5 tokens | GPT-2 vocab size |
| 100,000 | ~4 tokens | GPT-4 estimated vocab size |

## The Ġ prefix convention (GPT-2 style)

"marathon" at the start of a word is encoded as "Ġmarathon" (Ġ = space prefix).
This lets the tokenizer distinguish "marathon" as a standalone word vs
"marathon" appearing mid-word (in "ultramarathon").

## Exercises

1. Train a tokenizer with `vocab_size=100` and inspect `tokenizer.vocab`.
   What tokens got merged first? Why?

2. Encode this sentence with different vocab sizes (200, 500, 2000).
   `"My marathon PR is 3:45, finished in Boston with negative splits"`
   How does token count change?

3. Find a number like "3:45:22" in your corpus. Does BPE handle it correctly?
   What happens to rare number patterns?

4. Read the `encode()` method carefully. What happens when a character
   appears in the test text but not in the training corpus?
