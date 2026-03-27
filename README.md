# LLM From Scratch — Running Performance Domain

> Build a GPT-style language model from the ground up, trained on real running data.

This project teaches every layer of a large language model through hands-on implementation —
no black boxes. Each phase builds on the last, from raw CSV files to a working text generator.

## Who This Is For

Data engineers, analysts, and scientists who understand pipelines, transformations,
and feature stores — and want to map those mental models onto how LLMs actually work.

## Mental Model Bridge

| Data Engineering Concept | LLM Equivalent |
|--------------------------|----------------|
| ETL pipeline | Forward pass |
| Schema / data types | Tensor shapes |
| Lookup table / foreign key | Token embedding |
| JOIN between tables | Attention mechanism |
| Feature normalization | Layer normalization |
| Index / inverted index | Vocabulary / token IDs |
| Optimizer / scheduler | Gradient descent + LR schedule |
| Model registry | Checkpoint saving |

## Project Phases

| Phase | Module | What You Build |
|-------|--------|----------------|
| 1 | `01_data/` | Download, clean, and tokenize running corpus |
| 2 | `02_tokenization/` | BPE tokenizer from scratch |
| 3 | `03_embeddings/` | Token + positional embeddings |
| 4 | `04_attention/` | Scaled dot-product + multi-head attention |
| 5 | `05_transformer/` | Full transformer block + GPT model |
| 6 | `06_training/` | Training loop, loss, AdamW, LR schedule |
| 7 | `07_inference/` | Greedy, top-k, nucleus sampling, beam search |
| 8 | `08_evaluation/` | Perplexity, BLEU, benchmark suite |
| 9 | `09_fine_tuning/` | LoRA fine-tuning on custom run logs |
| 10 | `10_scaling/` | Scaling laws, Chinchilla, efficiency |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare running data
python 01_data/download_data.py
python 01_data/preprocessing.py

# 3. Train the BPE tokenizer on your corpus
python 02_tokenization/train_tokenizer.py --vocab-size 8000

# 4. Train the mini GPT model (~8M params, ~2hrs on CPU)
python 06_training/train.py --config configs/small.yaml

# 5. Generate running log completions
python 07_inference/demo.py --prompt "My long run today was 18 miles at"
```

## Starter Model Config (runs on a laptop)

```yaml
vocab_size: 8000
d_model: 128
num_heads: 4
num_layers: 4
d_ff: 512
max_seq_len: 256
dropout: 0.1
batch_size: 32
learning_rate: 3.0e-4
warmup_steps: 200
max_steps: 5000
```

## Recommended Reading Order

1. Work through `notebooks/` in order (01 → 06)
2. Read each `docs/learning_notes/phaseN_*.md` before its module
3. Build each module, run its tests
4. Read the original papers in `docs/papers/` after implementing each concept

## Key Papers

- **Attention Is All You Need** (Vaswani et al., 2017) — the transformer paper
- **Language Models are Unsupervised Multitask Learners** (GPT-2, Radford et al., 2019)
- **Training Compute-Optimal LLMs** (Chinchilla, Hoffmann et al., 2022)

## License

MIT

## Notes
| Step | What | Notes & Learnings |
|-------|--------|----------------|
| 01_data | `Setup` | `Running wiht an old anaconda install caused issues. Deleted and uninstalled anaconda and setup a python instance. Running on venv also helped overcome package issues. Numpy 2.0 caused some headache as well.` |