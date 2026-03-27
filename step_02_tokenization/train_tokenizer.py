"""
Train the BPE tokenizer on the running corpus.

Usage:
    python 02_tokenization/train_tokenizer.py
    python 02_tokenization/train_tokenizer.py --vocab-size 4000 --corpus 01_data/processed/corpus.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from step_02_tokenization.bpe_tokenizer import BPETokenizer


def train(corpus_path: Path, vocab_size: int, output_path: Path) -> BPETokenizer:
    print(f"Loading corpus from {corpus_path} ...")
    with open(corpus_path) as f:
        corpus = [line.strip() for line in f if line.strip()]
    print(f"  {len(corpus):,} documents loaded")

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus, verbose=True)
    tokenizer.save(output_path)

    # Quick round-trip sanity check
    sample = corpus[0][:120]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"\nRound-trip check:")
    print(f"  Original:  {sample[:80]}...")
    print(f"  Tokens:    {encoded[:12]}...")
    print(f"  Decoded:   {decoded[:80]}...")
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("step_01_data/processed/corpus.txt"))
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=Path("step_02_tokenization/tokenizer.json"))
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Corpus not found at {args.corpus}.")
        print("Run: python step_01_data/download_data.py && python step_01_data/preprocessing.py")
        sys.exit(1)

    train(args.corpus, args.vocab_size, args.output)
