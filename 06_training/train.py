"""
Training Entry Point
====================
Wire everything together and launch a training run.

Usage:
    python 06_training/train.py
    python 06_training/train.py --config configs/small.yaml
    python 06_training/train.py --resume experiments/baseline/latest.pt
"""

import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from 02_tokenization.bpe_tokenizer import BPETokenizer
from 01_data.data_loader import RunningDataset, build_dataloaders
from 05_transformer.gpt_model import RunningGPT
from trainer import Trainer

# ── Default small config (runs on any laptop) ──────────────────────────────
DEFAULT_CONFIG = {
    "vocab_size":              8000,
    "d_model":                 128,
    "num_heads":               4,
    "num_layers":              4,
    "d_ff":                    512,
    "max_seq_len":             256,
    "dropout":                 0.1,
    "activation":              "gelu",
    "bias":                    False,

    "batch_size":              32,
    "learning_rate":           3e-4,
    "weight_decay":            0.1,
    "grad_clip":               1.0,
    "warmup_steps":            200,
    "max_steps":               5000,
    "val_interval":            250,
    "log_interval":            50,
    "grad_accumulation_steps": 1,

    "tokenizer_path":          "02_tokenization/tokenizer.json",
    "corpus_path":             "01_data/processed/corpus.txt",
    "experiment_dir":          "experiments/baseline",
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        overrides = yaml.safe_load(f)
    config = DEFAULT_CONFIG.copy()
    config.update(overrides or {})
    return config


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else DEFAULT_CONFIG.copy()

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tok_path = Path(config["tokenizer_path"])
    if not tok_path.exists():
        print(f"Tokenizer not found at {tok_path}.")
        print("Run: python 02_tokenization/train_tokenizer.py")
        sys.exit(1)
    tokenizer = BPETokenizer.load(tok_path)
    config["vocab_size"] = len(tokenizer)
    print(f"Loaded tokenizer: {len(tokenizer):,} tokens")

    # ── Dataset ────────────────────────────────────────────────────────────
    corpus_path = Path(config["corpus_path"])
    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}.")
        print("Run: python 01_data/download_data.py && python 01_data/preprocessing.py")
        sys.exit(1)

    dataset = RunningDataset.from_corpus_file(
        corpus_path, tokenizer, context_len=config["max_seq_len"]
    )
    train_loader, val_loader = build_dataloaders(
        dataset, batch_size=config["batch_size"]
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = RunningGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        activation=config["activation"],
        bias=config["bias"],
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=Path(config["experiment_dir"]),
    )

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    results = trainer.train()
    print("\nFinal results:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RunningGPT")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to YAML config file (overrides defaults)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to checkpoint to resume from")
    main(parser.parse_args())
