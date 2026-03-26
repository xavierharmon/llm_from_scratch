"""
Benchmark Suite
================
Evaluate a trained model across multiple metrics and report a full card.

Metrics computed:
    - Perplexity         (lower = better)
    - Token accuracy     (higher = better)
    - Generation quality (sample + inspect)
    - Throughput         (tokens/second at inference)

Usage:
    python 08_evaluation/benchmark.py --checkpoint experiments/baseline/best.pt
"""

import time
import math
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from perplexity import compute_perplexity, compute_token_accuracy


def measure_throughput(
    model: torch.nn.Module,
    seq_len: int = 256,
    batch_size: int = 1,
    n_steps: int = 20,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Measure tokens generated per second at inference time."""
    model.eval()
    dummy = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            _ = model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0

    total_tokens = batch_size * seq_len * n_steps
    tps = total_tokens / elapsed
    return {
        "tokens_per_second": round(tps),
        "ms_per_batch": round(elapsed / n_steps * 1000, 2),
        "seq_len": seq_len,
        "batch_size": batch_size,
    }


def count_parameters(model: torch.nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_M": round(total / 1e6, 3),
    }


def run_benchmark(checkpoint_path: Path) -> dict:
    from 02_tokenization.bpe_tokenizer import BPETokenizer
    from 01_data.generate_synthetic import SyntheticRunGenerator
    from 01_data.data_loader import RunningDataset, build_dataloaders
    from 05_transformer.gpt_model import RunningGPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    tokenizer = BPETokenizer.load(Path(config["tokenizer_path"]))

    model = RunningGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # Build a small eval dataset from fresh synthetic data
    gen = SyntheticRunGenerator(seed=999)
    test_corpus = gen.generate_run_notes(500)
    all_ids = []
    for doc in test_corpus:
        all_ids.extend(tokenizer.encode(doc, add_special=True))

    dataset = RunningDataset(all_ids, context_len=config["max_seq_len"])
    _, val_loader = build_dataloaders(dataset, batch_size=16)

    results = {}

    print("Running benchmark...")
    print(f"  Model: {checkpoint_path}")
    print(f"  Device: {device}")
    print()

    # Parameter count
    params = count_parameters(model)
    results.update(params)
    print(f"Parameters:       {params['total_params_M']:.3f}M")

    # Perplexity
    ppl = compute_perplexity(model, val_loader, device=device)
    results["perplexity"] = round(ppl, 2)
    print(f"Perplexity:       {ppl:.2f}")

    # Token accuracy
    acc = compute_token_accuracy(model, val_loader, device=device)
    results["token_accuracy"] = round(acc, 4)
    print(f"Token accuracy:   {acc*100:.2f}%")

    # Throughput
    tput = measure_throughput(model, seq_len=config["max_seq_len"], device=device)
    results.update(tput)
    print(f"Throughput:       {tput['tokens_per_second']:,} tokens/sec")
    print(f"Latency:          {tput['ms_per_batch']} ms/batch")

    print()
    print("Benchmark complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    run_benchmark(args.checkpoint)
