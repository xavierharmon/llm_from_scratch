"""
Download public running datasets for LLM training.

Sources:
    - Synthetic generator (always available, no API key needed)
    - Kaggle running dataset (requires kaggle CLI configured)

Usage:
    python 01_data/download_data.py --source synthetic --n 10000
    python 01_data/download_data.py --source kaggle
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_synthetic import SyntheticRunGenerator


def download_synthetic(n: int = 10_000, output_dir: Path = Path("01_data/raw")) -> None:
    """Generate a synthetic running corpus — no external dependencies needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gen = SyntheticRunGenerator(seed=42)

    print(f"Generating {n:,} synthetic runs...")
    df = gen.generate_activities(n)
    df.to_csv(output_dir / "running_activities.csv", index=False)
    print(f"  Saved running_activities.csv ({len(df):,} rows)")

    notes = gen.generate_run_notes(n)
    with open(output_dir / "run_notes.txt", "w") as f:
        f.write("\n\n".join(notes))
    print(f"  Saved run_notes.txt ({len(notes):,} entries)")

    splits_df = gen.generate_splits(min(n, 2000))
    splits_df.to_csv(output_dir / "splits.csv", index=False)
    print(f"  Saved splits.csv ({len(splits_df):,} rows)")

    print("\nDone. Run 'python step_01_data/preprocessing.py' next.")


def download_kaggle(output_dir: Path = Path("01_data/raw")) -> None:
    """Download from Kaggle — requires: pip install kaggle && kaggle API key configured."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Install kaggle: pip install kaggle")
        print("Then set up your API key: https://www.kaggle.com/docs/api")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "olegoaer/running-races",
        "-p", str(output_dir), "--unzip"
    ], check=True)
    print(f"Downloaded to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download running data")
    parser.add_argument("--source", choices=["synthetic", "kaggle"], default="synthetic")
    parser.add_argument("--n", type=int, default=10_000, help="Rows for synthetic source")
    parser.add_argument("--output-dir", type=Path, default=Path("01_data/raw"))
    args = parser.parse_args()

    if args.source == "synthetic":
        download_synthetic(args.n, args.output_dir)
    else:
        download_kaggle(args.output_dir)
