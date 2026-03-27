"""
Visualize learned token embeddings using t-SNE.

Run after training to see if semantically similar tokens cluster together.
Expects a saved model checkpoint.

Usage:
    python 03_embeddings/visualize_embeddings.py --checkpoint experiments/baseline/best.pt
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_embedding_clusters(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "Token Embedding Space (t-SNE)",
    output_path: Path = Path("docs/embedding_tsne.png"),
) -> None:
    """Reduce embeddings to 2D with t-SNE and plot."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("Install scikit-learn: pip install scikit-learn")
        return

    print("Running t-SNE (this may take a minute)...")
    reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.4, s=8, c="steelblue")

    # Label the most common tokens
    for i, label in enumerate(labels[:200]):
        if len(label) > 1 and not label.startswith("<|"):
            ax.annotate(label.replace("Ġ", ""), (coords[i, 0], coords[i, 1]),
                        fontsize=7, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("step_02_tokenization/tokenizer.json"))
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("docs/embedding_tsne.png"))
    args = parser.parse_args()

    from step_02_tokenization.bpe_tokenizer import BPETokenizer  # noqa
    tokenizer = BPETokenizer.load(args.tokenizer)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # Extract embedding weights (first layer of the model)
    emb_weights = checkpoint["model_state"]["token_embedding.embedding.weight"]
    emb_np = emb_weights.numpy()

    n = min(args.max_tokens, len(tokenizer.vocab))
    labels = [tokenizer.vocab[i] for i in range(n)]

    plot_embedding_clusters(emb_np[:n], labels, output_path=args.output)
