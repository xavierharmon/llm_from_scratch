"""
Visualize Attention Weights
============================
Generates heatmaps showing which tokens attend to which.
Run after training to inspect what the model has learned.

Usage:
    python 04_attention/visualize_attention.py \
        --checkpoint experiments/baseline/best.pt \
        --prompt "My marathon pace was 8:30 per mile and I felt strong"
"""

import argparse
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_attention_heatmap(
    weights: torch.Tensor,
    tokens: list[str],
    layer: int = 0,
    output_path: Path = Path("docs/attention_heatmap.png"),
) -> None:
    """
    Plot attention weights for all heads in a given layer.

    Args:
        weights:     [num_layers, num_heads, T, T] attention weight tensor
        tokens:      list of decoded token strings
        layer:       which transformer layer to visualize
        output_path: where to save the figure
    """
    layer_weights = weights[layer]          # [num_heads, T, T]
    num_heads = layer_weights.shape[0]
    T = len(tokens)

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for h in range(num_heads):
        ax = axes[h]
        data = layer_weights[h].detach().cpu().numpy()[:T, :T]
        im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f"Head {h + 1}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for h in range(num_heads, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(f"Attention weights — Layer {layer + 1}", fontsize=13)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("step_02_tokenization/tokenizer.json"))
    parser.add_argument("--prompt", type=str,
                        default="My marathon pace was 8:30 per mile and I felt strong")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("docs/attention_heatmap.png"))
    args = parser.parse_args()

    from step_02_tokenization.bpe_tokenizer import BPETokenizer
    from step_05_transformer.gpt_model import RunningGPT

    tokenizer = BPETokenizer.load(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = RunningGPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ids = tokenizer.encode(args.prompt)
    x = torch.tensor(ids).unsqueeze(0)
    tokens = [tokenizer.vocab[i].replace("Ġ", "") for i in ids]

    with torch.no_grad():
        _, all_weights = model(x, return_all_weights=True)

    plot_attention_heatmap(all_weights, tokens, layer=args.layer, output_path=args.output)
