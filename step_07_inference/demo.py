"""
Interactive Running Log Completion Demo
========================================
Load a trained model and generate running log completions
from user-supplied prompts.

Usage:
    python 07_inference/demo.py --checkpoint experiments/baseline/best.pt
    python 07_inference/demo.py --checkpoint experiments/baseline/best.pt \
        --prompt "My long run today was 18 miles at" \
        --strategy nucleus --temperature 0.8
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from step_02_tokenization.bpe_tokenizer import BPETokenizer
from step_05_transformer.gpt_model import RunningGPT
from step_07_inference.generate import generate, greedy_generate


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    """Load model and tokenizer from a checkpoint."""
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
        dropout=0.0,   # disable dropout at inference time
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def complete(
    prompt: str,
    model,
    tokenizer,
    strategy: str = "nucleus",
    temperature: float = 0.8,
    max_new_tokens: int = 80,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate a completion for a running log prompt."""
    ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(ids).unsqueeze(0).to(device)

    if strategy == "greedy":
        out_ids = greedy_generate(model, token_ids, max_new_tokens=max_new_tokens,
                                  eos_id=tokenizer.eos_id)
    elif strategy == "topk":
        out_ids = generate(model, token_ids, max_new_tokens=max_new_tokens,
                           temperature=temperature, top_k=50, eos_id=tokenizer.eos_id)
    else:  # nucleus
        out_ids = generate(model, token_ids, max_new_tokens=max_new_tokens,
                           temperature=temperature, top_p=0.9, eos_id=tokenizer.eos_id)

    # Decode only the new tokens (exclude the prompt)
    new_ids = out_ids[0, len(ids):].tolist()
    return tokenizer.decode(new_ids)


def interactive_repl(model, tokenizer, device, strategy, temperature):
    """Run an interactive prompt → completion loop."""
    print("\n" + "═" * 60)
    print("  RunningGPT — Running Log Completion")
    print("  Type a partial running log entry. Press Enter to complete.")
    print("  Type 'quit' to exit.")
    print("═" * 60 + "\n")

    while True:
        try:
            prompt = input("Prompt > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not prompt or prompt.lower() in ("quit", "exit"):
            break

        completion = complete(prompt, model, tokenizer, strategy, temperature,
                              device=device)
        print(f"\nCompletion:\n  {prompt}{completion}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (skips interactive mode)")
    parser.add_argument("--strategy", choices=["greedy", "topk", "nucleus"],
                        default="nucleus")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    if args.prompt:
        result = complete(args.prompt, model, tokenizer,
                          strategy=args.strategy,
                          temperature=args.temperature,
                          max_new_tokens=args.max_new_tokens,
                          device=device)
        print(f"\n{args.prompt}{result}")
    else:
        interactive_repl(model, tokenizer, device, args.strategy, args.temperature)
