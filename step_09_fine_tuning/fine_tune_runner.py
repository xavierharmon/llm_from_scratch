"""
LoRA Fine-Tuning Runner
========================
Fine-tune a pretrained RunningGPT on a custom dataset using LoRA.

Example use cases:
    - Adapt a general running model to trail running specifically
    - Fine-tune on your own personal run logs
    - Specialize for race reports vs training logs

Fine-tuning workflow:
    1. Load a pretrained checkpoint (from Phase 6 training)
    2. Apply LoRA adapters to Q, V attention projections
    3. Freeze all base model weights — only train LoRA params
    4. Train on the new dataset with a lower learning rate
    5. Optionally merge LoRA weights back into the base model

Usage:
    python 09_fine_tuning/fine_tune_runner.py \
        --base-checkpoint experiments/baseline/best.pt \
        --data 01_data/raw/my_personal_logs.txt \
        --lora-rank 8 \
        --output experiments/finetuned
"""

import argparse
import sys
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from lora import apply_lora
from step_02_tokenization.bpe_tokenizer import BPETokenizer
from step_01_data.data_loader import RunningDataset, build_dataloaders
from step_05_transformer.gpt_model import RunningGPT
from step_06_training.loss import cross_entropy_loss
from step_06_training.optimizer import CosineWarmupScheduler


class LoRAFineTuner:
    """
    Fine-tunes a pretrained RunningGPT with LoRA adapters.

    Args:
        base_checkpoint: path to pretrained model checkpoint
        lora_rank:       LoRA rank r (4, 8, or 16 are common)
        lora_alpha:      LoRA scaling alpha (typically == rank)
        learning_rate:   LR for LoRA params (10x lower than pretraining)
        output_dir:      where to save fine-tuned checkpoints
    """

    def __init__(
        self,
        base_checkpoint: Path,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        learning_rate: float = 2e-4,
        output_dir: Path = Path("experiments/finetuned"),
    ):
        self.base_checkpoint = Path(base_checkpoint)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_base_model()

    def _load_base_model(self) -> None:
        """Load the pretrained model and apply LoRA."""
        ckpt = torch.load(self.base_checkpoint, map_location=self.device)
        self.config = ckpt["config"]

        self.tokenizer = BPETokenizer.load(Path(self.config["tokenizer_path"]))

        self.model = RunningGPT(
            vocab_size=self.config["vocab_size"],
            d_model=self.config["d_model"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            d_ff=self.config["d_ff"],
            max_seq_len=self.config["max_seq_len"],
            dropout=self.config.get("dropout", 0.1),
        )
        self.model.load_state_dict(ckpt["model_state"])

        # Apply LoRA — freezes base weights, adds trainable adapters
        apply_lora(self.model, rank=self.lora_rank, alpha=self.lora_alpha)
        self.model.to(self.device)

        # Only optimize LoRA parameters
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(lora_params, lr=self.learning_rate, weight_decay=0.01)
        print(f"\nLoaded base model from {self.base_checkpoint}")
        print(f"LoRA rank={self.lora_rank}, alpha={self.lora_alpha}")

    def fine_tune(
        self,
        train_text: list[str],
        max_steps: int = 1000,
        batch_size: int = 8,
        warmup_steps: int = 50,
    ) -> dict:
        """
        Run the LoRA fine-tuning loop.

        Args:
            train_text:   list of training documents (strings)
            max_steps:    number of gradient steps
            batch_size:   training batch size
            warmup_steps: LR warmup steps

        Returns:
            dict of final train_loss, val_loss
        """
        # Tokenize fine-tuning corpus
        all_ids = []
        for doc in train_text:
            all_ids.extend(self.tokenizer.encode(doc, add_special=True))

        dataset = RunningDataset(all_ids, context_len=self.config["max_seq_len"])
        train_loader, val_loader = build_dataloaders(dataset, batch_size=batch_size)

        scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps=warmup_steps, max_steps=max_steps
        )

        data_iter = iter(train_loader)
        self.model.train()
        self.optimizer.zero_grad()

        train_losses = []
        t0 = time.time()

        print(f"\nFine-tuning for {max_steps} steps...")
        for step in range(1, max_steps + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            loss = cross_entropy_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()
            scheduler.step()
            self.optimizer.zero_grad()
            train_losses.append(loss.item())

            if step % 100 == 0:
                avg = sum(train_losses[-100:]) / 100
                elapsed = time.time() - t0
                print(f"  step {step:5d} | loss={avg:.4f} | ppl={math.exp(avg):.1f} | {elapsed:.1f}s")
                t0 = time.time()

        # Validation
        self.model.eval()
        val_total = sum(
            cross_entropy_loss(self.model(xv.to(self.device))[0], yv.to(self.device)).item()
            for xv, yv in val_loader
        )
        val_loss = val_total / len(val_loader)

        # Save checkpoint
        out_path = self.output_dir / "lora_finetuned.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }, out_path)
        print(f"\nSaved fine-tuned checkpoint to {out_path}")
        print(f"Val loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.1f}")

        return {"train_loss": train_losses[-1], "val_loss": val_loss}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None,
                        help="Text file with fine-tuning documents (one per line)")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("experiments/finetuned"))
    args = parser.parse_args()

    # Load fine-tuning data or use synthetic
    if args.data and args.data.exists():
        with open(args.data) as f:
            train_text = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(train_text):,} fine-tuning documents from {args.data}")
    else:
        from 01_data.generate_synthetic import SyntheticRunGenerator
        print("No data file provided — using synthetic trail running data")
        gen = SyntheticRunGenerator(seed=777)
        train_text = gen.generate_run_notes(500)

    tuner = LoRAFineTuner(
        base_checkpoint=args.base_checkpoint,
        lora_rank=args.lora_rank,
        output_dir=args.output,
    )
    tuner.fine_tune(train_text, max_steps=args.max_steps)
