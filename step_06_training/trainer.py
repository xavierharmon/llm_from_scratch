"""
Training Loop
=============
Orchestrates model training: forward pass, loss, backward, optimizer step.

Key concepts demonstrated here:
    - Gradient clipping: prevents exploding gradients in early training
    - Mixed precision (AMP): use float16 for forward pass, float32 for gradients
    - Gradient accumulation: simulate larger batches without more memory
    - Checkpoint saving: save model state for resuming / inference
    - Validation loop: estimate held-out loss to detect overfitting

Data engineering analogy:
    The training loop is your ETL job scheduler.
    Each step = one micro-batch SELECT + transform.
    Gradient accumulation = buffered writes before COMMIT.
    Checkpointing = CHECKPOINT in your pipeline.
    Validation = data quality assertion on a hold-out partition.
"""

import time
import math
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .loss import cross_entropy_loss, perplexity
from .optimizer import build_optimizer, CosineWarmupScheduler


class Trainer:
    """
    Training orchestrator for RunningGPT.

    Args:
        model:            the RunningGPT model
        train_loader:     DataLoader for training data
        val_loader:       DataLoader for validation data
        config:           dict of training hyperparameters
        checkpoint_dir:   directory to save model checkpoints
        device:           torch device ('cpu', 'cuda', 'mps')
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        checkpoint_dir: Path = Path("experiments/baseline"),
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model = self.model.to(device)
        print(f"Training on: {device}")

        # Optimizer + scheduler
        self.optimizer = build_optimizer(
            model, lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.1)
        )
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"],
        )

        # Mixed precision scaler (no-op on CPU/MPS)
        self.use_amp = (device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Training state
        self.step = 0
        self.best_val_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        One forward + backward + optimizer step.

        Returns:
            float loss value for logging
        """
        x, y = x.to(self.device), y.to(self.device)
        self.model.train()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logits, _ = self.model(x)
            loss = cross_entropy_loss(logits, y)

        # Scale loss for gradient accumulation
        accum_steps = self.config.get("grad_accumulation_steps", 1)
        (self.scaler.scale(loss / accum_steps)).backward()

        return loss.item()

    def _optimizer_step(self) -> None:
        """Unscale, clip gradients, step optimizer and scheduler."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.get("grad_clip", 1.0)
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute average loss over the entire validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits, _ = self.model(x)
                loss = cross_entropy_loss(logits, y)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest") -> Path:
        path = self.checkpoint_dir / f"{tag}.pt"
        torch.save({
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from step {self.step}, best val loss {self.best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """
        Full training loop.

        Returns:
            dict with final train_loss, val_loss, perplexity, total_steps
        """
        max_steps = self.config["max_steps"]
        log_interval = self.config.get("log_interval", 50)
        val_interval = self.config.get("val_interval", 200)
        accum_steps = self.config.get("grad_accumulation_steps", 1)

        print(f"\nTraining for {max_steps:,} steps...")
        print(f"  Batch size:            {self.config['batch_size']}")
        print(f"  Gradient accumulation: {accum_steps}")
        print(f"  Effective batch size:  {self.config['batch_size'] * accum_steps}")
        print(f"  Peak LR:               {self.config['learning_rate']}")
        print()

        data_iter = iter(self.train_loader)
        running_loss = 0.0
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        while self.step < max_steps:
            # Refill iterator if exhausted
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)

            loss = self._train_step(x, y)
            running_loss += loss

            # Optimizer step every accum_steps micro-batches
            if (self.step + 1) % accum_steps == 0:
                self._optimizer_step()

            self.step += 1

            # Logging
            if self.step % log_interval == 0:
                avg_loss = running_loss / log_interval
                pp = math.exp(avg_loss)
                lr_now = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  step {self.step:5d}/{max_steps} | "
                      f"loss {avg_loss:.4f} | ppl {pp:.1f} | "
                      f"lr {lr_now:.2e} | {elapsed:.1f}s")
                self.train_losses.append(avg_loss)
                running_loss = 0.0
                t0 = time.time()

            # Validation + checkpointing
            if self.step % val_interval == 0:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                print(f"  ── Val loss: {val_loss:.4f} | ppl {math.exp(val_loss):.1f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                    print(f"  ★  New best! Saved checkpoint.")

                self.save_checkpoint("latest")

        final_val = self.evaluate()
        self.save_checkpoint("final")
        print(f"\nTraining complete.")
        print(f"  Best val loss:  {self.best_val_loss:.4f} (ppl {math.exp(self.best_val_loss):.1f})")
        print(f"  Final val loss: {final_val:.4f} (ppl {math.exp(final_val):.1f})")

        return {
            "train_loss": self.train_losses[-1] if self.train_losses else None,
            "val_loss": final_val,
            "perplexity": math.exp(final_val),
            "total_steps": self.step,
        }
