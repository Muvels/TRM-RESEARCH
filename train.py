"""
Training Script for TRM on Mimi Tokens.

Trains the Tiny Recursive Model to predict audio tokens
using recursive self-improvement.
"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models import TRM, TRMConfig, MimiEncoder
from dataset import AudioDataModule, PreTokenizedDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train TRM on Mimi tokens")

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="test_dataset",
        help="Directory with audio files",
    )
    parser.add_argument(
        "--max-audio-length",
        type=float,
        default=10.0,
        help="Maximum audio length in seconds",
    )
    parser.add_argument(
        "--use-pretokenized",
        action="store_true",
        help="Use pre-tokenized data (much faster, run pretokenize.py first)",
    )
    parser.add_argument(
        "--token-dir",
        type=str,
        default=None,
        help="Directory with pre-tokenized .pt files (default: data-dir/mimi_tokens)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames for pre-tokenized data (default: use all)",
    )

    # Model
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--L-layers", type=int, default=2, help="Recursive block layers")
    parser.add_argument("--H-cycles", type=int, default=3, help="High-level improvement cycles")
    parser.add_argument("--L-cycles", type=int, default=6, help="Low-level recursive cycles")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--use-mlp", action="store_true", help="Use MLP instead of attention")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logging")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")

    # System
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


class Trainer:
    """Training loop for TRM."""

    def __init__(
        self,
        model: TRM,
        mimi_encoder: Optional[MimiEncoder],
        train_dataloader,
        val_dataloader,
        optimizer: AdamW,
        scheduler,
        args,
        device: torch.device,
        use_pretokenized: bool = False,
    ):
        self.model = model
        self.mimi_encoder = mimi_encoder
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.device = device
        self.use_pretokenized = use_pretokenized

        self.scaler = GradScaler() if args.fp16 else None
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Logging
        self.wandb_run = None
        if args.wandb:
            import wandb

            run_name = args.run_name or f"trm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.wandb_run = wandb.init(
                project="trm-mimi",
                name=run_name,
                config=vars(args),
            )

        # Output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            loss = self.train_step(batch)

            total_loss += loss
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.args.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

                if self.wandb_run:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/avg_loss": avg_loss,
                            "train/lr": lr,
                            "train/step": self.global_step,
                        }
                    )

            # Evaluation
            if self.global_step % self.args.eval_interval == 0:
                val_loss = self.evaluate()
                self.model.train()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")

            # Checkpointing
            if self.global_step % self.args.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}")

        return total_loss / num_batches

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step."""
        self.optimizer.zero_grad()

        # Get tokens - either pre-tokenized or encode on-the-fly
        if self.use_pretokenized:
            tokens = batch["tokens"].to(self.device)
        else:
            with torch.no_grad():
                if "tokens" in batch:
                    tokens = batch["tokens"].to(self.device)
                else:
                    waveforms = batch["waveform"].to(self.device)
                    tokens = self.mimi_encoder.encode(waveforms)

        # Forward pass
        if self.args.fp16:
            with autocast():
                # Use tokens as both input and target (reconstruction task)
                input_tokens = tokens
                target_tokens = tokens
                logits, loss = self.model(input_tokens, target_tokens)
        else:
            input_tokens = tokens
            target_tokens = tokens
            logits, loss = self.model(input_tokens, target_tokens)

        # Backward pass
        if self.args.fp16:
            self.scaler.scale(loss).backward()
            if self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            # Get tokens
            if self.use_pretokenized:
                tokens = batch["tokens"].to(self.device)
            else:
                if "tokens" in batch:
                    tokens = batch["tokens"].to(self.device)
                else:
                    waveforms = batch["waveform"].to(self.device)
                    tokens = self.mimi_encoder.encode(waveforms)

            _, loss = self.model(tokens, tokens)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        print(f"\nValidation Loss: {avg_loss:.4f}")

        if self.wandb_run:
            import wandb

            wandb.log(
                {
                    "val/loss": avg_loss,
                    "val/step": self.global_step,
                }
            )

        return avg_loss

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.model.config.__dict__,
            "args": vars(self.args),
        }

        path = checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        print(f"Loaded checkpoint from step {self.global_step}")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Use num_workers=0 on MPS to avoid multiprocessing issues
    num_workers = 0 if device.type == "mps" else args.num_workers
    pin_memory = device.type == "cuda"  # Only pin memory on CUDA

    # Setup data and encoder based on mode
    mimi_encoder = None
    
    if args.use_pretokenized:
        # Use pre-tokenized data (much faster!)
        print("📁 Using pre-tokenized data...")
        
        token_dir = Path(args.token_dir) if args.token_dir else Path(args.data_dir) / "mimi_tokens"
        
        if not token_dir.exists():
            print(f"❌ Token directory not found: {token_dir}")
            print(f"   Run: python pretokenize.py --data-dir {args.data_dir}")
            return
        
        data_module = PreTokenizedDataModule(
            data_dir=args.data_dir,
            token_dir=token_dir,
            batch_size=args.batch_size,
            num_workers=num_workers,
            max_frames=args.max_frames,
            pin_memory=pin_memory,
        )
        data_module.setup()
        
        # Get codec info from metadata
        num_codebooks = data_module.num_codebooks
        codebook_size = data_module.codebook_size
        
        print(f"   Codebooks: {num_codebooks}, Vocab size: {codebook_size}")
    else:
        # On-the-fly encoding with Mimi
        print("🎵 Loading Mimi codec for on-the-fly encoding...")
        mimi_encoder = MimiEncoder(device=str(device))
        mimi_encoder.load()
        
        print("Setting up data...")
        data_module = AudioDataModule(
            data_dir=args.data_dir,
            mimi_encoder=None,  # Don't pass encoder to avoid pickle issues
            batch_size=args.batch_size,
            num_workers=num_workers,
            max_audio_length=args.max_audio_length,
            pin_memory=pin_memory,
        )
        data_module.setup()
        
        num_codebooks = mimi_encoder.num_codebooks
        codebook_size = mimi_encoder.codebook_size

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Create model
    print("Creating TRM model...")
    config = TRMConfig(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=codebook_size,
        num_codebooks=num_codebooks,
        L_layers=args.L_layers,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        num_heads=args.num_heads,
        use_attention=not args.use_mlp,
    )

    model = TRM(config).to(device)

    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # Trainer
    trainer = Trainer(
        model=model,
        mimi_encoder=mimi_encoder,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        device=device,
        use_pretokenized=args.use_pretokenized,
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Total steps: {total_steps}")

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        print(f"\nEpoch {epoch} - Average Train Loss: {train_loss:.4f}")

        # End of epoch evaluation
        val_loss = trainer.evaluate()

        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_checkpoint("best")

    # Final save
    trainer.save_checkpoint("final")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

