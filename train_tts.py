"""
TTS Training Script for TRM.

Trains the TTS TRM model: text + speaker_id → ALL 32 mimi_tokens codebooks

Multi-codebook training with weighted loss:
- Codebook 0 (semantic): highest weight - captures linguistic content
- Codebooks 1-7: moderate weight - early acoustic refinement
- Codebooks 8-31: equal weight - fine acoustic details
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models import TTSTRM, TTSTRMConfig
from models.mimi_wrapper import MimiEncoder
from dataset import TTSDataModule

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train TTS TRM model")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="test_dataset", help="Data directory")
    parser.add_argument("--token-dir", type=str, default=None, help="Pre-tokenized data directory")
    parser.add_argument("--max-text-length", type=int, default=512, help="Max text length")
    parser.add_argument("--max-audio-frames", type=int, default=256, help="Max audio frames")
    
    # Model
    parser.add_argument("--text-embed-dim", type=int, default=256, help="Text embedding dim")
    parser.add_argument("--audio-embed-dim", type=int, default=256, help="Audio embedding dim")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--text-layers", type=int, default=4, help="Text encoder layers")
    parser.add_argument("--L-layers", type=int, default=2, help="Recursive block layers")
    parser.add_argument("--H-cycles", type=int, default=3, help="Improvement cycles")
    parser.add_argument("--L-cycles", type=int, default=6, help="Recursive cycles")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--share-output-heads", action="store_true", 
                        help="Share output projection across codebooks (fewer params)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--fp16", action="store_true", help="Mixed precision")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Eval every N steps")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--output-dir", type=str, default="outputs/tts", help="Output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--wandb", action="store_true", help="Use W&B")
    
    # Sample generation during training
    parser.add_argument("--sample-interval", type=int, default=0, 
                        help="Generate samples every N steps (0 = disabled)")
    parser.add_argument("--num-samples", type=int, default=3, 
                        help="Number of samples to generate (if using dataset prompts)")
    parser.add_argument("--sample-prompts", type=str, nargs="+", default=None,
                        help="Custom text prompts for sample generation (e.g., 'Hello world' 'Testing one two three')")
    parser.add_argument("--sample-prompts-file", type=str, default=None,
                        help="Path to file with custom prompts (one per line, format: 'speaker_id|text')")
    parser.add_argument("--sample-speaker", type=int, default=0,
                        help="Speaker ID for custom prompts")
    parser.add_argument("--sample-temperature", type=float, default=0.8, 
                        help="Temperature for sample generation")
    parser.add_argument("--sample-top-k", type=int, default=50, 
                        help="Top-k for sample generation")
    
    # System
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


class TTSTrainer:
    """Trainer for TTS TRM model."""
    
    def __init__(
        self,
        model: TTSTRM,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        args,
        device: torch.device,
        val_dataset=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.device = device
        
        self.scaler = GradScaler() if args.fp16 else None
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample generation
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        self.mimi = None  # Lazy load
        self.sample_prompts = None  # Will be set up on first sample generation
        
        # W&B logging
        self.wandb_run = None
        if args.wandb:
            import wandb
            run_name = args.run_name or f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.wandb_run = wandb.init(
                project="trm-tts",
                name=run_name,
                config=vars(args),
            )
    
    def train_epoch(self, epoch: int) -> float:
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
                    wandb.log({
                        "train/loss": loss,
                        "train/avg_loss": avg_loss,
                        "train/lr": lr,
                        "train/step": self.global_step,
                    })
            
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
            
            # Sample generation
            if self.args.sample_interval > 0 and self.global_step % self.args.sample_interval == 0:
                self.generate_samples()
                self.model.train()
        
        return total_loss / num_batches
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        self.optimizer.zero_grad()
        
        # Move to device
        text_ids = batch["text_ids"].to(self.device)
        speaker_id = batch["speaker_id"].to(self.device)
        audio_tokens = batch["audio_tokens"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)
        
        # Forward
        if self.args.fp16:
            with autocast():
                logits, loss = self.model(
                    text_ids=text_ids,
                    speaker_id=speaker_id,
                    target_audio_tokens=audio_tokens,
                    text_mask=text_mask,
                )
        else:
            logits, loss = self.model(
                text_ids=text_ids,
                speaker_id=speaker_id,
                target_audio_tokens=audio_tokens,
                text_mask=text_mask,
            )
        
        # Backward
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
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Track per-codebook metrics
        num_codebooks = self.model.config.num_codebooks
        codebook_correct = torch.zeros(num_codebooks, device=self.device)
        codebook_total = torch.zeros(num_codebooks, device=self.device)
        
        # Track length prediction metrics
        length_errors = []
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            text_ids = batch["text_ids"].to(self.device)
            speaker_id = batch["speaker_id"].to(self.device)
            audio_tokens = batch["audio_tokens"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            actual_frames = batch["num_audio_frames"].to(self.device)
            
            logits, loss = self.model(
                text_ids=text_ids,
                speaker_id=speaker_id,
                target_audio_tokens=audio_tokens,
                text_mask=text_mask,
            )
            # logits: [B, num_codebooks, T, V]
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute per-codebook accuracy
            preds = logits.argmax(dim=-1)  # [B, num_codebooks, T]
            for c in range(num_codebooks):
                pred_c = preds[:, c, :]  # [B, T]
                target_c = audio_tokens[:, c, :]  # [B, T]
                
                # Ignore padding (token 0)
                mask = target_c != 0
                codebook_correct[c] += ((pred_c == target_c) & mask).sum()
                codebook_total[c] += mask.sum()
            
            # Compute length prediction error
            predicted_frames = self.model.predict_length(text_ids, speaker_id, text_mask)
            length_error = (predicted_frames - actual_frames.float()).abs()
            length_errors.append(length_error)
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute accuracies
        codebook_acc = codebook_correct / codebook_total.clamp(min=1)
        semantic_acc = codebook_acc[0].item()
        acoustic_acc = codebook_acc[1:].mean().item()
        overall_acc = codebook_acc.mean().item()
        
        # Compute length metrics
        all_length_errors = torch.cat(length_errors)
        mae_frames = all_length_errors.mean().item()
        
        print(f"\nValidation Loss: {avg_loss:.4f}")
        print(f"  Semantic (CB0) Accuracy: {semantic_acc:.2%}")
        print(f"  Acoustic (CB1-31) Accuracy: {acoustic_acc:.2%}")
        print(f"  Overall Accuracy: {overall_acc:.2%}")
        print(f"  Length MAE: {mae_frames:.1f} frames")
        
        if self.wandb_run:
            import wandb
            log_dict = {
                "val/loss": avg_loss,
                "val/semantic_acc": semantic_acc,
                "val/acoustic_acc": acoustic_acc,
                "val/overall_acc": overall_acc,
                "val/length_mae": mae_frames,
                "val/step": self.global_step,
            }
            # Log per-codebook accuracies
            for c in range(min(8, num_codebooks)):  # Log first 8 codebooks
                log_dict[f"val/codebook_{c}_acc"] = codebook_acc[c].item()
            wandb.log(log_dict)
        
        return avg_loss
    
    def _load_mimi(self):
        """Lazy load Mimi decoder for sample generation."""
        if self.mimi is None:
            print("\n🔊 Loading Mimi decoder for sample generation...")
            self.mimi = MimiEncoder(device=str(self.device))
            self.mimi.load()
        return self.mimi
    
    def _setup_sample_prompts(self):
        """Set up fixed prompts for consistent sample generation."""
        if self.sample_prompts is not None:
            return
        
        self.sample_prompts = []
        
        # Option 1: Custom prompts from CLI
        if self.args.sample_prompts:
            print(f"📝 Using custom prompts from CLI:")
            for text in self.args.sample_prompts:
                self._add_custom_prompt(text, self.args.sample_speaker)
        
        # Option 2: Custom prompts from file
        elif self.args.sample_prompts_file:
            prompts_file = Path(self.args.sample_prompts_file)
            if prompts_file.exists():
                print(f"📝 Loading prompts from: {prompts_file}")
                with open(prompts_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # Format: "speaker_id|text" or just "text"
                        if "|" in line:
                            speaker_str, text = line.split("|", 1)
                            speaker_id = int(speaker_str.strip())
                        else:
                            speaker_id = self.args.sample_speaker
                            text = line
                        self._add_custom_prompt(text.strip(), speaker_id)
            else:
                print(f"⚠️  Prompts file not found: {prompts_file}")
        
        # Option 3: Fall back to dataset samples
        else:
            if self.val_dataset is None:
                print("⚠️  No validation dataset available for sample generation")
                return
            
            print(f"📝 Using prompts from validation dataset:")
            num_samples = min(self.args.num_samples, len(self.val_dataset))
            
            for i in range(num_samples):
                sample = self.val_dataset[i]
                self.sample_prompts.append({
                    "text": sample["text"],
                    "text_ids": sample["text_ids"],
                    "speaker_id": sample["speaker_id"],
                    "num_frames": sample["num_audio_frames"],
                    "gt_tokens": sample["audio_tokens"],  # Ground truth for comparison
                    "has_gt": True,
                })
        
        # Print prompts
        for i, p in enumerate(self.sample_prompts):
            text_preview = f"\"{p['text'][:50]}...\"" if len(p['text']) > 50 else f"\"{p['text']}\""
            gt_marker = " (has GT)" if p.get("has_gt") else ""
            print(f"   {i+1}. [speaker {p['speaker_id']}] {text_preview}{gt_marker}")
    
    def _add_custom_prompt(self, text: str, speaker_id: int):
        """Add a custom text prompt."""
        if self.val_dataset is None:
            print(f"⚠️  Cannot encode text without dataset vocabulary")
            return
        
        # Encode text using the dataset's vocabulary
        text_ids = self.val_dataset.encode_text(text)
        
        # Estimate number of frames (~12.5 frames per second, ~3 chars per frame)
        num_frames = min(len(text) * 3, self.model.config.max_audio_frames)
        
        self.sample_prompts.append({
            "text": text,
            "text_ids": text_ids,
            "speaker_id": speaker_id,
            "num_frames": num_frames,
            "gt_tokens": None,
            "has_gt": False,
        })
    
    @torch.no_grad()
    def generate_samples(self):
        """Generate and save audio samples to monitor training progress."""
        if self.args.sample_interval <= 0:
            return
        
        if not HAS_SOUNDFILE:
            print("⚠️  soundfile not installed, skipping sample generation")
            return
        
        # Setup
        self._setup_sample_prompts()
        if not self.sample_prompts:
            return
        
        mimi = self._load_mimi()
        self.model.eval()
        
        step_dir = self.samples_dir / f"step_{self.global_step:06d}"
        step_dir.mkdir(exist_ok=True)
        
        print(f"\n🎵 Generating {len(self.sample_prompts)} samples at step {self.global_step}...")
        
        generated_paths = []
        
        for i, prompt in enumerate(self.sample_prompts):
            # Prepare input
            text_ids = prompt["text_ids"].unsqueeze(0).to(self.device)
            speaker_id = torch.tensor([prompt["speaker_id"]], device=self.device)
            target_frames = prompt["num_frames"]
            
            # Create text mask
            text_mask = torch.ones_like(text_ids, dtype=torch.float)
            
            # Get predicted length
            predicted_frames = self.model.predict_length(text_ids, speaker_id, text_mask)
            pred_frames_int = int(predicted_frames.item())
            pred_frames_int = max(1, min(pred_frames_int, self.model.config.max_audio_frames))
            
            # Use predicted length for generation (not target)
            tokens = self.model.generate(
                text_ids=text_ids,
                speaker_id=speaker_id,
                num_frames=None,  # Use predicted length
                text_mask=text_mask,
                temperature=self.args.sample_temperature,
                top_k=self.args.sample_top_k,
            )
            # tokens: [1, num_codebooks, T]
            
            # Decode to audio
            try:
                audio = mimi.decode(tokens)  # [1, 1, T_samples]
                audio = audio.squeeze().cpu().numpy()
                
                # Save generated audio
                gen_path = step_dir / f"sample_{i+1}_generated.wav"
                sf.write(str(gen_path), audio, mimi.sample_rate)
                generated_paths.append(gen_path)
                
                # Also save ground truth on first generation for comparison (if available)
                if self.global_step == self.args.sample_interval and prompt.get("has_gt") and prompt["gt_tokens"] is not None:
                    gt_tokens = prompt["gt_tokens"].unsqueeze(0).to(self.device)
                    gt_audio = mimi.decode(gt_tokens)
                    gt_audio = gt_audio.squeeze().cpu().numpy()
                    gt_path = self.samples_dir / f"sample_{i+1}_ground_truth.wav"
                    sf.write(str(gt_path), gt_audio, mimi.sample_rate)
                
                text_preview = f"\"{prompt['text'][:30]}...\"" if len(prompt['text']) > 30 else f"\"{prompt['text']}\""
                length_info = f"({pred_frames_int} frames"
                if prompt.get("has_gt"):
                    length_info += f", target: {target_frames})"
                else:
                    length_info += ")"
                print(f"   ✓ Sample {i+1}: {text_preview} {length_info} → {gen_path.name}")
                
            except Exception as e:
                print(f"   ✗ Sample {i+1} failed: {e}")
        
        # Save sample info with predicted lengths
        sample_info = []
        for p in self.sample_prompts:
            text_ids = p["text_ids"].unsqueeze(0).to(self.device)
            speaker_id = torch.tensor([p["speaker_id"]], device=self.device)
            text_mask = torch.ones_like(text_ids, dtype=torch.float)
            pred_len = int(self.model.predict_length(text_ids, speaker_id, text_mask).item())
            sample_info.append({
                "text": p["text"],
                "speaker_id": p["speaker_id"],
                "predicted_frames": pred_len,
                "target_frames": p["num_frames"] if p.get("has_gt") else None,
            })
        
        info = {
            "step": self.global_step,
            "temperature": self.args.sample_temperature,
            "top_k": self.args.sample_top_k,
            "prompts": sample_info,
        }
        with open(step_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        # Log to W&B
        if self.wandb_run:
            try:
                import wandb
                audio_logs = {}
                
                # Log generated samples
                for i, path in enumerate(generated_paths):
                    prompt = self.sample_prompts[i]
                    caption = f"[spk {prompt['speaker_id']}] {prompt['text'][:80]}"
                    audio_logs[f"samples/generated_{i+1}"] = wandb.Audio(
                        str(path),
                        sample_rate=mimi.sample_rate,
                        caption=caption,
                    )
                
                # Log ground truth on first generation (for comparison)
                if self.global_step == self.args.sample_interval:
                    for i, prompt in enumerate(self.sample_prompts):
                        if prompt.get("has_gt"):
                            gt_path = self.samples_dir / f"sample_{i+1}_ground_truth.wav"
                            if gt_path.exists():
                                caption = f"[GT] [spk {prompt['speaker_id']}] {prompt['text'][:80]}"
                                audio_logs[f"samples/ground_truth_{i+1}"] = wandb.Audio(
                                    str(gt_path),
                                    sample_rate=mimi.sample_rate,
                                    caption=caption,
                                )
                
                # Log as a table for better organization
                if generated_paths:
                    columns = ["step", "sample_id", "speaker", "text", "pred_frames", "audio"]
                    data = []
                    for i, path in enumerate(generated_paths):
                        prompt = self.sample_prompts[i]
                        text_ids = prompt["text_ids"].unsqueeze(0).to(self.device)
                        speaker_id_t = torch.tensor([prompt["speaker_id"]], device=self.device)
                        text_mask = torch.ones_like(text_ids, dtype=torch.float)
                        pred_frames = int(self.model.predict_length(text_ids, speaker_id_t, text_mask).item())
                        
                        data.append([
                            self.global_step,
                            i + 1,
                            prompt["speaker_id"],
                            prompt["text"][:100],
                            pred_frames,
                            wandb.Audio(str(path), sample_rate=mimi.sample_rate),
                        ])
                    
                    table = wandb.Table(columns=columns, data=data)
                    audio_logs["samples/table"] = table
                
                wandb.log(audio_logs, step=self.global_step)
                print(f"   📤 Uploaded {len(generated_paths)} samples to W&B")
                
            except Exception as e:
                print(f"   ⚠️  W&B audio logging failed: {e}")
        
        print(f"   Samples saved to: {step_dir}")
        self.model.train()
    
    def save_checkpoint(self, name: str):
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


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # System settings
    num_workers = 0 if device.type == "mps" else args.num_workers
    pin_memory = device.type == "cuda"
    
    # Setup data
    print("\n📁 Setting up TTS data...")
    token_dir = Path(args.token_dir) if args.token_dir else Path(args.data_dir) / "mimi_tokens"
    
    if not token_dir.exists():
        print(f"❌ Token directory not found: {token_dir}")
        print(f"   Run: python pretokenize.py --data-dir {args.data_dir} --tts")
        return
    
    data_module = TTSDataModule(
        data_dir=args.data_dir,
        token_dir=token_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
        max_text_length=args.max_text_length,
        max_audio_frames=args.max_audio_frames,
        pin_memory=pin_memory,
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Get vocabulary info
    text_vocab_size = data_module.text_vocab_size
    num_speakers = data_module.num_speakers
    num_codebooks = data_module.num_codebooks
    codebook_size = data_module.codebook_size
    
    print(f"   Text vocab size: {text_vocab_size}")
    print(f"   Speakers: {num_speakers}")
    print(f"   Codebooks: {num_codebooks}")
    print(f"   Codebook size: {codebook_size}")
    
    # Create model
    print("\n📦 Creating TTS TRM model (full multi-codebook)...")
    config = TTSTRMConfig(
        text_vocab_size=text_vocab_size,
        text_embed_dim=args.text_embed_dim,
        text_hidden_dim=args.hidden_dim,
        text_num_layers=args.text_layers,
        text_num_heads=args.num_heads,
        num_speakers=num_speakers,
        audio_embed_dim=args.audio_embed_dim,
        audio_hidden_dim=args.hidden_dim,
        audio_vocab_size=codebook_size,
        num_codebooks=num_codebooks,
        L_layers=args.L_layers,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        num_heads=args.num_heads,
        max_text_len=args.max_text_length,
        max_audio_frames=args.max_audio_frames,
        share_output_heads=args.share_output_heads,
    )
    
    model = TTSTRM(config).to(device)
    
    num_params = model.count_parameters()
    print(f"   Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"   Output heads: {'shared' if args.share_output_heads else 'separate per codebook'}")
    
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
    trainer = TTSTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        device=device,
        val_dataset=data_module.val_dataset,
    )
    
    # Print codebook weights
    print("\n📊 Codebook loss weights:")
    weights = model.get_codebook_weights()
    print(f"   Semantic (CB0): {weights[0]:.2f}")
    print(f"   Early acoustic (CB1-7): {weights[1:8].mean():.2f} avg")
    print(f"   Late acoustic (CB8-31): {weights[8:].mean():.2f} avg")
    
    # Train
    print(f"\n🚀 Starting TTS training for {args.epochs} epochs...")
    print(f"   Total steps: {total_steps}")
    print(f"   Input: text + speaker_id → Output: ALL {num_codebooks} mimi codebooks")
    if args.sample_interval > 0:
        print(f"   Sample generation: every {args.sample_interval} steps ({args.num_samples} samples)")
    else:
        print(f"   Sample generation: disabled")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
        
        val_loss = trainer.evaluate()
        
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_checkpoint("best")
    
    trainer.save_checkpoint("final")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()

