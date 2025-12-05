"""
TRM-Research: Tiny Recursive Model for Audio Token Prediction.

Quick demo script showing how to use the TRM with Mimi codec.
"""

import argparse
import torch

from models import TRM, TRMConfig, MimiEncoder


def demo():
    """Run a quick demonstration of TRM with Mimi."""
    parser = argparse.ArgumentParser(description="TRM Demo")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create TRM model
    print("\n📦 Creating TRM model...")
    config = TRMConfig(
        embed_dim=256,
        hidden_dim=512,
        vocab_size=2048,
        num_codebooks=8,
        H_cycles=3,
        L_cycles=6,
        L_layers=2,
    )
    model = TRM(config).to(device)

    num_params = model.count_parameters()
    print(f"   Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"   H_cycles (improvement steps): {config.H_cycles}")
    print(f"   L_cycles (recursive steps): {config.L_cycles}")

    if args.audio:
        # Load and encode audio with Mimi
        print(f"\n🎵 Loading Mimi codec...")
        mimi = MimiEncoder(device=device)
        mimi.load()

        print(f"   Encoding audio: {args.audio}")
        tokens = mimi.encode(args.audio)
        print(f"   Token shape: {tokens.shape}")  # [B, 8, T]

        # Run through TRM
        print("\n🔄 Running TRM recursive refinement...")
        model.eval()
        with torch.no_grad():
            logits, _ = model(tokens)
        print(f"   Output logits shape: {logits.shape}")

        # Get predictions
        predictions = logits.argmax(dim=-1)
        print(f"   Predicted tokens shape: {predictions.shape}")

        print("\n✅ Demo complete!")
    else:
        # Demo with random input
        print("\n🎲 Running with random input (no audio provided)...")
        batch_size, num_codebooks, seq_len = 2, 8, 100
        dummy_tokens = torch.randint(0, 2048, (batch_size, num_codebooks, seq_len)).to(device)

        model.eval()
        with torch.no_grad():
            logits, _ = model(dummy_tokens)

        print(f"   Input shape: {dummy_tokens.shape}")
        print(f"   Output shape: {logits.shape}")
        print("\n✅ Model works! Run with --audio <path> to test with real audio.")


if __name__ == "__main__":
    demo()
