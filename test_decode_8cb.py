#!/usr/bin/env python3
"""
Test script to decode 8-codebook samples.

Mimi natively supports decoding with fewer codebooks!
No padding needed - just pass [8, T] tokens directly.

Usage:
    source .venv/bin/activate
    python test_decode_8cb.py
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Try to import pyarrow
try:
    import pyarrow.parquet as pq
except ImportError:
    print("Installing pyarrow...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "pandas", "-q"])
    import pyarrow.parquet as pq


def load_mimi():
    """Load the Mimi model."""
    from transformers import MimiModel
    
    print("Loading Mimi model...")
    model = MimiModel.from_pretrained("kyutai/mimi")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"  Loaded on {next(model.parameters()).device}")
    return model


def decode_8cb(model, tokens_8cb: np.ndarray):
    """
    Decode 8-codebook tokens directly (Mimi supports num_quantizers=8).
    
    Args:
        model: Mimi model
        tokens_8cb: [8, T] array of tokens
    
    Returns:
        audio: numpy array of audio samples
    """
    device = next(model.parameters()).device
    
    # tokens_8cb is [8, T]
    num_cb, T = tokens_8cb.shape
    print(f"  Input tokens: [{num_cb}, {T}]")
    
    # Convert to tensor [1, 8, T]
    tokens_tensor = torch.from_numpy(tokens_8cb).unsqueeze(0).long().to(device)
    
    # Decode - Mimi natively supports 8 codebooks!
    with torch.no_grad():
        outputs = model.decode(tokens_tensor)
        audio = outputs.audio_values  # [1, 1, T_samples]
    
    audio = audio.squeeze().cpu().numpy()
    print(f"  Audio samples: {len(audio)} ({len(audio)/24000:.2f}s)")
    
    return audio


def main():
    # Config
    parquet_path = Path("train-00000-of-00031.parquet")
    output_dir = Path("test_audio_8cb")
    num_samples = 3
    num_codebooks = 8
    sample_rate = 24000
    
    # Check parquet exists
    if not parquet_path.exists():
        # Try in emilia directory
        alt_path = Path("emilia-en-mimi-small/data/train-00000-of-00031.parquet")
        if alt_path.exists():
            parquet_path = alt_path
        else:
            print(f"Error: Parquet file not found at {parquet_path}")
            print("Please specify the correct path.")
            return
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading parquet: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.slice(0, num_samples).to_pandas()
    
    print(f"Found {len(df)} samples")
    
    # Load Mimi
    model = load_mimi()
    
    # Process each sample
    for i in range(num_samples):
        print(f"\n--- Sample {i+1} ---")
        
        row = df.iloc[i]
        codes = np.array(row["codes"])
        text = row["text"]
        duration = row.get("duration", 0)
        
        print(f"  Text: {text[:80]}...")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Codes length: {len(codes)}")
        
        # Reshape sequential codes to [8, T]
        # Format: [c0_t0, c0_t1, ..., c0_tN, c1_t0, c1_t1, ..., c1_tN, ...]
        num_frames = len(codes) // num_codebooks
        tokens = codes.reshape(num_codebooks, num_frames)  # [8, T]
        
        # Decode directly (Mimi supports 8 codebooks natively!)
        audio = decode_8cb(model, tokens)
        
        # Save
        output_path = output_dir / f"sample_{i+1}.wav"
        sf.write(str(output_path), audio, sample_rate)
        print(f"  Saved: {output_path}")
        
        # Also save text
        text_path = output_dir / f"sample_{i+1}.txt"
        with open(text_path, "w") as f:
            f.write(f"Text: {text}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Frames: {num_frames}\n")
    
    print(f"\n✅ Done! Audio files saved to: {output_dir}/")
    print("   Mimi natively supports 8 codebooks - full quality decoding!")


if __name__ == "__main__":
    main()

