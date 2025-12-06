#!/usr/bin/env python3
"""
Debug script to figure out the correct token format for the Emilia dataset.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "pandas", "-q"])
    import pyarrow.parquet as pq


def load_mimi():
    from transformers import MimiModel
    print("Loading Mimi model...")
    model = MimiModel.from_pretrained("kyutai/mimi")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, device


def try_decode(model, device, tokens, name, output_dir):
    """Try to decode tokens and save."""
    print(f"\n  Trying: {name}")
    print(f"    Shape: {tokens.shape}")
    print(f"    Range: [{tokens.min()}, {tokens.max()}]")
    
    try:
        tokens_tensor = torch.from_numpy(tokens).unsqueeze(0).long().to(device)
        with torch.no_grad():
            outputs = model.decode(tokens_tensor)
            audio = outputs.audio_values.squeeze().cpu().numpy()
        
        output_path = output_dir / f"{name}.wav"
        sf.write(str(output_path), audio, 24000)
        print(f"    Saved: {output_path}")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    parquet_path = Path("train-00000-of-00031.parquet")
    if not parquet_path.exists():
        parquet_path = Path("emilia-en-mimi-small/data/train-00000-of-00031.parquet")
    
    output_dir = Path("debug_audio")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.slice(0, 1).to_pandas()
    
    row = df.iloc[0]
    codes = np.array(row["codes"])
    text = row["text"]
    duration = row.get("duration", 0)
    
    print(f"\nSample info:")
    print(f"  Text: {text[:80]}...")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Codes length: {len(codes)}")
    print(f"  Codes range: [{codes.min()}, {codes.max()}]")
    
    model, device = load_mimi()
    
    # Try different interpretations
    print("\n" + "="*60)
    print("Trying different token interpretations...")
    print("="*60)
    
    # Interpretation 1: Interleaved [c0_t0, c1_t0, ..., c7_t0, c0_t1, ...]
    # Reshape to [T, 8] then transpose to [8, T]
    num_frames = len(codes) // 8
    tokens_v1 = codes[:num_frames*8].reshape(num_frames, 8).T  # [8, T]
    try_decode(model, device, tokens_v1, "v1_interleaved_8xT", output_dir)
    
    # Interpretation 2: Sequential [c0_t0, c0_t1, ..., c0_tN, c1_t0, ...]
    # Reshape to [8, T] directly
    tokens_v2 = codes[:num_frames*8].reshape(8, num_frames)  # [8, T]
    try_decode(model, device, tokens_v2, "v2_sequential_8xT", output_dir)
    
    # Interpretation 3: Maybe it's not 8 codebooks?
    # Try with different numbers
    for num_cb in [1, 4, 16, 32]:
        if len(codes) % num_cb == 0:
            nf = len(codes) // num_cb
            # Interleaved
            tokens = codes.reshape(nf, num_cb).T
            try_decode(model, device, tokens, f"v3_interleaved_{num_cb}xT", output_dir)
    
    # Interpretation 4: First 8 codebooks from 32 interleaved
    # Maybe it's actually 32 codebooks but only first 8 stored?
    # Try treating as if there are hidden codebooks
    
    # Interpretation 5: Raw 1D (single codebook)
    tokens_v5 = codes.reshape(1, -1)  # [1, T]
    try_decode(model, device, tokens_v5, "v5_single_codebook", output_dir)
    
    print(f"\n✅ Check {output_dir}/ and listen to which one sounds correct!")


if __name__ == "__main__":
    main()


