"""
Pre-tokenize audio files with Mimi codec.

This script encodes all audio files to Mimi tokens and saves them,
making training much faster by avoiding on-the-fly encoding.

For TTS mode (--tts), it also extracts text and speaker info from transcripts.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Dict, List
import time

import torch
from tqdm import tqdm

from models import MimiEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize audio with Mimi")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="test_dataset",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tokens (default: data-dir/mimi_tokens)",
    )
    parser.add_argument(
        "--max-audio-length",
        type=float,
        default=30.0,
        help="Maximum audio length in seconds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto/cuda/mps/cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for encoding (use 1 for memory efficiency)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing token files",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="TTS mode: extract text and speaker info from transcripts",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def find_audio_files(data_dir: Path) -> List[Path]:
    """Find all audio files in directory."""
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(data_dir.rglob(f"*{ext}"))
    return sorted(audio_files)


def get_token_path(audio_path: Path, data_dir: Path, output_dir: Path) -> Path:
    """Get the corresponding token file path for an audio file."""
    relative_path = audio_path.relative_to(data_dir)
    token_path = output_dir / relative_path.with_suffix(".pt")
    return token_path


def parse_transcript_file(transcript_path: Path) -> List[Dict]:
    """
    Parse transcript file to extract speaker and text pairs.
    
    Expected format:
        [1]: Text from speaker 1
        [2]: Text from speaker 2
    
    Returns list of {"speaker_id": int, "text": str, "line_num": int}
    """
    samples = []
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse [speaker_id]: text format
                match = re.match(r"\[(\d+)\]:\s*(.+)", line)
                if match:
                    speaker_id = int(match.group(1))
                    text = match.group(2).strip()
                    
                    samples.append({
                        "speaker_id": speaker_id,
                        "text": text,
                        "line_num": line_num,
                    })
    except Exception as e:
        print(f"Warning: Could not parse {transcript_path}: {e}")
    
    return samples


def find_tts_samples(data_dir: Path) -> List[Dict]:
    """
    Find TTS samples by matching audio segments with transcript lines.
    
    Expects structure:
        conversation_folder/
            segments/
                001_speaker1.wav  <- matches line 1 of transcript
                002_speaker2.wav  <- matches line 2 of transcript
                vibevoice-podcast-script.txt
    """
    samples = []
    
    # Find all segment folders
    for segment_dir in data_dir.rglob("**/segments"):
        # Find transcript
        transcript_path = segment_dir / "vibevoice-podcast-script.txt"
        if not transcript_path.exists():
            continue
        
        # Parse transcript
        transcript_samples = parse_transcript_file(transcript_path)
        if not transcript_samples:
            continue
        
        # Find audio files and match with transcript
        audio_files = sorted(segment_dir.glob("*.wav"))
        
        for audio_path in audio_files:
            # Extract segment number from filename (e.g., "001_speaker1.wav" -> 1)
            match = re.match(r"(\d+)_speaker\d+\.wav", audio_path.name)
            if match:
                segment_num = int(match.group(1))
                
                # Find matching transcript line
                if segment_num <= len(transcript_samples):
                    transcript = transcript_samples[segment_num - 1]
                    
                    samples.append({
                        "audio_path": audio_path,
                        "text": transcript["text"],
                        "speaker_id": transcript["speaker_id"],
                        "segment_num": segment_num,
                        "conversation_folder": segment_dir.parent,
                    })
    
    return samples


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "mimi_tokens"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Initialize Mimi encoder
    print("\n📦 Loading Mimi codec...")
    mimi = MimiEncoder(device=device)
    mimi.load()
    
    # Find samples
    if args.tts:
        print(f"\n🔍 Scanning {data_dir} for TTS samples (audio + transcript pairs)...")
        tts_samples = find_tts_samples(data_dir)
        print(f"   Found {len(tts_samples)} TTS samples")
        
        if not tts_samples:
            print("❌ No TTS samples found!")
            print("   Expected structure:")
            print("     conversation_folder/")
            print("       segments/")
            print("         001_speaker1.wav")
            print("         002_speaker2.wav")
            print("         vibevoice-podcast-script.txt")
            return
        
        # Track unique speakers
        speaker_ids = set(s["speaker_id"] for s in tts_samples)
        print(f"   Speakers found: {sorted(speaker_ids)}")
    else:
        print(f"\n🔍 Scanning {data_dir} for audio files...")
        audio_files = find_audio_files(data_dir)
        print(f"   Found {len(audio_files)} audio files")
        tts_samples = None
    
    # Stats
    stats = {
        "total": len(tts_samples) if tts_samples else len(audio_files),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "total_tokens": 0,
        "total_duration_sec": 0,
    }
    
    print(f"\n🎵 Pre-tokenizing audio files...")
    start_time = time.time()
    
    if args.tts:
        # TTS mode: process with text/speaker info
        for sample in tqdm(tts_samples, desc="Tokenizing TTS"):
            audio_path = sample["audio_path"]
            token_path = get_token_path(audio_path, data_dir, output_dir)
            
            # Skip if exists
            if token_path.exists() and not args.overwrite:
                stats["skipped"] += 1
                continue
            
            try:
                # Encode audio
                tokens = mimi.encode(audio_path)
                tokens = tokens.squeeze(0).cpu()  # [num_codebooks, T]
                
                # Create directories
                token_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save with TTS info
                torch.save({
                    "tokens": tokens,
                    "text": sample["text"],
                    "speaker_id": sample["speaker_id"],
                    "source": str(audio_path),
                    "num_codebooks": tokens.size(0),
                    "num_frames": tokens.size(1),
                    "segment_num": sample["segment_num"],
                }, token_path)
                
                stats["processed"] += 1
                stats["total_tokens"] += tokens.numel()
                stats["total_duration_sec"] += tokens.size(1) / 12.5
                
            except Exception as e:
                print(f"\n❌ Error processing {audio_path}: {e}")
                stats["errors"] += 1
    else:
        # Regular mode: just audio tokens
        audio_files = find_audio_files(data_dir)
        
        for audio_path in tqdm(audio_files, desc="Tokenizing"):
            token_path = get_token_path(audio_path, data_dir, output_dir)
            
            if token_path.exists() and not args.overwrite:
                stats["skipped"] += 1
                continue
            
            try:
                tokens = mimi.encode(audio_path)
                tokens = tokens.squeeze(0).cpu()
                
                token_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    "tokens": tokens,
                    "source": str(audio_path),
                    "num_codebooks": tokens.size(0),
                    "num_frames": tokens.size(1),
                }, token_path)
                
                stats["processed"] += 1
                stats["total_tokens"] += tokens.numel()
                stats["total_duration_sec"] += tokens.size(1) / 12.5
                
            except Exception as e:
                print(f"\n❌ Error processing {audio_path}: {e}")
                stats["errors"] += 1
    
    elapsed = time.time() - start_time
    
    # Save metadata
    metadata = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "num_codebooks": mimi.num_codebooks,
        "codebook_size": mimi.codebook_size,
        "frame_rate": 12.5,
        "sample_rate": 24000,
        "mode": "tts" if args.tts else "audio",
        "num_speakers": len(speaker_ids) if args.tts else 0,
        "stats": stats,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n✅ Pre-tokenization complete!")
    print(f"   Mode: {'TTS (text + speaker + audio)' if args.tts else 'Audio only'}")
    print(f"   Output directory: {output_dir}")
    print(f"   Processed: {stats['processed']}")
    print(f"   Skipped (existing): {stats['skipped']}")
    print(f"   Errors: {stats['errors']}")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Total audio duration: {stats['total_duration_sec']:.1f}s ({stats['total_duration_sec']/60:.1f}min)")
    print(f"   Time elapsed: {elapsed:.1f}s")
    print(f"\n📝 Metadata saved to: {metadata_path}")
    
    if args.tts:
        print(f"\n💡 To train TTS, run:")
        print(f"   python train_tts.py --data-dir {data_dir}")
    else:
        print(f"\n💡 To train with pre-tokenized data, run:")
        print(f"   python train.py --data-dir {data_dir} --use-pretokenized")


if __name__ == "__main__":
    main()
