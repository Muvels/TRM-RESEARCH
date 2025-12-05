"""
TTS Dataset for TRM Training.

Pairs text transcripts with speaker IDs and Mimi audio tokens
for Text-to-Speech training.

Input format: "[speaker_id]text of the audio"
Output: Mimi tokens
"""

import json
import random
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class TTSDataset(Dataset):
    """
    TTS Dataset that pairs text+speaker with audio tokens.
    
    Expects pre-tokenized data with text/speaker info.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        token_dir: Optional[Union[str, Path]] = None,
        max_text_length: int = 512,
        max_audio_frames: Optional[int] = None,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize TTS dataset.
        
        Args:
            data_dir: Base data directory
            token_dir: Directory with pre-tokenized .pt files
            max_text_length: Maximum text length in characters
            max_audio_frames: Maximum audio frames (None = use all)
            split: "train" or "val"
            val_ratio: Validation ratio
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.token_dir = Path(token_dir) if token_dir else self.data_dir / "mimi_tokens"
        self.max_text_length = max_text_length
        self.max_audio_frames = max_audio_frames
        self.split = split
        
        # Load metadata
        metadata_path = self.token_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Find all samples with text+audio pairs
        self.samples = self._find_samples()
        
        if not self.samples:
            raise ValueError(
                f"No TTS samples found. Run 'python pretokenize.py --data-dir {self.data_dir} --tts' first."
            )
        
        # Split into train/val
        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)
        
        n_val = int(len(indices) * val_ratio)
        if split == "val":
            self.indices = indices[:n_val]
        else:
            self.indices = indices[n_val:]
        
        # Build character vocabulary
        self.char_to_idx, self.idx_to_char = self._build_vocab()
        
        print(f"TTSDataset ({split}): {len(self.indices)} samples")
        print(f"   Vocabulary size: {len(self.char_to_idx)}")
    
    def _find_samples(self) -> List[Dict]:
        """Find all TTS samples (text + audio token pairs)."""
        samples = []
        
        # Look for .pt files with TTS info
        for token_file in self.token_dir.rglob("*.pt"):
            if token_file.name == "metadata.pt":
                continue
            
            try:
                data = torch.load(token_file, map_location="cpu", weights_only=False)
                
                # Check if this is a TTS sample (has text and speaker)
                if "text" in data and "speaker_id" in data:
                    samples.append({
                        "path": token_file,
                        "text": data["text"],
                        "speaker_id": data["speaker_id"],
                    })
            except Exception as e:
                continue
        
        return sorted(samples, key=lambda x: str(x["path"]))
    
    def _build_vocab(self) -> tuple:
        """Build character vocabulary from all texts."""
        # Collect all unique characters
        chars = set()
        for sample in self.samples:
            chars.update(sample["text"])
        
        # Special tokens
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        
        # Build mappings
        char_to_idx = {tok: i for i, tok in enumerate(special_tokens)}
        for char in sorted(chars):
            if char not in char_to_idx:
                char_to_idx[char] = len(char_to_idx)
        
        idx_to_char = {v: k for k, v in char_to_idx.items()}
        
        return char_to_idx, idx_to_char
    
    @property
    def vocab_size(self) -> int:
        return len(self.char_to_idx)
    
    @property
    def pad_token_id(self) -> int:
        return self.char_to_idx["<pad>"]
    
    @property
    def bos_token_id(self) -> int:
        return self.char_to_idx["<bos>"]
    
    @property
    def eos_token_id(self) -> int:
        return self.char_to_idx["<eos>"]
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token ids."""
        # Add BOS and EOS
        ids = [self.bos_token_id]
        
        for char in text[:self.max_text_length - 2]:  # Leave room for BOS/EOS
            ids.append(self.char_to_idx.get(char, self.char_to_idx["<unk>"]))
        
        ids.append(self.eos_token_id)
        
        return torch.tensor(ids, dtype=torch.long)
    
    def decode_text(self, ids: torch.Tensor) -> str:
        """Decode token ids to text."""
        chars = []
        for idx in ids.tolist():
            if idx == self.pad_token_id:
                continue
            if idx == self.bos_token_id or idx == self.eos_token_id:
                continue
            chars.append(self.idx_to_char.get(idx, "?"))
        return "".join(chars)
    
    def _chunk_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Chunk or pad audio tokens."""
        if self.max_audio_frames is None:
            return tokens
        
        T = tokens.size(-1)
        
        if T > self.max_audio_frames:
            if self.split == "train":
                start = random.randint(0, T - self.max_audio_frames)
            else:
                start = 0  # Use beginning for validation
            tokens = tokens[..., start : start + self.max_audio_frames]
        
        return tokens
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a TTS sample.
        
        Returns:
            dict with:
                - text_ids: [T_text] encoded text tokens
                - speaker_id: int speaker identifier
                - audio_tokens: [num_codebooks, T_audio] Mimi tokens
                - text: original text string
        """
        sample = self.samples[self.indices[idx]]
        
        # Load audio tokens
        data = torch.load(sample["path"], map_location="cpu", weights_only=False)
        audio_tokens = data["tokens"]
        
        # Chunk if needed
        audio_tokens = self._chunk_tokens(audio_tokens)
        
        # Encode text
        text_ids = self.encode_text(sample["text"])
        
        return {
            "text_ids": text_ids,
            "speaker_id": sample["speaker_id"],
            "audio_tokens": audio_tokens,
            "text": sample["text"],
            "num_audio_frames": audio_tokens.size(-1),
        }


class TTSDataModule:
    """Data module for TTS training."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        token_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        max_text_length: int = 512,
        max_audio_frames: Optional[int] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.token_dir = Path(token_dir) if token_dir else self.data_dir / "mimi_tokens"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_text_length = max_text_length
        self.max_audio_frames = max_audio_frames
        self.val_ratio = val_ratio
        self.seed = seed
        self.pin_memory = pin_memory
        
        self.train_dataset = None
        self.val_dataset = None
        
        # Load metadata
        metadata_path = self.token_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    @property
    def num_codebooks(self) -> int:
        return self.metadata.get("num_codebooks", 32)
    
    @property
    def codebook_size(self) -> int:
        return self.metadata.get("codebook_size", 2048)
    
    @property
    def text_vocab_size(self) -> int:
        if self.train_dataset:
            return self.train_dataset.vocab_size
        return 256  # Default
    
    @property
    def num_speakers(self) -> int:
        return self.metadata.get("num_speakers", 2)
    
    def setup(self):
        """Create train and validation datasets."""
        common_args = {
            "data_dir": self.data_dir,
            "token_dir": self.token_dir,
            "max_text_length": self.max_text_length,
            "max_audio_frames": self.max_audio_frames,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
        }
        
        self.train_dataset = TTSDataset(**common_args, split="train")
        self.val_dataset = TTSDataset(**common_args, split="val")
        
        # Share vocabulary between train and val
        self.val_dataset.char_to_idx = self.train_dataset.char_to_idx
        self.val_dataset.idx_to_char = self.train_dataset.idx_to_char
    
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self._collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate with padding."""
        # Pad text
        max_text_len = max(b["text_ids"].size(0) for b in batch)
        padded_text = []
        text_mask = []
        
        for b in batch:
            text = b["text_ids"]
            T = text.size(0)
            if T < max_text_len:
                text = F.pad(text, (0, max_text_len - T), value=0)  # pad token = 0
            padded_text.append(text)
            
            mask = torch.ones(max_text_len)
            mask[T:] = 0
            text_mask.append(mask)
        
        # Pad audio tokens
        max_audio_len = max(b["audio_tokens"].size(-1) for b in batch)
        padded_audio = []
        audio_mask = []
        
        for b in batch:
            audio = b["audio_tokens"]
            T = audio.size(-1)
            if T < max_audio_len:
                audio = F.pad(audio, (0, max_audio_len - T), value=0)
            padded_audio.append(audio)
            
            mask = torch.ones(max_audio_len)
            mask[T:] = 0
            audio_mask.append(mask)
        
        return {
            "text_ids": torch.stack(padded_text),
            "text_mask": torch.stack(text_mask),
            "speaker_id": torch.tensor([b["speaker_id"] for b in batch]),
            "audio_tokens": torch.stack(padded_audio),
            "audio_mask": torch.stack(audio_mask),
            "text": [b["text"] for b in batch],
            "num_audio_frames": torch.tensor([b["num_audio_frames"] for b in batch]),
        }


def parse_transcript(transcript_path: Path) -> List[Dict]:
    """
    Parse transcript file to extract speaker and text pairs.
    
    Expected format:
        [1]: Text from speaker 1
        [2]: Text from speaker 2
    
    Returns list of {"speaker_id": int, "text": str, "line_num": int}
    """
    samples = []
    
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
    
    return samples

