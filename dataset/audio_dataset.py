"""
Audio Dataset for TRM Training.

Loads audio files, encodes them with Mimi codec, and prepares
them for training the Tiny Recursive Model.

Supports both on-the-fly encoding and pre-tokenized data.
"""

import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


class PreTokenizedDataset(Dataset):
    """
    Dataset for loading pre-tokenized Mimi tokens.
    
    Much faster than on-the-fly encoding since tokens are loaded directly.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        token_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize pre-tokenized dataset.
        
        Args:
            data_dir: Base data directory
            token_dir: Directory containing .pt token files (default: data_dir/mimi_tokens)
            max_frames: Maximum number of frames to use (None = use all)
            split: "train" or "val"
            val_ratio: Validation ratio
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.token_dir = Path(token_dir) if token_dir else self.data_dir / "mimi_tokens"
        self.max_frames = max_frames
        self.split = split
        
        # Load metadata
        metadata_path = self.token_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self.num_codebooks = self.metadata.get("num_codebooks", 32)
            self.codebook_size = self.metadata.get("codebook_size", 2048)
            self.frame_rate = self.metadata.get("frame_rate", 12.5)
        else:
            # Defaults if no metadata
            self.metadata = {}
            self.num_codebooks = 32
            self.codebook_size = 2048
            self.frame_rate = 12.5
        
        # Find all token files
        self.token_files = self._find_token_files()
        
        if not self.token_files:
            raise ValueError(
                f"No token files found in {self.token_dir}. "
                f"Run 'python pretokenize.py --data-dir {self.data_dir}' first."
            )
        
        # Split into train/val
        random.seed(seed)
        indices = list(range(len(self.token_files)))
        random.shuffle(indices)
        
        n_val = int(len(indices) * val_ratio)
        if split == "val":
            self.indices = indices[:n_val]
        else:
            self.indices = indices[n_val:]
        
        print(f"PreTokenizedDataset ({split}): {len(self.indices)} samples")
    
    def _find_token_files(self) -> List[Path]:
        """Find all .pt token files."""
        token_files = list(self.token_dir.rglob("*.pt"))
        # Filter out metadata files
        token_files = [f for f in token_files if f.name != "metadata.pt"]
        return sorted(token_files)
    
    def _chunk_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Chunk or pad tokens to max_frames if specified."""
        if self.max_frames is None:
            return tokens
        
        T = tokens.size(-1)
        
        if T > self.max_frames:
            # Random crop for training, center crop for val
            if self.split == "train":
                start = random.randint(0, T - self.max_frames)
            else:
                start = (T - self.max_frames) // 2
            tokens = tokens[..., start : start + self.max_frames]
        elif T < self.max_frames:
            # Pad with zeros (will be masked out)
            pad_length = self.max_frames - T
            tokens = F.pad(tokens, (0, pad_length), value=0)
        
        return tokens
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get pre-tokenized sample.
        
        Returns:
            dict with:
                - tokens: [num_codebooks, T] Mimi tokens
                - path: original source path
                - num_frames: actual number of frames (before padding)
        """
        file_idx = self.indices[idx]
        token_path = self.token_files[file_idx]
        
        # Load tokens
        data = torch.load(token_path, map_location="cpu", weights_only=True)
        tokens = data["tokens"]  # [num_codebooks, T]
        
        original_frames = tokens.size(-1)
        
        # Chunk/pad if needed
        tokens = self._chunk_tokens(tokens)
        
        return {
            "tokens": tokens,
            "path": data.get("source", str(token_path)),
            "num_frames": min(original_frames, self.max_frames or original_frames),
        }


class PreTokenizedDataModule:
    """
    Data module for pre-tokenized datasets.
    
    Significantly faster than AudioDataModule since no encoding is needed.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        token_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        max_frames: Optional[int] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
    ):
        """
        Initialize pre-tokenized data module.
        
        Args:
            data_dir: Base data directory
            token_dir: Directory with .pt token files
            batch_size: Batch size
            num_workers: Dataloader workers
            max_frames: Max frames per sample (None = variable length)
            val_ratio: Validation ratio
            seed: Random seed
            pin_memory: Pin memory for GPU transfer
        """
        self.data_dir = Path(data_dir)
        self.token_dir = Path(token_dir) if token_dir else self.data_dir / "mimi_tokens"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_frames = max_frames
        self.val_ratio = val_ratio
        self.seed = seed
        self.pin_memory = pin_memory
        
        self.train_dataset = None
        self.val_dataset = None
        
        # Load metadata for model config
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
    
    def setup(self):
        """Create train and validation datasets."""
        common_args = {
            "data_dir": self.data_dir,
            "token_dir": self.token_dir,
            "max_frames": self.max_frames,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
        }
        
        self.train_dataset = PreTokenizedDataset(
            **common_args,
            split="train",
        )
        
        self.val_dataset = PreTokenizedDataset(
            **common_args,
            split="val",
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
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
        """Get validation dataloader."""
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
    
    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """Collate batch of samples with padding."""
        # Find max length in batch
        max_len = max(b["tokens"].size(-1) for b in batch)
        
        # Pad tokens to same length
        padded_tokens = []
        attention_masks = []
        
        for b in batch:
            tokens = b["tokens"]
            T = tokens.size(-1)
            
            if T < max_len:
                # Pad with zeros
                tokens = F.pad(tokens, (0, max_len - T), value=0)
            
            padded_tokens.append(tokens)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = torch.ones(max_len)
            mask[b["num_frames"]:] = 0
            attention_masks.append(mask)
        
        return {
            "tokens": torch.stack(padded_tokens),
            "attention_mask": torch.stack(attention_masks),
            "path": [b["path"] for b in batch],
            "num_frames": torch.tensor([b["num_frames"] for b in batch]),
        }


class AudioDataset(Dataset):
    """
    Dataset for loading audio files and encoding with Mimi.

    Supports various audio formats and handles preprocessing,
    chunking, and augmentation.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        mimi_encoder=None,
        max_audio_length: float = 10.0,  # seconds
        sample_rate: int = 24000,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        precompute_tokens: bool = False,
        cache_dir: Optional[Path] = None,
        augment: bool = True,
    ):
        """
        Initialize the audio dataset.

        Args:
            data_dir: Directory containing audio files
            mimi_encoder: MimiEncoder instance for tokenization
            max_audio_length: Maximum audio length in seconds
            sample_rate: Target sample rate (24000 for Mimi)
            split: "train" or "val"
            val_ratio: Fraction of data for validation
            seed: Random seed for reproducibility
            precompute_tokens: Pre-encode all audio to tokens
            cache_dir: Directory to cache tokenized data
            augment: Apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.mimi_encoder = mimi_encoder
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        self.split = split
        self.augment = augment and split == "train"
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Find all audio files
        self.audio_files = self._find_audio_files()

        # Split into train/val
        random.seed(seed)
        indices = list(range(len(self.audio_files)))
        random.shuffle(indices)

        n_val = int(len(indices) * val_ratio)
        if split == "val":
            self.indices = indices[:n_val]
        else:
            self.indices = indices[n_val:]

        print(f"AudioDataset ({split}): {len(self.indices)} samples")

        # Precompute tokens if requested
        self.token_cache = {}
        if precompute_tokens and mimi_encoder is not None:
            self._precompute_tokens()

    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in the data directory."""
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(self.data_dir.rglob(f"*{ext}"))

        # Sort for reproducibility
        audio_files = sorted(audio_files)

        if not audio_files:
            raise ValueError(f"No audio files found in {self.data_dir}")

        print(f"Found {len(audio_files)} audio files")
        return audio_files

    def _precompute_tokens(self):
        """Pre-encode all audio files to tokens."""
        print("Pre-computing Mimi tokens...")
        for idx in self.indices:
            if idx not in self.token_cache:
                audio_path = self.audio_files[idx]
                waveform = self._load_audio(audio_path)
                tokens = self.mimi_encoder.encode(waveform)
                self.token_cache[idx] = tokens.cpu()

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load and preprocess audio file."""
        if HAS_SOUNDFILE:
            data, sr = sf.read(str(path))
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T  # [C, T]
        elif HAS_TORCHAUDIO:
            waveform, sr = torchaudio.load(str(path))
        else:
            raise ImportError("Neither soundfile nor torchaudio available")

        # Resample if needed
        if sr != self.sample_rate:
            if HAS_TORCHAUDIO:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            else:
                # Simple linear interpolation resample
                ratio = self.sample_rate / sr
                new_len = int(waveform.size(-1) * ratio)
                waveform = F.interpolate(
                    waveform.unsqueeze(0), size=new_len, mode='linear', align_corners=False
                ).squeeze(0)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    def _chunk_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Chunk or pad audio to max_samples."""
        T = waveform.size(-1)

        if T > self.max_samples:
            # Random crop for training, center crop for val
            if self.split == "train":
                start = random.randint(0, T - self.max_samples)
            else:
                start = (T - self.max_samples) // 2
            waveform = waveform[..., start : start + self.max_samples]
        elif T < self.max_samples:
            # Pad with zeros
            pad_length = self.max_samples - T
            waveform = F.pad(waveform, (0, pad_length))

        return waveform

    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation."""
        if not self.augment:
            return waveform

        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain

        # Random noise
        if random.random() < 0.3:
            noise_level = random.uniform(0.001, 0.01)
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise

        # Normalize
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / (max_val + 1e-8)

        return waveform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single audio sample.

        Returns:
            dict with:
                - waveform: [1, T] audio tensor
                - tokens: [num_codebooks, T_frames] Mimi tokens (if encoder available)
                - path: original file path
        """
        file_idx = self.indices[idx]
        audio_path = self.audio_files[file_idx]

        # Check cache first
        if file_idx in self.token_cache:
            tokens = self.token_cache[file_idx]
            # Load waveform for reference
            waveform = self._load_audio(audio_path)
            waveform = self._chunk_audio(waveform)
            return {
                "waveform": waveform,
                "tokens": tokens,
                "path": str(audio_path),
            }

        # Load and preprocess audio
        waveform = self._load_audio(audio_path)
        waveform = self._chunk_audio(waveform)
        waveform = self._augment_audio(waveform)

        result = {
            "waveform": waveform,
            "path": str(audio_path),
        }

        # Encode with Mimi if available
        if self.mimi_encoder is not None:
            tokens = self.mimi_encoder.encode(waveform.unsqueeze(0))
            result["tokens"] = tokens.squeeze(0)  # Remove batch dim

        return result


class AudioDataModule:
    """
    PyTorch Lightning-style data module for audio datasets.

    Handles train/val splits and dataloaders.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        mimi_encoder=None,
        batch_size: int = 16,
        num_workers: int = 4,
        max_audio_length: float = 10.0,
        val_ratio: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
        precompute_tokens: bool = False,
    ):
        """
        Initialize data module.

        Args:
            data_dir: Directory with audio files
            mimi_encoder: MimiEncoder for tokenization
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            max_audio_length: Max audio length in seconds
            val_ratio: Validation split ratio
            seed: Random seed
            pin_memory: Pin memory for faster GPU transfer
            precompute_tokens: Pre-encode all audio
        """
        self.data_dir = Path(data_dir)
        self.mimi_encoder = mimi_encoder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_audio_length = max_audio_length
        self.val_ratio = val_ratio
        self.seed = seed
        self.pin_memory = pin_memory
        self.precompute_tokens = precompute_tokens

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """Create train and validation datasets."""
        common_args = {
            "data_dir": self.data_dir,
            "mimi_encoder": self.mimi_encoder,
            "max_audio_length": self.max_audio_length,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
            "precompute_tokens": self.precompute_tokens,
        }

        self.train_dataset = AudioDataset(
            **common_args,
            split="train",
            augment=True,
        )

        self.val_dataset = AudioDataset(
            **common_args,
            split="val",
            augment=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
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
        """Get validation dataloader."""
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

    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """Collate batch of samples."""
        waveforms = torch.stack([b["waveform"] for b in batch])
        paths = [b["path"] for b in batch]

        result = {
            "waveform": waveforms,
            "path": paths,
        }

        # Stack tokens if available
        if "tokens" in batch[0]:
            tokens = torch.stack([b["tokens"] for b in batch])
            result["tokens"] = tokens

        return result


class ConversationDataset(Dataset):
    """
    Dataset for conversation-style audio data.

    Loads full conversations and their segments,
    useful for dialogue modeling.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        mimi_encoder=None,
        max_segment_length: float = 5.0,
        max_segments_per_conv: int = 10,
        sample_rate: int = 24000,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize conversation dataset.

        Args:
            data_dir: Root directory with conversation folders
            mimi_encoder: MimiEncoder for tokenization
            max_segment_length: Max length per segment in seconds
            max_segments_per_conv: Max segments to use per conversation
            sample_rate: Target sample rate
            split: "train" or "val"
            val_ratio: Validation ratio
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.mimi_encoder = mimi_encoder
        self.max_segment_length = max_segment_length
        self.max_segment_samples = int(max_segment_length * sample_rate)
        self.max_segments = max_segments_per_conv
        self.sample_rate = sample_rate
        self.split = split

        # Find conversation folders
        self.conversations = self._find_conversations()

        # Split
        random.seed(seed)
        indices = list(range(len(self.conversations)))
        random.shuffle(indices)

        n_val = int(len(indices) * val_ratio)
        if split == "val":
            self.indices = indices[:n_val]
        else:
            self.indices = indices[n_val:]

        print(f"ConversationDataset ({split}): {len(self.indices)} conversations")

    def _find_conversations(self) -> List[Dict]:
        """Find all conversation folders with segments."""
        conversations = []

        # Look for folders with segments subdirectory
        for folder in self.data_dir.rglob("**/segments"):
            conv_folder = folder.parent
            segments = sorted(folder.glob("*.wav"))

            if segments:
                # Look for transcript
                transcript_file = folder / "vibevoice-podcast-script.txt"
                transcript = None
                if transcript_file.exists():
                    transcript = transcript_file.read_text()

                conversations.append({
                    "folder": conv_folder,
                    "segments": segments,
                    "full_audio": conv_folder / "full_conversation.wav",
                    "transcript": transcript,
                })

        return conversations

    def _load_segment(self, path: Path) -> torch.Tensor:
        """Load a single audio segment."""
        if HAS_SOUNDFILE:
            data, sr = sf.read(str(path))
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
        elif HAS_TORCHAUDIO:
            waveform, sr = torchaudio.load(str(path))
        else:
            raise ImportError("Neither soundfile nor torchaudio available")

        if sr != self.sample_rate:
            if HAS_TORCHAUDIO:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            else:
                ratio = self.sample_rate / sr
                new_len = int(waveform.size(-1) * ratio)
                waveform = F.interpolate(
                    waveform.unsqueeze(0), size=new_len, mode='linear', align_corners=False
                ).squeeze(0)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate or pad
        if waveform.size(-1) > self.max_segment_samples:
            waveform = waveform[..., : self.max_segment_samples]
        elif waveform.size(-1) < self.max_segment_samples:
            pad_len = self.max_segment_samples - waveform.size(-1)
            waveform = F.pad(waveform, (0, pad_len))

        return waveform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a conversation with its segments.

        Returns:
            dict with segments, transcript, and tokens
        """
        conv = self.conversations[self.indices[idx]]

        # Load segments
        segments = conv["segments"][: self.max_segments]
        waveforms = [self._load_segment(s) for s in segments]
        waveforms = torch.stack(waveforms)  # [num_segments, 1, T]

        result = {
            "waveforms": waveforms,
            "num_segments": len(segments),
            "transcript": conv["transcript"],
            "folder": str(conv["folder"]),
        }

        # Encode with Mimi
        if self.mimi_encoder is not None:
            tokens_list = []
            for waveform in waveforms:
                tokens = self.mimi_encoder.encode(waveform.unsqueeze(0))
                tokens_list.append(tokens.squeeze(0))
            result["tokens"] = torch.stack(tokens_list)

        return result

