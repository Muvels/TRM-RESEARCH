"""
Mimi Codec Wrapper for Audio Tokenization.

Mimi is Kyutai's neural audio codec that compresses 24kHz audio
to discrete tokens at 12.5Hz with ~1.1kbps bandwidth.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn

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


class MimiEncoder:
    """
    Wrapper for Kyutai's Mimi audio codec.

    Handles encoding audio waveforms to discrete tokens and
    decoding tokens back to audio.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Mimi encoder.

        Args:
            device: Device to run the model on
            dtype: Data type for model weights
        """
        self.device = device
        self.dtype = dtype
        self.model = None
        self.feature_extractor = None
        self.sample_rate = 24000  # Mimi native sample rate
        self.frame_rate = 12.5  # Mimi frame rate in Hz
        self.num_codebooks = None  # Will be set after loading model
        self.codebook_size = 2048  # Vocabulary size per codebook

    def load(self):
        """Load the Mimi model from HuggingFace."""
        try:
            from transformers import MimiModel, AutoFeatureExtractor

            self.model = MimiModel.from_pretrained("kyutai/mimi")
            self.model = self.model.to(self.device).to(self.dtype)
            self.model.eval()

            self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
            
            # Get actual number of codebooks from model config
            if hasattr(self.model.config, 'num_quantizers'):
                self.num_codebooks = self.model.config.num_quantizers
            else:
                self.num_codebooks = 32  # Default for Mimi

            print(f"✓ Mimi model loaded on {self.device}")
            print(f"   Codebooks: {self.num_codebooks}, Vocab size: {self.codebook_size}")
            return self
        except ImportError:
            raise ImportError(
                "Please install transformers>=4.45.1: pip install transformers"
            )
        except Exception as e:
            print(f"Failed to load Mimi from HuggingFace: {e}")
            print("Falling back to local mode (will need to implement)")
            raise

    def preprocess_audio(
        self,
        audio: Union[torch.Tensor, str, Path],
        target_sample_rate: int = 24000,
    ) -> torch.Tensor:
        """
        Preprocess audio for Mimi encoding.

        Args:
            audio: Audio tensor [C, T] or path to audio file
            target_sample_rate: Target sample rate (Mimi uses 24kHz)

        Returns:
            Preprocessed audio tensor [1, T]
        """
        if isinstance(audio, (str, Path)):
            # Use soundfile for reading (most reliable)
            if HAS_SOUNDFILE:
                data, sr = sf.read(str(audio))
                # soundfile returns [T] or [T, C], convert to [C, T]
                waveform = torch.from_numpy(data).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T  # [C, T]
            elif HAS_TORCHAUDIO:
                waveform, sr = torchaudio.load(str(audio))
            else:
                raise ImportError("Neither soundfile nor torchaudio available")
        else:
            waveform = audio
            sr = target_sample_rate

        # Resample if needed
        if sr != target_sample_rate:
            if HAS_TORCHAUDIO:
                resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                waveform = resampler(waveform)
            else:
                # Simple linear interpolation resample
                ratio = target_sample_rate / sr
                new_len = int(waveform.size(-1) * ratio)
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0), size=new_len, mode='linear', align_corners=False
                ).squeeze(0)

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform

    @torch.no_grad()
    def encode(
        self,
        audio: Union[torch.Tensor, str, Path, List],
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode audio to Mimi tokens.

        Args:
            audio: Audio waveform [B, C, T] or [C, T] or file path(s)
            return_embeddings: Also return continuous embeddings

        Returns:
            tokens: Discrete token indices [B, num_codebooks, T_frames]
            embeddings: (optional) Continuous embeddings [B, T_frames, D]
        """
        if self.model is None:
            self.load()

        # Handle different input types
        if isinstance(audio, (str, Path)):
            waveform = self.preprocess_audio(audio)
            waveform = waveform.unsqueeze(0)  # Add batch dim [1, 1, T]
        elif isinstance(audio, list):
            waveforms = [self.preprocess_audio(a) for a in audio]
            # Pad to same length
            max_len = max(w.size(-1) for w in waveforms)
            waveforms = [
                torch.nn.functional.pad(w, (0, max_len - w.size(-1)))
                for w in waveforms
            ]
            waveform = torch.stack(waveforms)  # [B, 1, T]
        else:
            waveform = audio
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # [B, 1, T]

        # Ensure mono: if waveform is [B, C, T] with C > 1, average channels
        if waveform.dim() == 3 and waveform.size(1) > 1:
            waveform = waveform.mean(dim=1, keepdim=True)  # [B, 1, T]

        waveform = waveform.to(self.device).to(self.dtype)

        # Use feature extractor if available
        if self.feature_extractor is not None:
            # Feature extractor expects list of [T] arrays or single [T] array
            # Our waveform is [B, 1, T]
            B = waveform.size(0)
            audio_list = [waveform[i, 0].cpu().numpy() for i in range(B)]  # List of [T]
            
            # If single sample, pass as single array
            if B == 1:
                audio_input = audio_list[0]
            else:
                audio_input = audio_list
            
            inputs = self.feature_extractor(
                audio_input,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.encode(**inputs)
            tokens = outputs.audio_codes  # [B, num_codebooks, T]
        else:
            # Direct encoding (if using custom implementation)
            outputs = self.model.encode(waveform)
            tokens = outputs.audio_codes

        if return_embeddings:
            # Get continuous embeddings from encoder
            embeddings = outputs.encoder_outputs if hasattr(outputs, 'encoder_outputs') else None
            return tokens, embeddings

        return tokens

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode Mimi tokens back to audio.

        Args:
            tokens: Token indices [B, num_codebooks, T_frames]

        Returns:
            audio: Reconstructed waveform [B, 1, T_samples]
        """
        if self.model is None:
            self.load()

        tokens = tokens.to(self.device)

        # Decode tokens
        outputs = self.model.decode(tokens)
        audio = outputs.audio_values  # [B, 1, T]

        return audio

    def tokens_to_embeddings(
        self,
        tokens: torch.Tensor,
        embed_dim: int = 256,
    ) -> torch.Tensor:
        """
        Convert discrete tokens to continuous embeddings.

        This creates a simple embedding from the discrete tokens
        that can be fed into the TRM model.

        Args:
            tokens: Token indices [B, num_codebooks, T]
            embed_dim: Target embedding dimension

        Returns:
            embeddings: [B, T, embed_dim]
        """
        B, C, T = tokens.shape

        # Simple learned embedding projection
        # In practice, this should use the codec's learned embeddings
        if not hasattr(self, 'token_projector'):
            self.token_projector = nn.Embedding(
                self.codebook_size, embed_dim // C
            ).to(tokens.device)

        # Get embeddings for each codebook
        embeddings = []
        for c in range(C):
            emb = self.token_projector(tokens[:, c, :])  # [B, T, D/C]
            embeddings.append(emb)

        # Concatenate codebook embeddings
        embeddings = torch.cat(embeddings, dim=-1)  # [B, T, D]

        return embeddings

    def get_frame_count(self, audio_length_samples: int) -> int:
        """Get number of Mimi frames for given audio length."""
        return int(audio_length_samples * self.frame_rate / self.sample_rate)

    def get_audio_length(self, num_frames: int) -> int:
        """Get audio length in samples for given number of frames."""
        return int(num_frames * self.sample_rate / self.frame_rate)


class MimiTokenizer:
    """
    Tokenizer interface for Mimi tokens.

    Provides a text-tokenizer-like interface for audio tokens.
    """

    def __init__(
        self,
        num_codebooks: int = 8,
        codebook_size: int = 2048,
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        # Special tokens
        self.bos_token_id = codebook_size * num_codebooks
        self.eos_token_id = codebook_size * num_codebooks + 1
        self.pad_token_id = codebook_size * num_codebooks + 2

        self.vocab_size = codebook_size * num_codebooks + 3

    def flatten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Flatten multi-codebook tokens to single sequence.

        Args:
            tokens: [B, num_codebooks, T]

        Returns:
            flat_tokens: [B, T * num_codebooks]
        """
        B, C, T = tokens.shape
        # Add codebook offset to distinguish tokens
        offsets = (
            torch.arange(C, device=tokens.device).view(1, C, 1) * self.codebook_size
        )
        tokens_offset = tokens + offsets
        # Interleave codebooks: [t0_c0, t0_c1, ..., t0_c7, t1_c0, ...]
        return tokens_offset.transpose(1, 2).reshape(B, T * C)

    def unflatten_tokens(self, flat_tokens: torch.Tensor) -> torch.Tensor:
        """
        Unflatten single sequence back to multi-codebook format.

        Args:
            flat_tokens: [B, T * num_codebooks]

        Returns:
            tokens: [B, num_codebooks, T]
        """
        B = flat_tokens.size(0)
        T = flat_tokens.size(1) // self.num_codebooks

        # Reshape to [B, T, C]
        tokens = flat_tokens.view(B, T, self.num_codebooks)

        # Remove offsets
        offsets = (
            torch.arange(self.num_codebooks, device=flat_tokens.device)
            * self.codebook_size
        )
        tokens = tokens - offsets

        # Transpose to [B, C, T]
        return tokens.transpose(1, 2)

    def add_special_tokens(
        self,
        tokens: torch.Tensor,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> torch.Tensor:
        """Add BOS and/or EOS tokens."""
        B = tokens.size(0)
        device = tokens.device

        if add_bos:
            bos = torch.full((B, 1), self.bos_token_id, device=device)
            tokens = torch.cat([bos, tokens], dim=1)

        if add_eos:
            eos = torch.full((B, 1), self.eos_token_id, device=device)
            tokens = torch.cat([tokens, eos], dim=1)

        return tokens

    def pad_sequence(
        self,
        tokens: torch.Tensor,
        max_length: int,
        padding_side: str = "right",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad token sequence to max_length.

        Returns:
            padded_tokens: [B, max_length]
            attention_mask: [B, max_length]
        """
        B, T = tokens.shape
        device = tokens.device

        if T >= max_length:
            return tokens[:, :max_length], torch.ones(B, max_length, device=device)

        pad_length = max_length - T
        pad = torch.full((B, pad_length), self.pad_token_id, device=device)
        mask = torch.zeros(B, max_length, device=device)

        if padding_side == "right":
            padded = torch.cat([tokens, pad], dim=1)
            mask[:, :T] = 1
        else:
            padded = torch.cat([pad, tokens], dim=1)
            mask[:, pad_length:] = 1

        return padded, mask

