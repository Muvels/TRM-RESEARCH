"""
TTS TRM Model - Text-to-Speech with Tiny Recursive Model.

Takes text + speaker_id as input and predicts ALL 32 Mimi audio codebooks.
Uses recursive refinement to progressively generate audio.

Multi-codebook prediction uses:
- Shared representation from recursive refinement
- Codebook embeddings to condition on codebook index
- Per-codebook output heads for specialized predictions
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTSTRMConfig:
    """Configuration for TTS TRM model."""
    
    # Text encoder
    text_vocab_size: int = 256  # Character vocabulary size
    text_embed_dim: int = 256
    text_hidden_dim: int = 512
    text_num_layers: int = 4
    text_num_heads: int = 8
    
    # Speaker embedding
    num_speakers: int = 10
    speaker_embed_dim: int = 64
    
    # Audio decoder (TRM)
    audio_embed_dim: int = 256
    audio_hidden_dim: int = 512
    audio_vocab_size: int = 2048  # Mimi codebook size
    num_codebooks: int = 32
    
    # Recursive parameters
    L_layers: int = 2
    H_cycles: int = 3  # High-level improvement cycles
    L_cycles: int = 6  # Low-level recursive cycles
    
    # Multi-codebook settings
    codebook_embed_dim: int = 64  # Dimension for codebook embeddings
    share_output_heads: bool = False  # If True, use single head with codebook conditioning
    codebook_loss_weights: Optional[List[float]] = None  # Per-codebook loss weights
    
    # Length prediction
    length_loss_weight: float = 0.1  # Weight for duration prediction loss
    
    # General
    num_heads: int = 8
    dropout: float = 0.1
    max_text_len: int = 512
    max_audio_frames: int = 1024


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encodings."""
    
    def __init__(self, embed_dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextEncoder(nn.Module):
    """
    Transformer encoder for text input.
    
    Encodes character-level text into contextual embeddings.
    """
    
    def __init__(self, config: TTSTRMConfig):
        super().__init__()
        self.config = config
        
        # Character embedding
        self.embed = nn.Embedding(config.text_vocab_size, config.text_embed_dim)
        
        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            config.text_embed_dim, config.max_text_len
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.text_embed_dim,
            nhead=config.text_num_heads,
            dim_feedforward=config.text_hidden_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.text_num_layers
        )
        
        self.ln = nn.LayerNorm(config.text_embed_dim)
    
    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text_ids: [B, T] text token ids
            text_mask: [B, T] attention mask (1 = valid, 0 = padding)
        
        Returns:
            Text embeddings [B, T, D]
        """
        x = self.embed(text_ids)
        x = self.pos_enc(x)
        
        # Convert mask for transformer (True = masked out)
        if text_mask is not None:
            key_padding_mask = (text_mask == 0)
        else:
            key_padding_mask = None
        
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.ln(x)


class LengthPredictor(nn.Module):
    """
    Predicts the number of audio frames from text encoding.
    
    Uses attention pooling over text positions followed by an MLP
    to predict a scalar (number of frames).
    """
    
    def __init__(self, config: TTSTRMConfig):
        super().__init__()
        self.config = config
        
        # Attention pooling - learn to weight text positions
        self.attn_pool = nn.Sequential(
            nn.Linear(config.audio_embed_dim, config.audio_embed_dim // 4),
            nn.Tanh(),
            nn.Linear(config.audio_embed_dim // 4, 1),
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config.audio_embed_dim, config.audio_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.audio_hidden_dim, config.audio_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.audio_hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        text_ctx: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict number of audio frames.
        
        Args:
            text_ctx: Text context [B, T_text, D] (already includes speaker info)
            text_mask: [B, T_text] attention mask (1=valid, 0=padding)
        
        Returns:
            predicted_frames: [B] predicted number of frames (continuous)
        """
        # Compute attention weights
        attn_logits = self.attn_pool(text_ctx).squeeze(-1)  # [B, T_text]
        
        # Mask padding positions
        if text_mask is not None:
            attn_logits = attn_logits.masked_fill(text_mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, T_text]
        
        # Weighted sum
        pooled = torch.einsum("bt,btd->bd", attn_weights, text_ctx)  # [B, D]
        
        # Predict frames
        predicted_frames = self.predictor(pooled).squeeze(-1)  # [B]
        
        # Ensure positive with softplus
        predicted_frames = F.softplus(predicted_frames)
        
        return predicted_frames


class RecursiveBlock(nn.Module):
    """
    Recursive reasoning block for TTS.
    
    Updates latent z given text context, speaker, current answer y, and z.
    """
    
    def __init__(self, config: TTSTRMConfig):
        super().__init__()
        self.config = config
        
        # Layer norms
        self.ln_text = nn.LayerNorm(config.audio_embed_dim)
        self.ln_y = nn.LayerNorm(config.audio_embed_dim)
        self.ln_z = nn.LayerNorm(config.audio_embed_dim)
        
        # Self-attention on z
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.audio_embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        # Cross-attention to text context
        self.cross_attn_text = nn.MultiheadAttention(
            embed_dim=config.audio_embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        # Cross-attention to current answer
        self.cross_attn_y = nn.MultiheadAttention(
            embed_dim=config.audio_embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.audio_embed_dim, config.audio_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.audio_hidden_dim, config.audio_embed_dim),
            nn.Dropout(config.dropout),
        )
        self.ln_ffn = nn.LayerNorm(config.audio_embed_dim)
    
    def forward(
        self,
        text_ctx: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update latent z.
        
        Args:
            text_ctx: Text context [B, T_text, D]
            y: Current answer [B, T_audio, D]
            z: Current latent [B, T_audio, D]
            text_mask: Text attention mask
        
        Returns:
            Updated z [B, T_audio, D]
        """
        z_norm = self.ln_z(z)
        
        # Self-attention
        z_self, _ = self.self_attn(z_norm, z_norm, z_norm)
        z = z + z_self
        
        # Cross-attention to text
        text_key_mask = (text_mask == 0) if text_mask is not None else None
        z_text, _ = self.cross_attn_text(
            self.ln_z(z), self.ln_text(text_ctx), self.ln_text(text_ctx),
            key_padding_mask=text_key_mask,
        )
        z = z + z_text
        
        # Cross-attention to y
        z_y, _ = self.cross_attn_y(
            self.ln_z(z), self.ln_y(y), self.ln_y(y)
        )
        z = z + z_y
        
        # FFN
        z = z + self.ffn(self.ln_ffn(z))
        
        return z


class AnswerUpdater(nn.Module):
    """Updates the audio prediction y given latent z."""
    
    def __init__(self, config: TTSTRMConfig):
        super().__init__()
        
        self.ln_y = nn.LayerNorm(config.audio_embed_dim)
        self.ln_z = nn.LayerNorm(config.audio_embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.audio_embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.audio_embed_dim, config.audio_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.audio_hidden_dim, config.audio_embed_dim),
            nn.Dropout(config.dropout),
        )
        self.ln_ffn = nn.LayerNorm(config.audio_embed_dim)
    
    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y_norm = self.ln_y(y)
        z_norm = self.ln_z(z)
        
        y_update, _ = self.cross_attn(y_norm, z_norm, z_norm)
        y = y + y_update
        
        y = y + self.ffn(self.ln_ffn(y))
        
        return y


class MultiCodebookHead(nn.Module):
    """
    Multi-codebook output head.
    
    Takes shared representation and produces logits for all codebooks.
    Uses codebook embeddings to specialize predictions for each codebook level.
    """
    
    def __init__(self, config: TTSTRMConfig):
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        
        # Codebook embeddings - learned representations for each codebook level
        # These help the model understand the hierarchical structure:
        # - Codebook 0: semantic/content
        # - Codebooks 1-31: progressive acoustic detail
        self.codebook_embed = nn.Embedding(config.num_codebooks, config.codebook_embed_dim)
        
        # Project codebook embedding to audio dimension
        self.codebook_proj = nn.Linear(config.codebook_embed_dim, config.audio_embed_dim)
        
        if config.share_output_heads:
            # Single shared head with codebook conditioning
            self.output_proj = nn.Linear(config.audio_embed_dim, config.audio_vocab_size)
        else:
            # Separate head per codebook (more expressive but more parameters)
            self.output_projs = nn.ModuleList([
                nn.Linear(config.audio_embed_dim, config.audio_vocab_size)
                for _ in range(config.num_codebooks)
            ])
        
        self.ln = nn.LayerNorm(config.audio_embed_dim)
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for all codebooks.
        
        Args:
            y: Shared representation [B, T_audio, D]
        
        Returns:
            logits: [B, num_codebooks, T_audio, vocab_size]
        """
        B, T, D = y.shape
        y = self.ln(y)
        
        all_logits = []
        
        for c in range(self.num_codebooks):
            # Get codebook embedding
            codebook_idx = torch.tensor([c], device=y.device)
            cb_emb = self.codebook_embed(codebook_idx)  # [1, cb_dim]
            cb_emb = self.codebook_proj(cb_emb)  # [1, D]
            
            # Add codebook embedding to representation
            y_c = y + cb_emb.unsqueeze(0)  # [B, T, D]
            
            # Project to logits
            if self.config.share_output_heads:
                logits_c = self.output_proj(y_c)  # [B, T, V]
            else:
                logits_c = self.output_projs[c](y_c)  # [B, T, V]
            
            all_logits.append(logits_c)
        
        # Stack: [B, num_codebooks, T, V]
        logits = torch.stack(all_logits, dim=1)
        
        return logits


class TTSTRM(nn.Module):
    """
    TTS TRM Model - Full Multi-Codebook Version.
    
    Text + Speaker → All 32 Mimi Audio Codebooks
    
    Architecture:
    1. Encode text with transformer
    2. Add speaker embedding
    3. Initialize audio prediction y and latent z
    4. For K improvement steps:
        - For n recursive steps: z = RecursiveBlock(text, y, z)
        - y = AnswerUpdater(y, z)
    5. Project y to audio token logits for ALL codebooks
    
    The multi-codebook head uses codebook embeddings to capture the
    hierarchical structure of the audio codec:
    - Codebook 0: Core semantic/linguistic content
    - Codebooks 1-31: Progressive acoustic refinement
    """
    
    def __init__(self, config: TTSTRMConfig):
        super().__init__()
        self.config = config
        
        # Text encoder
        self.text_encoder = TextEncoder(config)
        
        # Project text to audio dimension if different
        if config.text_embed_dim != config.audio_embed_dim:
            self.text_proj = nn.Linear(config.text_embed_dim, config.audio_embed_dim)
        else:
            self.text_proj = nn.Identity()
        
        # Speaker embedding
        self.speaker_embed = nn.Embedding(config.num_speakers + 1, config.speaker_embed_dim)
        self.speaker_proj = nn.Linear(config.speaker_embed_dim, config.audio_embed_dim)
        
        # Length predictor (predict audio frames from text + speaker)
        self.length_predictor = LengthPredictor(config)
        
        # Learnable initial audio prediction
        self.init_y = nn.Parameter(torch.randn(1, 1, config.audio_embed_dim) * 0.02)
        
        # Learnable initial latent
        self.init_z = nn.Parameter(torch.randn(1, 1, config.audio_embed_dim) * 0.02)
        
        # Positional encoding for audio
        self.audio_pos_enc = SinusoidalPositionalEncoding(
            config.audio_embed_dim, config.max_audio_frames
        )
        
        # Recursive blocks
        self.recursive_blocks = nn.ModuleList([
            RecursiveBlock(config) for _ in range(config.L_layers)
        ])
        
        # Answer updater
        self.answer_updater = AnswerUpdater(config)
        
        # Multi-codebook output head
        self.output_head = MultiCodebookHead(config)
        
        # Codebook loss weights (higher weight for semantic codebooks)
        if config.codebook_loss_weights is not None:
            self.register_buffer(
                "codebook_weights",
                torch.tensor(config.codebook_loss_weights)
            )
        else:
            # Default: weight earlier codebooks more heavily
            # Semantic codebook (0) gets highest weight, acoustic codebooks get less
            weights = self._default_codebook_weights(config.num_codebooks)
            self.register_buffer("codebook_weights", weights)
        
        self._init_weights()
    
    def _default_codebook_weights(self, num_codebooks: int) -> torch.Tensor:
        """
        Create default loss weights for codebooks.
        
        Strategy: Semantic codebooks (especially first few) are more important
        for intelligibility, so weight them higher. Acoustic codebooks add
        detail but are less critical.
        
        Weight distribution:
        - Codebook 0 (semantic): 2.0
        - Codebooks 1-7: Linear decay from 1.5 to 1.0
        - Codebooks 8-31: 1.0 (equal weight)
        """
        weights = torch.ones(num_codebooks)
        weights[0] = 2.0  # Semantic codebook most important
        
        # Decay for early acoustic codebooks
        for i in range(1, min(8, num_codebooks)):
            weights[i] = 1.5 - (0.5 * i / 7)
        
        # Normalize to mean 1.0
        weights = weights / weights.mean()
        
        return weights
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        text_ids: torch.Tensor,
        speaker_id: torch.Tensor,
        target_audio_tokens: Optional[torch.Tensor] = None,
        target_audio_frames: Optional[int] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            text_ids: [B, T_text] text token indices
            speaker_id: [B] speaker indices
            target_audio_tokens: [B, num_codebooks, T_audio] target tokens
            target_audio_frames: int, number of audio frames to generate
            text_mask: [B, T_text] text attention mask
            return_all_steps: return predictions from all H_cycles
        
        Returns:
            logits: [B, num_codebooks, T_audio, vocab_size] 
                    or [B, H, num_codebooks, T_audio, vocab_size] if return_all_steps
            loss: Cross-entropy loss if targets provided (averaged across all codebooks)
        """
        B = text_ids.size(0)
        device = text_ids.device
        
        # Encode text
        text_enc = self.text_encoder(text_ids, text_mask)  # [B, T_text, D]
        text_ctx = self.text_proj(text_enc)
        
        # Add speaker embedding (add to all text positions)
        speaker_emb = self.speaker_embed(speaker_id)  # [B, D_speaker]
        speaker_emb = self.speaker_proj(speaker_emb)  # [B, D_audio]
        text_ctx = text_ctx + speaker_emb.unsqueeze(1)  # [B, T_text, D]
        
        # Predict audio length
        predicted_frames = self.length_predictor(text_ctx, text_mask)  # [B]
        
        # Determine audio length to use
        if target_audio_tokens is not None:
            T_audio = target_audio_tokens.size(-1)
            target_length = torch.full((B,), T_audio, device=device, dtype=torch.float)
        elif target_audio_frames is not None:
            T_audio = target_audio_frames
            target_length = None
        else:
            # Use predicted length (rounded, clamped)
            T_audio = int(predicted_frames.mean().item())
            T_audio = max(1, min(T_audio, self.config.max_audio_frames))
            target_length = None
        
        # Initialize y and z
        y = self.init_y.expand(B, T_audio, -1).clone()
        z = self.init_z.expand(B, T_audio, -1).clone()
        
        # Add positional encoding
        y = self.audio_pos_enc(y)
        z = self.audio_pos_enc(z)
        
        all_logits = []
        
        # Recursive improvement
        for h in range(self.config.H_cycles):
            # Inner recursive reasoning
            for l in range(self.config.L_cycles):
                for block in self.recursive_blocks:
                    z = block(text_ctx, y, z, text_mask)
            
            # Update answer
            y = self.answer_updater(y, z)
            
            if return_all_steps or h == self.config.H_cycles - 1:
                # Get logits for all codebooks: [B, num_codebooks, T, V]
                logits = self.output_head(y)
                all_logits.append(logits)
        
        if return_all_steps:
            logits = torch.stack(all_logits, dim=1)  # [B, H, C, T, V]
        else:
            logits = all_logits[-1]  # [B, C, T, V]
        
        # Compute loss across all codebooks
        loss = None
        if target_audio_tokens is not None:
            # Token prediction loss
            token_loss = self._compute_multi_codebook_loss(
                all_logits, target_audio_tokens, return_all_steps, device
            )
            
            # Length prediction loss (MSE on log scale for stability)
            length_loss = F.mse_loss(
                torch.log(predicted_frames + 1),
                torch.log(target_length + 1),
            )
            
            # Combined loss
            loss = token_loss + self.config.length_loss_weight * length_loss
        
        return logits, loss
    
    def _compute_multi_codebook_loss(
        self,
        all_logits: List[torch.Tensor],
        target_audio_tokens: torch.Tensor,
        return_all_steps: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss across all codebooks.
        
        Args:
            all_logits: List of [B, num_codebooks, T, V] for each H step
            target_audio_tokens: [B, num_codebooks, T]
            return_all_steps: Whether to weight losses across H steps
            device: Device for tensors
        
        Returns:
            Weighted average loss
        """
        B, C, T = target_audio_tokens.shape
        
        def compute_codebook_losses(logits: torch.Tensor) -> torch.Tensor:
            """Compute per-codebook losses."""
            # logits: [B, C, T, V]
            # target: [B, C, T]
            losses = []
            for c in range(C):
                logits_c = logits[:, c, :, :]  # [B, T, V]
                target_c = target_audio_tokens[:, c, :]  # [B, T]
                
                loss_c = F.cross_entropy(
                    logits_c.reshape(-1, self.config.audio_vocab_size),
                    target_c.reshape(-1),
                    ignore_index=0,  # Padding
                    reduction='mean',
                )
                losses.append(loss_c)
            
            return torch.stack(losses)  # [C]
        
        if return_all_steps:
            # Compute losses for each H step, then weight
            step_losses = []
            for h, step_logits in enumerate(all_logits):
                codebook_losses = compute_codebook_losses(step_logits)  # [C]
                # Weight by codebook importance
                weighted_loss = (codebook_losses * self.codebook_weights).mean()
                step_losses.append(weighted_loss)
            
            # Weight later steps more (they should be better)
            step_weights = torch.linspace(0.5, 1.0, len(step_losses), device=device)
            step_weights = step_weights / step_weights.sum()
            loss = sum(w * l for w, l in zip(step_weights, step_losses))
        else:
            codebook_losses = compute_codebook_losses(all_logits[-1])  # [C]
            # Weight by codebook importance
            loss = (codebook_losses * self.codebook_weights).mean()
        
        return loss
    
    @torch.no_grad()
    def predict_length(
        self,
        text_ids: torch.Tensor,
        speaker_id: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict audio length in frames from text.
        
        Args:
            text_ids: [B, T_text] text tokens
            speaker_id: [B] speaker indices
            text_mask: Text attention mask
        
        Returns:
            predicted_frames: [B] predicted number of frames
        """
        # Encode text
        text_enc = self.text_encoder(text_ids, text_mask)
        text_ctx = self.text_proj(text_enc)
        
        # Add speaker
        speaker_emb = self.speaker_embed(speaker_id)
        speaker_emb = self.speaker_proj(speaker_emb)
        text_ctx = text_ctx + speaker_emb.unsqueeze(1)
        
        # Predict length
        predicted_frames = self.length_predictor(text_ctx, text_mask)
        
        return predicted_frames
    
    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        speaker_id: torch.Tensor,
        num_frames: Optional[int] = None,
        text_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate audio tokens from text for ALL codebooks.
        
        Args:
            text_ids: [B, T_text] text tokens
            speaker_id: [B] speaker indices
            num_frames: Number of audio frames to generate (None = use predicted length)
            text_mask: Text attention mask
            temperature: Sampling temperature (can be float or list per codebook)
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
        
        Returns:
            tokens: [B, num_codebooks, T_audio] predicted audio tokens
        """
        self.eval()
        
        # Use predicted length if not specified
        if num_frames is None:
            predicted = self.predict_length(text_ids, speaker_id, text_mask)
            # Use mean of batch, round, and clamp
            num_frames = int(predicted.mean().item())
            num_frames = max(1, min(num_frames, self.config.max_audio_frames))
        
        # Forward pass
        logits, _ = self.forward(
            text_ids, speaker_id,
            target_audio_frames=num_frames,
            text_mask=text_mask,
        )
        # logits: [B, num_codebooks, T, V]
        
        B, C, T, V = logits.shape
        
        # Handle per-codebook temperature
        if isinstance(temperature, (list, tuple)):
            temps = temperature
        else:
            temps = [temperature] * C
        
        all_tokens = []
        
        for c in range(C):
            logits_c = logits[:, c, :, :]  # [B, T, V]
            
            # Apply temperature
            logits_c = logits_c / temps[c]
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits_c, min(top_k, V), dim=-1)
                logits_c = torch.where(
                    logits_c < v[..., [-1]],
                    torch.full_like(logits_c, float("-inf")),
                    logits_c,
                )
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits_c, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                # Scatter back
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits_c = torch.where(
                    indices_to_remove,
                    torch.full_like(logits_c, float("-inf")),
                    logits_c,
                )
            
            # Sample
            probs = F.softmax(logits_c, dim=-1)
            tokens_c = torch.multinomial(probs.view(-1, V), num_samples=1)
            tokens_c = tokens_c.view(B, T)
            
            all_tokens.append(tokens_c)
        
        # Stack: [B, num_codebooks, T]
        tokens = torch.stack(all_tokens, dim=1)
        
        return tokens
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_codebook_weights(self) -> torch.Tensor:
        """Return current codebook loss weights."""
        return self.codebook_weights
