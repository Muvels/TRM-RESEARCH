"""
Tiny Recursive Model (TRM) for Audio Token Prediction.

Based on the paper: "Less is More: Recursive Reasoning with Tiny Networks"
by Samsung AI Lab Montreal.

The TRM architecture uses recursive self-improvement to refine predictions.
It iteratively updates a latent state (z) and answer (y) over K improvement steps.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TRMConfig:
    """Configuration for TRM model."""

    # Model dimensions
    embed_dim: int = 256  # Embedding dimension for latent/answer
    hidden_dim: int = 512  # Hidden dimension for MLPs

    # Vocabulary
    vocab_size: int = 2048  # Mimi codebook size (typically 2048)
    num_codebooks: int = 8  # Number of Mimi codebooks

    # Recursive parameters
    L_layers: int = 2  # Number of layers in recursive block
    H_cycles: int = 3  # High-level improvement cycles (K steps)
    L_cycles: int = 6  # Low-level recursive reasoning cycles (n steps)

    # Architecture choices
    use_attention: bool = True  # Use attention or MLP-only (mlp_t)
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.1  # Dropout rate

    # Positional encoding
    max_seq_len: int = 4096  # Maximum sequence length
    pos_encodings: str = "sinusoidal"  # "sinusoidal", "learned", or "none"


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encodings."""

    def __init__(self, embed_dim: int, max_seq_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class RecursiveBlock(nn.Module):
    """
    Recursive reasoning block that updates latent z given (x, y, z).

    This is the core of TRM - it recursively refines the latent representation.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Layer normalization
        self.ln_x = nn.LayerNorm(config.embed_dim)
        self.ln_y = nn.LayerNorm(config.embed_dim)
        self.ln_z = nn.LayerNorm(config.embed_dim)

        if config.use_attention:
            # Attention-based recursive update
            self.self_attn = nn.MultiheadAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.cross_attn_x = nn.MultiheadAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.cross_attn_y = nn.MultiheadAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
        else:
            # MLP-based recursive update (mlp_t variant)
            self.mlp_combine = nn.Sequential(
                nn.Linear(config.embed_dim * 3, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.embed_dim),
            )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )
        self.ln_ffn = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update latent z using question x and current answer y.

        Args:
            x: Input embeddings [B, Lx, D]
            y: Current answer embeddings [B, Ly, D]
            z: Current latent embeddings [B, Lz, D]
            x_mask: Optional mask for x
            y_mask: Optional mask for y

        Returns:
            Updated z [B, Lz, D]
        """
        z_norm = self.ln_z(z)
        x_norm = self.ln_x(x)
        y_norm = self.ln_y(y)

        if self.config.use_attention:
            # Self-attention on z
            z_self, _ = self.self_attn(z_norm, z_norm, z_norm)
            z = z + z_self

            # Cross-attention to x (question)
            z_cross_x, _ = self.cross_attn_x(
                self.ln_z(z), x_norm, x_norm, key_padding_mask=x_mask
            )
            z = z + z_cross_x

            # Cross-attention to y (current answer)
            z_cross_y, _ = self.cross_attn_y(
                self.ln_z(z), y_norm, y_norm, key_padding_mask=y_mask
            )
            z = z + z_cross_y
        else:
            # MLP-based combination
            # Pool x and y to match z dimension
            x_pooled = x_norm.mean(dim=1, keepdim=True).expand_as(z_norm)
            y_pooled = y_norm.mean(dim=1, keepdim=True).expand_as(z_norm)
            combined = torch.cat([z_norm, x_pooled, y_pooled], dim=-1)
            z = z + self.mlp_combine(combined)

        # Feed-forward
        z = z + self.ffn(self.ln_ffn(z))

        return z


class AnswerUpdater(nn.Module):
    """Updates the answer y given current y and latent z."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        self.ln_y = nn.LayerNorm(config.embed_dim)
        self.ln_z = nn.LayerNorm(config.embed_dim)

        if config.use_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
        else:
            self.mlp_update = nn.Sequential(
                nn.Linear(config.embed_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.embed_dim),
            )

        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )
        self.ln_ffn = nn.LayerNorm(config.embed_dim)

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Update answer y using latent z.

        Args:
            y: Current answer embeddings [B, Ly, D]
            z: Latent embeddings [B, Lz, D]

        Returns:
            Updated y [B, Ly, D]
        """
        y_norm = self.ln_y(y)
        z_norm = self.ln_z(z)

        if self.config.use_attention:
            y_update, _ = self.cross_attn(y_norm, z_norm, z_norm)
            y = y + y_update
        else:
            z_pooled = z_norm.mean(dim=1, keepdim=True).expand_as(y_norm)
            combined = torch.cat([y_norm, z_pooled], dim=-1)
            y = y + self.mlp_update(combined)

        y = y + self.ffn(self.ln_ffn(y))

        return y


class TRM(nn.Module):
    """
    Tiny Recursive Model for Mimi Token Prediction.

    The model works as follows:
    1. Encode input audio x using Mimi
    2. Initialize answer y and latent z
    3. For K improvement steps (H_cycles):
        a. For n recursive steps (L_cycles):
            - Update z = RecursiveBlock(x, y, z)
        b. Update y = AnswerUpdater(y, z)
    4. Project y to Mimi token predictions
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Token embeddings (for each codebook)
        self.token_embedding = nn.Embedding(
            config.vocab_size * config.num_codebooks + 2,  # +2 for BOS, PAD
            config.embed_dim,
        )

        # Special tokens
        self.bos_token = config.vocab_size * config.num_codebooks
        self.pad_token = config.vocab_size * config.num_codebooks + 1

        # Codebook embedding to distinguish different codebooks
        self.codebook_embedding = nn.Embedding(config.num_codebooks, config.embed_dim)

        # Positional encoding
        if config.pos_encodings == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(
                config.embed_dim, config.max_seq_len
            )
        elif config.pos_encodings == "learned":
            self.pos_enc = nn.Embedding(config.max_seq_len, config.embed_dim)
        else:
            self.pos_enc = None

        # Input projection (from Mimi features)
        self.input_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Learnable initial latent
        self.init_z = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        # Learnable initial answer (start token)
        self.init_y = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        # Recursive blocks (L_layers deep)
        self.recursive_blocks = nn.ModuleList(
            [RecursiveBlock(config) for _ in range(config.L_layers)]
        )

        # Answer updater
        self.answer_updater = AnswerUpdater(config)

        # Output projection to token logits (per codebook)
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)

        # Layer norms
        self.ln_input = nn.LayerNorm(config.embed_dim)
        self.ln_output = nn.LayerNorm(config.embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values to prevent instability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_tokens(
        self, tokens: torch.Tensor, codebook_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode Mimi tokens to embeddings.

        Args:
            tokens: Token indices [B, num_codebooks, T] or [B, T]
            codebook_indices: Which codebook each token belongs to [B, T]

        Returns:
            Token embeddings [B, T, D]
        """
        if tokens.dim() == 3:
            # [B, num_codebooks, T] -> flatten to [B, T * num_codebooks]
            B, C, T = tokens.shape
            # Add codebook offset to distinguish tokens from different codebooks
            offsets = (
                torch.arange(C, device=tokens.device).view(1, C, 1) * self.config.vocab_size
            )
            tokens_offset = tokens + offsets
            tokens_flat = tokens_offset.transpose(1, 2).reshape(B, T * C)

            # Get embeddings
            embeddings = self.token_embedding(tokens_flat)

            # Add codebook position information
            codebook_ids = (
                torch.arange(C, device=tokens.device)
                .view(1, C, 1)
                .expand(B, C, T)
                .transpose(1, 2)
                .reshape(B, T * C)
            )
            embeddings = embeddings + self.codebook_embedding(codebook_ids)
        else:
            # [B, T] - single codebook or pre-flattened
            embeddings = self.token_embedding(tokens)
            if codebook_indices is not None:
                embeddings = embeddings + self.codebook_embedding(codebook_indices)

        # Add positional encoding
        if self.pos_enc is not None:
            if isinstance(self.pos_enc, SinusoidalPositionalEncoding):
                embeddings = self.pos_enc(embeddings)
            else:
                positions = torch.arange(embeddings.size(1), device=embeddings.device)
                embeddings = embeddings + self.pos_enc(positions)

        return self.ln_input(embeddings)

    def forward(
        self,
        input_tokens: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with recursive refinement.

        Args:
            input_tokens: Input Mimi tokens [B, num_codebooks, T] or features [B, T, D]
            target_tokens: Target tokens for loss computation [B, num_codebooks, T]
            return_all_steps: Whether to return predictions from all H_cycles

        Returns:
            logits: Token predictions [B, T, vocab_size] or [B, H, T, vocab_size]
            loss: Cross-entropy loss if targets provided
        """
        # Encode input
        if input_tokens.dim() == 3 and input_tokens.size(-1) == self.config.embed_dim:
            # Already embeddings
            x = self.input_proj(input_tokens)
            x = self.ln_input(x)
        else:
            # Token indices
            x = self.encode_tokens(input_tokens)

        B, T, D = x.shape

        # Initialize latent z (expand to sequence length)
        z = self.init_z.expand(B, T, D).clone()

        # Initialize answer y
        y = self.init_y.expand(B, T, D).clone()

        all_logits = []

        # Recursive improvement loop
        for h in range(self.config.H_cycles):
            # Inner recursive reasoning loop
            for l in range(self.config.L_cycles):
                for block in self.recursive_blocks:
                    z = block(x, y, z)

            # Update answer
            y = self.answer_updater(y, z)

            if return_all_steps or h == self.config.H_cycles - 1:
                # Project to logits
                logits = self.output_proj(self.ln_output(y))
                all_logits.append(logits)

        if return_all_steps:
            logits = torch.stack(all_logits, dim=1)  # [B, H, T, V]
        else:
            logits = all_logits[-1]  # [B, T, V]

        # Compute loss if targets provided
        loss = None
        if target_tokens is not None:
            if target_tokens.dim() == 3:
                # [B, C, T] -> [B, T * C]
                B, C, T_target = target_tokens.shape
                target_flat = target_tokens.transpose(1, 2).reshape(B, -1)
            else:
                target_flat = target_tokens

            # Reshape logits for loss computation
            if return_all_steps:
                # Average loss over all steps (progressive improvement)
                losses = []
                for h in range(self.config.H_cycles):
                    step_logits = all_logits[h].reshape(-1, self.config.vocab_size)
                    step_loss = F.cross_entropy(
                        step_logits,
                        target_flat.reshape(-1) % self.config.vocab_size,
                        ignore_index=self.pad_token % self.config.vocab_size,
                    )
                    losses.append(step_loss)
                # Weight later steps more heavily
                weights = torch.linspace(0.5, 1.0, len(losses), device=x.device)
                weights = weights / weights.sum()
                loss = sum(w * l for w, l in zip(weights, losses))
            else:
                logits_flat = logits.reshape(-1, self.config.vocab_size)
                loss = F.cross_entropy(
                    logits_flat,
                    target_flat.reshape(-1) % self.config.vocab_size,
                    ignore_index=self.pad_token % self.config.vocab_size,
                )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_tokens: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate Mimi tokens autoregressively.

        Args:
            input_tokens: Input tokens or embeddings
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated token indices [B, T]
        """
        self.eval()

        # Get initial embeddings
        if input_tokens.dim() == 3 and input_tokens.size(-1) == self.config.embed_dim:
            x = self.input_proj(input_tokens)
            x = self.ln_input(x)
        else:
            x = self.encode_tokens(input_tokens)

        B, T, D = x.shape

        # Initialize
        z = self.init_z.expand(B, max_length, D).clone()
        y = self.init_y.expand(B, max_length, D).clone()

        generated_tokens = []

        # Full recursive refinement on context
        for h in range(self.config.H_cycles):
            for l in range(self.config.L_cycles):
                for block in self.recursive_blocks:
                    z = block(x, y, z)
            y = self.answer_updater(y, z)

        # Get logits and sample
        logits = self.output_proj(self.ln_output(y))

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., [-1]]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
        tokens = tokens.view(B, max_length)

        return tokens

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


