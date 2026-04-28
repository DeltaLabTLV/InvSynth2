"""Transformer encoder for synthesizer inversion.

§3.2 of the paper:
- Operates on log-magnitude STFT spectrogram.
- Each time step is a token; the frequency axis is the token feature dimension.
- 3.5M parameters total.

Sized to hit ~3.5M with reasonable depth/width for a 1-second @ 16 kHz clip
(≈63 STFT frames at hop=256, 513 frequency bins at n_fft=1024).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10_000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoder(nn.Module):
    """Self-attention encoder for STFT frames.

    Inputs/outputs shape:
        Input  log_mag_norm: (B, F, T)   — F=513 frequency bins, T=63 frames
        Output features:     (B, T, D)   — D = embed_dim

    The model also exposes a learnable mask token so the same module can be
    called with a pre-computed boolean mask during pre-training.
    """

    def __init__(
        self,
        n_freq: int = 513,
        embed_dim: int = 192,
        depth: int = 6,
        n_heads: int = 4,
        ff_dim: int = 768,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # Project the frequency-bin column → embed_dim.
        self.input_proj = nn.Linear(n_freq, embed_dim)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=max_len)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        log_mag_norm: torch.Tensor,                # (B, F, T)
        frame_mask: torch.Tensor | None = None,    # (B, T) bool — True = MASKED
    ) -> torch.Tensor:
        # (B, F, T) → (B, T, F) → (B, T, D)
        x = log_mag_norm.transpose(1, 2)
        x = self.input_proj(x)
        if frame_mask is not None:
            mask_token = self.mask_token.expand(x.shape[0], x.shape[1], -1)
            x = torch.where(frame_mask.unsqueeze(-1), mask_token, x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.final_norm(x)
        return x  # (B, T, D)


def make_transformer_encoder(target_params: float = 3.5e6) -> TransformerEncoder:
    """Build a Transformer near the target parameter budget (3.5M)."""
    # Configuration tuned to land near 3.5M for n_freq=513.
    return TransformerEncoder(
        n_freq=513,
        embed_dim=224,
        depth=6,
        n_heads=4,
        ff_dim=896,
        dropout=0.1,
        max_len=256,
    )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
