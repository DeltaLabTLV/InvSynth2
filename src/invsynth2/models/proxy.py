"""IS2-style differentiable synthesizer proxy.

Maps θ → log-magnitude spectrogram. We follow the IS2 design: a small
convolutional generator that takes (B, n_total) parameters and produces
(B, F, T) spectrograms.

This proxy is trained in Stage 2 (`train_proxy.py`) once per dataset and then
frozen. During encoder fine-tuning, gradients flow through P to update the
encoder + PEN, but P's own weights are not updated.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IS2Proxy(nn.Module):
    """Lightweight conv generator that synthesizes (B, F, T) log-mag spectrograms
    from a parameter vector (B, n_total).

    Architecture: project to (B, C, F0, T0), then 4 transposed-conv up-blocks
    interpolated to the exact (F, T) target.
    """

    def __init__(
        self,
        n_params: int,
        out_freq: int = 513,
        out_time: int = 63,
        base_channels: int = 64,
    ):
        super().__init__()
        self.out_freq = out_freq
        self.out_time = out_time
        # Initial small grid that we will upsample.
        self.f0, self.t0 = 33, 4
        self.proj = nn.Sequential(
            nn.Linear(n_params, base_channels * self.f0 * self.t0),
            nn.GELU(),
        )
        c = base_channels
        self.up = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c // 2), nn.GELU(),
            nn.ConvTranspose2d(c // 2, c // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c // 2), nn.GELU(),
            nn.Conv2d(c // 2, c // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c // 4), nn.GELU(),
            nn.ConvTranspose2d(c // 4, c // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c // 4), nn.GELU(),
            nn.Conv2d(c // 4, c // 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c // 8), nn.GELU(),
            nn.ConvTranspose2d(c // 8, c // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c // 8), nn.GELU(),
            nn.Conv2d(c // 8, 1, kernel_size=3, padding=1),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # theta: (B, n_params) — produces (B, F, T) log-magnitude spectrogram.
        B = theta.shape[0]
        x = self.proj(theta).view(B, -1, self.f0, self.t0)
        x = self.up(x)  # (B, 1, F', T')
        # Resize to exact output target (F, T).
        if x.shape[-2:] != (self.out_freq, self.out_time):
            x = nn.functional.interpolate(x, size=(self.out_freq, self.out_time), mode="bilinear", align_corners=False)
        return x.squeeze(1)  # (B, F, T)
