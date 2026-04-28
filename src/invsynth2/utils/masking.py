"""Masking strategies for SSL pre-training.

- Transformer: BERT-style independent per-frame masking. 45% of time frames
  replaced with a learnable mask embedding. Span length M=1.

- U-Net: MAE-style rectangular pixel masking. 6.5% of pixels chosen as mask
  centers, 3x3 mask placed around each, ~45% total coverage after overlap.
"""

from __future__ import annotations

import torch


def make_frame_mask(
    n_frames: int,
    mask_ratio: float = 0.45,
    batch: int | None = None,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Bernoulli per-frame mask used by the Transformer.

    Returns
    -------
    mask: (B, n_frames) or (n_frames,) bool, True = MASKED.
    """
    if batch is None:
        return torch.rand(n_frames, device=device, generator=generator) < mask_ratio
    return torch.rand(batch, n_frames, device=device, generator=generator) < mask_ratio


def make_unet_mask(
    n_freq: int,
    n_time: int,
    center_ratio: float = 0.065,
    region_size: int = 3,
    batch: int | None = None,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """MAE-style rectangular pixel mask for the U-Net.

    Roughly `center_ratio` of pixels are sampled as mask centers, then a
    `region_size x region_size` square is placed around each. Final coverage
    after overlap is ~ 1 - (1 - region_size**2 / total) ** (center_ratio * total).

    For the paper's defaults (6.5% centers, 3x3 region), expected coverage ≈ 45%.
    """

    def _one(_n_freq: int, _n_time: int) -> torch.Tensor:
        total = _n_freq * _n_time
        n_centers = max(1, int(round(total * center_ratio)))
        # Sample center positions
        flat_idx = torch.randperm(total, device=device, generator=generator)[:n_centers]
        cf = flat_idx // _n_time  # center freq indices
        ct = flat_idx % _n_time   # center time indices

        mask = torch.zeros(_n_freq, _n_time, dtype=torch.bool, device=device)
        half = region_size // 2
        for df in range(-half, half + 1):
            for dt in range(-half, half + 1):
                ff = (cf + df).clamp(0, _n_freq - 1)
                tt = (ct + dt).clamp(0, _n_time - 1)
                mask[ff, tt] = True
        return mask

    if batch is None:
        return _one(n_freq, n_time)
    return torch.stack([_one(n_freq, n_time) for _ in range(batch)], dim=0)
