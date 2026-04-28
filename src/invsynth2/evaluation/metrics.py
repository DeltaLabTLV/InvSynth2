"""Evaluation metrics matching the paper's Table 1, 2, and 4.

All metrics expect linear-magnitude STFT spectrograms unless otherwise noted.

Implemented:
- Spec   : ||X̂ - X||_1 + ||X̂ - X||_2^2  (paper Eq. 5; reported as ×100)
- SC     : Spectral Convergence = ||X̂ - X||_F / ||X||_F
- Melspec: MSE between mel spectrograms (×100)
- MFCC   : MSE between MFCCs (×100)
- ACC    : per-parameter accuracy (continuous within-tolerance + categorical argmax)
- Band-decomposed MSE (Low/Mid/High) — used in Table 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Spec, SC
# ----------------------------------------------------------------------------
def spec_metric(pred_lin: torch.Tensor, target_lin: torch.Tensor) -> torch.Tensor:
    """Eq. (5) without α scaling — reported as ×100 in tables."""
    l1 = (pred_lin - target_lin).abs().mean()
    l2 = ((pred_lin - target_lin) ** 2).mean()
    return l1 + l2


def spectral_convergence(pred_lin: torch.Tensor, target_lin: torch.Tensor) -> torch.Tensor:
    """SC = ||X̂ - X||_F / ||X||_F. Computed per-sample then averaged."""
    # Frobenius norm per (B, F, T) sample.
    if pred_lin.dim() == 2:
        pred_lin = pred_lin.unsqueeze(0)
        target_lin = target_lin.unsqueeze(0)
    diff = (pred_lin - target_lin).reshape(pred_lin.shape[0], -1)
    targ = target_lin.reshape(target_lin.shape[0], -1)
    num = torch.norm(diff, dim=-1)
    denom = torch.norm(targ, dim=-1).clamp(min=1e-8)
    return (num / denom).mean()


# ----------------------------------------------------------------------------
# Mel and MFCC
# ----------------------------------------------------------------------------
def _make_mel_filterbank(
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Standard HTK-style mel filterbank, returned as a (n_mels, n_fft//2+1) matrix."""
    if f_max is None:
        f_max = sample_rate / 2.0

    def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freq = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sample_rate / 2, n_freq, device=device)

    m_min = hz_to_mel(torch.tensor(f_min, device=device))
    m_max = hz_to_mel(torch.tensor(f_max, device=device))
    mel_pts = torch.linspace(m_min, m_max, n_mels + 2, device=device)
    hz_pts = mel_to_hz(mel_pts)

    fb = torch.zeros(n_mels, n_freq, device=device)
    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        rising = (fft_freqs - left) / (center - left).clamp(min=1e-8)
        falling = (right - fft_freqs) / (right - center).clamp(min=1e-8)
        fb[i] = torch.maximum(torch.zeros_like(fft_freqs), torch.minimum(rising, falling))
    return fb


def melspec_metric(
    pred_lin: torch.Tensor,
    target_lin: torch.Tensor,
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    n_mels: int = 80,
) -> torch.Tensor:
    """MSE between mel-projected magnitude spectrograms."""
    fb = _make_mel_filterbank(sample_rate, n_fft, n_mels, device=pred_lin.device)
    # (B, F, T) → (B, n_mels, T)
    pred_mel = torch.einsum("mf,bft->bmt", fb, pred_lin)
    targ_mel = torch.einsum("mf,bft->bmt", fb, target_lin)
    return F.mse_loss(pred_mel, targ_mel)


def mfcc_metric(
    pred_lin: torch.Tensor,
    target_lin: torch.Tensor,
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    n_mels: int = 80,
    n_mfcc: int = 13,
) -> torch.Tensor:
    """MSE between MFCCs computed via DCT-II of log-mel."""
    fb = _make_mel_filterbank(sample_rate, n_fft, n_mels, device=pred_lin.device)
    pred_mel = torch.einsum("mf,bft->bmt", fb, pred_lin).clamp(min=1e-7).log()
    targ_mel = torch.einsum("mf,bft->bmt", fb, target_lin).clamp(min=1e-7).log()

    # DCT-II matrix (n_mfcc, n_mels)
    n = torch.arange(n_mels, device=pred_lin.device, dtype=pred_lin.dtype)
    k = torch.arange(n_mfcc, device=pred_lin.device, dtype=pred_lin.dtype)
    dct = torch.cos(
        torch.pi * k.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * n_mels)
    ) * (2.0 / n_mels) ** 0.5
    pred_mfcc = torch.einsum("km,bmt->bkt", dct, pred_mel)
    targ_mfcc = torch.einsum("km,bmt->bkt", dct, targ_mel)
    return F.mse_loss(pred_mfcc, targ_mfcc)


# ----------------------------------------------------------------------------
# Frequency-band decomposition (Table 2)
# ----------------------------------------------------------------------------
@dataclass
class BandConfig:
    """Frequency band edges in Hz. Defaults match the paper (0-1, 1-4, 4-8 kHz)."""

    sample_rate: int = 16_000
    n_fft: int = 1024
    low: tuple[float, float] = (0.0, 1_000.0)
    mid: tuple[float, float] = (1_000.0, 4_000.0)
    high: tuple[float, float] = (4_000.0, 8_000.0)


def band_errors(
    pred_lin: torch.Tensor,
    target_lin: torch.Tensor,
    cfg: BandConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Returns the band-decomposed MSE (×100) used in Table 2.

    The metric is mean squared error between linear-magnitude target and
    reconstructed spectrograms, summed over frequency bins within each band,
    averaged over frames and the test set, scaled by ×100.
    """
    cfg = cfg or BandConfig()
    n_freq = pred_lin.shape[-2]
    fft_freqs = torch.linspace(0, cfg.sample_rate / 2, n_freq, device=pred_lin.device)
    out: dict[str, torch.Tensor] = {}
    for name, (lo, hi) in (("low", cfg.low), ("mid", cfg.mid), ("high", cfg.high)):
        band_mask = (fft_freqs >= lo) & (fft_freqs < hi)
        if band_mask.sum() == 0:
            out[name] = torch.zeros((), device=pred_lin.device)
            continue
        diff_sq = ((pred_lin - target_lin) ** 2)        # (B, F, T)
        band_mse = diff_sq[..., band_mask, :].sum(dim=-2).mean()  # sum over freq, mean over (B, T)
        out[name] = band_mse * 100.0
    return out


# ----------------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------------
@dataclass
class MetricResults:
    spec: float
    sc: float
    melspec: float
    mfcc: float
    band_low: float
    band_mid: float
    band_high: float
    acc: float | None = None

    def to_dict(self, scale_x100: bool = True) -> dict[str, float]:
        scale = 100.0 if scale_x100 else 1.0
        out = {
            "Spec_x100": self.spec * scale,
            "SC": self.sc,
            "Melspec_x100": self.melspec * scale,
            "MFCC_x100": self.mfcc * scale,
            "BandLow_x100": self.band_low,
            "BandMid_x100": self.band_mid,
            "BandHigh_x100": self.band_high,
        }
        if self.acc is not None:
            out["ACC_pct"] = self.acc * 100.0
        return out
