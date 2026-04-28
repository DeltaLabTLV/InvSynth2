"""STFT computation, normalization, and inversion.

The encoder consumes a log-magnitude STFT. The IMW loss is computed in the
linear-magnitude domain. We expose both representations.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class STFTConfig:
    """STFT parameters matching the paper (Hann 1024, hop 256, 16 kHz audio)."""

    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    eps: float = 1e-7  # for log-magnitude floor
    # Global normalization stats (set by GlobalNormalizer; see below)
    log_mag_mean: float = 0.0
    log_mag_std: float = 1.0


class STFTComputer(torch.nn.Module):
    """Computes STFT magnitude and log-magnitude representations.

    Input:  waveform (B, T) or (T,)
    Output: dict with
        - "linear_mag": (B, F, T') linear magnitude
        - "log_mag": (B, F, T') log magnitude
        - "log_mag_norm": (B, F, T') globally-normalized log magnitude
                          (this is what the encoder sees)
    """

    def __init__(self, cfg: STFTConfig):
        super().__init__()
        self.cfg = cfg
        # Pre-compute the Hann window as a buffer so it follows the module to GPU.
        window = torch.hann_window(cfg.win_length)
        self.register_buffer("window", window, persistent=False)

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        spec = torch.stft(
            wav,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        linear_mag = spec.abs()  # (B, F, T')
        log_mag = torch.log(linear_mag + self.cfg.eps)
        log_mag_norm = (log_mag - self.cfg.log_mag_mean) / (self.cfg.log_mag_std + 1e-9)
        return {
            "linear_mag": linear_mag,
            "log_mag": log_mag,
            "log_mag_norm": log_mag_norm,
        }


def denormalize_log_mag(log_mag_norm: torch.Tensor, cfg: STFTConfig) -> torch.Tensor:
    """Inverse of the global normalization."""
    return log_mag_norm * (cfg.log_mag_std + 1e-9) + cfg.log_mag_mean


def log_to_linear_mag(log_mag: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Convert log-magnitude back to linear magnitude (for the IMW loss)."""
    return torch.exp(log_mag) - eps


# ----------------------------------------------------------------------------
# Global normalization
# ----------------------------------------------------------------------------
def compute_global_log_mag_stats(loader, stft: STFTComputer, max_batches: int = 200):
    """Estimate global log-mag mean/std from training data.

    Runs once at training startup; the resulting (mean, std) are baked into the
    STFTConfig so that all forward passes apply identical normalization.
    """
    n = 0
    running_sum = 0.0
    running_sum_sq = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        wav = batch["wav"]
        log_mag = stft(wav)["log_mag"]
        running_sum += log_mag.sum().item()
        running_sum_sq += (log_mag**2).sum().item()
        n += log_mag.numel()
    if n == 0:
        raise RuntimeError("Empty loader during stats computation.")
    mean = running_sum / n
    var = max(running_sum_sq / n - mean**2, 0.0)
    std = var**0.5
    return float(mean), float(std)
