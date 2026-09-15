"""Paper-locked spectrogram construction and invertible feature scaling.

The name ``STFTConfig`` is retained for checkpoint compatibility, although the
DX7 profile also applies a mel projection. All logarithms, inverse transforms,
reciprocals, and reductions in this module are explicitly evaluated in FP32.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class STFTConfig:
    sample_rate: int = 16_000
    source_num_samples: int = 16_000
    analysis_num_samples: int = 16_384
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    center: bool = True
    pad_mode: str = "constant"
    hann_periodic: bool = False
    n_mels: int | None = None
    mel_fmin: float = 0.0
    mel_fmax: float | None = None
    magnitude_floor: float = 1e-6
    db_floor: float = -120.0
    feature_min: float | None = None
    feature_max: float | None = None
    expected_freq_bins: int | None = None
    expected_frames: int | None = None
    # Legacy z-score fields are accepted when loading an older checkpoint but
    # are not used by the ICASSP entry points.
    eps: float = 1e-7
    log_mag_mean: float = 0.0
    log_mag_std: float = 1.0

    @property
    def output_freq_bins(self) -> int:
        return self.n_mels if self.n_mels is not None else self.n_fft // 2 + 1

    @property
    def output_frames(self) -> int:
        if self.center:
            return 1 + self.analysis_num_samples // self.hop_length
        return 1 + (self.analysis_num_samples - self.n_fft) // self.hop_length

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(freq_mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, freq_mel / 2595.0) - 1.0)


def make_mel_filterbank(cfg: STFTConfig, device: torch.device) -> torch.Tensor:
    if cfg.n_mels is None:
        raise ValueError("A mel filterbank requires n_mels")
    fmax = cfg.mel_fmax if cfg.mel_fmax is not None else cfg.sample_rate / 2.0
    if not 0.0 <= cfg.mel_fmin < fmax <= cfg.sample_rate / 2.0:
        raise ValueError(
            f"Invalid mel range [{cfg.mel_fmin}, {fmax}] for Fs={cfg.sample_rate}"
        )
    fft_freqs = torch.linspace(
        0.0, cfg.sample_rate / 2.0, cfg.n_fft // 2 + 1,
        dtype=torch.float32, device=device,
    )
    low = _hz_to_mel(torch.tensor(cfg.mel_fmin, dtype=torch.float32, device=device))
    high = _hz_to_mel(torch.tensor(fmax, dtype=torch.float32, device=device))
    mel_points = torch.linspace(low, high, cfg.n_mels + 2, device=device)
    hz_points = _mel_to_hz(mel_points)
    left = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    right = hz_points[2:, None]
    rising = (fft_freqs[None, :] - left) / (center - left).clamp_min(1e-12)
    falling = (right - fft_freqs[None, :]) / (right - center).clamp_min(1e-12)
    return torch.minimum(rising, falling).clamp_min(0.0)


class STFTComputer(torch.nn.Module):
    """Compute native magnitudes, dB features, and normalized encoder inputs."""

    def __init__(self, cfg: STFTConfig):
        super().__init__()
        self.cfg = cfg
        window = torch.hann_window(cfg.win_length, periodic=cfg.hann_periodic)
        self.register_buffer("window", window, persistent=False)
        if cfg.expected_freq_bins is not None and cfg.output_freq_bins != cfg.expected_freq_bins:
            raise ValueError(
                f"Frequency profile produces {cfg.output_freq_bins} bins; "
                f"expected {cfg.expected_freq_bins}"
            )
        if cfg.expected_frames is not None and cfg.output_frames != cfg.expected_frames:
            raise ValueError(
                f"Frame profile produces {cfg.output_frames} frames; expected {cfg.expected_frames}"
            )

    def _fixed_analysis_support(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[-1] > self.cfg.source_num_samples:
            wav = wav[..., : self.cfg.source_num_samples]
        elif wav.shape[-1] < self.cfg.source_num_samples:
            wav = F.pad(wav, (0, self.cfg.source_num_samples - wav.shape[-1]))
        if self.cfg.analysis_num_samples > wav.shape[-1]:
            wav = F.pad(wav, (0, self.cfg.analysis_num_samples - wav.shape[-1]))
        elif self.cfg.analysis_num_samples < wav.shape[-1]:
            wav = wav[..., : self.cfg.analysis_num_samples]
        return wav

    def native_magnitude(self, wav: torch.Tensor) -> torch.Tensor:
        """Return linear-frequency or mel-bin magnitude in FP32."""

        wav = self._fixed_analysis_support(wav)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav32 = wav.float()
            spec = torch.stft(
                wav32,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=self.window.float(),
                center=self.cfg.center,
                return_complex=True,
                pad_mode=self.cfg.pad_mode,
            )
            magnitude = spec.abs()
            if self.cfg.n_mels is not None:
                filterbank = make_mel_filterbank(self.cfg, magnitude.device)
                magnitude = torch.einsum("mf,bft->bmt", filterbank, magnitude)
        self._assert_shape(magnitude)
        return magnitude

    def raw_db(self, wav: torch.Tensor) -> torch.Tensor:
        magnitude = self.native_magnitude(wav)
        with torch.autocast(device_type=magnitude.device.type, enabled=False):
            db = 20.0 * torch.log10(magnitude.float().clamp_min(self.cfg.magnitude_floor))
            db = db.clamp_min(self.cfg.db_floor)
        return db

    def normalize_db(self, db: torch.Tensor) -> torch.Tensor:
        if self.cfg.feature_min is None or self.cfg.feature_max is None:
            raise ValueError(
                "feature_min/feature_max are required. Compute them on the complete "
                "run-specific dataset and reload the saved feature_stats.json."
            )
        span = self.cfg.feature_max - self.cfg.feature_min
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError(f"Invalid feature range: {self.cfg.feature_min}, {self.cfg.feature_max}")
        with torch.autocast(device_type=db.device.type, enabled=False):
            return -1.0 + 2.0 * (db.float() - self.cfg.feature_min) / span

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        native = self.native_magnitude(wav)
        with torch.autocast(device_type=native.device.type, enabled=False):
            db = 20.0 * torch.log10(native.float().clamp_min(self.cfg.magnitude_floor))
            db = db.clamp_min(self.cfg.db_floor)
            normalized = self.normalize_db(db)
        return {
            "native_mag": native,
            "feature_db": db,
            "feature_norm": normalized,
            # Backward-compatible names used by the inherited training modules.
            "linear_mag": native,
            "log_mag": db,
            "log_mag_norm": normalized,
        }

    def _assert_shape(self, tensor: torch.Tensor) -> None:
        expected_f = self.cfg.expected_freq_bins
        expected_t = self.cfg.expected_frames
        if expected_f is not None and tensor.shape[-2] != expected_f:
            raise RuntimeError(f"Expected {expected_f} frequency bins, got {tensor.shape[-2]}")
        if expected_t is not None and tensor.shape[-1] != expected_t:
            raise RuntimeError(f"Expected {expected_t} frames, got {tensor.shape[-1]}")


def denormalize_log_mag(feature_norm: torch.Tensor, cfg: STFTConfig) -> torch.Tensor:
    """Undo the run-specific affine normalization, returning dB features."""

    if cfg.feature_min is None or cfg.feature_max is None:
        # Compatibility with checkpoints produced by the earlier z-score code.
        return feature_norm.float() * (cfg.log_mag_std + 1e-9) + cfg.log_mag_mean
    with torch.autocast(device_type=feature_norm.device.type, enabled=False):
        return (
            (feature_norm.float() + 1.0) * (cfg.feature_max - cfg.feature_min) / 2.0
            + cfg.feature_min
        )


def log_to_linear_mag(feature_db: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Convert amplitude dB to nonnegative native magnitude in FP32."""

    del eps  # retained in the signature for older call sites
    with torch.autocast(device_type=feature_db.device.type, enabled=False):
        return torch.pow(10.0, feature_db.float() / 20.0)


def compute_feature_extrema(loader, computer: STFTComputer) -> tuple[float, float]:
    """Compute exact dB extrema over the complete run-specific dataset."""

    feature_min = float("inf")
    feature_max = float("-inf")
    count = 0
    for batch in loader:
        db = computer.raw_db(batch["wav"])
        feature_min = min(feature_min, float(db.amin().item()))
        feature_max = max(feature_max, float(db.amax().item()))
        count += db.numel()
    if count == 0 or not math.isfinite(feature_min) or not math.isfinite(feature_max):
        raise RuntimeError("Cannot estimate feature extrema from an empty/non-finite loader")
    return feature_min, feature_max


def compute_global_log_mag_stats(loader, stft: STFTComputer, max_batches: int = 200):
    """Legacy mean/std helper retained for old checkpoints only."""

    n = 0
    running_sum = 0.0
    running_sum_sq = 0.0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        values = stft.raw_db(batch["wav"])
        running_sum += values.sum().item()
        running_sum_sq += values.square().sum().item()
        n += values.numel()
    if n == 0:
        raise RuntimeError("Empty loader during feature-statistics computation")
    mean = running_sum / n
    variance = max(running_sum_sq / n - mean**2, 0.0)
    return float(mean), float(variance**0.5)
