"""Synthesizer dataset.

Each dataset folder must look like:

    <root>/<dataset_name>/
        data/
            1.wav, 2.wav, 3.wav, ...
        labels/
            1.npy, 2.npy, 3.npy, ...

Files are matched by stem. Labels are 1D float arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class SynthDataset(Dataset):
    """Single-synthesizer dataset (no train/val/test split here; see SynthDataModule)."""

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        sample_indices: list[int] | None = None,
        target_seconds: float = 1.0,
        sample_rate: int = 16_000,
    ):
        self.root = Path(root) / dataset_name
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        self.audio_dir = self.root / "data"
        self.label_dir = self.root / "labels"

        if not self.audio_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {self.audio_dir} and {self.label_dir} to both exist."
            )

        self.target_samples = int(target_seconds * sample_rate)
        self.sample_rate = sample_rate

        # Discover all samples by walking the audio directory.
        all_audio = sorted(self.audio_dir.glob("*.wav"))
        if not all_audio:
            raise FileNotFoundError(f"No .wav files in {self.audio_dir}")

        # Cross-check labels exist for every audio file.
        verified = []
        for audio_path in all_audio:
            stem = audio_path.stem
            label_path = self.label_dir / f"{stem}.npy"
            if label_path.exists():
                verified.append((audio_path, label_path, stem))
        if not verified:
            raise RuntimeError(
                f"Found .wav files but no matching .npy labels in {self.label_dir}"
            )

        # Subset by index list if provided (used for split-based subsetting).
        if sample_indices is not None:
            verified = [verified[i] for i in sample_indices]

        self._items = verified

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        audio_path, label_path, stem = self._items[idx]
        wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        # Convert to mono if needed.
        if wav.ndim > 1:
            wav = wav.mean(axis=-1).astype(np.float32)
        # Resample if necessary (cheap branch — most data is already 16 kHz).
        if sr != self.sample_rate:
            try:
                import librosa  # local import: optional dependency for resampling
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            except ImportError as e:
                raise RuntimeError(
                    f"Audio file {audio_path} is at {sr} Hz but expected "
                    f"{self.sample_rate} Hz, and librosa is not installed."
                ) from e
        # Trim or pad to fixed length.
        if wav.shape[0] >= self.target_samples:
            wav = wav[: self.target_samples]
        else:
            pad = self.target_samples - wav.shape[0]
            wav = np.pad(wav, (0, pad), mode="constant")
        # RMS normalize to avoid scale drift across clips.
        rms = float(np.sqrt(np.mean(wav**2) + 1e-9))
        if rms > 1e-8:
            wav = wav * (0.1 / rms)  # peak-safe target RMS

        theta = np.load(label_path).astype(np.float32).reshape(-1)
        return {
            "wav": torch.from_numpy(wav),
            "theta": torch.from_numpy(theta),
            "stem": stem,
        }
