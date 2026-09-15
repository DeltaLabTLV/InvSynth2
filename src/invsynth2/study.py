"""Paper-locked ICASSP study configuration and path helpers.

This module deliberately has no PyTorch dependency so that dataset manifests
and configuration invariants can be audited on a CPU-only machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PAPER_CONFIGURATIONS = (
    "masked_imw",
    "masked_spectral",
    "masked_log",
    "supervised_imw",
)


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    directory: str
    expected_examples: int
    expected_continuous_controls: int
    expected_categorical_controls: int
    source_sample_rate: int
    source_num_samples: int
    analysis_num_samples: int
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int | None
    mel_fmin: float
    mel_fmax: float | None
    expected_freq_bins: int
    expected_frames: int

    @property
    def computed_freq_bins(self) -> int:
        return self.n_mels if self.n_mels is not None else self.n_fft // 2 + 1

    @property
    def computed_frames(self) -> int:
        # torch.stft with center=True gives 1 + floor(L / hop) frames.
        return 1 + self.analysis_num_samples // self.hop_length


def load_study_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_study_config(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def dataset_profile(cfg: dict[str, Any], dataset: str) -> DatasetProfile:
    key = canonical_dataset_name(dataset)
    try:
        values = cfg["datasets"][key]
    except KeyError as exc:
        raise KeyError(f"Dataset {dataset!r} is not defined in the study config") from exc
    return DatasetProfile(name=key, **values)


def canonical_dataset_name(name: str) -> str:
    aliases = {"talnoise": "tal", "tal_noisemaker": "tal", "yamaha_dx7": "dx7"}
    return aliases.get(name.lower(), name.lower())


def run_dataset_dir(data_root: str | Path, profile: DatasetProfile, seed: int) -> Path:
    return Path(data_root) / profile.directory / f"seed_{seed}"


def run_output_dir(run_root: str | Path, dataset: str, seed: int) -> Path:
    return Path(run_root) / canonical_dataset_name(dataset) / f"seed_{seed}"


def validate_study_config(cfg: dict[str, Any]) -> None:
    if cfg.get("schema_version") != 1:
        raise ValueError("Expected study schema_version: 1")
    if tuple(cfg.get("seeds", ())) != (0, 1, 2, 3, 4):
        raise ValueError("The paper protocol requires seeds [0, 1, 2, 3, 4]")
    if tuple(cfg.get("configurations", {}).keys()) != PAPER_CONFIGURATIONS:
        raise ValueError(
            "The ICASSP configuration order must be exactly "
            f"{PAPER_CONFIGURATIONS!r}"
        )

    opt = cfg["optimization"]
    if opt["masked_pretraining_updates"] != 50_000:
        raise ValueError("Masked pretraining must use 50,000 updates")
    if opt["pretrained_supervised_updates"] != 10_000:
        raise ValueError("Pretrained configurations must use 10,000 supervised updates")
    if opt["supervised_only_updates"] != 60_000:
        raise ValueError("The supervised-only configuration must use 60,000 updates")
    if opt["refinement_updates"] != 100:
        raise ValueError("All paper rows must use 100 refinement updates")

    for name, values in cfg["datasets"].items():
        profile = DatasetProfile(name=name, **values)
        if profile.computed_freq_bins != profile.expected_freq_bins:
            raise ValueError(
                f"{name}: configured frequency dimension {profile.computed_freq_bins} "
                f"!= expected {profile.expected_freq_bins}"
            )
        if profile.computed_frames != profile.expected_frames:
            raise ValueError(
                f"{name}: centered-frame count {profile.computed_frames} "
                f"!= expected {profile.expected_frames}"
            )
        if profile.analysis_num_samples < profile.source_num_samples:
            raise ValueError(f"{name}: analysis support cannot truncate the source waveform")

    fractions = (
        cfg["data"]["train_fraction"],
        cfg["data"]["validation_fraction"],
        cfg["data"]["test_fraction"],
    )
    if abs(sum(fractions) - 1.0) > 1e-12:
        raise ValueError(f"Encoder split fractions must sum to 1, got {fractions!r}")
