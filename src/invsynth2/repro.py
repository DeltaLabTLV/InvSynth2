"""Factories shared by the ICASSP command-line entry points."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from invsynth2.data import RunDataModule
from invsynth2.study import DatasetProfile, dataset_profile, run_dataset_dir, run_output_dir
from invsynth2.utils.parameters import ParameterSpec, load_parameter_spec
from invsynth2.utils.stft import STFTConfig


def make_data_module(
    study_cfg: dict[str, Any],
    profile: DatasetProfile,
    seed: int,
    data_root: str | Path,
    *,
    num_workers: int | None = None,
) -> RunDataModule:
    data_cfg = study_cfg["data"]
    return RunDataModule(
        run_dir=run_dataset_dir(data_root, profile, seed),
        source_num_samples=profile.source_num_samples,
        sample_rate=profile.source_sample_rate,
        batch_size=study_cfg["optimization"]["batch_size"],
        num_workers=data_cfg["num_workers"] if num_workers is None else num_workers,
        split_filename=data_cfg["split_filename"],
        audio_directory=data_cfg["audio_directory"],
        label_directory=data_cfg["label_directory"],
        expected_examples=profile.expected_examples,
    )


def parameter_spec_for_run(
    study_cfg: dict[str, Any],
    profile: DatasetProfile,
    seed: int,
    data_root: str | Path,
) -> ParameterSpec:
    path = (
        run_dataset_dir(data_root, profile, seed)
        / study_cfg["data"]["parameter_schema_filename"]
    )
    spec = load_parameter_spec(path, expected_name=profile.name)
    if spec.n_continuous != profile.expected_continuous_controls:
        raise ValueError(
            f"{path}: {spec.n_continuous} continuous controls; "
            f"paper profile expects {profile.expected_continuous_controls}"
        )
    if spec.n_categorical != profile.expected_categorical_controls:
        raise ValueError(
            f"{path}: {spec.n_categorical} categorical controls; "
            f"paper profile expects {profile.expected_categorical_controls}"
        )
    return spec


def base_feature_config(
    study_cfg: dict[str, Any], profile: DatasetProfile
) -> STFTConfig:
    feature_cfg = study_cfg["features"]
    if feature_cfg["log_scale"] != "db_amplitude":
        raise ValueError("The paper artifact supports amplitude dB only")
    if feature_cfg["normalization"] != "min_max_to_minus_one_one":
        raise ValueError("The paper artifact supports min/max normalization only")
    if feature_cfg["stft_pad_mode"] != "constant":
        raise ValueError("The paper artifact requires zero boundary padding")
    return STFTConfig(
        sample_rate=profile.source_sample_rate,
        source_num_samples=profile.source_num_samples,
        analysis_num_samples=profile.analysis_num_samples,
        n_fft=profile.n_fft,
        hop_length=profile.hop_length,
        win_length=profile.win_length,
        center=feature_cfg["center"],
        pad_mode=feature_cfg["stft_pad_mode"],
        hann_periodic=feature_cfg["hann_periodic"],
        n_mels=profile.n_mels,
        mel_fmin=profile.mel_fmin,
        mel_fmax=profile.mel_fmax,
        magnitude_floor=feature_cfg["magnitude_floor"],
        db_floor=feature_cfg["db_floor"],
        expected_freq_bins=profile.expected_freq_bins,
        expected_frames=profile.expected_frames,
        eps=study_cfg["loss"]["epsilon"],
    )


def save_feature_stats(
    path: str | Path,
    *,
    profile: DatasetProfile,
    seed: int,
    feature_min: float,
    feature_max: float,
    source: str = "complete_run_dataset",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": profile.name,
        "seed": int(seed),
        "scope": source,
        "feature_min_db": float(feature_min),
        "feature_max_db": float(feature_max),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_feature_config(
    study_cfg: dict[str, Any],
    profile: DatasetProfile,
    seed: int,
    stats_path: str | Path,
) -> STFTConfig:
    stats_path = Path(stats_path)
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    if stats.get("dataset") != profile.name or int(stats.get("seed", -1)) != int(seed):
        raise ValueError(
            f"{stats_path}: expected dataset={profile.name!r}, seed={seed}; got "
            f"dataset={stats.get('dataset')!r}, seed={stats.get('seed')!r}"
        )
    if stats.get("scope") != study_cfg["features"]["statistics_scope"]:
        raise ValueError(
            f"{stats_path}: statistics scope {stats.get('scope')!r} does not match "
            f"{study_cfg['features']['statistics_scope']!r}"
        )
    return replace(
        base_feature_config(study_cfg, profile),
        feature_min=float(stats["feature_min_db"]),
        feature_max=float(stats["feature_max_db"]),
    )


def default_stats_path(run_root: str | Path, dataset: str, seed: int) -> Path:
    return run_output_dir(run_root, dataset, seed) / "feature_stats.json"
