"""CPU-only, fail-closed audit of the paper configuration and data boundaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invsynth2.study import dataset_profile, load_study_config, run_dataset_dir


def _read_manifest(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["stem", "split"]:
            raise ValueError(f"{path}: expected exactly stem,split columns")
        groups = {"train": [], "val": [], "test": []}
        seen = set()
        for line, row in enumerate(reader, start=2):
            stem = row["stem"].strip()
            split = row["split"].strip().lower()
            if split not in groups:
                raise ValueError(f"{path}:{line}: invalid split {split!r}")
            if not stem or stem in seen:
                raise ValueError(f"{path}:{line}: empty or duplicate stem {stem!r}")
            seen.add(stem)
            groups[split].append(stem)
    return groups


def _audit_schema(path: Path, expected_continuous: int, expected_categorical: int) -> int:
    with path.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle)
    n_total = int(schema["n_total"])
    continuous = [int(value) for value in schema.get("continuous_indices", [])]
    blocks = schema.get("categorical_blocks", [])
    if len(continuous) != expected_continuous:
        raise ValueError(f"{path}: expected {expected_continuous} continuous controls")
    if len(blocks) != expected_categorical:
        raise ValueError(f"{path}: expected {expected_categorical} categorical controls")
    occupied = list(continuous)
    for block in blocks:
        indices = [int(value) for value in block["indices"]]
        if len(indices) < 2:
            raise ValueError(f"{path}: categorical block {block.get('name')!r} has <2 classes")
        occupied.extend(indices)
    if sorted(occupied) != list(range(n_total)):
        raise ValueError(f"{path}: schema does not cover 0..{n_total - 1} exactly once")
    return n_total


def _audit_run(cfg: dict, dataset: str, seed: int, data_root: Path, full_files: bool) -> None:
    import numpy as np
    import soundfile as sf

    profile = dataset_profile(cfg, dataset)
    run_dir = run_dataset_dir(data_root, profile, seed)
    manifest_path = run_dir / cfg["data"]["split_filename"]
    schema_path = run_dir / cfg["data"]["parameter_schema_filename"]
    if not manifest_path.is_file() or not schema_path.is_file():
        raise FileNotFoundError(f"Missing manifest/schema in {run_dir}")
    groups = _read_manifest(manifest_path)
    expected = profile.expected_examples
    expected_sizes = {
        "train": int(expected * cfg["data"]["train_fraction"]),
        "val": int(expected * cfg["data"]["validation_fraction"]),
        "test": expected
        - int(expected * cfg["data"]["train_fraction"])
        - int(expected * cfg["data"]["validation_fraction"]),
    }
    actual_sizes = {key: len(value) for key, value in groups.items()}
    if actual_sizes != expected_sizes:
        raise ValueError(f"{manifest_path}: split sizes {actual_sizes}; expected {expected_sizes}")
    n_total = _audit_schema(
        schema_path,
        profile.expected_continuous_controls,
        profile.expected_categorical_controls,
    )
    stems = groups["train"] + groups["val"] + groups["test"]
    checked = stems if full_files else stems[:16]
    for stem in checked:
        audio_path = run_dir / cfg["data"]["audio_directory"] / f"{stem}.wav"
        label_path = run_dir / cfg["data"]["label_directory"] / f"{stem}.npy"
        if not audio_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing audio/label pair for {dataset}/seed_{seed}/{stem}")
        info = sf.info(str(audio_path))
        if info.samplerate != profile.source_sample_rate:
            raise ValueError(f"{audio_path}: sample rate {info.samplerate}")
        label = np.load(label_path, allow_pickle=False).reshape(-1)
        if label.size != n_total:
            raise ValueError(f"{label_path}: width {label.size}; schema requires {n_total}")


def _audit_reported_table(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise ValueError(f"{path}: expected 24 aggregate rows, got {len(rows)}")
    keys = {(row["configuration"], row["dataset"], row["metric"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError(f"{path}: duplicate configuration/dataset/metric rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/icassp2027.yaml")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument(
        "--full-files", action="store_true",
        help="Inspect every WAV/label pair; default samples 16 per realization.",
    )
    args = parser.parse_args()

    cfg = load_study_config(args.config)
    _audit_reported_table(PROJECT_ROOT / "paper" / "reported_aggregate_results.csv")
    if not args.config_only:
        for dataset in cfg["datasets"]:
            for seed in cfg["seeds"]:
                _audit_run(cfg, dataset, seed, Path(args.data_root), args.full_files)
    mode = "configuration" if args.config_only else "configuration and data"
    print(f"PASS: ICASSP {mode} audit")


if __name__ == "__main__":
    main()
