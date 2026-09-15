"""Orchestrate only the four U-Net configurations reported at ICASSP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from invsynth2.study import (
    PAPER_CONFIGURATIONS,
    dataset_profile,
    load_study_config,
    run_output_dir,
)


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", default="configs/icassp2027.yaml")
    parser.add_argument("--datasets", nargs="+", default=["fm", "dx7", "tal"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--configurations", nargs="+", default=list(PAPER_CONFIGURATIONS),
        choices=PAPER_CONFIGURATIONS,
    )
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_study_config(args.study_config)
    python = sys.executable
    common = ["--study-config", args.study_config]

    for dataset in args.datasets:
        dataset_profile(cfg, dataset)  # validate name/profile before launching jobs
        for seed in args.seeds:
            if seed not in cfg["seeds"]:
                raise ValueError(f"Seed {seed} is outside the paper protocol")
            run_base = run_output_dir(args.run_dir, dataset, seed)
            stats = run_base / "feature_stats.json"
            proxy = run_base / "proxy" / "proxy.pt"
            encoder = run_base / "pretrain" / "encoder.pt"
            identity = [
                "--dataset", dataset,
                "--seed", str(seed),
                "--data-root", args.data_root,
                "--run-dir", args.run_dir,
            ]

            _run(
                [python, "scripts/train_proxy.py", *common, *identity],
                dry_run=args.dry_run,
            )
            _run(
                [
                    python, "scripts/pretrain.py", *common, *identity,
                    "--feature-stats", str(stats),
                ],
                dry_run=args.dry_run,
            )

            for configuration in args.configurations:
                finetune = [
                    python, "scripts/finetune.py", *common, *identity,
                    "--configuration", configuration,
                    "--feature-stats", str(stats),
                    "--proxy-ckpt", str(proxy),
                ]
                if cfg["configurations"][configuration]["pretraining"]:
                    finetune.extend(["--encoder-ckpt", str(encoder)])
                _run(finetune, dry_run=args.dry_run)

                metadata_path = run_base / configuration / "run_metadata.json"
                if args.dry_run:
                    checkpoint = run_base / configuration / "ckpts" / "<best.ckpt>"
                else:
                    with metadata_path.open("r", encoding="utf-8") as handle:
                        checkpoint = Path(json.load(handle)["best_checkpoint"])
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
                _run(
                    [
                        python, "scripts/evaluate.py", *common,
                        "--dataset", dataset,
                        "--seed", str(seed),
                        "--configuration", configuration,
                        "--data-root", args.data_root,
                        "--feature-stats", str(stats),
                        "--finetuned-ckpt", str(checkpoint),
                        "--apply-itf",
                        "--itf-steps", str(cfg["optimization"]["refinement_updates"]),
                        "--itf-lr", str(cfg["optimization"]["refinement_learning_rate"]),
                    ],
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()
