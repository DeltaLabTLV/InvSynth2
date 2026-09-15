"""Fit one fresh proxy without exposing encoder-test examples to its weights."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from invsynth2.repro import (
    base_feature_config,
    default_stats_path,
    load_feature_config,
    make_data_module,
    parameter_spec_for_run,
    save_feature_stats,
)
from invsynth2.study import dataset_profile, load_study_config, run_dataset_dir, run_output_dir
from invsynth2.training import ProxyTrainModule
from invsynth2.utils.stft import STFTComputer, compute_feature_extrema
from scripts.common import make_tb_logger


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", default="configs/icassp2027.yaml")
    parser.add_argument("--dataset", required=True, choices=["fm", "dx7", "tal"])
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_study_config(args.study_config)
    profile = dataset_profile(cfg, args.dataset)
    L.seed_everything(args.seed, workers=True)

    dm = make_data_module(cfg, profile, args.seed, args.data_root)
    dm.setup()
    spec = parameter_spec_for_run(cfg, profile, args.seed, args.data_root)

    # Normalization statistics follow the paper's complete-realization scope.
    # This preprocessing access is distinct from proxy fitting: learned proxy
    # weights use train examples, validation selects the checkpoint, and the
    # encoder-test partition is never passed to the trainer.
    raw_feature_cfg = base_feature_config(cfg, profile)
    raw_computer = STFTComputer(raw_feature_cfg)
    feature_min, feature_max = compute_feature_extrema(
        dm.full_dataloader(shuffle=False), raw_computer
    )
    stats_path = default_stats_path(args.run_dir, args.dataset, args.seed)
    save_feature_stats(
        stats_path,
        profile=profile,
        seed=args.seed,
        feature_min=feature_min,
        feature_max=feature_max,
    )
    feature_cfg = load_feature_config(cfg, profile, args.seed, stats_path)

    module = ProxyTrainModule(
        stft_cfg=feature_cfg,
        spec=spec,
        out_freq=profile.expected_freq_bins,
        out_time=profile.expected_frames,
        learning_rate=cfg["optimization"]["proxy_learning_rate"],
        weight_decay=cfg["optimization"]["weight_decay"],
    )

    output_dir = run_output_dir(args.run_dir, args.dataset, args.seed) / "proxy"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "ckpts"),
        monitor="val/proxy_mse",
        mode="min",
        save_last=True,
        save_top_k=1,
    )
    trainer = L.Trainer(
        max_epochs=args.max_epochs or cfg["optimization"]["proxy_max_epochs"],
        accelerator="auto",
        devices=1,
        precision=cfg["optimization"]["precision"],
        gradient_clip_val=cfg["optimization"]["gradient_clip_norm"],
        deterministic=cfg["optimization"]["deterministic"],
        callbacks=[checkpoint],
        logger=make_tb_logger(output_dir, "logs"),
        default_root_dir=str(output_dir),
    )
    trainer.fit(
        module,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    # Export the best validation checkpoint rather than the final epoch. The
    # historical stopping/checkpoint rule is not retained, so this is the
    # release implementation's explicit and auditable selection rule.
    if not checkpoint.best_model_path:
        raise RuntimeError("Proxy training did not produce a validation checkpoint")
    best = torch.load(checkpoint.best_model_path, map_location="cpu")
    module.load_state_dict(best["state_dict"])

    manifest = run_dataset_dir(args.data_root, profile, args.seed) / cfg["data"]["split_filename"]
    torch.save(
        {
            "proxy_state_dict": module.proxy.state_dict(),
            "stft_config": feature_cfg.to_dict(),
            "dataset": profile.name,
            "seed": args.seed,
            "training_scope": "encoder_train_split",
            "validation_scope": "encoder_validation_split",
            "test_excluded_from_fitting": True,
            "feature_statistics_scope": "complete_run_dataset",
            "n_train_examples": len(dm.train_set),
            "n_validation_examples": len(dm.val_set),
            "n_test_examples": len(dm.test_set),
            "best_validation_checkpoint": checkpoint.best_model_path,
            "split_manifest_sha256": _sha256(manifest),
            "parameter_schema": {
                "n_total": spec.n_total,
                "n_continuous": spec.n_continuous,
                "n_categorical": spec.n_categorical,
            },
            "proxy_max_epochs": trainer.max_epochs,
        },
        output_dir / "proxy.pt",
    )
    print(f"Saved feature statistics to {stats_path}")
    print(f"Saved run-specific proxy to {output_dir / 'proxy.pt'}")


if __name__ == "__main__":
    main()
