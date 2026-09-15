"""Run the 50,000-update masked-reconstruction U-Net stage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from invsynth2.repro import (
    default_stats_path,
    load_feature_config,
    make_data_module,
)
from invsynth2.study import dataset_profile, load_study_config, run_output_dir
from invsynth2.training import UNetPretrainModule
from scripts.common import make_tb_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", default="configs/icassp2027.yaml")
    parser.add_argument("--dataset", required=True, choices=["fm", "dx7", "tal"])
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--feature-stats", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_study_config(args.study_config)
    profile = dataset_profile(cfg, args.dataset)
    L.seed_everything(args.seed, workers=True)
    dm = make_data_module(cfg, profile, args.seed, args.data_root)
    dm.setup()

    stats_path = Path(args.feature_stats) if args.feature_stats else default_stats_path(
        args.run_dir, args.dataset, args.seed
    )
    feature_cfg = load_feature_config(cfg, profile, args.seed, stats_path)
    module = UNetPretrainModule(
        stft_cfg=feature_cfg,
        learning_rate=cfg["optimization"]["unet_learning_rate"],
        weight_decay=cfg["optimization"]["weight_decay"],
        mask_center_ratio=cfg["masking"]["center_ratio"],
        mask_region_size=cfg["masking"]["region_size"],
    )

    output_dir = run_output_dir(args.run_dir, args.dataset, args.seed) / "pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "ckpts"),
        monitor="val/mae_recon",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    max_steps = args.max_steps or cfg["optimization"]["masked_pretraining_updates"]
    trainer = L.Trainer(
        max_epochs=-1,
        max_steps=max_steps,
        accelerator="auto",
        devices=1,
        precision=cfg["optimization"]["precision"],
        gradient_clip_val=cfg["optimization"]["gradient_clip_norm"],
        deterministic=cfg["optimization"]["deterministic"],
        callbacks=[checkpoint],
        logger=make_tb_logger(output_dir, "logs"),
        default_root_dir=str(output_dir),
    )
    trainer.fit(module, datamodule=dm)
    if trainer.global_step != max_steps:
        raise RuntimeError(f"Expected {max_steps} pretraining updates, got {trainer.global_step}")

    torch.save(
        {
            "encoder_state_dict": module.get_encoder_state_dict(),
            "stft_config": feature_cfg.to_dict(),
            "stage": "masked_reconstruction",
            "dataset": profile.name,
            "seed": args.seed,
            "encoder_updates": trainer.global_step,
            "mask_center_ratio": cfg["masking"]["center_ratio"],
            "mask_region_size": cfg["masking"]["region_size"],
            "loss_support": cfg["masking"]["loss_support"],
        },
        output_dir / "encoder.pt",
    )
    print(f"Saved U-Net encoder to {output_dir / 'encoder.pt'}")


if __name__ == "__main__":
    main()
