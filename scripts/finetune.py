"""Run one paper-defined supervised U-Net configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from invsynth2.repro import load_feature_config, make_data_module, parameter_spec_for_run
from invsynth2.study import (
    PAPER_CONFIGURATIONS,
    dataset_profile,
    load_study_config,
    run_output_dir,
)
from invsynth2.training import FineTuneModule
from scripts.common import make_tb_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", default="configs/icassp2027.yaml")
    parser.add_argument("--dataset", required=True, choices=["fm", "dx7", "tal"])
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    parser.add_argument("--configuration", required=True, choices=PAPER_CONFIGURATIONS)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--feature-stats", required=True)
    parser.add_argument("--proxy-ckpt", required=True)
    parser.add_argument("--encoder-ckpt", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_study_config(args.study_config)
    profile = dataset_profile(cfg, args.dataset)
    row = cfg["configurations"][args.configuration]
    L.seed_everything(args.seed, workers=True)

    dm = make_data_module(cfg, profile, args.seed, args.data_root)
    dm.setup()
    spec = parameter_spec_for_run(cfg, profile, args.seed, args.data_root)
    feature_cfg = load_feature_config(
        cfg, profile, args.seed, args.feature_stats
    )

    proxy_checkpoint = torch.load(args.proxy_ckpt, map_location="cpu")
    if proxy_checkpoint.get("dataset") != profile.name or proxy_checkpoint.get("seed") != args.seed:
        raise ValueError("Proxy checkpoint does not belong to this dataset/seed realization")
    if (
        proxy_checkpoint.get("training_scope") != "encoder_train_split"
        or proxy_checkpoint.get("validation_scope") != "encoder_validation_split"
        or proxy_checkpoint.get("test_excluded_from_fitting") is not True
    ):
        raise ValueError(
            "Paper runs require a train-fitted, validation-monitored proxy "
            "whose weights never see encoder-test examples"
        )

    loss_cfg = cfg["loss"]
    opt_cfg = cfg["optimization"]
    module = FineTuneModule(
        stft_cfg=feature_cfg,
        spec=spec,
        encoder_kind="unet",
        loss_mode=row["reconstruction"],
        beta=loss_cfg["beta"],
        alpha1=loss_cfg["alpha_l1"],
        alpha2=loss_cfg["alpha_l2"],
        epsilon=loss_cfg["epsilon"],
        lambda_param=loss_cfg["lambda_parameter"],
        lambda_rec=loss_cfg["lambda_reconstruction"],
        lambda_reg=loss_cfg["lambda_regression"],
        lambda_cls=loss_cfg["lambda_classification"],
        learning_rate_unet=opt_cfg["unet_learning_rate"],
        weight_decay=opt_cfg["weight_decay"],
        max_grad_norm=opt_cfg["gradient_clip_norm"],
    )
    module.load_proxy_weights(proxy_checkpoint["proxy_state_dict"])

    if row["pretraining"]:
        if not args.encoder_ckpt:
            raise ValueError(f"{args.configuration} requires --encoder-ckpt")
        encoder_checkpoint = torch.load(args.encoder_ckpt, map_location="cpu")
        if (
            encoder_checkpoint.get("dataset") != profile.name
            or encoder_checkpoint.get("seed") != args.seed
        ):
            raise ValueError("Encoder checkpoint does not belong to this dataset/seed realization")
        if encoder_checkpoint.get("encoder_updates") != opt_cfg["masked_pretraining_updates"]:
            raise ValueError("Encoder checkpoint does not contain exactly 50,000 pretraining updates")
        module.load_encoder_weights(encoder_checkpoint["encoder_state_dict"], strict=True)
    elif args.encoder_ckpt:
        raise ValueError("supervised_imw must start from random weights; omit --encoder-ckpt")

    output_dir = run_output_dir(args.run_dir, args.dataset, args.seed) / args.configuration
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "ckpts"),
        monitor="val/rec_spec",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    expected_steps = int(row["supervised_updates"])
    max_steps = args.max_steps or expected_steps
    trainer = L.Trainer(
        max_epochs=-1,
        max_steps=max_steps,
        accelerator="auto",
        devices=1,
        precision=opt_cfg["precision"],
        gradient_clip_val=opt_cfg["gradient_clip_norm"],
        deterministic=opt_cfg["deterministic"],
        callbacks=[checkpoint],
        logger=make_tb_logger(output_dir, "logs"),
        default_root_dir=str(output_dir),
    )
    trainer.fit(module, datamodule=dm)
    if trainer.global_step != max_steps:
        raise RuntimeError(f"Expected {max_steps} supervised updates, got {trainer.global_step}")

    metadata = {
        "schema_version": 1,
        "dataset": profile.name,
        "seed": args.seed,
        "configuration": args.configuration,
        "display_name": row["display_name"],
        "pretrained": bool(row["pretraining"]),
        "masked_pretraining_updates": (
            opt_cfg["masked_pretraining_updates"] if row["pretraining"] else 0
        ),
        "supervised_updates": trainer.global_step,
        "reconstruction": row["reconstruction"],
        "best_checkpoint": checkpoint.best_model_path,
        "proxy_training_scope": proxy_checkpoint["training_scope"],
        "proxy_validation_scope": proxy_checkpoint["validation_scope"],
        "proxy_test_excluded_from_fitting": proxy_checkpoint[
            "test_excluded_from_fitting"
        ],
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
