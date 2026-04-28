"""Stage 1: SSL pre-training.

Usage
-----
    python scripts/pretrain.py --config configs/pretrain_transformer.yaml --dataset fm
    python scripts/pretrain.py --config configs/pretrain_unet.yaml        --dataset fm
"""

from __future__ import annotations

# --- Make sibling 'scripts' modules importable when launched as a script ---
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# --------------------------------------------------------------------------

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from invsynth2.data import SynthDataModule
from invsynth2.training import TransformerPretrainModule, UNetPretrainModule
from invsynth2.utils.stft import (
    STFTComputer,
    STFTConfig,
    compute_global_log_mag_stats,
)
from scripts.common import load_config, make_tb_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="fm | dx7 | talnoise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--max-epochs", type=int, default=None, help="override config max_epochs")
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    cfg = load_config(args.config)

    # ---- Data ----
    dm = SynthDataModule(
        root=cfg["dataset"]["root"],
        dataset_name=args.dataset,
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        target_seconds=cfg["dataset"]["target_seconds"],
        sample_rate=cfg["dataset"]["sample_rate"],
        train_frac=cfg["dataset"]["train_frac"],
        val_frac=cfg["dataset"]["val_frac"],
        seed=args.seed,
    )
    dm.setup()

    # ---- Compute & freeze global STFT normalization stats ----
    stft_cfg_init = STFTConfig(
        sample_rate=cfg["stft"]["sample_rate"],
        n_fft=cfg["stft"]["n_fft"],
        hop_length=cfg["stft"]["hop_length"],
        win_length=cfg["stft"]["win_length"],
        eps=cfg["stft"]["eps"],
    )
    probe_stft = STFTComputer(stft_cfg_init)
    mean, std = compute_global_log_mag_stats(dm.train_dataloader(), probe_stft, max_batches=200)
    print(f"[Global STFT] log-mag mean={mean:.4f} std={std:.4f}")
    stft_cfg = STFTConfig(
        sample_rate=stft_cfg_init.sample_rate,
        n_fft=stft_cfg_init.n_fft,
        hop_length=stft_cfg_init.hop_length,
        win_length=stft_cfg_init.win_length,
        eps=stft_cfg_init.eps,
        log_mag_mean=mean, log_mag_std=std,
    )

    # ---- Module ----
    stage = cfg["stage"]
    if stage == "pretrain_transformer":
        module = TransformerPretrainModule(stft_cfg=stft_cfg, **cfg["module"])
    elif stage == "pretrain_unet":
        module = UNetPretrainModule(stft_cfg=stft_cfg, **cfg["module"])
    else:
        raise ValueError(f"Unknown stage {stage!r}")

    # ---- Trainer ----
    run_name = f"{stage}_{args.dataset}_seed{args.seed}"
    logger = make_tb_logger(args.run_dir, run_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(Path(args.run_dir) / run_name / "ckpts"),
            **cfg["callbacks"]["ckpt"],
        ),
        EarlyStopping(**cfg["callbacks"]["early_stop"]),
        LearningRateMonitor(),
    ]
    trainer = L.Trainer(
        max_epochs=args.max_epochs or cfg["trainer"]["max_epochs"],
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        gradient_clip_val=cfg["trainer"]["gradient_clip_val"],
        deterministic=cfg["trainer"].get("deterministic", False),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(Path(args.run_dir) / run_name),
    )

    trainer.fit(module, datamodule=dm)

    # Save the encoder-only state dict for downstream stages.
    out_dir = Path(args.run_dir) / run_name
    encoder_sd = module.get_encoder_state_dict()
    torch.save(
        {
            "encoder_state_dict": encoder_sd,
            "stft_config": stft_cfg.__dict__,
            "stage": stage,
            "dataset": args.dataset,
            "seed": args.seed,
        },
        out_dir / "encoder.pt",
    )
    print(f"Saved encoder weights to {out_dir / 'encoder.pt'}")


if __name__ == "__main__":
    main()
