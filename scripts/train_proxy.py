"""Stage 2: Train the differentiable synthesizer proxy P.

Usage:
    python scripts/train_proxy.py --config configs/proxy.yaml --dataset fm
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
from invsynth2.training import ProxyTrainModule
from invsynth2.utils.parameters import DEFAULT_SPECS
from invsynth2.utils.stft import (
    STFTComputer,
    STFTConfig,
    compute_global_log_mag_stats,
)
from scripts.common import load_config, make_tb_logger


def _spec_for(dataset: str):
    if dataset in DEFAULT_SPECS:
        return DEFAULT_SPECS[dataset]
    if dataset == "talnoise":
        return DEFAULT_SPECS["tal"]
    raise ValueError(f"No ParameterSpec for dataset={dataset!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    cfg = load_config(args.config)

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

    stft_cfg_init = STFTConfig(
        sample_rate=cfg["stft"]["sample_rate"],
        n_fft=cfg["stft"]["n_fft"],
        hop_length=cfg["stft"]["hop_length"],
        win_length=cfg["stft"]["win_length"],
        eps=cfg["stft"]["eps"],
    )
    mean, std = compute_global_log_mag_stats(
        dm.train_dataloader(), STFTComputer(stft_cfg_init), max_batches=200,
    )
    stft_cfg = STFTConfig(
        sample_rate=stft_cfg_init.sample_rate,
        n_fft=stft_cfg_init.n_fft,
        hop_length=stft_cfg_init.hop_length,
        win_length=stft_cfg_init.win_length,
        eps=stft_cfg_init.eps,
        log_mag_mean=mean, log_mag_std=std,
    )

    spec = _spec_for(args.dataset)

    module = ProxyTrainModule(
        stft_cfg=stft_cfg,
        spec=spec,
        out_freq=cfg["module"]["out_freq"],
        out_time=cfg["module"]["out_time"],
        learning_rate=cfg["module"]["learning_rate"],
        weight_decay=cfg["module"]["weight_decay"],
    )

    run_name = f"proxy_{args.dataset}_seed{args.seed}"
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

    # Save proxy state dict for downstream use.
    out_dir = Path(args.run_dir) / run_name
    torch.save(
        {
            "proxy_state_dict": module.proxy.state_dict(),
            "stft_config": stft_cfg.__dict__,
            "spec": spec.__dict__,
            "dataset": args.dataset,
            "seed": args.seed,
        },
        out_dir / "proxy.pt",
    )
    print(f"Saved proxy weights to {out_dir / 'proxy.pt'}")


if __name__ == "__main__":
    main()
