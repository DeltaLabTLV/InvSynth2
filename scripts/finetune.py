"""Stage 3: Fine-tune for synthesizer inversion.

Usage examples
--------------
# Full Transformer with IMW loss
python scripts/finetune.py --config configs/finetune_transformer.yaml \
    --dataset fm \
    --encoder-ckpt runs/pretrain_transformer_fm_seed42/encoder.pt \
    --proxy-ckpt   runs/proxy_fm_seed42/proxy.pt

# Ablation: w/o IMW (= spec_only)
python scripts/finetune.py --config configs/finetune_transformer.yaml \
    --dataset fm \
    --encoder-ckpt ... --proxy-ckpt ... \
    --loss spec_only

# Ablation: w/o SSL (random encoder init; do not load --encoder-ckpt)
python scripts/finetune.py --config configs/finetune_transformer.yaml \
    --dataset fm \
    --proxy-ckpt ... \
    --skip-pretrain
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
from invsynth2.training import FineTuneModule
from invsynth2.utils.parameters import DEFAULT_SPECS, ParameterSpec
from invsynth2.utils.stft import STFTConfig
from scripts.common import load_config, make_tb_logger


def _spec_for(dataset: str) -> ParameterSpec:
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
    parser.add_argument("--encoder-ckpt", type=str, default=None,
                        help="Path to Stage-1 encoder.pt (omit with --skip-pretrain).")
    parser.add_argument("--proxy-ckpt", type=str, required=True,
                        help="Path to Stage-2 proxy.pt.")
    parser.add_argument("--loss", type=str, default=None, choices=["imw", "log_spec", "spec_only"],
                        help="Override the loss mode in the config (for ablation runs).")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Train encoder from random init (the 'w/o SSL' ablation row).")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    cfg = load_config(args.config)
    if args.loss is not None:
        cfg["module"]["loss_mode"] = args.loss

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

    # ---- Recover frozen STFT config from the proxy checkpoint ----
    proxy_ckpt = torch.load(args.proxy_ckpt, map_location="cpu")
    stft_cfg = STFTConfig(**proxy_ckpt["stft_config"])

    spec = _spec_for(args.dataset)

    module = FineTuneModule(stft_cfg=stft_cfg, spec=spec, **cfg["module"])
    module.load_proxy_weights(proxy_ckpt["proxy_state_dict"])

    if not args.skip_pretrain:
        if args.encoder_ckpt is None:
            raise ValueError("Provide --encoder-ckpt or pass --skip-pretrain.")
        enc_ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
        # Prefer the saved encoder-only state dict from pretrain.py:
        sd = enc_ckpt.get("encoder_state_dict", enc_ckpt)
        module.load_encoder_weights(sd, strict=False)
    else:
        print("=== Running in 'w/o SSL' ablation mode: encoder is randomly initialized ===")

    run_name = (
        f"finetune_{cfg['module']['encoder_kind']}"
        f"_{args.dataset}"
        f"_{cfg['module']['loss_mode']}"
        f"{'_noSSL' if args.skip_pretrain else ''}"
        f"_seed{args.seed}"
    )
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
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
