"""Orchestrate the full ablation grid from the paper.

For each (dataset × encoder × loss × pretrain) cell, runs all 5 seeds of:
    Stage 1: SSL pre-training (skipped if --skip-pretrain selected)
    Stage 2: Proxy training (one-time per dataset; reused across cells)
    Stage 3: Fine-tuning
    Stage 4: ITF (optional)
    Stage 5: Evaluation

Runs are launched as subprocesses to keep memory pressure low and to allow
restarting individual cells without losing previous progress. Use a job
scheduler (or a simple `for` loop) for parallel execution across multiple GPUs.

Usage:
    python scripts/run_full_ablation.py --datasets fm dx7 talnoise --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import subprocess
from itertools import product
from pathlib import Path


ENCODERS = ["transformer", "unet"]
LOSSES = ["imw", "log_spec", "spec_only"]
SSL_OPTIONS = [True, False]   # True = use pretrain; False = w/o-SSL ablation row


def _run(cmd: list[str], dry_run: bool):
    print(">> " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=["fm", "dx7", "talnoise"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--encoders", type=str, nargs="+", default=ENCODERS, choices=ENCODERS)
    parser.add_argument("--losses", type=str, nargs="+", default=LOSSES, choices=LOSSES)
    parser.add_argument("--include-noSSL", action="store_true",
                        help="Also run the w/o-SSL ablation row for each (encoder, loss).")
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--apply-itf", action="store_true",
                        help="Run ITF + evaluate after each fine-tuning.")
    parser.add_argument("--max-pretrain-epochs", type=int, default=None)
    parser.add_argument("--max-finetune-epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: pretrain encoders (one per dataset × encoder × seed) ----
    for ds, enc, seed in product(args.datasets, args.encoders, args.seeds):
        cfg = f"configs/pretrain_{enc}.yaml"
        cmd = ["python", "scripts/pretrain.py",
               "--config", cfg,
               "--dataset", ds,
               "--seed", str(seed),
               "--run-dir", str(run_dir)]
        if args.max_pretrain_epochs is not None:
            cmd += ["--max-epochs", str(args.max_pretrain_epochs)]
        _run(cmd, args.dry_run)

    # ---- Stage 2: train proxy (one per dataset × seed) ----
    for ds, seed in product(args.datasets, args.seeds):
        cmd = ["python", "scripts/train_proxy.py",
               "--config", "configs/proxy.yaml",
               "--dataset", ds,
               "--seed", str(seed),
               "--run-dir", str(run_dir)]
        _run(cmd, args.dry_run)

    # ---- Stage 3: fine-tune for every (encoder × loss × ±SSL × dataset × seed) ----
    ssl_options = [True, False] if args.include_noSSL else [True]
    for ds, enc, loss, use_ssl, seed in product(
        args.datasets, args.encoders, args.losses, ssl_options, args.seeds
    ):
        cfg = f"configs/finetune_{enc}.yaml"
        encoder_ckpt = run_dir / f"pretrain_{enc}_{ds}_seed{seed}" / "encoder.pt"
        proxy_ckpt = run_dir / f"proxy_{ds}_seed{seed}" / "proxy.pt"
        cmd = ["python", "scripts/finetune.py",
               "--config", cfg,
               "--dataset", ds,
               "--seed", str(seed),
               "--proxy-ckpt", str(proxy_ckpt),
               "--loss", loss,
               "--run-dir", str(run_dir)]
        if use_ssl:
            cmd += ["--encoder-ckpt", str(encoder_ckpt)]
        else:
            cmd += ["--skip-pretrain"]
        if args.max_finetune_epochs is not None:
            cmd += ["--max-epochs", str(args.max_finetune_epochs)]
        _run(cmd, args.dry_run)

        # ---- Stage 5: evaluate (and ITF if requested) ----
        suffix = "_noSSL" if not use_ssl else ""
        ckpt_dir = run_dir / f"finetune_{enc}_{ds}_{loss}{suffix}_seed{seed}" / "ckpts"
        # Resolve "best" model path: pick the only .ckpt that is not last.ckpt
        if ckpt_dir.is_dir() and not args.dry_run:
            ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
            best = sorted(ckpts)[0] if ckpts else (ckpt_dir / "last.ckpt")
            best = str(best)
        else:
            best = "<best.ckpt>"

        eval_cmd = ["python", "scripts/evaluate.py",
                    "--finetuned-ckpt", best,
                    "--dataset", ds,
                    "--seed", str(seed),
                    "--data-root", args.data_root]
        if args.apply_itf:
            eval_cmd.append("--apply-itf")
        _run(eval_cmd, args.dry_run)


if __name__ == "__main__":
    main()
