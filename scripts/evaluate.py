"""Run full test-set evaluation matching Tables 1, 2, and 4 of the paper.

Usage:
    python scripts/evaluate.py \
        --finetuned-ckpt runs/finetune_transformer_fm_imw_seed42/ckpts/best.ckpt \
        --dataset fm \
        --apply-itf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch

from invsynth2.data import SynthDataModule
from invsynth2.evaluation import run_evaluation
from invsynth2.training import FineTuneModule
from invsynth2.utils.parameters import DEFAULT_SPECS


def _spec_for(dataset: str):
    if dataset in DEFAULT_SPECS:
        return DEFAULT_SPECS[dataset]
    if dataset == "talnoise":
        return DEFAULT_SPECS["tal"]
    raise ValueError(f"No ParameterSpec for dataset={dataset!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned-ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--apply-itf", action="store_true",
                        help="Run with Inference-Time Fine-tuning before evaluating.")
    parser.add_argument("--itf-steps", type=int, default=100)
    parser.add_argument("--itf-lr", type=float, default=1e-2)
    parser.add_argument("--out-json", type=str, default=None,
                        help="Path for the JSON results file.")
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    spec = _spec_for(args.dataset)
    print(f"Loading checkpoint from {args.finetuned_ckpt}...")
    module = FineTuneModule.load_from_checkpoint(args.finetuned_ckpt, spec=spec)

    dm = SynthDataModule(
        root=args.data_root,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )
    dm.setup()

    print(f"Running evaluation (apply_itf={args.apply_itf})...")
    out = run_evaluation(
        module,
        dm.test_dataloader(),
        apply_itf=args.apply_itf,
        itf_steps=args.itf_steps,
        itf_lr=args.itf_lr,
    )

    print("\n=== Results ===")
    for k, v in out["as_dict"].items():
        print(f"  {k:>20s} : {v:.4f}")
    print(f"  n_samples = {out['n_samples']}")

    out_path = (
        Path(args.out_json)
        if args.out_json
        else Path(args.finetuned_ckpt).parent.parent
            / f"eval{'_itf' if args.apply_itf else ''}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "checkpoint": args.finetuned_ckpt,
                "dataset": args.dataset,
                "seed": args.seed,
                "apply_itf": args.apply_itf,
                "metrics": out["as_dict"],
                "n_samples": out["n_samples"],
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
