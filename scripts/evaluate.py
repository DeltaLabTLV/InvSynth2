"""Evaluate one U-Net configuration and retain paired per-example records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L

from invsynth2.evaluation import run_evaluation
from invsynth2.repro import load_feature_config, make_data_module, parameter_spec_for_run
from invsynth2.study import PAPER_CONFIGURATIONS, dataset_profile, load_study_config
from invsynth2.training import FineTuneModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", default="configs/icassp2027.yaml")
    parser.add_argument("--dataset", required=True, choices=["fm", "dx7", "tal"])
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    parser.add_argument("--configuration", required=True, choices=PAPER_CONFIGURATIONS)
    parser.add_argument("--finetuned-ckpt", required=True)
    parser.add_argument("--feature-stats", required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--apply-itf", action="store_true")
    parser.add_argument("--itf-steps", type=int, default=100)
    parser.add_argument("--itf-lr", type=float, default=1e-2)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    cfg = load_study_config(args.study_config)
    profile = dataset_profile(cfg, args.dataset)
    row = cfg["configurations"][args.configuration]
    L.seed_everything(args.seed, workers=True)
    if args.apply_itf:
        if args.itf_steps != cfg["optimization"]["refinement_updates"]:
            raise ValueError("Paper evaluation requires exactly 100 refinement updates")
        if args.itf_lr != cfg["optimization"]["refinement_learning_rate"]:
            raise ValueError("Paper evaluation requires refinement learning rate 1e-2")

    dm = make_data_module(cfg, profile, args.seed, args.data_root, num_workers=0)
    dm.setup()
    spec = parameter_spec_for_run(cfg, profile, args.seed, args.data_root)
    feature_cfg = load_feature_config(cfg, profile, args.seed, args.feature_stats)
    module = FineTuneModule.load_from_checkpoint(
        args.finetuned_ckpt,
        stft_cfg=feature_cfg,
        spec=spec,
    )
    if module.hparams.encoder_kind != "unet":
        raise ValueError("The focused ICASSP artifact evaluates U-Net checkpoints only")
    if module.hparams.loss_mode != row["reconstruction"]:
        raise ValueError("Checkpoint loss mode does not match --configuration")

    result = run_evaluation(
        module,
        dm.test_dataloader(),
        apply_itf=args.apply_itf,
        itf_steps=args.itf_steps,
        itf_lr=args.itf_lr,
    )
    for name, value in result["as_dict"].items():
        print(f"{name}: {value:.8g}")

    checkpoint = Path(args.finetuned_ckpt)
    out_path = Path(args.out_json) if args.out_json else checkpoint.parent.parent / (
        "eval_itf.json" if args.apply_itf else "eval_step0.json"
    )
    record_path = out_path.with_suffix(".per_example.jsonl")
    payload = {
        "schema_version": 1,
        "dataset": profile.name,
        "seed": args.seed,
        "configuration": args.configuration,
        "display_name": row["display_name"],
        "checkpoint": str(checkpoint),
        "apply_itf": args.apply_itf,
        "itf_steps": args.itf_steps if args.apply_itf else 0,
        "itf_learning_rate": args.itf_lr if args.apply_itf else None,
        "metrics": result["as_dict"],
        "n_samples": result["n_samples"],
        "per_example_file": record_path.name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with record_path.open("w", encoding="utf-8") as handle:
        for record in result["per_example"]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Saved aggregate metrics to {out_path}")
    print(f"Saved paired per-example metrics to {record_path}")


if __name__ == "__main__":
    main()
