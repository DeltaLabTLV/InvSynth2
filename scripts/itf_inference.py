"""Save relaxed and hard-decoded 100-step refinements for one paper run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L
import numpy as np
import torch

from invsynth2.repro import load_feature_config, make_data_module, parameter_spec_for_run
from invsynth2.study import dataset_profile, load_study_config
from invsynth2.training import FineTuneModule, itf_refine
from invsynth2.utils.parameters import hard_decode_theta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", default="configs/icassp2027.yaml")
    parser.add_argument("--dataset", required=True, choices=["fm", "dx7", "tal"])
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    parser.add_argument("--finetuned-ckpt", required=True)
    parser.add_argument("--feature-stats", required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_study_config(args.study_config)
    profile = dataset_profile(cfg, args.dataset)
    spec = parameter_spec_for_run(cfg, profile, args.seed, args.data_root)
    feature_cfg = load_feature_config(cfg, profile, args.seed, args.feature_stats)
    L.seed_everything(args.seed, workers=True)
    module = FineTuneModule.load_from_checkpoint(
        args.finetuned_ckpt, stft_cfg=feature_cfg, spec=spec
    )
    if module.hparams.encoder_kind != "unet":
        raise ValueError("The focused ICASSP artifact refines U-Net checkpoints only")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module.to(device)
    dm = make_data_module(cfg, profile, args.seed, args.data_root, num_workers=0)
    dm.setup()

    initial, relaxed, decoded, stems = [], [], [], []
    for batch in dm.test_dataloader():
        wav = batch["wav"].to(device)
        with torch.enable_grad():
            result = itf_refine(
                module,
                wav,
                n_steps=cfg["optimization"]["refinement_updates"],
                lr=cfg["optimization"]["refinement_learning_rate"],
            )
        initial.append(result["theta_hat_initial"].cpu().numpy())
        relaxed_tensor = result["theta_hat_refined"]
        relaxed.append(relaxed_tensor.cpu().numpy())
        decoded.append(hard_decode_theta(relaxed_tensor, spec).cpu().numpy())
        stems.extend(str(stem) for stem in batch["stem"])

    out = Path(args.out) if args.out else Path(args.finetuned_ckpt).parent.parent / "refinement.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        stems=np.asarray(stems),
        theta_initial=np.concatenate(initial),
        theta_relaxed=np.concatenate(relaxed),
        theta_hard_decoded=np.concatenate(decoded),
    )
    print(f"Saved {len(stems)} refinements to {out}")


if __name__ == "__main__":
    main()
