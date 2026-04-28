"""Stage 4: Inference-Time Fine-tuning (ITF).

Loads a fine-tuned checkpoint and runs ITF refinement over the test set,
saving the refined θ̂ vectors and (optionally) reconstructed spectrograms.

Usage:
    python scripts/itf_inference.py \
        --finetuned-ckpt runs/finetune_transformer_fm_imw_seed42/ckpts/best.ckpt \
        --dataset fm \
        --steps 100 --lr 1e-2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import numpy as np
import torch

from invsynth2.data import SynthDataModule
from invsynth2.training import FineTuneModule, itf_refine
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
    parser.add_argument("--steps", type=int, default=100, help="ITF optimization steps per sample.")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to save refined θ̂ tensors. Defaults to ckpt's parent.")
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    # Load the fine-tuned module from checkpoint.
    print(f"Loading checkpoint from {args.finetuned_ckpt}...")
    module = FineTuneModule.load_from_checkpoint(
        args.finetuned_ckpt,
        spec=_spec_for(args.dataset),
        # stft_cfg will be restored from the saved hyperparameters; if missing,
        # we let the constructor re-create it from defaults — caller should
        # pass --stft-* flags here if they used non-default STFT settings.
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module.to(device)

    # Build test loader (uses same seed → same split as training).
    dm = SynthDataModule(
        root=args.data_root,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )
    dm.setup()
    loader = dm.test_dataloader()

    # Run ITF over the test set.
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.finetuned_ckpt).parent / "itf_refined"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_initial = []
    all_refined = []
    all_stems = []
    for batch_idx, batch in enumerate(loader):
        wav = batch["wav"].to(device)
        with torch.enable_grad():
            res = itf_refine(module, wav, n_steps=args.steps, lr=args.lr)
        all_initial.append(res["theta_hat_initial"].cpu().numpy())
        all_refined.append(res["theta_hat_refined"].cpu().numpy())
        all_stems.extend(batch["stem"])
        if batch_idx % 10 == 0:
            mean_loss = float(np.mean(res["loss_history"]))
            print(f"[batch {batch_idx}] mean ITF loss = {mean_loss:.6f}")

    initial = np.concatenate(all_initial, axis=0)
    refined = np.concatenate(all_refined, axis=0)
    np.savez_compressed(
        out_dir / "itf_results.npz",
        theta_initial=initial,
        theta_refined=refined,
        stems=np.array(all_stems),
    )
    print(f"Saved {len(all_stems)} refined θ̂ vectors to {out_dir / 'itf_results.npz'}")


if __name__ == "__main__":
    main()
