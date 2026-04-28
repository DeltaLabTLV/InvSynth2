"""Test-set evaluation runner.

Loads a fine-tuned checkpoint, optionally applies ITF, and reports all
quantitative metrics over the test split — matching Tables 1, 2, and 4.
"""

from __future__ import annotations

from typing import Any

import torch

from invsynth2.evaluation.metrics import (
    BandConfig,
    band_errors,
    melspec_metric,
    mfcc_metric,
    spec_metric,
    spectral_convergence,
    MetricResults,
)
from invsynth2.training.finetune import FineTuneModule, nn_resize_time
from invsynth2.training.itf import itf_refine
from invsynth2.utils.parameters import parameter_accuracy, split_label
from invsynth2.utils.stft import denormalize_log_mag, log_to_linear_mag


@torch.no_grad()
def _forward_no_itf(module: FineTuneModule, wav: torch.Tensor) -> dict:
    """Single forward pass through encoder+PEN+frozen proxy. No ITF."""
    log_mag_norm = module.stft(wav)["log_mag_norm"]
    feats = module.encode(log_mag_norm)
    head_out = module.pen(feats)
    theta_hat = module._compose_theta(head_out, wav.new_zeros(wav.shape[0], module.spec.n_total))
    pred_log_mag_norm = module.proxy(theta_hat)
    pred_log_mag = denormalize_log_mag(pred_log_mag_norm, module.stft_cfg)
    pred_lin = log_to_linear_mag(pred_log_mag, eps=module.stft_cfg.eps)
    return {"theta_hat": theta_hat, "pred_lin": pred_lin, "head_out": head_out}


def _forward_with_itf(
    module: FineTuneModule, wav: torch.Tensor, n_steps: int, lr: float
) -> dict:
    """ITF-refined θ̂ + corresponding spectrogram."""
    res = itf_refine(module, wav, n_steps=n_steps, lr=lr)
    with torch.no_grad():
        pred_log_mag_norm = module.proxy(res["theta_hat_refined"])
        pred_log_mag = denormalize_log_mag(pred_log_mag_norm, module.stft_cfg)
        pred_lin = log_to_linear_mag(pred_log_mag, eps=module.stft_cfg.eps)
    return {"theta_hat": res["theta_hat_refined"], "pred_lin": pred_lin}


@torch.no_grad()
def run_evaluation(
    module: FineTuneModule,
    test_loader,
    apply_itf: bool = False,
    itf_steps: int = 100,
    itf_lr: float = 1e-2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Run all metrics across the test loader and return aggregated results."""
    module.to(device)
    module.eval()
    band_cfg = BandConfig(sample_rate=module.stft_cfg.sample_rate, n_fft=module.stft_cfg.n_fft)

    sums: dict[str, float] = {
        "spec": 0.0, "sc": 0.0, "mel": 0.0, "mfcc": 0.0,
        "band_low": 0.0, "band_mid": 0.0, "band_high": 0.0,
    }
    n_batches = 0
    n_samples = 0
    correct = 0
    total = 0

    for batch in test_loader:
        wav = batch["wav"].to(device)
        theta_gt = batch["theta"].to(device)
        target_lin = module.stft(wav)["linear_mag"]

        # Forward (with or without ITF)
        if apply_itf:
            # Re-enable grad context inside ITF.
            with torch.enable_grad():
                fwd = _forward_with_itf(module, wav, n_steps=itf_steps, lr=itf_lr)
        else:
            fwd = _forward_no_itf(module, wav)
        pred_lin = fwd["pred_lin"]
        if pred_lin.shape[-1] != target_lin.shape[-1]:
            pred_lin = nn_resize_time(pred_lin, target_lin.shape[-1])

        # Spectral metrics
        sums["spec"] += float(spec_metric(pred_lin, target_lin).item())
        sums["sc"]   += float(spectral_convergence(pred_lin, target_lin).item())
        sums["mel"]  += float(melspec_metric(pred_lin, target_lin, sample_rate=band_cfg.sample_rate, n_fft=band_cfg.n_fft).item())
        sums["mfcc"] += float(mfcc_metric(pred_lin, target_lin, sample_rate=band_cfg.sample_rate, n_fft=band_cfg.n_fft).item())
        bands = band_errors(pred_lin, target_lin, band_cfg)
        sums["band_low"]  += float(bands["low"].item())
        sums["band_mid"]  += float(bands["mid"].item())
        sums["band_high"] += float(bands["high"].item())
        n_batches += 1

        # ACC
        gt_split = split_label(theta_gt, module.spec)
        # For ACC we need the head output, not the composed θ̂. We re-run only
        # the encoder/PEN to avoid needing to keep head_out in the ITF path.
        log_mag_norm = module.stft(wav)["log_mag_norm"]
        head_out = module.pen(module.encode(log_mag_norm))
        # If ITF was applied, also report ACC on the refined θ̂ later.
        for_acc_cont = head_out["cont"]
        for_acc_logits = head_out["cat_logits"]
        # parameter_accuracy expects a count, not a proportion; we accumulate.
        acc_batch = parameter_accuracy(
            for_acc_cont, for_acc_logits, gt_split["cont"], gt_split["cat"], module.spec,
        )
        # acc_batch is already a proportion; convert to (correct, total) via numel.
        n_per = (gt_split["cont"].numel() + gt_split["cat"].numel())
        correct += int(round(acc_batch * n_per))
        total += n_per
        n_samples += wav.shape[0]

    means = {k: v / max(n_batches, 1) for k, v in sums.items()}
    acc = correct / total if total else 0.0

    results = MetricResults(
        spec=means["spec"], sc=means["sc"], melspec=means["mel"], mfcc=means["mfcc"],
        band_low=means["band_low"], band_mid=means["band_mid"], band_high=means["band_high"],
        acc=acc,
    )
    return {
        "results": results,
        "as_dict": results.to_dict(),
        "n_samples": n_samples,
        "n_batches": n_batches,
        "apply_itf": apply_itf,
    }
