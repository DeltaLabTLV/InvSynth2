"""Per-example evaluation for the focused ICASSP study."""

from __future__ import annotations

from typing import Any

import torch

from invsynth2.evaluation.metrics import (
    spec_per_example,
    spectral_convergence_per_example,
)
from invsynth2.training.finetune import FineTuneModule, nn_resize_time
from invsynth2.training.itf import itf_refine
from invsynth2.utils.parameters import hard_decode_theta
from invsynth2.utils.stft import denormalize_log_mag, log_to_linear_mag


@torch.no_grad()
def _forward_no_itf(module: FineTuneModule, wav: torch.Tensor) -> dict[str, torch.Tensor]:
    feature_norm = module.stft(wav)["feature_norm"]
    head_out = module.pen(module.encode(feature_norm))
    theta_hat = module._compose_theta(
        head_out, wav.new_zeros(wav.shape[0], module.spec.n_total)
    )
    pred_db = denormalize_log_mag(module.proxy(theta_hat), module.stft_cfg)
    return {
        "theta_hat": theta_hat,
        "pred_native": log_to_linear_mag(pred_db),
    }


def _forward_with_itf(
    module: FineTuneModule, wav: torch.Tensor, n_steps: int, lr: float
) -> dict[str, torch.Tensor]:
    result = itf_refine(module, wav, n_steps=n_steps, lr=lr)
    with torch.no_grad():
        pred_db = denormalize_log_mag(
            module.proxy(result["theta_hat_refined"]), module.stft_cfg
        )
        pred_native = log_to_linear_mag(pred_db)
    return {
        "theta_hat": result["theta_hat_refined"],
        "pred_native": pred_native,
    }


def _categorical_correct_matrix(
    pred_theta: torch.Tensor,
    target_theta: torch.Tensor,
    module: FineTuneModule,
) -> torch.Tensor:
    pred_theta = hard_decode_theta(pred_theta, module.spec)
    columns = []
    for block in module.spec.categorical_blocks:
        indices = list(block.indices)
        pred_label = pred_theta[..., indices].argmax(dim=-1)
        target_label = target_theta[..., indices].argmax(dim=-1)
        columns.append(pred_label == target_label)
    if not columns:
        return torch.empty(
            pred_theta.shape[0], 0, dtype=torch.bool, device=pred_theta.device
        )
    return torch.stack(columns, dim=-1)


def run_evaluation(
    module: FineTuneModule,
    test_loader,
    apply_itf: bool = True,
    itf_steps: int = 100,
    itf_lr: float = 1e-2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Evaluate relaxed proxy fit and hard-decoded categorical ACC.

    Per-example records are returned so paired differences can be reconstructed;
    no p-value is inferred from marginal standard deviations.
    """

    module.to(device)
    module.eval()
    records: list[dict[str, float | str]] = []
    spec_sum = 0.0
    sc_sum = 0.0
    categorical_correct = 0
    categorical_total = 0
    n_batches = 0

    for batch in test_loader:
        wav = batch["wav"].to(device)
        theta_gt = batch["theta"].to(device)
        with torch.no_grad():
            target_native = module.stft(wav)["native_mag"]

        if apply_itf:
            with torch.enable_grad():
                fwd = _forward_with_itf(module, wav, n_steps=itf_steps, lr=itf_lr)
        else:
            with torch.no_grad():
                fwd = _forward_no_itf(module, wav)
        pred_native = fwd["pred_native"]
        if pred_native.shape[-2:] != target_native.shape[-2:]:
            pred_native = torch.nn.functional.interpolate(
                pred_native.unsqueeze(1),
                size=target_native.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        spec_values = spec_per_example(pred_native, target_native)
        sc_values = spectral_convergence_per_example(pred_native, target_native)
        correct = _categorical_correct_matrix(fwd["theta_hat"], theta_gt, module)
        if correct.shape[-1] > 0:
            acc_values = correct.float().mean(dim=-1)
            categorical_correct += int(correct.sum().item())
            categorical_total += int(correct.numel())
        else:
            acc_values = torch.full_like(spec_values, float("nan"))

        spec_sum += float(spec_values.sum().item())
        sc_sum += float(sc_values.sum().item())
        for stem, spec_value, sc_value, acc_value in zip(
            batch["stem"], spec_values, sc_values, acc_values
        ):
            records.append(
                {
                    "stem": str(stem),
                    "Spec_x100": 100.0 * float(spec_value.item()),
                    "SC": float(sc_value.item()),
                    "ACC_pct": 100.0 * float(acc_value.item()),
                }
            )
        n_batches += 1

    n_samples = len(records)
    if n_samples == 0:
        raise RuntimeError("The test loader yielded no examples")
    metrics = {
        "Spec_x100": 100.0 * spec_sum / n_samples,
        "SC": sc_sum / n_samples,
    }
    if categorical_total:
        metrics["ACC_pct"] = 100.0 * categorical_correct / categorical_total
    return {
        "as_dict": metrics,
        "per_example": records,
        "n_samples": n_samples,
        "n_batches": n_batches,
        "apply_itf": apply_itf,
    }
