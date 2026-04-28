"""Spectral reconstruction losses.

This module implements three alternatives the paper compares:

1. Standard spectral loss (L1 + L2 in linear-magnitude domain). Eq. (5).
2. Inverse-Magnitude Weighted (IMW) loss — the paper's contribution. Eq. (3).
3. Log-magnitude spectral loss (L2 between log-mags) — competitor baseline.

The full reconstruction objective is:
    L_rec = β L_spec + (1 - β) L_IMW         (Eq. 4)

For the ablations:
    full       → L_rec with β = 0.7 (default)
    w/o IMW    → β = 1.0 (pure L_spec)
    w/ log-spec→ replace L_IMW with L_log_spec, same β
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IMWLoss(nn.Module):
    """Inverse-Magnitude Weighted spectral loss.

    L_IMW = sum_{t,f} w(|X_{t,f}|) (|X̂_{t,f}| - |X_{t,f}|)^2
    where w(x) = 1 / (x + epsilon).

    The weight depends only on the target |X|, so it is treated as a constant
    with respect to |X̂| (no gradient flows through it).

    The paper uses ε = 1e-7 in the linear-magnitude domain after global
    normalization. We keep that default and confirm in the paper that smaller
    epsilon increased gradient variance without improving val error.
    """

    def __init__(self, epsilon: float = 1e-7, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.reduction = reduction

    def forward(self, pred_lin_mag: torch.Tensor, target_lin_mag: torch.Tensor) -> torch.Tensor:
        # Detach the weights — gradient flows only through (pred - target)^2.
        with torch.no_grad():
            weight = 1.0 / (target_lin_mag + self.epsilon)
        sq_err = (pred_lin_mag - target_lin_mag) ** 2
        loss_per_bin = weight * sq_err
        if self.reduction == "mean":
            return loss_per_bin.mean()
        return loss_per_bin.sum()


class StandardSpecLoss(nn.Module):
    """L_spec = α1 |X̂ - X|_1 + α2 |X̂ - X|_2^2 in the linear-magnitude domain. Eq. (5)."""

    def __init__(self, alpha1: float = 1.0, alpha2: float = 1.0):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, pred_lin_mag: torch.Tensor, target_lin_mag: torch.Tensor) -> torch.Tensor:
        l1 = (pred_lin_mag - target_lin_mag).abs().mean()
        l2 = ((pred_lin_mag - target_lin_mag) ** 2).mean()
        return self.alpha1 * l1 + self.alpha2 * l2


class LogSpecLoss(nn.Module):
    """L_log_spec = || log|X̂| - log|X| ||_2^2 — the IMW competitor.

    Numerically guarded with a small epsilon to avoid log(0).
    """

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred_lin_mag: torch.Tensor, target_lin_mag: torch.Tensor) -> torch.Tensor:
        diff = torch.log(pred_lin_mag + self.epsilon) - torch.log(target_lin_mag + self.epsilon)
        return (diff**2).mean()


class ReconstructionLoss(nn.Module):
    """Composite reconstruction loss as defined in Eq. (4).

    Three modes:
        - 'imw'       → β L_spec + (1 - β) L_IMW  (paper's full loss)
        - 'spec_only' → L_spec                    (β = 1.0; w/o IMW ablation)
        - 'log_spec'  → β L_spec + (1 - β) L_log_spec  (IMW competitor ablation)
    """

    def __init__(
        self,
        mode: str = "imw",
        beta: float = 0.7,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        self.mode = mode
        self.beta = beta
        self.spec = StandardSpecLoss(alpha1=alpha1, alpha2=alpha2)
        if mode == "imw":
            self.aux = IMWLoss(epsilon=epsilon)
        elif mode == "log_spec":
            self.aux = LogSpecLoss(epsilon=epsilon)
        elif mode == "spec_only":
            self.aux = None
        else:
            raise ValueError(f"Unknown reconstruction-loss mode {mode!r}")

    def forward(self, pred_lin_mag: torch.Tensor, target_lin_mag: torch.Tensor) -> dict[str, torch.Tensor]:
        spec = self.spec(pred_lin_mag, target_lin_mag)
        if self.aux is None:
            total = spec
            aux = torch.zeros_like(spec)
        else:
            aux = self.aux(pred_lin_mag, target_lin_mag)
            total = self.beta * spec + (1.0 - self.beta) * aux
        return {"loss": total, "spec": spec, "aux": aux, "mode": self.mode}
