"""Parameter regression + classification losses.

Used during fine-tuning. Following IS2, parameters are split into:
- Continuous (regressed) — MSE loss in [0, 1]-normalized space
- Categorical (classified) — cross-entropy per categorical parameter

Total parameter loss = λ_reg * L_reg + λ_cls * L_cls. The default weights
follow IS2 (both = 1.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterLoss(nn.Module):
    """Combined regression + classification loss for synthesizer parameters."""

    def __init__(self, lambda_reg: float = 1.0, lambda_cls: float = 1.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls

    def forward(
        self,
        pred_cont: torch.Tensor,                  # (B, n_continuous)
        gt_cont: torch.Tensor,                    # (B, n_continuous)
        pred_cat_logits: list[torch.Tensor] | None,  # list of (B, num_classes)
        gt_cat: torch.Tensor | None,              # (B, n_categorical) integer
    ) -> dict[str, torch.Tensor]:
        # MSE on continuous predictions in [0, 1]
        if pred_cont.shape[-1] > 0:
            loss_reg = F.mse_loss(pred_cont, gt_cont)
        else:
            loss_reg = torch.zeros((), device=pred_cont.device, dtype=pred_cont.dtype)

        # Cross-entropy on each categorical parameter
        loss_cls = torch.zeros((), device=pred_cont.device, dtype=pred_cont.dtype)
        if pred_cat_logits is not None and len(pred_cat_logits) > 0:
            assert gt_cat is not None and gt_cat.shape[-1] == len(pred_cat_logits), (
                "Mismatch between number of categorical heads and gt_cat dimensions."
            )
            for k, logits in enumerate(pred_cat_logits):
                loss_cls = loss_cls + F.cross_entropy(logits, gt_cat[..., k])
            loss_cls = loss_cls / len(pred_cat_logits)

        total = self.lambda_reg * loss_reg + self.lambda_cls * loss_cls
        return {"loss": total, "reg": loss_reg, "cls": loss_cls}
