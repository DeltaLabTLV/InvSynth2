"""Synthesizer parameter handling.

IS2 (and our paper) split parameters into continuous (regression) and discrete
(classification) groups. The PEN has separate output heads for each group, and
the parameter loss is L_reg + L_cls.

The label files in the dataset are 1D float arrays. To support the split, each
dataset specifies a `ParameterSpec` describing which indices are continuous vs
categorical (and for categorical, the integer codomain).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class ParameterSpec:
    """Per-dataset parameter specification.

    Attributes
    ----------
    n_total
        Total number of parameter slots in the label .npy.
    cont_indices
        Indices in the label vector that are continuous (regressed in [0, 1]).
    cat_specs
        List of (index, num_classes) for categorical parameters.
    name
        Human-readable name (e.g., 'fm', 'dx7', 'tal').
    """

    n_total: int
    cont_indices: tuple[int, ...]
    cat_specs: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    name: str = ""

    @property
    def n_continuous(self) -> int:
        return len(self.cont_indices)

    @property
    def n_categorical(self) -> int:
        return len(self.cat_specs)


# Defaults matching the paper's parameter-count description (§5.1).
# Users should override these for their actual synthesizer if it differs.
DEFAULT_SPECS: dict[str, ParameterSpec] = {
    # FM: 12 parameters, treated as all-continuous in the paper.
    "fm": ParameterSpec(
        n_total=12,
        cont_indices=tuple(range(12)),
        cat_specs=(),
        name="fm",
    ),
    # DX7: 42 parameters with operator-routing/envelope-shape categorical fields.
    # Concrete index assignment depends on how labels were authored;
    # here we assume the last 7 are categorical with 4 options each (algorithm
    # routing, EG shape, etc.). EDIT this to match your labels.
    "dx7": ParameterSpec(
        n_total=42,
        cont_indices=tuple(range(35)),
        cat_specs=tuple((35 + i, 4) for i in range(7)),
        name="dx7",
    ),
    # TAL Noisemaker: 35 parameters, mostly continuous, with a few mode switches.
    "tal": ParameterSpec(
        n_total=35,
        cont_indices=tuple(range(32)),
        cat_specs=tuple((32 + i, 4) for i in range(3)),
        name="tal",
    ),
}


def split_label(theta: torch.Tensor, spec: ParameterSpec) -> dict[str, torch.Tensor]:
    """Split a (B, n_total) label tensor into continuous and categorical components.

    Returns
    -------
    {"cont": (B, n_continuous), "cat": (B, n_categorical) integer indices}
    """
    cont = theta[..., list(spec.cont_indices)]
    if spec.n_categorical:
        cat_idx = [i for (i, _) in spec.cat_specs]
        cat = theta[..., cat_idx].long()
    else:
        cat = torch.empty(*theta.shape[:-1], 0, dtype=torch.long, device=theta.device)
    return {"cont": cont, "cat": cat}


def merge_label(
    cont: torch.Tensor,
    cat: torch.Tensor,
    spec: ParameterSpec,
) -> torch.Tensor:
    """Recombine continuous + categorical predictions into a single (B, n_total) vector.

    Categorical values are expected to be integer class indices, not logits.
    The output has the same dtype as `cont`.
    """
    out = torch.zeros(*cont.shape[:-1], spec.n_total, dtype=cont.dtype, device=cont.device)
    out[..., list(spec.cont_indices)] = cont
    for k, (idx, _) in enumerate(spec.cat_specs):
        out[..., idx] = cat[..., k].to(cont.dtype)
    return out


def parameter_accuracy(
    pred_cont: torch.Tensor,
    pred_cat_logits: list[torch.Tensor],
    gt_cont: torch.Tensor,
    gt_cat: torch.Tensor,
    spec: ParameterSpec,
    cont_tol: float = 0.05,
) -> float:
    """ACC: per-parameter accuracy following IS2 conventions.

    A continuous parameter is "correct" if |pred - gt| <= cont_tol.
    A categorical parameter is "correct" if argmax(logits) == gt class index.
    Returns the mean across all parameters and batch elements (in [0, 1]).
    """
    correct = 0
    total = 0
    if spec.n_continuous > 0:
        cont_correct = (pred_cont - gt_cont).abs() <= cont_tol
        correct += cont_correct.sum().item()
        total += cont_correct.numel()
    for k, _ in enumerate(spec.cat_specs):
        pred_k = pred_cat_logits[k].argmax(dim=-1)
        cat_correct = pred_k == gt_cat[..., k]
        correct += cat_correct.sum().item()
        total += cat_correct.numel()
    if total == 0:
        return 0.0
    return correct / total
