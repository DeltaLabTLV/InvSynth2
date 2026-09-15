"""Explicit continuous coordinates and categorical one-hot blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class CategoricalBlock:
    name: str
    indices: tuple[int, ...]

    @property
    def num_classes(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class ParameterSpec:
    """Layout of the flat, proxy-facing parameter vector.

    Continuous controls occupy one normalized coordinate each. Every
    categorical control occupies one one-hot block. During refinement the
    complete vector is relaxed to a box; hard decoding restores one-hot blocks.
    """

    n_total: int
    cont_indices: tuple[int, ...]
    categorical_blocks: tuple[CategoricalBlock, ...] = field(default_factory=tuple)
    name: str = ""

    def __post_init__(self) -> None:
        occupied = list(self.cont_indices)
        for block in self.categorical_blocks:
            if block.num_classes < 2:
                raise ValueError(f"Categorical block {block.name!r} needs at least two classes")
            occupied.extend(block.indices)
        if len(occupied) != len(set(occupied)):
            raise ValueError("Parameter schema contains overlapping indices")
        if sorted(occupied) != list(range(self.n_total)):
            raise ValueError(
                "Parameter schema must cover every proxy-input coordinate exactly once; "
                f"covered {len(set(occupied))} of {self.n_total}"
            )

    @property
    def n_continuous(self) -> int:
        return len(self.cont_indices)

    @property
    def n_categorical(self) -> int:
        return len(self.categorical_blocks)

    @property
    def cat_specs(self) -> tuple[tuple[int, int], ...]:
        """Compatibility view: (first block index, number of classes)."""

        return tuple((block.indices[0], block.num_classes) for block in self.categorical_blocks)


def load_parameter_spec(path: str | Path, *, expected_name: str | None = None) -> ParameterSpec:
    """Load a run-local schema and fail closed on missing cardinalities."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Parameter schema not found: {path}. Do not use guessed default indices."
        )
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    name = str(raw.get("name", expected_name or ""))
    if expected_name is not None and name.lower() != expected_name.lower():
        raise ValueError(f"{path}: schema name {name!r} does not match {expected_name!r}")
    blocks = tuple(
        CategoricalBlock(
            name=str(block["name"]),
            indices=tuple(int(index) for index in block["indices"]),
        )
        for block in raw.get("categorical_blocks", [])
    )
    return ParameterSpec(
        n_total=int(raw["n_total"]),
        cont_indices=tuple(int(index) for index in raw.get("continuous_indices", [])),
        categorical_blocks=blocks,
        name=name,
    )


def split_label(theta: torch.Tensor, spec: ParameterSpec) -> dict[str, torch.Tensor]:
    """Split a proxy vector into continuous values and category indices."""

    if theta.shape[-1] != spec.n_total:
        raise ValueError(f"Expected theta width {spec.n_total}, got {theta.shape[-1]}")
    cont = theta[..., list(spec.cont_indices)] if spec.cont_indices else theta[..., :0]
    categories = []
    for block in spec.categorical_blocks:
        categories.append(theta[..., list(block.indices)].argmax(dim=-1))
    if categories:
        cat = torch.stack(categories, dim=-1)
    else:
        cat = torch.empty(*theta.shape[:-1], 0, dtype=torch.long, device=theta.device)
    return {"cont": cont, "cat": cat}


def compose_proxy_vector(
    cont: torch.Tensor,
    categorical_values: list[torch.Tensor],
    spec: ParameterSpec,
) -> torch.Tensor:
    """Compose continuous coordinates and relaxed categorical blocks."""

    batch_shape = cont.shape[:-1]
    out = cont.new_zeros(*batch_shape, spec.n_total)
    if spec.cont_indices:
        out[..., list(spec.cont_indices)] = cont
    if len(categorical_values) != spec.n_categorical:
        raise ValueError(
            f"Expected {spec.n_categorical} categorical blocks, got {len(categorical_values)}"
        )
    for values, block in zip(categorical_values, spec.categorical_blocks):
        if values.shape[-1] != block.num_classes:
            raise ValueError(
                f"Block {block.name!r}: expected {block.num_classes} values, got {values.shape[-1]}"
            )
        out[..., list(block.indices)] = values
    return out


def merge_label(cont: torch.Tensor, cat: torch.Tensor, spec: ParameterSpec) -> torch.Tensor:
    """Compose continuous coordinates and integer categories as one-hot blocks."""

    blocks = []
    for position, block in enumerate(spec.categorical_blocks):
        blocks.append(
            torch.nn.functional.one_hot(
                cat[..., position].long(), num_classes=block.num_classes
            ).to(dtype=cont.dtype)
        )
    return compose_proxy_vector(cont, blocks, spec)


def hard_decode_theta(theta: torch.Tensor, spec: ParameterSpec) -> torch.Tensor:
    """Clamp continuous values and replace each relaxed block by one-hot argmax."""

    decoded = theta.clamp(0.0, 1.0).clone()
    for block in spec.categorical_blocks:
        block_values = theta[..., list(block.indices)]
        labels = block_values.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(
            labels, num_classes=block.num_classes
        ).to(dtype=theta.dtype)
        decoded[..., list(block.indices)] = one_hot
    return decoded


def parameter_accuracy(
    pred_cont: torch.Tensor,
    pred_cat_logits: list[torch.Tensor],
    gt_cont: torch.Tensor,
    gt_cat: torch.Tensor,
    spec: ParameterSpec,
    cont_tol: float = 0.05,
) -> float:
    """Categorical ACC used by the paper; continuous controls are excluded."""

    del pred_cont, gt_cont, cont_tol
    if len(pred_cat_logits) != spec.n_categorical:
        raise ValueError("Categorical head count does not match the parameter schema")
    if spec.n_categorical == 0:
        return float("nan")
    predictions = torch.stack([logits.argmax(dim=-1) for logits in pred_cat_logits], dim=-1)
    return float((predictions == gt_cat).float().mean().item())


def categorical_accuracy_from_theta(
    pred_theta: torch.Tensor,
    target_theta: torch.Tensor,
    spec: ParameterSpec,
) -> tuple[int, int]:
    """Return exact correct/total counts after hard decoding relaxed blocks."""

    if spec.n_categorical == 0:
        return 0, 0
    pred_labels = []
    target_labels = []
    for block in spec.categorical_blocks:
        indices = list(block.indices)
        pred_labels.append(pred_theta[..., indices].argmax(dim=-1))
        target_labels.append(target_theta[..., indices].argmax(dim=-1))
    pred = torch.stack(pred_labels, dim=-1)
    target = torch.stack(target_labels, dim=-1)
    return int((pred == target).sum().item()), int(target.numel())


# Deliberately empty: the earlier repository contained guessed dataset layouts.
# Paper entry points always load run-local parameter_schema.yaml files.
DEFAULT_SPECS: dict[str, ParameterSpec] = {}
