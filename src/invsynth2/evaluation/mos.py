"""MOS (Mean Opinion Score) listening-test analysis.

Two functions:

1. `aggregate_mos(ratings_csv)` — given a CSV with columns
   `[listener_id, dataset, system, stimulus_id, rating]`, compute per-(dataset,
   system) means with 95% bootstrap confidence intervals over listener-level
   means. This matches Table 3 of the paper.

2. `paired_bootstrap(ratings_csv, dataset, sys_a, sys_b)` — paired bootstrap
   test on listener-level differences between two systems.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def _listener_cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw ratings to one row per (listener, dataset, system)."""
    return (
        df.groupby(["listener_id", "dataset", "system"], as_index=False)["rating"]
        .mean()
        .rename(columns={"rating": "listener_cell_mean"})
    )


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def aggregate_mos(
    ratings_csv: str,
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute Table 3-style MOS values: mean ± 95% bootstrap CI per (dataset, system).

    Inputs
    ------
    ratings_csv : path to a CSV with columns
        [listener_id, dataset, system, stimulus_id, rating]
        where rating ∈ {1, 2, 3, 4, 5}.

    Returns
    -------
    DataFrame with columns [dataset, system, mean, ci_lo, ci_hi, ci_halfwidth, n_listeners]
    """
    df = pd.read_csv(ratings_csv)
    expected = {"listener_id", "dataset", "system", "stimulus_id", "rating"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"ratings CSV missing columns: {missing}")

    cell_means = _listener_cell_means(df)
    rows: list[dict[str, Any]] = []
    for (dataset, system), group in cell_means.groupby(["dataset", "system"]):
        listener_means = group["listener_cell_mean"].to_numpy()
        mean = float(listener_means.mean())
        ci_lo, ci_hi = _bootstrap_ci(listener_means, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "dataset": dataset,
                "system": system,
                "mean": mean,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "ci_halfwidth": (ci_hi - ci_lo) / 2.0,
                "n_listeners": len(listener_means),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "system"]).reset_index(drop=True)


def paired_bootstrap(
    ratings_csv: str,
    dataset: str,
    sys_a: str,
    sys_b: str,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap test on (sys_a − sys_b) listener-level differences.

    Returns
    -------
    {
        "mean_diff": float,
        "ci_lo": float, "ci_hi": float,
        "p_two_sided": float,
        "n_paired": int,
    }
    """
    df = pd.read_csv(ratings_csv)
    cell_means = _listener_cell_means(df)
    sub = cell_means[cell_means["dataset"] == dataset]
    a = sub[sub["system"] == sys_a].sort_values("listener_id")["listener_cell_mean"].to_numpy()
    b = sub[sub["system"] == sys_b].sort_values("listener_id")["listener_cell_mean"].to_numpy()
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        raise ValueError(
            f"Mismatched/empty listener cohorts: |{sys_a}|={len(a)}, |{sys_b}|={len(b)}. "
            "Ensure every listener rated both systems on this dataset."
        )

    diff = a - b
    n = len(diff)
    obs = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    # Two-sided p-value: shift differences to mean zero under H0, then count.
    diff_centered = diff - obs
    boot_null = diff_centered[idx].mean(axis=1)
    p_two_sided = float((1 + np.sum(np.abs(boot_null) >= np.abs(obs))) / (1 + n_boot))

    return {
        "mean_diff": obs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_two_sided": p_two_sided,
        "n_paired": n,
    }
