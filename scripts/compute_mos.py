"""Compute descriptive MOS means and listener-bootstrap intervals only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invsynth2.evaluation.mos import aggregate_mos


REPORTED_SYSTEMS = {"Flow", "InverSynth", "IS2_no_ITF", "IS2", "UNet_full"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-listeners", type=int, default=30)
    parser.add_argument("--expected-trials-per-listener", type=int, default=54)
    args = parser.parse_args()

    raw = pd.read_csv(args.ratings)
    expected_columns = {"listener_id", "dataset", "system", "stimulus_id", "rating"}
    missing = expected_columns - set(raw.columns)
    if missing:
        raise ValueError(f"Ratings file is missing columns: {sorted(missing)}")
    if not raw["rating"].between(1, 5).all():
        raise ValueError("All ratings must lie on the closed 1--5 scale")
    if raw["listener_id"].nunique() != args.expected_listeners:
        raise ValueError("Listener count does not match the manuscript")
    per_listener = raw.groupby("listener_id").size()
    if not (per_listener == args.expected_trials_per_listener).all():
        raise ValueError(
            "Every listener must have the same 54-trial record used by the manuscript"
        )

    table = aggregate_mos(args.ratings, n_boot=args.n_boot, seed=args.seed)
    reported = table[table["system"].isin(REPORTED_SYSTEMS)].copy()
    observed_systems = set(reported["system"])
    if observed_systems != REPORTED_SYSTEMS:
        raise ValueError(f"Missing reported systems: {sorted(REPORTED_SYSTEMS - observed_systems)}")
    if not (reported["n_listeners"] == args.expected_listeners).all():
        raise ValueError("At least one reported cell is missing listener-level data")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "ratings_file": args.ratings,
                "bootstrap_unit": "listener_cell_mean",
                "bootstrap_resamples": args.n_boot,
                "bootstrap_seed": args.seed,
                "mos_table": reported.to_dict(orient="records"),
                "inferential_tests": None,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(reported.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved descriptive MOS results to {output}")


if __name__ == "__main__":
    main()
