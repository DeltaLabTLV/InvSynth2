"""Aggregate five run JSONs descriptively; never infer tests from marginal SDs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, stdev
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invsynth2.study import PAPER_CONFIGURATIONS, load_study_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", default="configs/icassp2027.yaml")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--out-csv", default="runs/aggregate_results.csv")
    args = parser.parse_args()

    cfg = load_study_config(args.study_config)
    rows = []
    for dataset in cfg["datasets"]:
        for configuration in PAPER_CONFIGURATIONS:
            values = {"Spec": [], "SC": []}
            observed_seeds = []
            for seed in cfg["seeds"]:
                path = Path(args.run_dir) / dataset / f"seed_{seed}" / configuration / "eval_itf.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing run aggregate: {path}")
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload["dataset"] != dataset or payload["configuration"] != configuration:
                    raise ValueError(f"Identity mismatch in {path}")
                observed_seeds.append(int(payload["seed"]))
                values["Spec"].append(float(payload["metrics"]["Spec_x100"]))
                values["SC"].append(float(payload["metrics"]["SC"]))
            if sorted(observed_seeds) != cfg["seeds"]:
                raise ValueError(f"Duplicate/missing seeds for {dataset}/{configuration}")
            for metric, metric_values in values.items():
                rows.append(
                    {
                        "configuration": configuration,
                        "display_name": cfg["configurations"][configuration]["display_name"],
                        "dataset": dataset.upper(),
                        "metric": metric,
                        "mean": fmean(metric_values),
                        "sd": stdev(metric_values),
                        "n_runs": len(metric_values),
                        "provenance": "recomputed_from_run_json",
                    }
                )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} descriptive aggregate rows to {out}")


if __name__ == "__main__":
    main()
