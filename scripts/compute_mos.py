"""Compute MOS results and paired bootstrap tests from a ratings CSV.

The input CSV must have columns:
    listener_id, dataset, system, stimulus_id, rating

`rating` is a 1–5 Likert score. Listener IDs should match across systems for
the paired tests to be valid.

Usage:
    python scripts/compute_mos.py --ratings ratings.csv --output mos_results.json

Optional:
    --paired-tests "is2:transformer,is2:unet,transformer:unet"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from invsynth2.evaluation.mos import aggregate_mos, paired_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paired-tests", type=str, default="",
                        help="Comma-separated 'sysA:sysB' pairs (run on every dataset).")
    args = parser.parse_args()

    print(f"Loading ratings from {args.ratings}...")
    table = aggregate_mos(args.ratings, n_boot=args.n_boot, seed=args.seed)
    print("\n=== MOS results (Table 3 format) ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    paired_results = []
    if args.paired_tests:
        for pair_spec in args.paired_tests.split(","):
            sys_a, sys_b = pair_spec.split(":")
            sys_a, sys_b = sys_a.strip(), sys_b.strip()
            print(f"\n=== Paired bootstrap test: {sys_a} vs {sys_b} ===")
            for ds in sorted(table["dataset"].unique()):
                try:
                    r = paired_bootstrap(
                        args.ratings, dataset=ds, sys_a=sys_a, sys_b=sys_b,
                        n_boot=args.n_boot, seed=args.seed,
                    )
                    print(
                        f"  {ds}: mean_diff={r['mean_diff']:+.4f} "
                        f"95% CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                        f"p={r['p_two_sided']:.4f}  (n_paired={r['n_paired']})"
                    )
                    paired_results.append({"dataset": ds, "sys_a": sys_a, "sys_b": sys_b, **r})
                except ValueError as e:
                    print(f"  {ds}: skipped ({e})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "ratings_file": args.ratings,
                "mos_table": table.to_dict(orient="records"),
                "paired_tests": paired_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
