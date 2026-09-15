"""Build a listening-test manifest from existing true-synthesizer renders.

This script never converts proxy spectrograms to audio. The manuscript's
perceptual evidence requires hard-decoded presets rendered by the target
synthesizers; those licensed renderers and audio files must be supplied by the
authors.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


REPORTED_SYSTEMS = ("Flow", "InverSynth", "IS2_no_ITF", "IS2", "UNet_full")
EXPLORATORY_SYSTEM = "Transformer_exploratory"


def _rms(path: Path) -> tuple[int, float]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return sample_rate, float(np.sqrt(np.mean(mono**2) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", required=True)
    parser.add_argument("--output-csv", default="mos_manifest.csv")
    parser.add_argument("--datasets", nargs="+", default=["fm", "dx7", "tal"])
    parser.add_argument("--n-stimuli", type=int, default=3)
    parser.add_argument("--include-exploratory-transformer", action="store_true")
    args = parser.parse_args()

    root = Path(args.render_root)
    systems = list(REPORTED_SYSTEMS)
    if args.include_exploratory_transformer:
        systems.append(EXPLORATORY_SYSTEM)
    rows = []
    for dataset in args.datasets:
        stimulus_dirs = sorted(path for path in (root / dataset).iterdir() if path.is_dir())
        if len(stimulus_dirs) != args.n_stimuli:
            raise ValueError(
                f"{dataset}: expected {args.n_stimuli} stimulus directories, "
                f"found {len(stimulus_dirs)}"
            )
        for stimulus_dir in stimulus_dirs:
            target = stimulus_dir / "target.wav"
            if not target.is_file():
                raise FileNotFoundError(target)
            target_rate, target_rms = _rms(target)
            for system in systems:
                render = stimulus_dir / f"{system}.wav"
                if not render.is_file():
                    raise FileNotFoundError(render)
                sample_rate, render_rms = _rms(render)
                if sample_rate != target_rate:
                    raise ValueError(f"Sample-rate mismatch: {target} vs {render}")
                rows.append(
                    {
                        "stimulus_id": stimulus_dir.name,
                        "dataset": dataset,
                        "system": system,
                        "reported": system != EXPLORATORY_SYSTEM,
                        "target_path": str(target),
                        "render_path": str(render),
                        "sample_rate": sample_rate,
                        "target_rms": target_rms,
                        "render_rms": render_rms,
                    }
                )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} true-render comparisons to {output}")


if __name__ == "__main__":
    main()
