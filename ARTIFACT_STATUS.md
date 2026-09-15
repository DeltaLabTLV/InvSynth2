# Artifact status and provenance

This branch was refactored from commit `69d413a` of the earlier ISMIR-oriented
repository. No model was trained and no GPU evaluation was run during the
refactor.

## What is supported by retained manuscript records

- The U-Net architecture and masked-region objective.
- The four ICASSP configuration labels and 50k/10k versus 60k encoder-update
  allocations.
- Adam, batch size 32, gradient clipping 1.0, U-Net learning rate `1e-4`,
  weight decay `1e-6`, and 100-step Adam refinement at `1e-2`.
- Five independently generated dataset/proxy realizations per synthesizer.
- A proxy fitted separately for each realization and then shared by all four
  configurations in that paired run.
- The aggregate means and standard deviations in
  `paper/reported_aggregate_results.csv`.

## Problems found in the pre-refactor repository

The checked-out ISMIR code could not be cited as an exact implementation of
the current paper. In particular, it:

- selected five random splits of one dataset path rather than five separately
  generated dataset directories;
- trained the proxy through the 80% encoder-training loader, which supports
  the corrected manuscript statement that encoder-test examples were held
  out from proxy fitting;
- used a single 16-kHz, FFT-1024/hop-256 natural-log/z-score front end for all
  datasets, inconsistent with the reported `257 x 129` and `257 x 347`
  interfaces;
- contained explicitly provisional parameter schemas (including an `EDIT this`
  comment) that conflict with the paper's parameter counts;
- controlled training by epoch limits and early stopping rather than the
  reported 50k/10k/60k update budgets;
- averaged some metrics per batch rather than per example; and
- generated Griffin-Lim audio from proxy spectrograms even though the paper's
  listening study used hard-decoded presets rendered by real synthesizers.

Those paths must not be represented as producing the ICASSP numbers.

## What this branch changes

The ICASSP entry points use explicit run directories and split manifests,
train each run-specific proxy on `train`, monitor it on `val`, and exclude
`test` from fitting and checkpoint selection. A separate complete-set pass
computes the run-specific normalization extrema disclosed in the manuscript.
The code locks encoder stages to update counts, validates output tensor sizes,
performs reciprocal/log/reductions in FP32, and keeps the Transformer outside
the default paper workflow. Evaluation emits per-example records before
computing means and sample standard deviations.

## Items the authors must supply before an exact-reproduction claim

1. The 15 run-specific dataset directories or stable download instructions.
2. The exact `splits.csv` for each directory.
3. The exact `parameter_schema.yaml` for each synthesizer.
4. Run-specific feature extrema or the complete data required to recompute
   them, plus confirmation that the dB/mel ordering in `configs/icassp2027.yaml`
   matches the historical runs.
5. Proxy and encoder checkpoints, or the proxy-training budget and checkpoint
   rule used historically.
6. The five per-run metric JSON files underlying each reported mean/SD.
7. The true-synthesizer listening-test renders and anonymized ratings table if
   those artifacts can legally and ethically be released.

Until these are present, the repository is a reviewed implementation of the
stated protocol plus a transcription of aggregate results—not proof that the
checked-in code generated those aggregates.
