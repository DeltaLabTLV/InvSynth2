# Masked Reconstruction Pretraining for Proxy-Based Synthesizer Inversion

Candidate reproducibility workspace for the ICASSP 2027 manuscript by Moshe
Laufer, Oren Barkan, and Noam Koenigstein. It must pass the artifact audit and
be checked against the authors' retained run records before being tagged as an
exact reproduction release.

Repository: <https://github.com/DeltaLabTLV/InvSynth2>

## Scope of this release

The ICASSP paper studies one 3.61M-parameter U-Net encoder under four complete
training configurations:

| Configuration | Masked updates | Supervised updates | Reconstruction objective |
|---|---:|---:|---|
| Masked + IMW | 50,000 | 10,000 | `0.7 * spec + 0.3 * imw` |
| Masked + spectral-only | 50,000 | 10,000 | `spec` |
| Masked + log mixture | 50,000 | 10,000 | `0.7 * spec + 0.3 * log` |
| Supervised + IMW | 0 | 60,000 | `0.7 * spec + 0.3 * imw` |

Every row receives 60,000 encoder updates. This matches update counts, not
FLOPs or wall-clock time: masked pretraining also updates a decoder that is
discarded before supervision.

The earlier Transformer implementation is recoverable from Git history but is
removed from the paper-facing source tree. It is not part of the ICASSP method,
results, commands, or release claims. Do not place Transformer outputs in an
ICASSP results directory or table.

The tracked `data/` tree is likewise a legacy media archive, not evidence for
the ICASSP ablations. It is omitted from the source-only submission package;
the paper-facing aggregate records live under `paper/`.

## Experimental unit and data boundary

The unit of replication is a complete run-specific data/proxy realization.
For each synthesizer and each seed `0..4`:

1. a new dataset is generated;
2. an explicit 80/10/10 encoder split is recorded in `splits.csv`;
3. one proxy is fitted on the encoder-training partition, monitored on the
   validation partition, and then frozen; encoder-test examples never update
   or select its weights; and
4. all four U-Net configurations share that dataset, split, feature statistics,
   and frozen proxy.

Accordingly, encoder-test examples are excluded from both proxy fitting and
masked/supervised encoder training. The spectral results remain
proxy-conditioned because the frozen learned proxy supplies the evaluation
domain; they are not true-synthesizer reconstruction metrics.

Expected layout:

```text
datasets/
  fm/seed_0/
    data/<stem>.wav
    labels/<stem>.npy
    splits.csv
    parameter_schema.yaml
  dx7/seed_0/...
  tal/seed_0/...
```

`splits.csv` has exactly two columns, `stem` and `split`, where `split` is one
of `train`, `val`, or `test`. A stem may occur once only. Proxy and encoder
optimization use `train`; proxy checkpoint monitoring uses `val`; final
evaluation uses `test`. The only complete-set pass computes the disclosed
run-specific normalization extrema. Use five independently generated
directories, not five random splits of one shared directory.

The original training datasets, commercial synthesizer plug-ins, run-specific
feature constants, checkpoints, and raw per-run metric logs are not committed
here. The checked-in aggregate table is a transcription of the authors'
retained records, not a substitute for those artifacts. See
[`ARTIFACT_STATUS.md`](ARTIFACT_STATUS.md) before making a reproduction claim.

## Paper-locked preprocessing

The ICASSP profiles in `configs/icassp2027.yaml` implement the manuscript's
fixed tensor supports:

| Dataset | Source audio | Analysis support | Transform | Output |
|---|---|---|---|---|
| FM | 16,000 samples at 16 kHz | right-zero-pad to 16,384 | Hann STFT, FFT 512, hop 128 | 257 x 129 |
| TAL | 16,000 samples at 16 kHz | right-zero-pad to 16,384 | Hann STFT, FFT 512, hop 128 | 257 x 129 |
| DX7 | 66,150 samples at 22.05 kHz | right-zero-pad to 88,576 | Hann STFT, FFT 1024, hop 256, 257 mel bins | 257 x 347 |

The centered STFT uses constant (zero) boundary padding. For analyzed length
`L`, its frame count is `1 + floor(L / hop)`. Magnitudes are converted to dB
with `20 log10(max(magnitude, 1e-6))`, floored at -120 dB, and affinely mapped
to `[-1, 1]` using run-specific extrema computed over the complete dataset.
Those extrema are saved and reused by the proxy, encoder, and evaluator.
Losses invert the affine map and dB compression; DX7 remains in mel-magnitude
coordinates and is never pseudo-inverted to a linear STFT.

## Installation

Python 3.10 and the pinned PyTorch/Lightning versions used for the reported
study are listed in `requirements-pip.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pip.txt
python -m pip install -e .
```

## Validate the artifact before training

```bash
python scripts/audit_icassp_artifact.py \
  --config configs/icassp2027.yaml \
  --data-root datasets
```

The audit checks all 15 dataset directories, split disjointness and expected
counts, audio/label pairing, feature shapes, declared parameter schemas, and
the paper's four-row training grid. It fails closed when required factual
artifacts are missing.

## Reproduce one run

The proxy is run-specific and must be trained before encoder configurations:

```bash
python scripts/train_proxy.py --study-config configs/icassp2027.yaml \
  --dataset fm --seed 0 --data-root datasets --run-dir runs

python scripts/pretrain.py --study-config configs/icassp2027.yaml \
  --dataset fm --seed 0 --data-root datasets --run-dir runs

python scripts/finetune.py --study-config configs/icassp2027.yaml \
  --dataset fm --seed 0 --configuration masked_imw \
  --feature-stats runs/fm/seed_0/feature_stats.json \
  --proxy-ckpt runs/fm/seed_0/proxy/proxy.pt \
  --encoder-ckpt runs/fm/seed_0/pretrain/encoder.pt --run-dir runs

python scripts/evaluate.py --study-config configs/icassp2027.yaml \
  --dataset fm --seed 0 --configuration masked_imw \
  --finetuned-ckpt runs/fm/seed_0/masked_imw/ckpts/<checkpoint>.ckpt \
  --feature-stats runs/fm/seed_0/feature_stats.json \
  --data-root datasets --apply-itf --itf-steps 100 --itf-lr 1e-2
```

To print or launch the full 3-dataset x 5-seed grid:

```bash
python scripts/run_icassp_study.py --dry-run
python scripts/run_icassp_study.py
```

The runner trains one proxy and one masked U-Net per dataset/seed, then runs
the three pretrained objectives and the 60k-update supervised control. It
never invokes Transformer code.

## Reported aggregate results

`paper/reported_aggregate_results.csv` contains the means and standard
deviations printed in the manuscript. `scripts/aggregate_results.py` computes
sample means/SDs from raw run JSON files and refuses to infer paired tests from
marginal summaries.

## Listening study

The paper used hard-decoded presets rendered by the target synthesizers.
Proxy spectrogram inversion or Griffin-Lim audio is not a substitute. The
repository includes only the listener-level aggregation utility; generation
of true-synthesizer stimuli requires the authors' licensed synthesizer setup.

## License

MIT. See [`LICENSE`](LICENSE).
