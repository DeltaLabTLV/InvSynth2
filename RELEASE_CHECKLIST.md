# ICASSP 2027 release checklist

Run these checks from the repository root before creating the public tag.

```bash
python scripts/audit_icassp_artifact.py \
  --config configs/icassp2027.yaml --data-root datasets
python -m pytest -q
git status --short
```

The full audit is expected to fail until the 15 run-specific datasets,
manifests, and parameter schemas are installed. Resolve every failure from
the retained experiment records; do not substitute guessed defaults.

Before pushing:

- confirm the dB/mel order, feature extrema, proxy architecture, proxy budget,
  and checkpoint rule against the executed runs;
- place the five raw run JSON records behind every aggregate mean/SD under a
  documented release path;
- remove or move the legacy `data/` media tree to a historical release so it
  cannot be mistaken for the ICASSP evidence;
- confirm no Transformer configuration or result has entered the ICASSP
  release tag; the earlier implementation remains recoverable from Git history;
- verify that the manuscript's repository link resolves to the public tag;
- obtain approval from all three authors before publishing checkpoints,
  listener records, or licensed synthesizer renders; and
- create a commit and immutable tag only after `ARTIFACT_STATUS.md` can be
  updated from “candidate” to “exact reproduction.”

No new GPU experiment is required for these release checks; they are
provenance, configuration, packaging, and authorship tasks.
