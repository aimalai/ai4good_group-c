# Current Group C fork audit — 2026-10-08

This note records the repository state that was reviewed before starting the TemporalGRU experiment.
The source reviewed was the ZIP of `aimalai/ai4good_group-c` supplied by Tristan on 2026-10-08.

## What changed relative to the starter snapshot previously reviewed

The core course code is unchanged in the supplied ZIP: the data pipeline, baseline model, loss,
metrics, training/evaluation scripts, tests, config and package metadata match the earlier starter
snapshot that was reviewed in this project.

The Group C fork adds four exploratory scripts:

- `scripts/explore_stats.py`
- `scripts/explore_granular.py`
- `scripts/explore_sample_timeseries.py`
- `scripts/explore_sentinel_lags.py`

and extends `.gitignore` to exclude data/large artifacts (`data/`, `*.zarr/`, `*.parquet`, `*.png`).

We are **not modifying those Group C exploratory files in this experiment**. The experiment lives
entirely under `experiments/tristan_temporal_gru/` so it can be developed and reviewed independently.

## Important cautions found during the audit

These are notes for discussion, not edits to somebody else's work.

### 1. Human-readable label names in two exploratory scripts do not match the repository labels

`explore_stats.py` and `explore_granular.py` contain ad-hoc dictionaries such as `110: "Minor Loss"`
and `212: "Clear-cut/Abrupt"`. According to the repository's canonical
`forest_disturbance/data/constants.py` and `configs/label_mapping.yaml`, examples include:

- 110 = Undisturbed Forest
- 120 = Revegetation
- 121 = With Trees (after clear cut)
- 122 = Canopy closing (after thinning/defoliation)
- 123 = Without Trees
- 211 = Clear Cut
- 212 = Thinning
- 222 = Salvage after abiotic disturbance
- 243 = Wind

Therefore plots from those scripts should not be interpreted using the current ad-hoc names until
that mapping is corrected or replaced by the repository's canonical mapping.

### 2. `explore_sample_timeseries.py` chooses DataFrame index values as sample IDs

The script uses `samples.index[:3].tolist()`. That is only guaranteed to be correct if the parquet
row index happens to equal `sample_id`. Using the explicit `sample_id` column would be safer.

### 3. The spatial panel in `explore_granular.py` looks for coordinates in `center_pixels.parquet`

The documented purpose of `center_pixels.parquet` is the spectral centre-pixel cache. Location
metadata belongs conceptually to `samples.parquet`. If coordinate columns are not present in the
cache, the script falls back to plotting its row index, which is not a geographic distribution.

### 4. `evidence_lag_days = end_evidence - start` is not automatically a detection delay

For event-like annotations this can be a short evidence interval, while continuous biotic labels can
span much longer periods. It should be described as annotation/evidence interval duration unless the
data semantics justify a stronger interpretation.

### 5. The new exploratory scripts import `pandas`, but the project environment does not declare it

All four new exploratory scripts import `pandas as pd`. The current `pyproject.toml` and `uv.lock`
do not declare/lock pandas. Therefore a clean `uv sync` environment is not guaranteed to run those
scripts. This should be fixed deliberately (either add pandas to the project dependencies or rewrite
the small analyses with the already-declared Polars dependency) before treating the scripts as
reproducible project tooling.

## Baseline facts we preserve

- Development: fold 0.
- Input: Sentinel-2 centre pixel, recent 30 days + same season one year earlier ±15 days.
- Classes: six disturbance agents + No Disturbance.
- Loss: `RecentDisturbanceLoss` with the existing forgetting schedule.
- Main evaluation: operational binary F1 and macro F1 at 14/30/60/365 days.
- Final reporting: train all five folds and concatenate validation predictions before one evaluation;
  do not average five fold-level F1 values.

## Why the first new experiment is a GRU

The baseline embeds each acquisition but then collapses time to mean/std statistics. That removes
most sequence-order information. A small GRU changes only that component while preserving the
existing data, labels, loss and evaluation. This makes the experiment easy to attribute and cheap
enough to iterate on the shared L40S.
