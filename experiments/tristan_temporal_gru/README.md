# Tristan experiment 01 — Temporal GRU

**Status:** ready for a data-free smoke test, then a short fold-0 run on `sushko`.

This folder is intentionally self-contained. It does **not** overwrite the professor's baseline
files or the Group C exploratory scripts.

## 1. Research question

> Does preserving temporal order improve **early operational forest-disturbance detection and
> attribution** compared with the provided StatisticalMLP, which summarizes each time window with
> mean/std statistics?

This is a deliberately conservative first experiment. We change one main modelling choice while
keeping the course data pipeline, labels, loss and evaluation exactly the same.

## 2. What changes vs. the baseline

Baseline:

```text
S2 centre-pixel frames -> per-frame MLP -> mean/std over time -> MLP -> 7 logits
```

This experiment:

```text
S2 centre-pixel frames -> per-frame MLP -> ordered GRU -> MLP -> 7 logits
```

There are separate GRUs for the recent window and same-season historical window. Their final hidden
states are concatenated before classification.

What deliberately stays unchanged for experiment 01:

- Sentinel-2 only
- centre pixel only (`pixel_cache`)
- 30-day recent window
- one yearly context window at `D - 1 year ± 15 days`
- class mapping
- rare-class sampling
- `RecentDisturbanceLoss`
- optimizer defaults
- operational alert filter and thresholds
- 14/30/60/365-day evaluation horizons

Keeping these fixed makes the comparison much easier to interpret.

## 3. Files

- `model.py` — new `TemporalGRU` model.
- `config.yaml` — independent experiment config copied from the baseline settings.
- `train.py` — experiment-only training entry point; reuses the existing course data/loss/metrics.
- `smoke_test.py` — no-data CPU forward/backward test.
- `CURRENT_FORK_AUDIT.md` — what was reviewed in the Group C fork before starting.
- `RESULTS.md` — append-only run log / comparison table.

## 4. Branch workflow

Create the branch from the **latest Group C main** before copying this folder in:

```bash
git checkout main
git pull
git checkout -b tristan/temporal-gru
```

Then copy `experiments/tristan_temporal_gru/` into the repository root. Check:

```bash
git status
```

At this stage, the only untracked/changed files should be under this experiment folder.

## 5. Cluster setup

On the course machines the dataset path posted for Group C is:

```bash
export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
```

Run commands from the repository root. The project uses `uv`, not Conda:

```bash
uv sync
source .venv/bin/activate
```

If the environment already exists and is current, activation is enough.

## 6. Step-by-step run plan

### Step A — data-free smoke test

Do this first. It should finish in seconds and does not need the dataset or GPU.

```bash
python experiments/tristan_temporal_gru/smoke_test.py
```

Expected ending:

```text
TemporalGRU smoke test PASSED
logits shape: (4, 7)
...
```

If this fails, stop and paste the complete traceback into our chat.

### Step B — verify the course tests still pass

Because we did not modify baseline files, they should behave exactly as before:

```bash
export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
pytest
```

### Step C — reproduce/confirm the provided baseline first

Before claiming improvement, run at least a short baseline check using the untouched baseline:

```bash
python scripts/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=200 \
  logger.kind=csv \
  run_name=baseline_smoke
```

This is only a pipeline check, not a score comparison.

### Step D — short GRU pipeline run

```bash
python experiments/tristan_temporal_gru/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=200 \
  logger.kind=csv \
  run_name=tristan_gru_smoke
```

Important things to capture from the terminal:

- number of training/validation examples
- printed parameter count
- any warnings/errors
- run folder path
- final validation metrics

If it works, paste the terminal tail and the generated
`predictions/epoch_000_metrics.json` back into chat.

### Step E — meaningful fold-0 comparison

Only after the smoke run is clean:

```bash
python scripts/train.py \
  --config configs/statistical_mlp.yaml \
  data.fold=0 \
  logger.kind=csv \
  run_name=baseline_fold0_reproduction
```

and

```bash
python experiments/tristan_temporal_gru/train.py \
  data.fold=0 \
  logger.kind=csv \
  run_name=tristan_gru_fold0
```

Use the same fold, epochs and other settings. Record the operational F1 values in `RESULTS.md`.

## 7. What counts as progress

The main target is **not frame accuracy**. Compare operational:

- `binary_f1_14d`, `binary_f1_30d`, `binary_f1_60d`, `binary_f1_365d`
- `macro_f1_14d`, `macro_f1_30d`, `macro_f1_60d`, `macro_f1_365d`

The most interesting gains are at 14/30/60 days because the project is about timely alerts.

A useful result can also be negative: if the GRU does not improve early F1, that tells us sequence
order alone is not enough and gives a principled reason to move to data quality, Sentinel-1 fusion,
or spatial context.

## 8. Planned next experiments (do not implement yet)

We should choose these based on experiment-01 results rather than changing everything at once:

1. **Time-gap feature** — give the sequence encoder explicit day gaps / relative time.
2. **Cloud/no-data handling** — mask or flag corrupted centre-pixel S2 observations.
3. **Sentinel-1 + Sentinel-2 fusion** — test whether radar improves early detection under cloud gaps.
4. **Small spatial encoder** — only if we need the 1.7 TB patch data and have a clear reason.
5. **Transformer** — only after a smaller recurrent model establishes whether temporal order helps.

## 9. Reproducibility rules for us

For every meaningful run:

1. use a descriptive `run_name`;
2. keep the automatically saved run `config.yaml`;
3. record branch + commit in `RESULTS.md`;
4. do not tune on multiple folds while developing — use fold 0;
5. for a final reported result, train all five folds and concatenate validation predictions before
   computing metrics once, as required by the course repository.

## 10. Literature rationale

The experiment is consistent with current forest-monitoring literature rather than being an arbitrary
architecture swap. Recent work has demonstrated Transformer-based Sentinel-2 time-series disturbance
detection in Central Europe, and the 2026 DISFOR description explicitly frames the dataset for
multi-temporal disturbance-agent classification. We start with a GRU because it is much smaller and
cheaper to debug; if temporal order helps, a Transformer becomes a justified next step.

See the project chat for the exact open-access paper links and citations used to motivate this choice.
