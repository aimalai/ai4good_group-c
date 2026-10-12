#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c
export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance

echo "=== GRU alpha=0.6 ==="
uv run --frozen python experiments/tristan_temporal_gru/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1000 \
  logger.kind=csv \
  data.sampling_alpha=0.6 \
  run_name=gru_alpha06_1000

echo "=== GRU alpha=0.7 ==="
uv run --frozen python experiments/tristan_temporal_gru/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1000 \
  logger.kind=csv \
  data.sampling_alpha=0.7 \
  run_name=gru_alpha07_1000

echo "=== FINAL COMPARISON ==="

python3 - <<'PY'
import csv
import glob

experiments = [
    ("Baseline", "baseline_1000"),
    ("GRU a=.5", "gru_1000"),
    ("GRU a=.6", "gru_alpha06_1000"),
    ("GRU a=.7", "gru_alpha07_1000"),
    ("GRU a=.8", "gru_alpha08_1000"),
]

def latest(prefix):
    files = sorted(glob.glob(f"runs/{prefix}_*/**/metrics.csv", recursive=True))
    if not files:
        return None
    return files[-1]

def read_metric(path, key):
    if path is None:
        return float("nan")
    rows = list(csv.DictReader(open(path)))
    for row in reversed(rows):
        if row.get(key):
            return float(row[key])
    return float("nan")

paths = {name: latest(prefix) for name, prefix in experiments}

metrics = [
    ("Binary 14d", "operational/binary_f1_14d"),
    ("Binary 30d", "operational/binary_f1_30d"),
    ("Binary 60d", "operational/binary_f1_60d"),
    ("Binary 365d", "operational/binary_f1_365d"),
    ("Macro 14d", "operational/macro_f1_14d"),
    ("Macro 30d", "operational/macro_f1_30d"),
    ("Macro 60d", "operational/macro_f1_60d"),
    ("Macro 365d", "operational/macro_f1_365d"),
]

print()
print(f"{'Metric':<14}", end="")
for name, _ in experiments:
    print(f"{name:>12}", end="")
print()
print("-" * 74)

for label, key in metrics:
    print(f"{label:<14}", end="")
    for name, _ in experiments:
        print(f"{read_metric(paths[name], key):>12.3f}", end="")
    print()

print("\nHigher = better.")
print("\nFiles used:")
for name, _ in experiments:
    print(f"{name:12}: {paths[name]}")
PY
