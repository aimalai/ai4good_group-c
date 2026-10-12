#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance

echo "=================================================="
echo " MATCHED LONG RUN: BASELINE vs TEMPORAL GRU"
echo " 28,000 training batches each"
echo " SAME fold / sampling / loss / evaluation metrics"
echo "=================================================="

START_ALL=$SECONDS

echo
echo "=== 1/2 BASELINE: 28,000 BATCHES ==="
START=$SECONDS

uv run --frozen python scripts/train.py \
  --config configs/statistical_mlp.yaml \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=28000 \
  data.sampling_alpha=0.5 \
  logger.kind=csv \
  run_name=matched_baseline_28k

echo "Baseline finished in $(( (SECONDS-START)/60 )) minutes"

echo
echo "=== 2/2 TEMPORAL GRU: 28,000 BATCHES ==="
START=$SECONDS

uv run --frozen python experiments/tristan_temporal_gru/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=28000 \
  data.sampling_alpha=0.5 \
  logger.kind=csv \
  run_name=matched_gru_28k

echo "GRU finished in $(( (SECONDS-START)/60 )) minutes"

echo
echo "=================================================="
echo " FINAL MATCHED COMPARISON"
echo "=================================================="

python3 - <<'PY'
import csv
import glob

def latest(prefix):
    files = sorted(glob.glob(f"runs/{prefix}_*/metrics.csv"))
    if not files:
        raise RuntimeError(f"No run found for {prefix}")
    return files[-1]

def get(path, key):
    rows = list(csv.DictReader(open(path)))
    for row in reversed(rows):
        if row.get(key):
            return float(row[key])
    return float("nan")

baseline = latest("matched_baseline_28k")
gru = latest("matched_gru_28k")

metrics = [
    ("Binary 14d",  "operational/binary_f1_14d"),
    ("Binary 30d",  "operational/binary_f1_30d"),
    ("Binary 60d",  "operational/binary_f1_60d"),
    ("Binary 365d", "operational/binary_f1_365d"),
    ("Macro 14d",   "operational/macro_f1_14d"),
    ("Macro 30d",   "operational/macro_f1_30d"),
    ("Macro 60d",   "operational/macro_f1_60d"),
    ("Macro 365d",  "operational/macro_f1_365d"),
]

print()
print(f"{'Metric':<15}{'Baseline':>12}{'GRU':>12}{'Difference':>14}")
print("-" * 53)

for label, key in metrics:
    b = get(baseline, key)
    g = get(gru, key)
    print(f"{label:<15}{b:>12.3f}{g:>12.3f}{g-b:>+14.3f}")

print()
print("Validation loss:")
print(f"Baseline: {get(baseline, 'val/loss'):.4f}")
print(f"GRU:      {get(gru, 'val/loss'):.4f}")

print()
print("Files:")
print("Baseline:", baseline)
print("GRU:     ", gru)
PY

echo
echo "=================================================="
echo " TOTAL TIME: $(( (SECONDS-START_ALL)/60 )) minutes"
echo " MATCHED 28K EXPERIMENT COMPLETE"
echo "=================================================="
