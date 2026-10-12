#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c
export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance

echo "=== TRAINING GRU WITH MORE RARE-CLASS SAMPLING ==="

uv run --frozen python experiments/tristan_temporal_gru/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1000 \
  logger.kind=csv \
  data.sampling_alpha=0.8 \
  run_name=gru_alpha08_1000

echo "=== COMPARING OPERATIONAL F1 SCORES ==="

python3 - <<'PY'
import csv
import glob

experiments = [
    ("Baseline", "baseline_1000"),
    ("Original GRU", "gru_1000"),
    ("GRU alpha=0.8", "gru_alpha08_1000"),
]

results = {}

for name, prefix in experiments:
    files = sorted(glob.glob(
        f"runs/{prefix}_*/**/metrics.csv", recursive=True
    ))
    if not files:
        raise SystemExit(f"Missing metrics for {name}")

    with open(files[-1], newline="") as f:
        results[name] = list(csv.DictReader(f))

def get_metric(rows, key):
    for row in reversed(rows):
        value = row.get(key)
        if value:
            return float(value)
    return float("nan")

print(f"{'Metric':<23} {'Baseline':>10} {'GRU':>10} {'Alpha 0.8':>10}")
print("-" * 58)

for label, prefix in [
    ("Binary F1", "operational/binary_f1"),
    ("Cause macro F1", "operational/macro_f1"),
]:
    for days in (14, 30, 60, 365):
        key = f"{prefix}_{days}d"
        values = [
            get_metric(results[name], key)
            for name, _ in experiments
        ]
        print(
            f"{label + ' ' + str(days) + 'd':<23}"
            + "".join(f"{v:>10.3f}" for v in values)
        )

print("\nHigher F1 is better.")
PY
