#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export HF_HOME=/zfs/ai4good/student/thein/hf_cache
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

EPOCHS=4
START_ALL=$SECONDS

echo "============================================================"
echo " FULL-DATA MATCHED COMPARISON"
echo " Baseline vs TESSERA v2"
echo " $EPOCHS epochs"
echo " 100% of training batches every epoch"
echo " fold 0 / alpha 0.5 / same loss / same operational metrics"
echo "============================================================"

echo
echo "=== 1/2 BASELINE ==="
START=$SECONDS

uv run --frozen python scripts/train.py \
  --config configs/statistical_mlp.yaml \
  data.fold=0 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  trainer.max_epochs=$EPOCHS \
  trainer.limit_train_batches=1.0 \
  logger.kind=csv \
  run_name=full_baseline_4ep

echo
echo "Baseline finished in $(( (SECONDS-START)/60 )) minutes"

echo
echo "=== 2/2 TESSERA v2 ==="
START=$SECONDS

uv run \
  --with "huggingface_hub>=0.26" \
  python experiments/tristan_tessera_v2/train.py \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.sensors='[s2]' \
  data.target_sensors='[s2]' \
  data.normalization=null \
  data.fold=0 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  trainer.max_epochs=$EPOCHS \
  trainer.limit_train_batches=1.0 \
  logger.kind=csv \
  run_name=full_tessera_v2_4ep

echo
echo "TESSERA finished in $(( (SECONDS-START)/60 )) minutes"

echo
echo "============================================================"
echo " FINAL FULL-DATA COMPARISON"
echo "============================================================"

python3 - <<'PY'
import csv
import glob
import json
from pathlib import Path

def latest(prefix):
    runs = sorted(glob.glob(f"runs/{prefix}_*"))
    if not runs:
        raise RuntimeError(f"No run found: {prefix}")
    return Path(runs[-1])

baseline = latest("full_baseline_4ep")
tessera = latest("full_tessera_v2_4ep")

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

def epoch_jsons(run):
    return sorted((run / "predictions").glob("epoch_*_metrics.json"))

def load_epochs(run):
    out = []
    for f in epoch_jsons(run):
        epoch = int(f.stem.split("_")[1])
        out.append((epoch, json.loads(f.read_text())))
    return out

def val_losses(run):
    rows = list(csv.DictReader(open(run / "metrics.csv")))
    result = {}
    for r in rows:
        if r.get("epoch") and r.get("val/loss"):
            result[int(float(r["epoch"]))] = float(r["val/loss"])
    return result

b_epochs = load_epochs(baseline)
t_epochs = load_epochs(tessera)
b_loss = val_losses(baseline)
t_loss = val_losses(tessera)

print()
print("RUNS")
print("Baseline:", baseline)
print("TESSERA :", tessera)

print()
print("PER-EPOCH SUMMARY")
print()

for e in range(min(len(b_epochs), len(t_epochs))):
    _, b = b_epochs[e]
    _, t = t_epochs[e]

    print(f"Epoch {e}")
    print(
        f"  val loss       "
        f"baseline={b_loss.get(e, float('nan')):.4f}  "
        f"TESSERA={t_loss.get(e, float('nan')):.4f}"
    )
    print(
        f"  binary 60d     "
        f"baseline={b['operational/binary_f1_60d']:.3f}  "
        f"TESSERA={t['operational/binary_f1_60d']:.3f}"
    )
    print(
        f"  macro 60d      "
        f"baseline={b['operational/macro_f1_60d']:.3f}  "
        f"TESSERA={t['operational/macro_f1_60d']:.3f}"
    )
    print(
        f"  binary 365d    "
        f"baseline={b['operational/binary_f1_365d']:.3f}  "
        f"TESSERA={t['operational/binary_f1_365d']:.3f}"
    )
    print(
        f"  macro 365d     "
        f"baseline={b['operational/macro_f1_365d']:.3f}  "
        f"TESSERA={t['operational/macro_f1_365d']:.3f}"
    )
    print()

b = b_epochs[-1][1]
t = t_epochs[-1][1]

print("FINAL EPOCH")
print()
print(
    f"{'Metric':<15}"
    f"{'Baseline':>12}"
    f"{'TESSERA':>12}"
    f"{'Difference':>14}"
)
print("-" * 53)

for label, key in metrics:
    bv = float(b[key])
    tv = float(t[key])
    print(
        f"{label:<15}"
        f"{bv:>12.3f}"
        f"{tv:>12.3f}"
        f"{tv-bv:>+14.3f}"
    )

last = len(b_epochs) - 1

print()
print("Final validation loss:")
print(f"Baseline: {b_loss.get(last, float('nan')):.4f}")
print(f"TESSERA : {t_loss.get(last, float('nan')):.4f}")

print()
print("Training scale:")
print("  ~1,017,043 sampled training examples per epoch")
print(f"  {len(b_epochs)} complete epochs per model")
print(
    f"  ~{1_017_043 * len(b_epochs):,} training draws per model"
)
PY

echo
echo "============================================================"
echo " TOTAL WALL TIME: $(( (SECONDS-START_ALL)/60 )) minutes"
echo " FULL COMPARISON COMPLETE"
echo "============================================================"
