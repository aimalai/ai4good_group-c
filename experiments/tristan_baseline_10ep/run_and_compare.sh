#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

EXP_DIR=experiments/tristan_baseline_10ep
mkdir -p "$EXP_DIR"

echo "============================================================"
echo "FULL 10-EPOCH BASELINE"
echo "fold=0"
echo "S2 only"
echo "pixel_cache"
echo "batch_size=16"
echo "num_workers=8"
echo "sampling_alpha=0.5"
echo "10 full epochs"
echo "same operational evaluation"
echo "============================================================"

START=$SECONDS

uv run python scripts/train.py \
  --config configs/statistical_mlp.yaml \
  data.fold=0 \
  data.sensors='[s2]' \
  data.target_sensors='[s2]' \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  trainer.max_epochs=10 \
  trainer.limit_train_batches=1.0 \
  logger.kind=csv \
  run_name=full_baseline_10ep \
  2>&1 | tee "$EXP_DIR/baseline_10ep.log"

echo
echo "Baseline finished in $(( (SECONDS-START)/60 )) minutes"
echo
echo "Creating saved result summaries..."

uv run python - <<'PY'
import csv
import glob
import json
from pathlib import Path

OUT = Path("experiments/tristan_baseline_10ep")
OUT.mkdir(parents=True, exist_ok=True)

def latest(prefix):
    runs = sorted(glob.glob(f"runs/{prefix}_*"))
    if not runs:
        raise RuntimeError(f"No run found for {prefix}")
    return Path(runs[-1])

def load_epochs(run):
    files = sorted((run / "predictions").glob("epoch_*_metrics.json"))
    rows = []
    for f in files:
        epoch = int(f.stem.split("_")[1])
        rows.append((epoch, json.loads(f.read_text())))
    if not rows:
        raise RuntimeError(f"No epoch metric files in {run}")
    return rows

def metric(m, key):
    v = m.get(key)
    return float("nan") if v is None else float(v)

KEYS = [
    ("Binary 14d", "operational/binary_f1_14d"),
    ("Binary 30d", "operational/binary_f1_30d"),
    ("Binary 60d", "operational/binary_f1_60d"),
    ("Binary 365d", "operational/binary_f1_365d"),
    ("Macro 14d", "operational/macro_f1_14d"),
    ("Macro 30d", "operational/macro_f1_30d"),
    ("Macro 60d", "operational/macro_f1_60d"),
    ("Macro 365d", "operational/macro_f1_365d"),
]

baseline_run = latest("full_baseline_10ep")
s2_run = latest("full_tessera_s2_10ep")
fusion_run = latest("full_tessera_s1s2_10ep")

baseline = load_epochs(baseline_run)
s2 = load_epochs(s2_run)
fusion = load_epochs(fusion_run)

# ------------------------------------------------------------
# Save baseline epoch-by-epoch metrics
# ------------------------------------------------------------

with (OUT / "baseline_epochs.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch"] + [name for name, _ in KEYS])
    for epoch, m in baseline:
        w.writerow([epoch] + [metric(m, key) for _, key in KEYS])

# ------------------------------------------------------------
# Final-epoch 3-way comparison
# ------------------------------------------------------------

models = {
    "Baseline": baseline[-1][1],
    "TESSERA S2": s2[-1][1],
    "TESSERA S1+S2": fusion[-1][1],
}

rows = []

for name, key in KEYS:
    b = metric(models["Baseline"], key)
    s = metric(models["TESSERA S2"], key)
    f = metric(models["TESSERA S1+S2"], key)

    rows.append({
        "metric": name,
        "baseline": b,
        "tessera_s2": s,
        "tessera_s1s2": f,
        "s2_minus_baseline": s - b,
        "fusion_minus_baseline": f - b,
        "fusion_minus_s2": f - s,
    })

with (OUT / "comparison_3way.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

summary = {
    "runs": {
        "baseline": str(baseline_run),
        "tessera_s2": str(s2_run),
        "tessera_s1s2": str(fusion_run),
    },
    "training": {
        "fold": 0,
        "epochs": 10,
        "batch_size": 16,
        "num_workers": 8,
        "sampling_alpha": 0.5,
        "target_sensor": "s2",
    },
    "final_epoch": rows,
}

best = {}

for label, epochs in [
    ("baseline", baseline),
    ("tessera_s2", s2),
    ("tessera_s1s2", fusion),
]:
    epoch, m = max(
        epochs,
        key=lambda x: metric(
            x[1],
            "operational/macro_f1_60d",
        ),
    )

    best[label] = {
        "epoch": epoch,
        "macro_f1_60d": metric(
            m,
            "operational/macro_f1_60d",
        ),
        "binary_f1_60d": metric(
            m,
            "operational/binary_f1_60d",
        ),
        "macro_f1_365d": metric(
            m,
            "operational/macro_f1_365d",
        ),
        "binary_f1_365d": metric(
            m,
            "operational/binary_f1_365d",
        ),
    }

summary["best_epoch_by_macro60"] = best

(OUT / "comparison_3way.json").write_text(
    json.dumps(summary, indent=2)
)

# ------------------------------------------------------------
# Markdown summary for documentation / paper work
# ------------------------------------------------------------

lines = [
    "# 10-Epoch Fold-0 Comparison",
    "",
    "All three models use the same fold and operational evaluation.",
    "",
    f"- Baseline: `{baseline_run}`",
    f"- TESSERA S2: `{s2_run}`",
    f"- TESSERA S1+S2: `{fusion_run}`",
    "",
    "## Final epoch",
    "",
    "| Metric | Baseline | TESSERA S2 | TESSERA S1+S2 | Fusion vs baseline |",
    "|---|---:|---:|---:|---:|",
]

for r in rows:
    lines.append(
        f"| {r['metric']} "
        f"| {r['baseline']:.3f} "
        f"| {r['tessera_s2']:.3f} "
        f"| {r['tessera_s1s2']:.3f} "
        f"| {r['fusion_minus_baseline']:+.3f} |"
    )

lines += [
    "",
    "## Best epoch by Macro F1 @ 60 days",
    "",
]

for label, x in best.items():
    lines.append(
        f"- **{label}**: epoch {x['epoch']}, "
        f"Macro60={x['macro_f1_60d']:.3f}, "
        f"Binary60={x['binary_f1_60d']:.3f}, "
        f"Macro365={x['macro_f1_365d']:.3f}, "
        f"Binary365={x['binary_f1_365d']:.3f}"
    )

(OUT / "README.md").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------
# Print final table
# ------------------------------------------------------------

print()
print("=" * 76)
print("FINAL 10-EPOCH THREE-WAY COMPARISON")
print("=" * 76)
print()

print(
    f"{'Metric':<15}"
    f"{'Baseline':>12}"
    f"{'TESSERA S2':>14}"
    f"{'S1+S2':>12}"
    f"{'Fusion-BL':>12}"
)

print("-" * 65)

for r in rows:
    print(
        f"{r['metric']:<15}"
        f"{r['baseline']:>12.3f}"
        f"{r['tessera_s2']:>14.3f}"
        f"{r['tessera_s1s2']:>12.3f}"
        f"{r['fusion_minus_baseline']:>+12.3f}"
    )

print()
print("Saved:")
print(" ", OUT / "baseline_epochs.csv")
print(" ", OUT / "comparison_3way.csv")
print(" ", OUT / "comparison_3way.json")
print(" ", OUT / "README.md")
print()
PY

echo
echo "============================================================"
echo "BASELINE + THREE-WAY COMPARISON COMPLETE"
echo "============================================================"
