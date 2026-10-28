#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

EXP=experiments/tristan_ltae_late_fusion
START=$SECONDS

echo "============================================================"
echo " DUAL L-TAE S1+S2 FINE-TUNE"
echo " base: completed 10-epoch Dual L-TAE"
echo " extra epochs: 5"
echo " fresh LR: 3e-5"
echo " aux weight: 0.10"
echo " acquisition dropout: S2=.05 S1=.05"
echo "============================================================"

uv run python "$EXP/finetune.py" \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.sensors='[s1,s2]' \
  data.target_sensors='[s2]' \
  data.normalization=null \
  data.fold=0 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  optimizer.lr=0.00003 \
  optimizer.weight_decay=0.01 \
  trainer.max_epochs=5 \
  trainer.limit_train_batches=1.0 \
  trainer.precision=bf16-mixed \
  logger.kind=csv \
  run_name=finetune_dual_ltae_s1s2_5ep

python3 - <<'PY'
import csv
import glob
import json
from pathlib import Path


EXP = Path("experiments/tristan_ltae_late_fusion")


def latest(prefix):
    paths = sorted(glob.glob(f"runs/{prefix}_*"))
    if not paths:
        raise RuntimeError(f"No run found for {prefix}")
    return Path(paths[-1])


def epoch_metrics(run):
    files = sorted(
        (run / "predictions").glob("epoch_*_metrics.json")
    )
    if not files:
        raise RuntimeError(f"No metrics in {run}")

    rows = []
    for f in files:
        epoch = int(f.stem.split("_")[1])
        rows.append((epoch, json.loads(f.read_text())))
    return rows


def v(m, key):
    x = m.get(key)
    return float("nan") if x is None else float(x)


base_run = latest("full_dual_ltae_s1s2_10ep")
ft_run = latest("finetune_dual_ltae_s1s2_5ep")

base_rows = epoch_metrics(base_run)
ft_rows = epoch_metrics(ft_run)

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

base_final = base_rows[-1][1]
ft_final = ft_rows[-1][1]

base_best_epoch, base_best = max(
    base_rows,
    key=lambda r: v(r[1], "operational/macro_f1_60d"),
)

ft_best_epoch, ft_best = max(
    ft_rows,
    key=lambda r: v(r[1], "operational/macro_f1_60d"),
)

out_csv = EXP / "finetune_comparison.csv"

with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "metric",
        "original_final",
        "finetune_final",
        "finetune_minus_original",
        "finetune_best_macro60_epoch_value",
    ])

    for label, key in metrics:
        a = v(base_final, key)
        b = v(ft_final, key)

        w.writerow([
            label,
            a,
            b,
            b - a,
            v(ft_best, key),
        ])


print()
print("============================================================")
print(" FINE-TUNE RESULT")
print("============================================================")

print(
    f"{'Metric':<15}"
    f"{'Original':>11}"
    f"{'FT final':>11}"
    f"{'Change':>10}"
    f"{'FT best*':>11}"
)

print("-" * 58)

for label, key in metrics:
    a = v(base_final, key)
    b = v(ft_final, key)
    c = v(ft_best, key)

    print(
        f"{label:<15}"
        f"{a:>11.3f}"
        f"{b:>11.3f}"
        f"{b-a:>+10.3f}"
        f"{c:>11.3f}"
    )

print()
print("* FT best = fine-tune epoch selected by Macro F1 @ 60d")
print()
print(
    f"Original best Macro60: "
    f"epoch {base_best_epoch} = "
    f"{v(base_best, 'operational/macro_f1_60d'):.3f}"
)

print(
    f"Fine-tune best Macro60: "
    f"epoch {ft_best_epoch} = "
    f"{v(ft_best, 'operational/macro_f1_60d'):.3f}"
)

gain = (
    v(ft_best, "operational/macro_f1_60d")
    - v(base_best, "operational/macro_f1_60d")
)

print(f"Macro60 improvement: {gain:+.3f}")

lines = [
    "# Dual L-TAE Fine-Tuning",
    "",
    f"- Original run: `{base_run}`",
    f"- Fine-tune run: `{ft_run}`",
    "",
    "Fine-tuning starts from the completed 10-epoch Dual L-TAE weights.",
    "The old optimizer/scheduler is NOT restored.",
    "",
    "- Additional epochs: 5",
    "- Learning rate: 3e-5 with cosine decay",
    "- Auxiliary branch loss weight: 0.10 each",
    "- S2 acquisition dropout: 0.05",
    "- S1 acquisition dropout: 0.05",
    "",
    "## Best Macro F1 @ 60d",
    "",
    f"- Original: {v(base_best, 'operational/macro_f1_60d'):.3f}",
    f"- Fine-tuned: {v(ft_best, 'operational/macro_f1_60d'):.3f}",
    f"- Change: {gain:+.3f}",
    "",
    "The fine-tuned checkpoint is selected only for comparison using "
    "Macro F1 @ 60d; all operational metrics should still be reported.",
]

(EXP / "FINETUNE_RESULTS.md").write_text(
    "\n".join(lines) + "\n"
)

print()
print("Saved:", out_csv)
print("Saved:", EXP / "FINETUNE_RESULTS.md")
PY

echo
echo "TOTAL WALL TIME: $(( (SECONDS-START)/60 )) minutes"
echo "FINE-TUNE COMPLETE"
