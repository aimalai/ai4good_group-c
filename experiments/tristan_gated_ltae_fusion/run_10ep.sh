#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

EXP=experiments/tristan_gated_ltae_fusion
START=$SECONDS

echo
echo "============================================================"
echo " GATED DUAL L-TAE S1+S2"
echo " clean matched fusion ablation"
echo
echo " fold 0"
echo " 10 epochs"
echo " batch 16"
echo " alpha 0.5"
echo " no foundation model"
echo " only major change: learned S1/S2 fusion gate"
echo "============================================================"
echo

uv run python \
  "$EXP/train.py" \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.sensors='[s1,s2]' \
  data.target_sensors='[s2]' \
  data.normalization=null \
  data.fold=0 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  optimizer.lr=0.0003 \
  optimizer.weight_decay=0.01 \
  trainer.max_epochs=10 \
  trainer.limit_train_batches=1.0 \
  trainer.precision=bf16-mixed \
  logger.kind=csv \
  run_name=full_gated_dual_ltae_s1s2_10ep

python3 - <<'PY'
import csv
import glob
import json
from pathlib import Path


EXP = Path(
    "experiments/"
    "tristan_gated_ltae_fusion"
)


def latest(prefix):
    paths = sorted(
        glob.glob(
            f"runs/{prefix}_*"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No run found: {prefix}"
        )

    return Path(
        paths[-1]
    )


def epochs(run):
    files = sorted(
        (
            run
            / "predictions"
        ).glob(
            "epoch_*_metrics.json"
        )
    )

    if not files:
        raise RuntimeError(
            f"No metrics found in {run}"
        )

    result = []

    for path in files:

        epoch = int(
            path.stem.split(
                "_"
            )[1]
        )

        result.append(
            (
                epoch,
                json.loads(
                    path.read_text()
                ),
            )
        )

    return result


def v(m, key):
    x = m.get(key)

    return (
        float("nan")
        if x is None
        else float(x)
    )


control_run = latest(
    "full_dual_ltae_s1s2_10ep"
)

gated_run = latest(
    "full_gated_dual_ltae_s1s2_10ep"
)

control = epochs(
    control_run
)

gated = epochs(
    gated_run
)


metrics = [
    (
        "Binary 14d",
        "operational/binary_f1_14d",
    ),

    (
        "Binary 30d",
        "operational/binary_f1_30d",
    ),

    (
        "Binary 60d",
        "operational/binary_f1_60d",
    ),

    (
        "Binary 365d",
        "operational/binary_f1_365d",
    ),

    (
        "Macro 14d",
        "operational/macro_f1_14d",
    ),

    (
        "Macro 30d",
        "operational/macro_f1_30d",
    ),

    (
        "Macro 60d",
        "operational/macro_f1_60d",
    ),

    (
        "Macro 365d",
        "operational/macro_f1_365d",
    ),
]


control_final = (
    control[-1][1]
)

gated_final = (
    gated[-1][1]
)


control_best_epoch, control_best = max(
    control,

    key=lambda row:
        v(
            row[1],
            "operational/macro_f1_60d",
        ),
)


gated_best_epoch, gated_best = max(
    gated,

    key=lambda row:
        v(
            row[1],
            "operational/macro_f1_60d",
        ),
)


csv_path = (
    EXP
    / "comparison_gated_vs_ltae.csv"
)


with csv_path.open(
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "metric",
            "dual_ltae",
            "gated_dual_ltae",
            "gated_minus_ltae",
        ]
    )

    for label, key in metrics:

        a = v(
            control_final,
            key,
        )

        b = v(
            gated_final,
            key,
        )

        writer.writerow(
            [
                label,
                a,
                b,
                b - a,
            ]
        )


print()
print(
    "============================================================"
)
print(
    " GATED FUSION RESULT"
)
print(
    "============================================================"
)

print(
    f"{'Metric':<15}"
    f"{'Dual LTAE':>12}"
    f"{'Gated':>12}"
    f"{'Change':>11}"
)

print("-" * 50)


for label, key in metrics:

    a = v(
        control_final,
        key,
    )

    b = v(
        gated_final,
        key,
    )

    print(
        f"{label:<15}"
        f"{a:>12.3f}"
        f"{b:>12.3f}"
        f"{b-a:>+11.3f}"
    )


print()
print(
    "Original best Macro60:",
    f"epoch {control_best_epoch}",
    f"= {v(control_best, 'operational/macro_f1_60d'):.3f}",
)

print(
    "Gated best Macro60:",
    f"epoch {gated_best_epoch}",
    f"= {v(gated_best, 'operational/macro_f1_60d'):.3f}",
)

gain = (
    v(
        gated_best,
        "operational/macro_f1_60d",
    )
    -
    v(
        control_best,
        "operational/macro_f1_60d",
    )
)

print(
    "Best-checkpoint change:",
    f"{gain:+.3f}",
)


lines = [
    "# Gated Dual L-TAE S1+S2",
    "",
    "Matched fold-0 ablation of learned multimodal gating versus the original Dual L-TAE concatenation fusion.",
    "",
    "- Same S1/S2 temporal encoders",
    "- Same fold 0",
    "- Same 10 epochs",
    "- Same batch size and sampling",
    "- Same operational evaluation",
    "- No foundation model",
    "",
    f"- Control: `{control_run}`",
    f"- Gated: `{gated_run}`",
    "",
    "## Final epoch",
    "",
    "| Metric | Dual L-TAE | Gated | Change |",
    "|---|---:|---:|---:|",
]


for label, key in metrics:

    a = v(
        control_final,
        key,
    )

    b = v(
        gated_final,
        key,
    )

    lines.append(
        f"| {label} | "
        f"{a:.3f} | "
        f"{b:.3f} | "
        f"{b-a:+.3f} |"
    )


lines += [
    "",
    "## Best Macro F1 @ 60d",
    "",
    (
        f"- Original Dual L-TAE: "
        f"{v(control_best, 'operational/macro_f1_60d'):.3f} "
        f"(epoch {control_best_epoch})"
    ),
    (
        f"- Gated Dual L-TAE: "
        f"{v(gated_best, 'operational/macro_f1_60d'):.3f} "
        f"(epoch {gated_best_epoch})"
    ),
    (
        f"- Change: {gain:+.3f}"
    ),
]


(
    EXP
    / "RESULTS.md"
).write_text(
    "\n".join(lines)
    + "\n"
)


print()
print(
    "Saved:",
    csv_path,
)

print(
    "Saved:",
    EXP / "RESULTS.md",
)
PY

echo
echo "TOTAL WALL TIME: $(( (SECONDS-START)/60 )) minutes"
echo "GATED L-TAE EXPERIMENT COMPLETE"
