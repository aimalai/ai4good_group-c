#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

EXP=experiments/tristan_firms_fusion
START=$SECONDS

echo "============================================================"
echo " NASA FIRMS + DUAL L-TAE S1/S2"
echo " fold 0 / 10 epochs"
echo " FIRMS features strictly BEFORE target date"
echo "============================================================"

uv run python \
  "$EXP/build_firms_features.py"

uv run python \
  "$EXP/train_firms.py" \
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
  run_name=full_dual_ltae_s1s2_firms_10ep

python3 - <<'PY'
import csv
import glob
import json
from pathlib import Path


EXP = Path(
    "experiments/"
    "tristan_firms_fusion"
)


def latest(prefix):
    paths = sorted(
        glob.glob(
            f"runs/{prefix}_*"
        )
    )

    if not paths:
        raise RuntimeError(
            prefix
        )

    return Path(
        paths[-1]
    )


def read_epochs(run):
    out = []

    for path in sorted(
        (
            run
            / "predictions"
        ).glob(
            "epoch_*_metrics.json"
        )
    ):
        epoch = int(
            path.stem.split(
                "_"
            )[1]
        )

        out.append(
            (
                epoch,
                json.loads(
                    path.read_text()
                ),
            )
        )

    return out


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

firms_run = latest(
    "full_dual_ltae_s1s2_firms_10ep"
)

control = read_epochs(
    control_run
)

firms = read_epochs(
    firms_run
)

metrics = [
    (
        "Binary 14d",
        "operational/"
        "binary_f1_14d",
    ),
    (
        "Binary 30d",
        "operational/"
        "binary_f1_30d",
    ),
    (
        "Binary 60d",
        "operational/"
        "binary_f1_60d",
    ),
    (
        "Binary 365d",
        "operational/"
        "binary_f1_365d",
    ),
    (
        "Macro 14d",
        "operational/"
        "macro_f1_14d",
    ),
    (
        "Macro 30d",
        "operational/"
        "macro_f1_30d",
    ),
    (
        "Macro 60d",
        "operational/"
        "macro_f1_60d",
    ),
    (
        "Macro 365d",
        "operational/"
        "macro_f1_365d",
    ),
]

c_final = control[-1][1]
f_final = firms[-1][1]

c_epoch, c_best = max(
    control,
    key=lambda x:
    v(
        x[1],
        "operational/"
        "macro_f1_60d",
    ),
)

f_epoch, f_best = max(
    firms,
    key=lambda x:
    v(
        x[1],
        "operational/"
        "macro_f1_60d",
    ),
)

out = (
    EXP
    / "comparison_firms_vs_ltae.csv"
)

with out.open(
    "w",
    newline="",
) as fh:

    w = csv.writer(fh)

    w.writerow(
        [
            "metric",
            "dual_ltae",
            "dual_ltae_plus_firms",
            "firms_minus_ltae",
        ]
    )

    for label, key in metrics:

        a = v(
            c_final,
            key,
        )

        b = v(
            f_final,
            key,
        )

        w.writerow(
            [
                label,
                a,
                b,
                b - a,
            ]
        )


print()
print(
    "============================================"
)
print(
    " FIRMS ABLATION RESULT"
)
print(
    "============================================"
)

print(
    f"{'Metric':<15}"
    f"{'L-TAE':>10}"
    f"{'+FIRMS':>10}"
    f"{'Change':>10}"
)

print(
    "-" * 45
)

for label, key in metrics:

    a = v(
        c_final,
        key,
    )

    b = v(
        f_final,
        key,
    )

    print(
        f"{label:<15}"
        f"{a:>10.3f}"
        f"{b:>10.3f}"
        f"{b-a:>+10.3f}"
    )


print()

print(
    "Original best Macro60:",
    f"epoch {c_epoch}",
    f"= {v(c_best, 'operational/macro_f1_60d'):.3f}",
)

print(
    "FIRMS best Macro60:",
    f"epoch {f_epoch}",
    f"= {v(f_best, 'operational/macro_f1_60d'):.3f}",
)

print(
    "Best-checkpoint change:",
    f"{v(f_best, 'operational/macro_f1_60d') - v(c_best, 'operational/macro_f1_60d'):+.3f}",
)


lines = [
    "# NASA FIRMS External-Data Ablation",
    "",
    "Dual L-TAE S1+S2 plus eight strictly past-only NASA FIRMS VIIRS active-fire features.",
    "",
    f"- Control: `{control_run}`",
    f"- +FIRMS: `{firms_run}`",
    "",
    "No FIRMS detection from the target date or future is used.",
    "",
    "## Final epoch",
    "",
    "| Metric | Dual L-TAE | +FIRMS | Change |",
    "|---|---:|---:|---:|",
]

for label, key in metrics:

    a = v(
        c_final,
        key,
    )

    b = v(
        f_final,
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
        f"- Dual L-TAE: "
        f"{v(c_best, 'operational/macro_f1_60d'):.3f} "
        f"(epoch {c_epoch})"
    ),
    (
        f"- +FIRMS: "
        f"{v(f_best, 'operational/macro_f1_60d'):.3f} "
        f"(epoch {f_epoch})"
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
    out,
)

print(
    "Saved:",
    EXP / "RESULTS.md",
)
PY

echo
echo "TOTAL WALL TIME: $(( (SECONDS-START)/60 )) minutes"
echo "FIRMS EXPERIMENT COMPLETE"
