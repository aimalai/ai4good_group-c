"""Data-free forward/backward smoke test for the TemporalGRU experiment.

Run from the repository root:
    python experiments/tristan_temporal_gru/smoke_test.py

No DISFOR data and no GPU are required.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import torch

# Make importing the sibling model reliable even if this file is invoked differently.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import TemporalGRU  # noqa: E402
from forest_disturbance.models.loss import RecentDisturbanceLoss  # noqa: E402


def padded_series(lengths: list[int], *, last_day: date) -> dict[str, torch.Tensor]:
    batch = len(lengths)
    max_t = max(lengths)
    values = torch.zeros(batch, max_t, 12, 1, 1)
    dates = torch.zeros(batch, max_t, dtype=torch.long)
    mask = torch.zeros(batch, max_t, dtype=torch.bool)

    generator = torch.Generator().manual_seed(123)
    for row, length in enumerate(lengths):
        values[row, :length] = torch.rand(length, 12, 1, 1, generator=generator) * 0.5
        days = [last_day - timedelta(days=5 * (length - 1 - i)) for i in range(length)]
        dates[row, :length] = torch.tensor([d.toordinal() for d in days], dtype=torch.long)
        mask[row, :length] = True
    return {"values": values, "dates": dates, "mask": mask}


def main() -> None:
    torch.manual_seed(0)
    batch_size = 4
    target = date(2020, 6, 1)
    batch = {
        "inputs": {
            "s2": {
                "recent": padded_series([2, 4, 5, 3], last_day=target),
                "yearly": [
                    padded_series([3, 2, 4, 3], last_day=date(2019, 6, 10))
                ],
            }
        },
        "target": {
            "label": torch.tensor([0, 3, 6, 6], dtype=torch.long),
            "days_since_event": torch.tensor([10.0, 10.0, float("inf"), float("inf")]),
        },
    }

    model = TemporalGRU(num_classes=7)
    loss_fn = RecentDisturbanceLoss(no_disturbance_id=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(3):
        logits = model(batch)
        loss = loss_fn(logits, batch["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert logits.shape == (batch_size, 7), logits.shape
    assert torch.isfinite(logits).all()
    assert all(torch.isfinite(torch.tensor(losses)))

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("TemporalGRU smoke test PASSED")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"trainable parameters: {params:,}")
    print("losses:", ", ".join(f"{x:.4f}" for x in losses))


if __name__ == "__main__":
    main()
