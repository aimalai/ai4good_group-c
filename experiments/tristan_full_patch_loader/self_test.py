"""Fast synthetic correctness tests before touching the real dataset."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import zarr

from loader import (
    CUDAPrefetcher,
    ZarrPatchDataset,
    discover,
    make_loader,
)


def create_array(path, values):
    arr = zarr.open_array(
        str(path),
        mode="w",
        shape=values.shape,
        dtype=values.dtype,
    )
    arr[:] = values


def check_layout(layout, values):
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "sample.zarr"
        create_array(store, values)

        specs = discover(
            Path(tmp),
            crop=16,
            layout=layout,
            max_stores=1,
        )

        dataset = ZarrPatchDataset(
            specs,
            crop=16,
            frames_per_store=4,
        )

        item = dataset[0]
        assert item.dtype == torch.float32
        assert item.shape[-2:] == (16, 16)
        assert item.shape[0] == specs[0].channels
        assert torch.isfinite(item).all()

        if layout == "TCHW":
            expected = values[0, :, 8:24, 10:26]
        elif layout == "THWC":
            expected = np.moveaxis(
                values[0, 8:24, 10:26, :], -1, 0
            )
        elif layout == "NTCHW":
            expected = values[0, 0, :, 8:24, 10:26]
        elif layout == "NTHWC":
            expected = np.moveaxis(
                values[0, 0, 8:24, 10:26, :], -1, 0
            )
        elif layout == "CHW":
            expected = values[:, 8:24, 10:26]
        elif layout == "HWC":
            expected = np.moveaxis(
                values[8:24, 10:26, :], -1, 0
            )
        else:
            expected = values[8:24, 10:26][None]

        np.testing.assert_array_equal(item.numpy(), expected)

        loader = make_loader(
            dataset, batch_size=2, workers=2
        )
        batch = next(iter(loader))
        assert batch.ndim == 4
        assert batch.shape[1:] == item.shape

        if torch.cuda.is_available():
            gpu_loader = make_loader(
                dataset,
                batch_size=2,
                workers=0,
                pin_memory=True,
            )
            gpu_batch = next(iter(CUDAPrefetcher(gpu_loader)))
            assert gpu_batch.is_cuda
            assert gpu_batch.shape[1:] == item.shape
            torch.cuda.synchronize()

        print(f"PASS: {layout}", flush=True)


def main():
    rng = np.random.default_rng(42)
    base = rng.normal(size=(5, 3, 32, 36)).astype(np.float32)

    check_layout("TCHW", base)
    check_layout("THWC", np.moveaxis(base, 1, -1))
    check_layout("NTCHW", base[None, ...])
    check_layout(
        "NTHWC", np.moveaxis(base[None, ...], 2, -1)
    )
    check_layout("CHW", base[0])
    check_layout("HWC", np.moveaxis(base[0], 0, -1))
    check_layout("HW", base[0, 0])

    print("\nALL SYNTHETIC LOADER TESTS PASSED")


if __name__ == "__main__":
    main()
