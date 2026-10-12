"""Independent, lazy Zarr patch loader for I/O benchmarks.

Supports:
    HW, CHW, HWC, TCHW, THWC, NTCHW, NTHWC

Each sample is one spatial image patch (C,H,W).
Time-series frames are indexed independently for benchmarking.
This is not a labelled training dataset.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


LAYOUTS = {
    2: ("HW",),
    3: ("CHW", "HWC"),
    4: ("TCHW", "THWC"),
    5: ("NTCHW", "NTHWC"),
}


@dataclass(frozen=True)
class ArraySpec:
    store: str
    key: str
    shape: tuple[int, ...]
    dtype: str
    layout: str

    @property
    def channels(self):
        if self.layout == "HW":
            return 1
        if self.layout.endswith("CHW"):
            return self.shape[-3]
        return self.shape[-1]

    @property
    def height(self):
        if self.layout == "HW" or self.layout.endswith("CHW"):
            return self.shape[-2]
        return self.shape[-3]

    @property
    def width(self):
        if self.layout == "HW" or self.layout.endswith("CHW"):
            return self.shape[-1]
        return self.shape[-2]

    @property
    def frames(self):
        if self.layout.startswith("NT"):
            return self.shape[0] * self.shape[1]
        if self.layout.startswith("T"):
            return self.shape[0]
        return 1


def find_stores(root, max_stores=12, max_depth=10, max_dirs=6000):
    """Find Zarr directories without walking into chunk directories."""
    root = Path(root).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    stack = [(root, 0)]
    seen = set()
    found = []

    while stack and len(found) < max_stores and len(seen) < max_dirs:
        path, depth = stack.pop()
        try:
            resolved = path.resolve()
        except OSError:
            continue

        if resolved in seen:
            continue
        seen.add(resolved)

        if any(
            (path / marker).is_file()
            for marker in ("zarr.json", ".zgroup", ".zarray")
        ):
            found.append(path)
            continue

        if depth >= max_depth:
            continue

        try:
            children = sorted(
                (
                    p for p in path.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                ),
                key=lambda p: p.name,
            )
        except (OSError, PermissionError):
            continue

        stack.extend((child, depth + 1) for child in reversed(children))

    return found


def inspect_store(store, max_arrays=128):
    """Inspect Zarr metadata only; never read full image arrays."""
    import zarr

    root = zarr.open(str(store), mode="r")
    arrays = []

    if hasattr(root, "shape"):
        return [("", tuple(root.shape), str(root.dtype))]

    pending = [("", root)]
    while pending and len(arrays) < max_arrays:
        prefix, group = pending.pop()

        for name, arr in group.arrays():
            key = f"{prefix}/{name}" if prefix else name
            arrays.append((key, tuple(arr.shape), str(arr.dtype)))
            if len(arrays) >= max_arrays:
                break

        if len(arrays) >= max_arrays:
            break

        children = list(group.groups())
        for name, child in reversed(children):
            key = f"{prefix}/{name}" if prefix else name
            pending.append((key, child))

    return arrays


def infer_layout(shape, requested="auto"):
    ndim = len(shape)
    choices = LAYOUTS.get(ndim, ())

    if requested != "auto":
        return requested if requested in choices else None

    if ndim == 2:
        return "HW"

    if ndim not in (3, 4, 5):
        return None

    first_channel_axis = ndim - 3
    c_first = shape[first_channel_axis]
    c_last = shape[-1]

    # Conservative heuristic. Explicit --layout overrides ambiguity.
    first_plausible = 1 <= c_first <= 24
    last_plausible = 1 <= c_last <= 24

    if first_plausible and (
        not last_plausible or c_first < c_last
    ):
        return choices[0]

    if last_plausible and (
        not first_plausible or c_last < c_first
    ):
        return choices[1]

    return None


def discover(
    root,
    crop=64,
    array_key=None,
    layout="auto",
    max_stores=12,
):
    stores = find_stores(root, max_stores=max_stores)

    if not stores:
        raise RuntimeError(
            f"No Zarr stores found under {root}. "
            "Check the inspect output and the dataset path."
        )

    candidates = []
    errors = []

    for store in stores:
        try:
            metadata = inspect_store(store)
        except Exception as exc:
            errors.append(f"{store}: {exc}")
            continue

        for key, shape, dtype in metadata:
            if array_key is not None and key != array_key:
                continue

            detected = infer_layout(shape, layout)
            if detected is None:
                continue

            spec = ArraySpec(
                store=str(store),
                key=key,
                shape=shape,
                dtype=dtype,
                layout=detected,
            )

            if (
                spec.height >= crop
                and spec.width >= crop
                and spec.channels >= 1
                and spec.frames >= 1
            ):
                candidates.append(spec)

    if not candidates:
        detail = "\n".join(errors[:5])
        raise RuntimeError(
            "No compatible image arrays found.\n"
            "Try --mode inspect, --layout TCHW/THWC, "
            "or a smaller --crop.\n" + detail
        )

    # Compare one consistent image type across stores.
    grouped = defaultdict(list)
    for spec in candidates:
        grouped[(spec.key, spec.layout, spec.channels)].append(spec)

    selected = max(
        grouped.values(),
        key=lambda group: (
            len(group),
            max(len(spec.shape) for spec in group),
            min(group[0].channels, 16),
        ),
    )

    print(
        "Selected:",
        f"key={selected[0].key!r}",
        f"layout={selected[0].layout}",
        f"channels={selected[0].channels}",
        f"stores={len(selected)}",
        flush=True,
    )

    return selected


class ZarrPatchDataset(Dataset):
    """Lazy center-cropped image patches with per-worker handle caching."""

    def __init__(
        self,
        specs,
        crop=64,
        frames_per_store=64,
        max_samples=2048,
        cache_size=4,
        seed=42,
        shuffle=False,
    ):
        if not specs:
            raise ValueError("No array specifications.")
        if min(crop, frames_per_store, max_samples, cache_size) < 1:
            raise ValueError("All dataset limits must be positive.")

        self.specs = list(specs)
        self.crop = crop
        self.cache_size = cache_size
        self._handles = OrderedDict()

        # Small consecutive groups improve locality without benchmarking
        # only a single Zarr store.
        per_store = []
        for spec_id, spec in enumerate(self.specs):
            n = min(spec.frames, frames_per_store)
            indices = np.linspace(
                0, spec.frames - 1, num=n, dtype=np.int64
            )
            per_store.append(
                [(spec_id, int(index)) for index in indices]
            )

        samples = []
        block_size = 16
        for offset in range(0, frames_per_store, block_size):
            for group in per_store:
                samples.extend(group[offset:offset + block_size])
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break

        self.samples = samples[:max_samples]
        if shuffle:
            random.Random(seed).shuffle(self.samples)

        if not self.samples:
            raise RuntimeError("No samples generated.")

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def _get_array(self, spec):
        import zarr

        cache_key = (spec.store, spec.key)
        if cache_key in self._handles:
            self._handles.move_to_end(cache_key)
            return self._handles[cache_key]

        root = zarr.open(spec.store, mode="r")
        array = root[spec.key] if spec.key else root
        self._handles[cache_key] = array

        while len(self._handles) > self.cache_size:
            self._handles.popitem(last=False)

        return array

    def __getitem__(self, index):
        spec_id, flat_frame = self.samples[index]
        spec = self.specs[spec_id]
        array = self._get_array(spec)

        y = (spec.height - self.crop) // 2
        x = (spec.width - self.crop) // 2
        ys = slice(y, y + self.crop)
        xs = slice(x, x + self.crop)

        layout = spec.layout

        if layout == "NTCHW":
            t = spec.shape[1]
            patch = array[flat_frame // t, flat_frame % t, :, ys, xs]
        elif layout == "NTHWC":
            t = spec.shape[1]
            patch = array[flat_frame // t, flat_frame % t, ys, xs, :]
        elif layout == "TCHW":
            patch = array[flat_frame, :, ys, xs]
        elif layout == "THWC":
            patch = array[flat_frame, ys, xs, :]
        elif layout == "CHW":
            patch = array[:, ys, xs]
        elif layout == "HWC":
            patch = array[ys, xs, :]
        elif layout == "HW":
            patch = array[ys, xs]
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        patch = np.asarray(patch)

        if layout == "HW":
            patch = patch[None, :, :]
        elif layout.endswith("HWC"):
            patch = np.moveaxis(patch, -1, 0)

        # Float32 keeps GPU convolution and comparison runs consistent.
        patch = np.ascontiguousarray(patch, dtype=np.float32)
        return torch.from_numpy(patch)


def _init_worker(worker_id):
    torch.set_num_threads(1)


def make_loader(dataset, batch_size=8, workers=4, pin_memory=False):
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid batch size or workers.")

    options = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_init_worker if workers else None,
    )

    if workers:
        options.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    return DataLoader(**options)


class CUDAPrefetcher:
    """Prefetch next pinned-memory batch using a separate CUDA stream."""

    def __init__(self, loader, device="cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        self.device = torch.device(device)
        self.iterator = iter(loader)
        self.stream = torch.cuda.Stream(device=self.device)
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            cpu_batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = cpu_batch.to(
                self.device, non_blocking=True
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)

        batch = self.next_batch
        batch.record_stream(current)

        self._preload()
        return batch
