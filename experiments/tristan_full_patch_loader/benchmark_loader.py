"""Inspect and benchmark actual Zarr patch loading."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import (
    CUDAPrefetcher,
    ZarrPatchDataset,
    discover,
    find_stores,
    inspect_store,
    make_loader,
)


HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/zfs/ai4good/datasets/forest_disturbance/patches"
        ),
    )
    p.add_argument(
        "--mode",
        choices=("inspect", "sweep", "single"),
        default="inspect",
    )
    p.add_argument("--array-key", default=None)
    p.add_argument("--layout", default="auto",
                   choices=(
                       "auto", "HW", "CHW", "HWC",
                       "TCHW", "THWC", "NTCHW", "NTHWC",
                   ))
    p.add_argument("--max-stores", type=int, default=12)
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--frames-per-store", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=2048)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--conv", action="store_true")
    return p.parse_args()


def inspect(args):
    stores = find_stores(args.root, max_stores=args.max_stores)
    print(f"Root: {args.root}")
    print(f"Zarr stores found: {len(stores)}")

    if not stores:
        raise RuntimeError(
            "No Zarr stores found. Check the root directory."
        )

    for store in stores:
        print(f"\nSTORE: {store}")
        try:
            for key, shape, dtype in inspect_store(store):
                print(
                    f"  key={key!r} shape={shape} dtype={dtype}"
                )
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")


def run_one(args, specs, workers, gpu, prefetch, conv):
    if prefetch and not gpu:
        raise ValueError("CUDA prefetch requires GPU.")
    if conv and not gpu:
        raise ValueError("Convolution requires GPU.")
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA unavailable.")

    dataset = ZarrPatchDataset(
        specs,
        crop=args.crop,
        frames_per_store=args.frames_per_store,
        max_samples=args.max_samples,
    )

    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=workers,
        pin_memory=gpu,
    )

    device = torch.device("cuda:0") if gpu else None
    network = None

    if conv:
        channels = specs[0].channels
        network = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        ).to(device).eval()

    def new_iterator():
        return (
            iter(CUDAPrefetcher(loader, device=device))
            if prefetch
            else iter(loader)
        )

    start_total = time.perf_counter()
    iterator = new_iterator()

    first_batch_seconds = None
    measurement_start = None
    measured_samples = 0
    measured_batches = 0
    logical_bytes = 0
    iterator_restarts = 0

    with torch.inference_mode():
        for step in range(args.warmup + args.steps):
            if step == args.warmup:
                if gpu:
                    torch.cuda.synchronize(device)
                measurement_start = time.perf_counter()

            try:
                batch = next(iterator)
            except StopIteration:
                iterator_restarts += 1
                iterator = new_iterator()
                batch = next(iterator)

            if gpu and not prefetch:
                batch = batch.to(device, non_blocking=True)

            if network is not None:
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_bf16_supported(),
                ):
                    result = network(batch)
                    _ = result.mean()

            if gpu:
                # Wait for this batch's work, but do not block the
                # independent prefetch stream unnecessarily.
                torch.cuda.current_stream(device).synchronize()

            if step == 0:
                first_batch_seconds = (
                    time.perf_counter() - start_total
                )

            if step >= args.warmup:
                measured_samples += int(batch.shape[0])
                measured_batches += 1
                logical_bytes += batch.numel() * batch.element_size()

    if gpu:
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - measurement_start

    # Release worker processes between benchmark configurations.
    del iterator, loader, dataset, network

    return {
        "workers": workers,
        "gpu": gpu,
        "prefetch": prefetch,
        "conv": conv,
        "crop": args.crop,
        "batch_size": args.batch_size,
        "stores": len(specs),
        "channels": specs[0].channels,
        "layout": specs[0].layout,
        "array_key": specs[0].key,
        "samples": measured_samples,
        "batches": measured_batches,
        "restarts": iterator_restarts,
        "first_batch_s": round(first_batch_seconds, 3),
        "measured_s": round(elapsed, 3),
        "samples_per_s": round(measured_samples / elapsed, 2),
        "batches_per_s": round(measured_batches / elapsed, 2),
        "logical_gib_per_s": round(
            logical_bytes / elapsed / (1024 ** 3), 4
        ),
    }


def main():
    args = parse_args()

    if args.mode == "inspect":
        inspect(args)
        return

    if args.steps < 1 or args.warmup < 0:
        raise ValueError("Invalid benchmark step count.")

    specs = discover(
        args.root,
        crop=args.crop,
        array_key=args.array_key,
        layout=args.layout,
        max_stores=args.max_stores,
    )

    if args.mode == "single":
        configs = [
            (args.workers, args.gpu, args.prefetch, args.conv)
        ]
    else:
        configs = [
            (0, False, False, False),
            (2, False, False, False),
            (4, False, False, False),
            (8, False, False, False),
        ]

        if torch.cuda.is_available():
            configs += [
                (4, True, False, False),
                (4, True, False, True),
                (4, True, True, True),
                (8, True, True, True),
            ]

    results = []
    for i, (workers, gpu, prefetch, conv) in enumerate(configs, 1):
        print(
            f"\n[{i}/{len(configs)}] "
            f"workers={workers}, gpu={gpu}, "
            f"prefetch={prefetch}, conv={conv}",
            flush=True,
        )
        result = run_one(
            args, specs, workers, gpu, prefetch, conv
        )
        results.append(result)
        print(
            f"  {result['samples_per_s']:.1f} samples/s; "
            f"first batch {result['first_batch_s']:.2f}s",
            flush=True,
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = HERE / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": stamp,
        "dataset_root": str(args.root),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "warmup": args.warmup,
        "steps": args.steps,
        "max_stores": args.max_stores,
        "results": results,
    }

    json_path = outdir / f"benchmark_{stamp}.json"
    csv_path = outdir / f"benchmark_{stamp}.csv"

    json_path.write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nBENCHMARK COMPLETE")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("\nThroughput summary:")
    for result in results:
        print(
            f"workers={result['workers']:<2} "
            f"gpu={str(result['gpu']):<5} "
            f"prefetch={str(result['prefetch']):<5} "
            f"conv={str(result['conv']):<5} "
            f"{result['samples_per_s']:>10.1f} samples/s"
        )


if __name__ == "__main__":
    main()
