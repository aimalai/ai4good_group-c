from __future__ import annotations

import csv
import io
import math
import os
import struct
import time
import urllib.request
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


DATA_ROOT = Path(
    os.environ.get(
        "DISFOR_DATA_ROOT",
        "/zfs/ai4good/datasets/forest_disturbance",
    )
)

OUT = Path(
    "experiments/tristan_firms_fusion/firms_cache"
)

CHUNKS = OUT / "chunks"
FEATURES = OUT / "firms_features.parquet"

SOURCE = "VIIRS_SNPP_SP"
CELL_DEG = 0.25


def point_wkb(raw):
    b = bytes(raw)

    endian = "<" if b[0] == 1 else ">"
    geom_type = struct.unpack(
        endian + "I",
        b[1:5],
    )[0]

    if (geom_type & 0xFF) != 1:
        raise ValueError(
            f"Expected POINT geometry, got {geom_type}"
        )

    lon, lat = struct.unpack(
        endian + "dd",
        b[5:21],
    )

    return float(lon), float(lat)


def load_sites():
    df = pl.read_parquet(
        DATA_ROOT / "samples.parquet",
        columns=[
            "sample_id",
            "geometry",
        ],
    )

    sites = []

    for row in df.iter_rows(named=True):
        lon, lat = point_wkb(
            row["geometry"]
        )

        sites.append(
            (
                int(row["sample_id"]),
                lat,
                lon,
            )
        )

    return sites


def load_targets():
    return (
        pl.scan_parquet(
            DATA_ROOT
            / "zarr_frames.parquet"
        )
        .filter(
            pl.col("sensor") == "s2"
        )
        .select(
            "sample_id",
            "date",
        )
        .unique()
        .sort(
            "sample_id",
            "date",
        )
        .collect()
    )


def fetch(url):
    last = None

    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent":
                    "uzh-ai4good-disfor/1.0"
                },
            )

            with urllib.request.urlopen(
                req,
                timeout=180,
            ) as r:
                text = r.read().decode(
                    "utf-8",
                    errors="replace",
                )

            if (
                "latitude" not in text.lower()
                or
                "longitude" not in text.lower()
            ):
                raise RuntimeError(
                    "Unexpected FIRMS response: "
                    + text[:300]
                )

            return text

        except Exception as exc:
            last = exc

            time.sleep(
                min(
                    30,
                    2 ** attempt,
                )
            )

    raise RuntimeError(
        f"FIRMS request failed: {last}"
    )


def download_fires(
    key,
    sites,
    targets,
):
    CHUNKS.mkdir(
        parents=True,
        exist_ok=True,
    )

    min_lat = (
        min(x[1] for x in sites)
        - 0.35
    )
    max_lat = (
        max(x[1] for x in sites)
        + 0.35
    )
    min_lon = (
        min(x[2] for x in sites)
        - 0.35
    )
    max_lon = (
        max(x[2] for x in sites)
        + 0.35
    )

    area = (
        f"{min_lon:.4f},"
        f"{min_lat:.4f},"
        f"{max_lon:.4f},"
        f"{max_lat:.4f}"
    )

    dates = targets[
        "date"
    ].to_list()

    start = (
        min(dates)
        - timedelta(days=30)
    )

    # Strictly before the target date.
    end = (
        max(dates)
        - timedelta(days=1)
    )

    print(
        "FIRMS source:",
        SOURCE,
    )
    print(
        "Bounding box:",
        area,
    )
    print(
        "Date range:",
        start,
        "to",
        end,
    )

    current = start
    n = 0

    while current <= end:

        remaining = (
            end - current
        ).days + 1

        days = min(
            5,
            remaining,
        )

        path = (
            CHUNKS
            / (
                f"{current.isoformat()}"
                f"_{days}.csv"
            )
        )

        if (
            not path.exists()
            or path.stat().st_size == 0
        ):
            url = (
                "https://firms.modaps."
                "eosdis.nasa.gov/"
                "api/area/csv/"
                f"{key}/"
                f"{SOURCE}/"
                f"{area}/"
                f"{days}/"
                f"{current.isoformat()}"
            )

            path.write_text(
                fetch(url)
            )

            # Be polite to API.
            time.sleep(0.15)

        n += 1

        if n % 50 == 0:
            print(
                f"Cached {n} FIRMS "
                f"chunks through "
                f"{current}"
            )

        current += timedelta(
            days=days
        )


def valid_confidence(x):
    x = str(
        x or ""
    ).strip().lower()

    # Drop VIIRS low-confidence
    # detections.
    return x not in {
        "l",
        "low",
    }


def load_fires():
    fires = []

    for path in sorted(
        CHUNKS.glob("*.csv")
    ):
        text = path.read_text(
            errors="replace"
        )

        for row in csv.DictReader(
            io.StringIO(text)
        ):
            if not valid_confidence(
                row.get(
                    "confidence",
                    "",
                )
            ):
                continue

            try:
                lat = float(
                    row["latitude"]
                )
                lon = float(
                    row["longitude"]
                )

                day = (
                    date.fromisoformat(
                        row["acq_date"]
                    ).toordinal()
                )

                frp = max(
                    0.0,
                    float(
                        row.get("frp")
                        or 0.0
                    ),
                )

            except Exception:
                continue

            fires.append(
                (
                    lat,
                    lon,
                    day,
                    frp,
                )
            )

    print(
        "Loaded",
        f"{len(fires):,}",
        "nominal/high-confidence "
        "FIRMS detections",
    )

    return fires


def cell(lat, lon):
    return (
        int(
            math.floor(
                lat / CELL_DEG
            )
        ),
        int(
            math.floor(
                lon / CELL_DEG
            )
        ),
    )


def distance_km(
    lat0,
    lon0,
    lat,
    lon,
):
    r = 6371.0088

    p0 = np.radians(lat0)
    p = np.radians(lat)

    dp = p - p0

    dl = np.radians(
        lon - lon0
    )

    a = (
        np.sin(
            dp / 2
        ) ** 2
        +
        np.cos(p0)
        * np.cos(p)
        * np.sin(
            dl / 2
        ) ** 2
    )

    return (
        2
        * r
        * np.arcsin(
            np.sqrt(
                np.clip(
                    a,
                    0,
                    1,
                )
            )
        )
    )


def build_features(
    sites,
    targets,
    fires,
):
    fire_cells = defaultdict(
        list
    )

    for lat, lon, day, frp in fires:
        fire_cells[
            cell(lat, lon)
        ].append(
            (
                lat,
                lon,
                day,
                frp,
            )
        )

    target_by_site = {
        int(
            key[0]
            if isinstance(
                key,
                tuple,
            )
            else key
        ):
        rows["date"].to_list()

        for key, rows
        in targets.partition_by(
            "sample_id",
            as_dict=True,
        ).items()
    }

    records = []

    for i, (
        sample_id,
        site_lat,
        site_lon,
    ) in enumerate(
        sites,
        start=1,
    ):

        ci, cj = cell(
            site_lat,
            site_lon,
        )

        candidates = []

        for di in range(-2, 3):
            for dj in range(-2, 3):

                candidates.extend(
                    fire_cells.get(
                        (
                            ci + di,
                            cj + dj,
                        ),
                        (),
                    )
                )

        if candidates:

            arr = np.asarray(
                candidates,
                dtype=np.float64,
            )

            dist = distance_km(
                site_lat,
                site_lon,
                arr[:, 0],
                arr[:, 1],
            )

            keep = dist <= 10.0

            arr = arr[keep]
            dist = dist[keep]

            fire_day = (
                arr[:, 2].astype(
                    np.int64,
                    copy=False,
                )
                if len(arr)
                else np.empty(
                    0,
                    dtype=np.int64,
                )
            )

            frp = (
                arr[:, 3]
                if len(arr)
                else np.empty(
                    0,
                    dtype=np.float64,
                )
            )

        else:
            dist = np.empty(
                0,
                dtype=np.float64,
            )
            fire_day = np.empty(
                0,
                dtype=np.int64,
            )
            frp = np.empty(
                0,
                dtype=np.float64,
            )

        for target_date in (
            target_by_site.get(
                sample_id,
                (),
            )
        ):
            target = (
                target_date.toordinal()
            )

            age = (
                target
                - fire_day
            )

            # THIS IS THE LEAKAGE GUARD:
            # age >= 1 means the FIRMS
            # detection must be before the
            # S2 target date.
            past30 = (
                (age >= 1)
                & (age <= 30)
            )

            def count(
                radius,
                days,
            ):
                return int(
                    np.count_nonzero(
                        (dist <= radius)
                        & (age >= 1)
                        & (age <= days)
                    )
                )

            c1 = count(
                1.0,
                1,
            )

            c3 = count(
                3.0,
                3,
            )

            c5 = count(
                5.0,
                7,
            )

            c10 = count(
                10.0,
                30,
            )

            near_fire = (
                (dist <= 5.0)
                & (age >= 1)
                & (age <= 7)
            )

            max_frp = (
                float(
                    frp[
                        near_fire
                    ].max()
                )
                if np.any(
                    near_fire
                )
                else 0.0
            )

            min_dist = (
                float(
                    dist[
                        past30
                    ].min()
                )
                if np.any(
                    past30
                )
                else 10.0
            )

            days_since = (
                float(
                    age[
                        past30
                    ].min()
                )
                if np.any(
                    past30
                )
                else 30.0
            )

            records.append(
                {
                    "sample_id":
                    sample_id,

                    "target_date_ordinal":
                    target,

                    "firms_log_count_1km_1d":
                    math.log1p(c1),

                    "firms_log_count_3km_3d":
                    math.log1p(c3),

                    "firms_log_count_5km_7d":
                    math.log1p(c5),

                    "firms_log_count_10km_30d":
                    math.log1p(c10),

                    "firms_log_max_frp_5km_7d":
                    math.log1p(
                        max_frp
                    ),

                    "firms_min_dist_30d_scaled":
                    min(
                        min_dist,
                        10.0,
                    )
                    / 10.0,

                    "firms_days_since_10km_scaled":
                    min(
                        days_since,
                        30.0,
                    )
                    / 30.0,

                    "firms_any_5km_7d":
                    float(c5 > 0),
                }
            )

        if i % 250 == 0:
            print(
                "Features:",
                f"{i:,}/"
                f"{len(sites):,}",
                "sites",
            )

    table = (
        pl.DataFrame(records)
        .sort(
            "sample_id",
            "target_date_ordinal",
        )
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.write_parquet(
        FEATURES
    )

    print(
        "Saved:",
        FEATURES,
    )

    print(
        "Feature rows:",
        f"{table.height:,}",
    )

    print(
        "Examples with fire "
        "within 5 km / 7 days:",
        int(
            table[
                "firms_any_5km_7d"
            ].sum()
        ),
    )


def main():
    key = os.environ.get(
        "FIRMS_MAP_KEY",
        "",
    ).strip()

    if not key:
        raise SystemExit(
            "FIRMS_MAP_KEY is not set"
        )

    sites = load_sites()
    targets = load_targets()

    print(
        "DISFOR sites:",
        f"{len(sites):,}",
    )

    print(
        "Unique S2 dates:",
        f"{targets.height:,}",
    )

    if FEATURES.exists():
        print(
            "Features already exist; "
            "reusing:",
            FEATURES,
        )
        return

    download_fires(
        key,
        sites,
        targets,
    )

    fires = load_fires()

    build_features(
        sites,
        targets,
        fires,
    )


if __name__ == "__main__":
    main()
