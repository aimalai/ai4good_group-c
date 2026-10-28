from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from experiments.tristan_ltae_late_fusion.model import (
    DualLTAEFusion,
)


FEATURE_COLUMNS = [
    "firms_log_count_1km_1d",
    "firms_log_count_3km_3d",
    "firms_log_count_5km_7d",
    "firms_log_count_10km_30d",
    "firms_log_max_frp_5km_7d",
    "firms_min_dist_30d_scaled",
    "firms_days_since_10km_scaled",
    "firms_any_5km_7d",
]


class FIRMSStore:
    def __init__(
        self,
        path,
    ):
        table = (
            pl.read_parquet(path)
            .sort(
                "sample_id",
                "target_date_ordinal",
            )
        )

        self.data = {}

        for key, rows in (
            table.partition_by(
                "sample_id",
                as_dict=True,
            ).items()
        ):
            sample_id = int(
                key[0]
                if isinstance(
                    key,
                    tuple,
                )
                else key
            )

            dates = (
                rows[
                    "target_date_ordinal"
                ]
                .to_numpy()
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            features = (
                rows.select(
                    FEATURE_COLUMNS
                )
                .to_numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            self.data[
                sample_id
            ] = (
                dates,
                features,
            )

        print(
            "Loaded FIRMS features "
            "for",
            f"{len(self.data):,}",
            "sites",
        )

    def lookup(
        self,
        sample_ids,
        target_dates,
        *,
        device,
        dtype,
    ):
        ids = (
            sample_ids
            .detach()
            .cpu()
            .numpy()
        )

        dates = (
            target_dates
            .detach()
            .cpu()
            .numpy()
        )

        out = np.zeros(
            (
                len(ids),
                len(
                    FEATURE_COLUMNS
                ),
            ),
            dtype=np.float32,
        )

        for i, (
            sample_id,
            target,
        ) in enumerate(
            zip(
                ids,
                dates,
                strict=True,
            )
        ):
            item = self.data.get(
                int(sample_id)
            )

            if item is None:
                continue

            sample_dates, features = (
                item
            )

            pos = int(
                np.searchsorted(
                    sample_dates,
                    target,
                )
            )

            if (
                pos
                < len(sample_dates)
                and
                sample_dates[pos]
                == target
            ):
                out[i] = (
                    features[pos]
                )

        return torch.as_tensor(
            out,
            device=device,
            dtype=dtype,
        )


class DualLTAEFIRMS(
    DualLTAEFusion
):
    def __init__(
        self,
        num_classes,
        firms_features_path,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            **kwargs,
        )

        self.firms_store = (
            FIRMSStore(
                firms_features_path
            )
        )

        s2_features = (
            4 * self.s2_dim
        )

        s1_features = (
            4 * self.s1_dim
        )

        firms_features = len(
            FEATURE_COLUMNS
        )

        fusion_dim = kwargs.get(
            "fusion_dim",
            512,
        )

        dropout = kwargs.get(
            "dropout",
            0.10,
        )

        total = (
            s2_features
            + s1_features
            + firms_features
        )

        self.fusion_head = (
            nn.Sequential(
                nn.LayerNorm(
                    total
                ),

                nn.Linear(
                    total,
                    fusion_dim,
                ),

                nn.GELU(),
                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    fusion_dim,
                    256,
                ),

                nn.GELU(),
                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    256,
                    num_classes,
                ),
            )
        )

    def forward(
        self,
        batch,
        return_aux=False,
    ):
        target = batch[
            "target_date"
        ]

        s2 = (
            self._modality_features(
                self.s2_branch,
                batch[
                    "inputs"
                ]["s2"],
                target,
                self.s2_dim,
            )
        )

        s1 = (
            self._modality_features(
                self.s1_branch,
                batch[
                    "inputs"
                ]["s1"],
                target,
                self.s1_dim,
            )
        )

        firms = (
            self.firms_store.lookup(
                batch["sample_id"],
                target,
                device=s2.device,
                dtype=s2.dtype,
            )
        )

        logits = self.fusion_head(
            torch.cat(
                (
                    s2,
                    s1,
                    firms,
                ),
                dim=-1,
            )
        )

        if not return_aux:
            return logits

        return (
            logits,
            self.s2_aux_head(s2),
            self.s1_aux_head(s1),
        )
