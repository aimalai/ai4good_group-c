import math

import torch
from torch import nn


class LightTemporalAttention(nn.Module):
    """Compact multi-head temporal attention pooling."""

    def __init__(self, dim, num_heads, key_dim=16, dropout=0.10):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = key_dim

        self.key_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads * key_dim),
        )

        self.query = nn.Parameter(
            torch.empty(num_heads, key_dim)
        )

        nn.init.normal_(
            self.query,
            mean=0.0,
            std=key_dim ** -0.5,
        )

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        # x: [B,T,D], mask: [B,T]
        B, T, D = x.shape

        if T == 0:
            return x.new_zeros(B, D)

        keys = self.key_net(x).view(
            B,
            T,
            self.num_heads,
            self.key_dim,
        )

        logits = torch.einsum(
            "bthk,hk->bth",
            keys,
            self.query,
        )

        logits = logits / math.sqrt(
            self.key_dim
        )

        valid = mask.unsqueeze(-1)

        logits = logits.masked_fill(
            ~valid,
            -1e4,
        )

        weights = (
            torch.softmax(logits, dim=1)
            * valid.to(dtype=logits.dtype)
        )

        weights = weights / (
            weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-6)
        )

        weights = self.dropout(weights)

        values = x.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )

        pooled = torch.einsum(
            "bth,bthd->bhd",
            weights,
            values,
        ).reshape(B, D)

        return self.out_norm(pooled)


class SensorLTAEBranch(nn.Module):
    def __init__(
        self,
        in_dim,
        model_dim,
        num_heads,
        *,
        sensor,
        acquisition_dropout,
        dropout=0.10,
    ):
        super().__init__()

        self.sensor = sensor
        self.acquisition_dropout = (
            acquisition_dropout
        )

        self.frame = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(
                model_dim,
                model_dim,
            ),
            nn.GELU(),
        )

        self.time_proj = nn.Linear(
            10,
            model_dim,
            bias=False,
        )

        self.pre_attention_norm = (
            nn.LayerNorm(model_dim)
        )

        self.temporal = (
            LightTemporalAttention(
                dim=model_dim,
                num_heads=num_heads,
                key_dim=16,
                dropout=dropout,
            )
        )

    def forward(
        self,
        series,
        target_date,
    ):
        values = series["values"]
        dates = series["dates"]
        mask = series["mask"]

        B, T = values.shape[:2]

        if T == 0:
            return values.new_zeros(
                B,
                self.temporal.dim,
            )

        h = values.shape[-2]
        w = values.shape[-1]

        x = values[
            :,
            :,
            :,
            h // 2,
            w // 2,
        ]

        x = torch.nan_to_num(
            x.float()
        )

        # Ignore true all-zero no-data frames.
        mask = mask & (
            x.abs().sum(dim=-1) > 0
        )

        if self.sensor == "s2":

            # DISFOR S2:
            # reflectance * 10000 + 1000
            x = (
                x - 1000.0
            ) / 10000.0

            x = x.clamp(
                -0.2,
                1.5,
            )

        elif self.sensor == "s1":

            # DISFOR S1:
            # dB * 100
            x = x / 100.0

            x = x.clamp(
                -60.0,
                10.0,
            )

        else:
            raise ValueError(
                f"Unknown sensor "
                f"{self.sensor!r}"
            )

        if (
            self.training
            and self.acquisition_dropout > 0
        ):
            mask = (
                self._drop_observations(
                    mask,
                    self.acquisition_dropout,
                )
            )

        z = self.frame(x)

        z = z + self.time_proj(
            self._time_features(
                dates,
                target_date,
                dtype=z.dtype,
            )
        )

        z = self.pre_attention_norm(z)

        return self.temporal(
            z,
            mask,
        )

    @staticmethod
    def _drop_observations(
        mask,
        probability,
    ):
        keep = (
            torch.rand(
                mask.shape,
                device=mask.device,
            )
            >= probability
        )

        dropped = mask & keep

        # Never remove the last real
        # observation of a sample.
        had_any = mask.any(dim=1)
        has_any = dropped.any(dim=1)

        restore = (
            had_any
            & ~has_any
        )

        if restore.any():

            first = (
                mask.to(torch.int64)
                .argmax(dim=1)
            )

            rows = torch.nonzero(
                restore,
                as_tuple=False,
            ).flatten()

            dropped[
                rows,
                first[rows],
            ] = True

        return dropped

    @staticmethod
    def _time_features(
        dates,
        target_date,
        *,
        dtype,
    ):
        # Exact timing relative to
        # target acquisition.
        rel = (
            dates
            - target_date[:, None]
        ).to(torch.float32)

        feats = []

        for period in (
            7.0,
            30.0,
            90.0,
            365.25,
        ):
            angle = (
                2.0
                * math.pi
                * rel
                / period
            )

            feats.extend(
                (
                    torch.sin(angle),
                    torch.cos(angle),
                )
            )

        # Annual seasonal phase.
        doy_angle = (
            2.0
            * math.pi
            * torch.remainder(
                dates.to(
                    torch.float32
                ),
                365.25,
            )
            / 365.25
        )

        feats.extend(
            (
                torch.sin(doy_angle),
                torch.cos(doy_angle),
            )
        )

        return torch.stack(
            feats,
            dim=-1,
        ).to(dtype=dtype)


class DualLTAEFusion(nn.Module):
    """
    Fully trainable S1/S2 model.

      S2 -> MLP -> L-TAE --\
                            > fusion -> 7 classes
      S1 -> MLP -> L-TAE --/

    No TESSERA.
    No pretrained foundation model.
    """

    def __init__(
        self,
        num_classes,
        s2_dim=128,
        s1_dim=96,
        s2_heads=8,
        s1_heads=4,
        fusion_dim=512,
        dropout=0.10,
        s2_acquisition_dropout=0.20,
        s1_acquisition_dropout=0.10,
    ):
        super().__init__()

        self.s2_dim = s2_dim
        self.s1_dim = s1_dim

        self.s2_branch = (
            SensorLTAEBranch(
                12,
                s2_dim,
                s2_heads,
                sensor="s2",
                acquisition_dropout=(
                    s2_acquisition_dropout
                ),
                dropout=dropout,
            )
        )

        self.s1_branch = (
            SensorLTAEBranch(
                2,
                s1_dim,
                s1_heads,
                sensor="s1",
                acquisition_dropout=(
                    s1_acquisition_dropout
                ),
                dropout=dropout,
            )
        )

        s2_feature_dim = (
            4 * s2_dim
        )

        s1_feature_dim = (
            4 * s1_dim
        )

        self.s2_aux_head = (
            nn.Sequential(
                nn.LayerNorm(
                    s2_feature_dim
                ),
                nn.Linear(
                    s2_feature_dim,
                    256,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    256,
                    num_classes,
                ),
            )
        )

        self.s1_aux_head = (
            nn.Sequential(
                nn.LayerNorm(
                    s1_feature_dim
                ),
                nn.Linear(
                    s1_feature_dim,
                    192,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    192,
                    num_classes,
                ),
            )
        )

        self.fusion_head = (
            nn.Sequential(
                nn.LayerNorm(
                    s2_feature_dim
                    + s1_feature_dim
                ),

                nn.Linear(
                    s2_feature_dim
                    + s1_feature_dim,
                    fusion_dim,
                ),

                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(
                    fusion_dim,
                    256,
                ),

                nn.GELU(),
                nn.Dropout(dropout),

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
        target_date = (
            batch["target_date"]
        )

        s2_features = (
            self._modality_features(
                self.s2_branch,
                batch["inputs"]["s2"],
                target_date,
                self.s2_dim,
            )
        )

        s1_features = (
            self._modality_features(
                self.s1_branch,
                batch["inputs"]["s1"],
                target_date,
                self.s1_dim,
            )
        )

        logits = self.fusion_head(
            torch.cat(
                (
                    s2_features,
                    s1_features,
                ),
                dim=-1,
            )
        )

        if not return_aux:
            return logits

        return (
            logits,
            self.s2_aux_head(
                s2_features
            ),
            self.s1_aux_head(
                s1_features
            ),
        )

    @staticmethod
    def _modality_features(
        branch,
        windows,
        target_date,
        dim,
    ):
        recent = branch(
            windows["recent"],
            target_date,
        )

        years = windows["yearly"]

        if years:

            yearly_embeddings = [
                branch(
                    year,
                    target_date,
                )
                for year in years
            ]

            yearly = torch.stack(
                yearly_embeddings,
                dim=0,
            ).mean(dim=0)

        else:
            yearly = recent.new_zeros(
                recent.shape[0],
                dim,
            )

        delta = (
            recent - yearly
        )

        return torch.cat(
            (
                recent,
                yearly,
                delta,
                delta.abs(),
            ),
            dim=-1,
        )
