import torch
from torch import nn

from experiments.tristan_ltae_late_fusion.model import (
    DualLTAEFusion,
)


class GatedDualLTAEFusion(DualLTAEFusion):
    """
    Dual L-TAE S1/S2 with learned multimodal gating.

    The temporal encoders are unchanged from the successful
    Dual L-TAE model. The main difference is fusion:

        S2 temporal features -> projection --\
                                              -> learned gate -> fused representation
        S1 temporal features -> projection --/

    The gate is feature-wise, so it can learn to emphasize
    Sentinel-1 or Sentinel-2 differently for each sample.
    """

    def __init__(
        self,
        num_classes,
        s2_dim=128,
        s1_dim=96,
        s2_heads=8,
        s1_heads=4,
        common_dim=512,
        dropout=0.10,
        s2_acquisition_dropout=0.20,
        s1_acquisition_dropout=0.10,
    ):
        super().__init__(
            num_classes=num_classes,
            s2_dim=s2_dim,
            s1_dim=s1_dim,
            s2_heads=s2_heads,
            s1_heads=s1_heads,
            fusion_dim=512,
            dropout=dropout,
            s2_acquisition_dropout=s2_acquisition_dropout,
            s1_acquisition_dropout=s1_acquisition_dropout,
        )

        s2_feature_dim = 4 * s2_dim
        s1_feature_dim = 4 * s1_dim

        self.common_dim = common_dim

        self.s2_projection = nn.Sequential(
            nn.LayerNorm(s2_feature_dim),
            nn.Linear(
                s2_feature_dim,
                common_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.s1_projection = nn.Sequential(
            nn.LayerNorm(s1_feature_dim),
            nn.Linear(
                s1_feature_dim,
                common_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Standard feature-wise multimodal gate.
        #
        # gate ~= 1 -> emphasize S2
        # gate ~= 0 -> emphasize S1
        self.gate = nn.Sequential(
            nn.LayerNorm(
                2 * common_dim
            ),
            nn.Linear(
                2 * common_dim,
                256,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                common_dim,
            ),
            nn.Sigmoid(),
        )

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(common_dim),

            nn.Linear(
                common_dim,
                512,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                512,
                256,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                256,
                num_classes,
            ),
        )

        # Useful for later inspection without affecting training.
        self.last_gate_mean = None

    def forward(
        self,
        batch,
        return_aux=False,
    ):
        target_date = batch[
            "target_date"
        ]

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

        s2_projected = (
            self.s2_projection(
                s2_features
            )
        )

        s1_projected = (
            self.s1_projection(
                s1_features
            )
        )

        gate = self.gate(
            torch.cat(
                (
                    s2_projected,
                    s1_projected,
                ),
                dim=-1,
            )
        )

        fused = (
            gate
            * s2_projected
            +
            (1.0 - gate)
            * s1_projected
        )

        # Diagnostic only.
        self.last_gate_mean = (
            gate.detach()
            .float()
            .mean()
        )

        logits = (
            self.fusion_head(
                fused
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
