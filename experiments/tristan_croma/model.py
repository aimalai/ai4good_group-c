import torch
from torch import nn

from torchgeo.models import croma_base, CROMABase_Weights


class FrozenCROMAClassifier(nn.Module):
    """
    Frozen pretrained CROMA spatial encoder.

    Each Sentinel-2 acquisition:
        12 x 120 x 120 patch -> CROMA -> 768-d embedding

    We then preserve the benchmark's recent-vs-yearly comparison:
        CROMA embeddings -> projection -> mean/std over time
        -> recent + yearly -> classifier -> 7 logits
    """

    def __init__(
        self,
        num_classes: int,
        projection_dim: int = 128,
        trunk_dim: int = 256,
        encoder_batch_size: int = 16,
    ):
        super().__init__()

        self.encoder_batch_size = encoder_batch_size

        self.encoder = croma_base(
            weights=CROMABase_Weights.CROMA_VIT,
            modalities=["optical"],
            image_size=120,
        )

        self.encoder.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, projection_dim),
            nn.GELU(),
        )

        # mean + std for recent, mean + std for yearly
        feature_dim = projection_dim * 4

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, trunk_dim),
            nn.GELU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(trunk_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)

        # Never put the pretrained encoder into training mode.
        self.encoder.eval()

        return self

    def forward(self, batch):
        s2 = batch["inputs"]["s2"]

        recent = self._summarize(s2["recent"])

        if s2["yearly"]:
            yearly_series = {
                key: torch.cat(
                    [year[key] for year in s2["yearly"]],
                    dim=1,
                )
                for key in ("values", "dates", "mask")
            }

            yearly = self._summarize(yearly_series)

        else:
            yearly = torch.zeros_like(recent)

        features = torch.cat((recent, yearly), dim=-1)

        return self.classifier(self.trunk(features))

    def _summarize(self, series):
        embeddings, mask = self._encode_series(series)

        projected = self.projection(embeddings)

        valid = mask.unsqueeze(-1)

        count = valid.sum(dim=1).clamp_min(1)

        mean = (
            projected.masked_fill(~valid, 0.0).sum(dim=1)
            / count
        )

        centered = (
            projected - mean.unsqueeze(1)
        ).masked_fill(~valid, 0.0)

        var = centered.square().sum(dim=1) / count

        std = torch.sqrt(
            var + torch.finfo(projected.dtype).eps
        )

        return torch.cat((mean, std), dim=-1)

    def _encode_series(self, series):
        values = series["values"]
        original_mask = series["mask"]

        B, T, C, H, W = values.shape

        assert C == 12, C
        assert H == 120 and W == 120, (H, W)

        flat = values.reshape(B * T, C, H, W)
        mask = original_mask.reshape(-1)

        result = values.new_zeros((B * T, 768))

        if not mask.any():
            return result.reshape(B, T, 768), original_mask

        real = flat[mask]

        # Reject completely empty / constant acquisitions.
        nonzero_fraction = (
            (real != 0)
            .float()
            .mean(dim=(1, 2, 3))
        )

        variation = (
            real.flatten(2)
            .std(dim=2)
            .mean(dim=1)
        )

        usable = (
            (nonzero_fraction > 0.5)
            & (variation > 1.0)
        )

        positions = torch.nonzero(mask, as_tuple=False).squeeze(1)
        usable_positions = positions[usable]

        real = real[usable]

        if len(real):
            # Same per-image/per-band normalisation used in our
            # successful real-image CROMA test.
            mean = real.mean(
                dim=(-2, -1),
                keepdim=True,
            )

            std = real.std(
                dim=(-2, -1),
                keepdim=True,
            ).clamp_min(1e-6)

            real = (
                (
                    real - (mean - 2 * std)
                )
                / (4 * std)
            ).clamp(0, 1)

            self.encoder.eval()

            encoded_parts = []

            with torch.no_grad():
                for start in range(
                    0,
                    len(real),
                    self.encoder_batch_size,
                ):
                    chunk = real[
                        start:
                        start + self.encoder_batch_size
                    ]

                    out = self.encoder(
                        x_optical=chunk
                    )

                    encoded_parts.append(
                        out["optical_GAP"]
                    )

            encoded = torch.cat(
                encoded_parts,
                dim=0,
            )

            result[usable_positions] = encoded.to(
                dtype=result.dtype
            )

        # Remove bad frames from temporal statistics too.
        quality_mask = torch.zeros_like(mask)

        quality_mask[usable_positions] = True

        return (
            result.reshape(B, T, 768),
            quality_mask.reshape(B, T),
        )
