"""Temporal GRU experiment for near-real-time forest disturbance classification.

This experiment intentionally reuses the course data loader, labels, loss and metrics.
The only substantive change from the provided StatisticalMLP baseline is the temporal
aggregator: instead of reducing each time window to mean/std, an ordered GRU encodes
all valid Sentinel-2 acquisitions in that window.

Input contract (same batch format as the course models):
    batch["inputs"]["s2"]["recent"] -> padded time series
    batch["inputs"]["s2"]["yearly"] -> list of padded time series

Output:
    class logits [B, num_classes]
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from forest_disturbance.data.constants import S2_BANDS
from forest_disturbance.models.layers import MLP
from forest_disturbance.models.statistical_mlp import spectral_indices


class TemporalGRU(nn.Module):
    """Encode recent and same-season historical Sentinel-2 series with GRUs.

    The model stays deliberately small so the comparison against the provided
    StatisticalMLP is easy to interpret. It still uses only the labelled centre pixel
    and Sentinel-2; Sentinel-1 and spatial patches are left for later experiments.

    Architecture:
        12 S2 bands + NDVI/NDMI/NDWI
            -> per-frame MLP
            -> separate GRU for recent / yearly context
            -> concatenate both final states
            -> MLP trunk
            -> 7 class logits

    Padded time steps are compacted to the end using acquisition dates before the GRU,
    then ignored with ``pack_padded_sequence``. This also makes ``years_context > 1``
    safe: all yearly observations are merged and sorted chronologically.
    """

    def __init__(
        self,
        num_classes: int,
        frame_dim: int = 32,
        frame_hidden_dims: list[int] = (),
        hidden_dim: int = 64,
        gru_layers: int = 1,
        trunk_dim: int = 128,
        trunk_hidden_dims: list[int] = (128,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if gru_layers < 1:
            raise ValueError("gru_layers must be >= 1")

        self.hidden_dim = hidden_dim
        self.frame_mlp = MLP(
            len(S2_BANDS) + 3,
            list(frame_hidden_dims),
            frame_dim,
            dropout,
        )

        gru_dropout = dropout if gru_layers > 1 else 0.0
        self.recent_gru = nn.GRU(
            input_size=frame_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.yearly_gru = nn.GRU(
            input_size=frame_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.recent_norm = nn.LayerNorm(hidden_dim)
        self.yearly_norm = nn.LayerNorm(hidden_dim)
        self.trunk = MLP(
            2 * hidden_dim,
            list(trunk_hidden_dims),
            trunk_dim,
            dropout,
        )
        self.class_head = nn.Linear(trunk_dim, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        """Return class logits with shape ``[B, num_classes]``."""
        if "s2" not in batch["inputs"]:
            raise KeyError("TemporalGRU requires Sentinel-2 input (data.sensors must include s2).")

        s2 = batch["inputs"]["s2"]
        recent = self._encode_series(s2["recent"], self.recent_gru)
        yearly_series = self._merge_yearly(s2["yearly"], like=s2["recent"])
        yearly = self._encode_series(yearly_series, self.yearly_gru)

        features = torch.cat(
            (self.recent_norm(recent), self.yearly_norm(yearly)),
            dim=-1,
        )
        return self.class_head(self.trunk(features))

    def _encode_series(self, series: dict[str, torch.Tensor], gru: nn.GRU) -> torch.Tensor:
        """Encode one padded series and return the final valid GRU state per example.

        ``series`` contains:
          values: [B, T, 12, H, W]
          dates:  [B, T] ordinal day numbers, 0 where padded
          mask:   [B, T] True for real acquisitions
        """
        values = series["values"]
        dates = series["dates"]
        mask = series["mask"].bool()
        batch_size = values.shape[0]

        if values.shape[1] == 0:
            return values.new_zeros((batch_size, self.hidden_dim))

        height, width = values.shape[-2:]
        pixels = values[:, :, :, height // 2, width // 2]
        frames = torch.cat((pixels, spectral_indices(pixels)), dim=-1)
        features = self.frame_mlp(frames)

        # ``yearly`` windows can be concatenated in block order (D-1y, D-2y, ...),
        # and padding may sit between those blocks. Sort real observations by date and
        # move all padded entries to the end before packing the sequence.
        pad_date = torch.iinfo(dates.dtype).max
        sort_key = dates.masked_fill(~mask, pad_date)
        order = torch.argsort(sort_key, dim=1)
        features = features.gather(1, order[..., None].expand(-1, -1, features.shape[-1]))
        mask = mask.gather(1, order)

        lengths = mask.sum(dim=1)
        encoded = features.new_zeros((batch_size, self.hidden_dim))
        valid_rows = lengths > 0
        if not valid_rows.any():
            return encoded

        packed = pack_padded_sequence(
            features[valid_rows],
            lengths[valid_rows].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = gru(packed)
        encoded[valid_rows] = hidden[-1].to(dtype=encoded.dtype)
        return encoded

    @staticmethod
    def _merge_yearly(
        yearly: list[dict[str, torch.Tensor]],
        *,
        like: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Merge yearly windows; return a correctly-shaped empty series if none exist."""
        if yearly:
            return {
                key: torch.cat([year[key] for year in yearly], dim=1)
                for key in ("values", "dates", "mask")
            }

        values = like["values"][:, :0]
        dates = like["dates"][:, :0]
        mask = like["mask"][:, :0]
        return {"values": values, "dates": dates, "mask": mask}
