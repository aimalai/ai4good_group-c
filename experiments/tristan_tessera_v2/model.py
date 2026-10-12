from pathlib import Path
import importlib.util

import torch
from torch import nn
from huggingface_hub import snapshot_download


# DISFOR order:
# B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
#
# TESSERA order:
# B04 B02 B03 B08 B8A B05 B06 B07 B11 B12
TESSERA_S2_IDX = [3, 1, 2, 7, 8, 4, 5, 6, 10, 11]


def load_tessera_s2(cache_dir):
    print("Loading TESSERA v2 Medium foundation model...")

    path = snapshot_download(
        repo_id="geotessera/TESSERA-V-2.0-2B-M",
        cache_dir=cache_dir,
    )

    model_file = Path(path) / "model.py"

    spec = importlib.util.spec_from_file_location(
        "_external_tessera_v2",
        model_file,
    )

    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)

    full = ext.load_model(
        str(Path(path) / "ckpt/student_medium.pt"),
        torch.device("cpu"),
    )

    # We deliberately use only the pretrained Sentinel-2 temporal backbone.
    encoder = full.s2_backbone

    mean = torch.tensor(
        ext.S2_BAND_MEAN,
        dtype=torch.float32,
    )

    std = torch.tensor(
        ext.S2_BAND_STD,
        dtype=torch.float32,
    )

    del full

    return encoder, mean, std


def ordinal_to_doy(ordinals):
    """Exact Gregorian ordinal -> day of year, fully vectorized."""
    n = ordinals.long() - 1

    rem = torch.remainder(n, 146097)

    n100 = torch.clamp(
        torch.div(rem, 36524, rounding_mode="floor"),
        max=3,
    )
    rem = rem - n100 * 36524

    n4 = torch.div(rem, 1461, rounding_mode="floor")
    rem = rem - n4 * 1461

    n1 = torch.clamp(
        torch.div(rem, 365, rounding_mode="floor"),
        max=3,
    )
    rem = rem - n1 * 365

    return rem + 1


class FrozenTesseraV2(nn.Module):
    """
    DISFOR S2 centre-pixel time series
        ->
    pretrained TESSERA-v2 temporal encoder
        ->
    recent/yearly embedding comparison
        ->
    small nonlinear disturbance classifier.
    """

    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        cache_dir="/zfs/ai4good/student/thein/tessera_cache",
    ):
        super().__init__()

        encoder, mean, std = load_tessera_s2(cache_dir)

        self.encoder = encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        self.register_buffer(
            "s2_mean",
            mean,
        )

        self.register_buffer(
            "s2_std",
            std,
        )

        # Output dimension of the S2 temporal encoder.
        embed_dim = self.encoder.attn_pool.query.in_features

        print("TESSERA S2 embedding dim:", embed_dim)

        # recent + yearly + signed change + absolute change
        in_dim = embed_dim * 4

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(hidden_dim, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)

        # Foundation model stays frozen.
        self.encoder.eval()

        return self

    def forward(self, batch):
        s2 = batch["inputs"]["s2"]

        recent = self._encode(s2["recent"])

        if s2["yearly"]:
            yearly_series = {
                key: torch.cat(
                    [year[key] for year in s2["yearly"]],
                    dim=1,
                )
                for key in ("values", "dates", "mask")
            }

            yearly = self._encode(yearly_series)

        else:
            yearly = torch.zeros_like(recent)

        delta = recent - yearly

        features = torch.cat(
            (
                recent,
                yearly,
                delta,
                delta.abs(),
            ),
            dim=-1,
        )

        return self.head(features)

    def _encode(self, series):
        # [B,T,12,1,1] -> [B,T,12]
        values = series["values"]
        h, w = values.shape[-2:]

        x = values[
            :, :, :, h // 2, w // 2
        ]

        # Exact TESSERA-v2 S2 band order.
        x = x[:, :, TESSERA_S2_IDX]

        mask = series["mask"]

        # Remove true no-data acquisitions.
        mask = mask & (
            x.abs().sum(dim=-1) > 0
        )

        # TESSERA was trained with observation counts bucketed to
        # multiples of 8. Our DISFOR windows are short, so use the
        # minimum training-time bucket of eight observations.
        x, dates = self._to_eight(
            x,
            series["dates"],
            mask,
        )

        # Official TESSERA-v2 training normalization.
        x = (
            x - self.s2_mean[None, None]
        ) / (
            self.s2_std[None, None] + 1e-6
        )

        doy = ordinal_to_doy(
            dates
        ).to(dtype=x.dtype)

        # [B,8,10 bands + DOY]
        x = torch.cat(
            (
                x,
                doy.unsqueeze(-1),
            ),
            dim=-1,
        )

        self.encoder.eval()

        with torch.no_grad():
            embedding = self.encoder(x)

        return embedding

    @staticmethod
    def _to_eight(values, dates, mask):
        """
        Pad/subsample each real time series to exactly 8 observations.

        Padding mirrors TESSERA's published inference procedure:
        real observations first, then repeated observations.
        """
        B = values.shape[0]

        out_x = []
        out_dates = []

        for i in range(B):
            idx = torch.nonzero(
                mask[i],
                as_tuple=False,
            ).flatten()

            # DISFOR target examples should have a valid window,
            # but keep a safe fallback.
            if idx.numel() == 0:
                idx = torch.zeros(
                    1,
                    dtype=torch.long,
                    device=values.device,
                )

            n = idx.numel()

            if n >= 8:
                positions = torch.linspace(
                    0,
                    n - 1,
                    8,
                    device=values.device,
                ).long()

            else:
                remain = 8 - n

                base = torch.arange(
                    n,
                    device=values.device,
                )

                if remain <= n:
                    groups = torch.tensor_split(
                        base,
                        remain,
                    )

                    fill = torch.stack(
                        [
                            g[len(g) // 2]
                            for g in groups
                        ]
                    )

                else:
                    fill = (
                        torch.arange(
                            remain,
                            device=values.device,
                        )
                        % n
                    )

                positions = torch.cat(
                    (
                        base,
                        fill,
                    )
                )

            chosen = idx[positions]

            out_x.append(
                values[i, chosen]
            )

            out_dates.append(
                dates[i, chosen]
            )

        return (
            torch.stack(out_x),
            torch.stack(out_dates),
        )
