import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from experiments.tristan_tessera_v2.model import (
    TESSERA_S2_IDX,
    load_tessera_s2,
    ordinal_to_doy,
)


class TesseraS1S2Fusion(nn.Module):
    """
    Late fusion of:

      Sentinel-2:
        frozen pretrained TESSERA-v2 temporal encoder

      Sentinel-1:
        small trainable GRU over VV/VH centre-pixel time series

    Both sensors retain the benchmark's:
        recent window
        previous-year window

    Target dates remain Sentinel-2 dates so evaluation stays directly
    comparable to the S2-only TESSERA experiment.
    """

    def __init__(
        self,
        num_classes: int,
        s1_frame_dim: int = 64,
        s1_hidden_dim: int = 128,
        head_dim: int = 512,
        dropout: float = 0.10,
        cache_dir: str = "/zfs/ai4good/student/thein/tessera_cache",
    ):
        super().__init__()

        print("Loading frozen TESSERA-v2 Sentinel-2 encoder...")

        encoder, s2_mean, s2_std = load_tessera_s2(cache_dir)

        self.s2_encoder = encoder
        self.s2_encoder.eval()

        for p in self.s2_encoder.parameters():
            p.requires_grad = False

        self.register_buffer(
            "s2_mean",
            s2_mean.float(),
        )

        self.register_buffer(
            "s2_std",
            s2_std.float(),
        )

        self.s2_dim = self.s2_encoder.attn_pool.query.in_features

        print("TESSERA S2 embedding dim:", self.s2_dim)

        # DISFOR Sentinel-1 is stored as dB * 100.
        # Convert back to dB, then LayerNorm learns a stable frame representation.
        self.s1_frame = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, s1_frame_dim),
            nn.GELU(),
        )

        self.s1_gru = nn.GRU(
            input_size=s1_frame_dim,
            hidden_size=s1_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # For each modality:
        # recent, yearly, signed change, absolute change.
        feature_dim = (
            4 * self.s2_dim
            + 4 * s1_hidden_dim
        )

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),

            nn.Linear(feature_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(head_dim, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)

        # TESSERA remains frozen and deterministic.
        self.s2_encoder.eval()

        return self

    def forward(self, batch):
        s2 = batch["inputs"]["s2"]
        s1 = batch["inputs"]["s1"]

        # ---------------------------------------------------------
        # Sentinel-2 TESSERA embeddings
        # ---------------------------------------------------------

        s2_recent = self._encode_s2(
            s2["recent"]
        )

        s2_yearly = self._encode_s2_yearly(
            s2["yearly"],
            reference=s2_recent,
        )

        s2_delta = s2_recent - s2_yearly

        # ---------------------------------------------------------
        # Sentinel-1 GRU embeddings
        # ---------------------------------------------------------

        s1_recent = self._encode_s1(
            s1["recent"]
        )

        s1_yearly = self._encode_s1_yearly(
            s1["yearly"],
            reference=s1_recent,
        )

        s1_delta = s1_recent - s1_yearly

        # ---------------------------------------------------------
        # Late multimodal fusion
        # ---------------------------------------------------------

        features = torch.cat(
            (
                s2_recent,
                s2_yearly,
                s2_delta,
                s2_delta.abs(),

                s1_recent,
                s1_yearly,
                s1_delta,
                s1_delta.abs(),
            ),
            dim=-1,
        )

        return self.head(features)

    def _encode_s2_yearly(self, years, reference):
        if not years:
            return torch.zeros_like(reference)

        merged = {
            key: torch.cat(
                [year[key] for year in years],
                dim=1,
            )
            for key in ("values", "dates", "mask")
        }

        return self._encode_s2(merged)

    def _encode_s1_yearly(self, years, reference):
        if not years:
            return torch.zeros_like(reference)

        merged = {
            key: torch.cat(
                [year[key] for year in years],
                dim=1,
            )
            for key in ("values", "dates", "mask")
        }

        return self._encode_s1(merged)

    def _encode_s1(self, series):
        values = series["values"]
        mask = series["mask"]

        B = values.shape[0]
        T = values.shape[1]

        if T == 0:
            return values.new_zeros(
                B,
                self.s1_gru.hidden_size,
            )

        h = values.shape[-2]
        w = values.shape[-1]

        # [B,T,2,H,W] -> [B,T,2]
        x = values[
            :,
            :,
            :,
            h // 2,
            w // 2,
        ]

        # Stored as dB * 100.
        x = x / 100.0

        x = self.s1_frame(x)

        lengths = mask.sum(dim=1)

        output = x.new_zeros(
            B,
            self.s1_gru.hidden_size,
        )

        valid_rows = lengths > 0

        if valid_rows.any():
            packed = pack_padded_sequence(
                x[valid_rows],
                lengths[valid_rows].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            _, hidden = self.s1_gru(packed)

            output[valid_rows] = hidden[-1].to(
                dtype=output.dtype
            )

        return output

    def _encode_s2(self, series):
        values = series["values"]
        mask = series["mask"]

        B = values.shape[0]
        T = values.shape[1]

        if T == 0:
            return values.new_zeros(
                B,
                self.s2_dim,
            )

        h = values.shape[-2]
        w = values.shape[-1]

        # [B,T,12,H,W] -> [B,T,12]
        x = values[
            :,
            :,
            :,
            h // 2,
            w // 2,
        ]

        # DISFOR:
        # B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
        #
        # TESSERA:
        # B04 B02 B03 B08 B8A B05 B06 B07 B11 B12
        x = x[:, :, TESSERA_S2_IDX]

        # Remove true no-data observations.
        mask = mask & (
            x.abs().sum(dim=-1) > 0
        )

        x, dates = self._to_training_bucket(
            x,
            series["dates"],
            mask,
        )

        # Exact TESSERA S2 standardisation.
        x = (
            x
            - self.s2_mean[None, None]
        ) / (
            self.s2_std[None, None]
            + 1e-6
        )

        doy = ordinal_to_doy(
            dates
        ).to(dtype=x.dtype)

        x = torch.cat(
            (
                x,
                doy.unsqueeze(-1),
            ),
            dim=-1,
        )

        self.s2_encoder.eval()

        with torch.no_grad():
            embedding = self.s2_encoder(x)

        return embedding

    @staticmethod
    def _bucket_size(n):
        if n <= 0:
            return 8

        for b in range(8, 257, 8):
            if n <= b:
                return b

        return 256

    @classmethod
    def _to_training_bucket(
        cls,
        values,
        dates,
        mask,
    ):
        """
        Reproduce TESSERA-style temporal bucketing.

        Every sample is padded/subsampled to a multiple of 8.
        Because a batch must have one common sequence length,
        we use the largest bucket required by any sample in
        the current batch.
        """

        B = values.shape[0]

        lengths = [
            int(mask[i].sum().item())
            for i in range(B)
        ]

        bucket = max(
            cls._bucket_size(n)
            for n in lengths
        )

        out_x = []
        out_dates = []

        for i in range(B):
            idx = torch.nonzero(
                mask[i],
                as_tuple=False,
            ).flatten()

            if idx.numel() == 0:
                idx = torch.zeros(
                    1,
                    dtype=torch.long,
                    device=values.device,
                )

            n = idx.numel()

            if n >= bucket:
                positions = torch.linspace(
                    0,
                    n - 1,
                    bucket,
                    device=values.device,
                ).long()

            else:
                remain = bucket - n

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
