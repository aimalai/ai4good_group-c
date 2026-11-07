"""Gated Residual Temporal Convolutional Network (GR-TCN) with Hierarchical Heads & Positional Encoding."""

import math
import torch
from torch import nn
from forest_disturbance.data.constants import S2_BANDS
from forest_disturbance.models.layers import MLP


class MultiHeadAttentionPooling1d(nn.Module):
    """Multi-head attention pooling for Tcn sequences."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.10):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.key_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T], mask: [B, 1, T]
        x_t = x.transpose(1, 2)  # [B, T, D]
        B, T, D = x_t.shape

        if T == 0:
            return x_t.new_zeros(B, D)

        logits = self.key_net(x_t)  # [B, T, num_heads]
        logits = logits / math.sqrt(self.head_dim)

        mask_bool = mask.transpose(1, 2).bool()  # [B, T, 1]
        logits = logits.masked_fill(~mask_bool, -1e4)

        weights = torch.softmax(logits, dim=1) * mask_bool.to(dtype=logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = self.dropout(weights)

        values = x_t.view(B, T, self.num_heads, self.head_dim)
        pooled = torch.einsum("bth,bthd->bhd", weights, values).reshape(B, D)
        return self.out_norm(pooled)


class GatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x_masked = x * mask.float()
        
        out = self.conv(x_masked)
        out, gate = out.chunk(2, dim=1)
        out = out * torch.sigmoid(gate)
        
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(out)
        
        return (out + residual) * mask.float()


class TemporalTCNModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        s2_hidden_dim: int = 96,
        s1_hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.10,
        num_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Projections (S2: 12 bands + 4 indices/mask = 16 | S1: 2 bands + 1 mask = 3)
        self.s2_proj = nn.Sequential(
            nn.Linear(len(S2_BANDS) + 4, s2_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(s2_hidden_dim),
            nn.Linear(s2_hidden_dim, s2_hidden_dim),
            nn.GELU(),
        )
        self.s2_time_proj = nn.Linear(10, s2_hidden_dim, bias=False)

        self.s1_proj = nn.Sequential(
            nn.Linear(3, s1_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(s1_hidden_dim),
            nn.Linear(s1_hidden_dim, s1_hidden_dim),
            nn.GELU(),
        )
        self.s1_time_proj = nn.Linear(10, s1_hidden_dim, bias=False)
        
        dilations = [2**i for i in range(num_layers)]  # [1, 2, 4, 8]
        self.s2_blocks = nn.ModuleList([
            GatedResidualBlock(s2_hidden_dim, dilation=d, dropout=dropout) for d in dilations
        ])
        self.s1_blocks = nn.ModuleList([
            GatedResidualBlock(s1_hidden_dim, dilation=d, dropout=dropout) for d in dilations
        ])
        
        self.s2_pooler = MultiHeadAttentionPooling1d(s2_hidden_dim, num_heads=num_heads, dropout=dropout)
        self.s1_pooler = MultiHeadAttentionPooling1d(s1_hidden_dim, num_heads=num_heads, dropout=dropout)
        
        feature_dim = (s2_hidden_dim * 4) + (s1_hidden_dim * 4)
        
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Hierarchical Heads
        self.binary_head = nn.Linear(128, 1)
        self.attr_head = nn.Linear(128, num_classes - 1)

    def forward(self, batch: dict) -> torch.Tensor:
        target_date = batch["target_date"]
        s2 = batch["inputs"]["s2"]
        s1 = batch["inputs"].get("s1", None)
        
        s2_rec = self.encode_series(s2["recent"], target_date, self.s2_proj, self.s2_time_proj, self.s2_blocks, self.s2_pooler, is_s2=True)
        s2_year = self.encode_series(self._merge_yearly(s2["yearly"]), target_date, self.s2_proj, self.s2_time_proj, self.s2_blocks, self.s2_pooler, is_s2=True)
        s2_diff = s2_rec - s2_year
        s2_features = torch.cat((s2_rec, s2_year, s2_diff, s2_diff.abs()), dim=-1)
        
        if s1 is not None:
            s1_rec = self.encode_series(s1["recent"], target_date, self.s1_proj, self.s1_time_proj, self.s1_blocks, self.s1_pooler, is_s2=False)
            s1_year = self.encode_series(self._merge_yearly(s1["yearly"]), target_date, self.s1_proj, self.s1_time_proj, self.s1_blocks, self.s1_pooler, is_s2=False)
            s1_diff = s1_rec - s1_year
            s1_features = torch.cat((s1_rec, s1_year, s1_diff, s1_diff.abs()), dim=-1)
        else:
            s1_features = torch.zeros_like(s2_features)
            
        combined = torch.cat((s2_features, s1_features), dim=-1)
        trunk_out = self.trunk(combined)
        
        b_logits = self.binary_head(trunk_out)
        a_logits = self.attr_head(trunk_out)
        
        no_dist_logit = -b_logits
        dist_logits = b_logits + a_logits
        
        return torch.cat([no_dist_logit, dist_logits], dim=-1)

    def _merge_yearly(self, yearly_list: list) -> dict | None:
        if not yearly_list:
            return None
        return {
            key: torch.cat([year[key] for year in yearly_list], dim=1)
            for key in ("values", "dates", "mask")
        }

    @staticmethod
    def _time_features(dates, target_date, *, dtype):
        rel = (dates - target_date[:, None]).to(torch.float32)
        feats = []
        for period in (7.0, 30.0, 90.0, 365.25):
            angle = 2.0 * math.pi * rel / period
            feats.extend((torch.sin(angle), torch.cos(angle)))
        doy_angle = 2.0 * math.pi * torch.remainder(dates.to(torch.float32), 365.25) / 365.25
        feats.extend((torch.sin(doy_angle), torch.cos(doy_angle)))
        return torch.stack(feats, dim=-1).to(dtype=dtype)

    def encode_series(self, series, target_date, proj, time_proj, blocks, pooler, is_s2: bool) -> torch.Tensor:
        if not series or series["values"].shape[1] == 0:
            device = next(self.parameters()).device
            dim = time_proj.out_features
            return torch.zeros(1, dim, device=device)

        vals = series["values"]
        dates = series["dates"]
        mask_seq = series["mask"]
        
        h, w = vals.shape[-2:]
        pixels = vals[:, :, :, h // 2, w // 2]
        
        feature_mask = mask_seq.unsqueeze(-1).float()
        if is_s2:
            frames = torch.cat((pixels, spectral_indices(pixels), feature_mask), dim=-1)
        else:
            frames = torch.cat((pixels, feature_mask), dim=-1)
            
        frames = torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
        
        z = proj(frames)
        z = z + time_proj(self._time_features(dates, target_date, dtype=z.dtype))
        
        conv_mask = mask_seq.unsqueeze(1)  # [B, 1, T]
        x = z.transpose(1, 2)              # [B, D, T]
        
        for block in blocks:
            x = block(x, conv_mask)
            
        return pooler(x, conv_mask)

def spectral_indices(pixels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    band = {name: pixels[..., i] for i, name in enumerate(S2_BANDS)}
    red, green, nir, swir = band["B04"], band["B03"], band["B08"], band["B11"]
    
    def normalized_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b) / (a + b).clamp_min(eps)

    return torch.stack(
        (
            normalized_difference(nir, red),
            normalized_difference(nir, swir),
            normalized_difference(green, nir),
        ),
        dim=-1,
    )
