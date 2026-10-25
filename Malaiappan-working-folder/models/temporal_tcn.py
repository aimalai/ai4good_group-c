"""Gated Residual Temporal Convolutional Network (GR-TCN) with Attention Pooling."""

import torch
from torch import nn
from forest_disturbance.data.constants import S2_BANDS
from forest_disturbance.models.layers import MLP


class AttentionPooling1d(nn.Module):
    """Learns a dynamic weighted average over the temporal sequence, ignoring clouds."""
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.Tanh(),
            nn.Linear(channels // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, T] -> transpose to [B, T, C] for linear layers
        x_t = x.transpose(1, 2)
        
        # Calculate unnormalized attention scores
        attn_scores = self.attention(x_t) # [B, T, 1]
        
        # Apply strict cloud/padding mask before softmax
        mask_bool = mask.transpose(1, 2).bool()
        attn_scores = attn_scores.masked_fill(~mask_bool, -1e9)
        
        # Normalize weights
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Handle entirely cloudy sequences (NaN prevention)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)
        
        # Calculate the weighted temporal average
        # Batch matrix multiply: [B, C, T] x [B, T, 1] -> [B, C, 1]
        pooled = torch.bmm(x, attn_weights)
        return pooled.squeeze(-1)


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
        
        # Gated Linear Unit activation
        out, gate = out.chunk(2, dim=1)
        out = out * torch.sigmoid(gate)
        
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(out)
        
        return (out + residual) * mask.float()


class TemporalTCNModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        frame_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        temperature: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        
        self.s2_proj = MLP(len(S2_BANDS) + 3, [], hidden_dim, dropout)
        self.s1_proj = MLP(2, [], hidden_dim, dropout)
        
        self.s2_blocks = nn.ModuleList([
            GatedResidualBlock(hidden_dim, dilation=2**i, dropout=dropout) 
            for i in range(num_layers)
        ])
        
        self.s1_blocks = nn.ModuleList([
            GatedResidualBlock(hidden_dim, dilation=2**i, dropout=dropout) 
            for i in range(num_layers)
        ])
        
        self.pooler = AttentionPooling1d(hidden_dim)
        
        # Fusion dim: S2 (Rec, Yr, Diff, Abs) + S1 equivalents
        feature_dim = (hidden_dim * 4) + (hidden_dim * 4)
        
        self.trunk = MLP(feature_dim, [128], 64, dropout)
        self.class_head = nn.Linear(64, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        s2 = batch["inputs"]["s2"]
        s1 = batch["inputs"].get("s1", None)
        
        s2_rec = self.encode_series(s2["recent"], self.s2_proj, self.s2_blocks, is_s2=True)
        s2_year = self.encode_series(self._merge_yearly(s2["yearly"]), self.s2_proj, self.s2_blocks, is_s2=True)
        s2_diff = s2_rec - s2_year
        s2_features = torch.cat((s2_rec, s2_year, s2_diff, s2_diff.abs()), dim=-1)
        
        if s1 is not None:
            s1_rec = self.encode_series(s1["recent"], self.s1_proj, self.s1_blocks, is_s2=False)
            s1_year = self.encode_series(self._merge_yearly(s1["yearly"]), self.s1_proj, self.s1_blocks, is_s2=False)
            s1_diff = s1_rec - s1_year
            s1_features = torch.cat((s1_rec, s1_year, s1_diff, s1_diff.abs()), dim=-1)
        else:
            s1_features = torch.zeros_like(s2_features)
            
        combined = torch.cat((s2_features, s1_features), dim=-1)
        logits = self.class_head(self.trunk(combined))
        
        # Scaled Logits forcing operational confidence
        return logits / self.temperature

    def _merge_yearly(self, yearly_list: list) -> dict | None:
        if not yearly_list:
            return None
        return {
            key: torch.cat([year[key] for year in yearly_list], dim=1)
            for key in ("values", "dates", "mask")
        }

    def encode_series(self, series: dict | None, proj: nn.Module, blocks: nn.ModuleList, is_s2: bool) -> torch.Tensor:
        if not series or series["values"].shape[1] == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, proj.layers[-1].out_features, device=device)

        vals = series["values"]
        mask = series["mask"].unsqueeze(1) # [B, 1, T]
        h, w = vals.shape[-2:]
        pixels = vals[:, :, :, h // 2, w // 2]
        
        if is_s2:
            frames = torch.cat((pixels, spectral_indices(pixels)), dim=-1)
        else:
            frames = pixels
            
        frames = torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Embed and transpose for Conv1d [B, C, T]
        x = proj(frames).transpose(1, 2)
        
        for block in blocks:
            x = block(x, mask)
            
        # Dynamic Attention-Weighted Smoothing
        return self.pooler(x, mask)

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
