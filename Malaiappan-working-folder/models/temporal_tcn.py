"""Gated Residual Temporal Convolutional Network (GR-TCN) with SE Blocks."""

import torch
from torch import nn
from forest_disturbance.data.constants import S2_BANDS
from forest_disturbance.models.layers import MLP

class SqueezeExcitation1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Masked Global Average Pooling for SE
        mask_float = mask.float()
        valid_counts = mask_float.sum(dim=2, keepdim=True).clamp(min=1.0)
        avg_pool = (x * mask_float).sum(dim=2) / valid_counts.squeeze(2)
        
        weights = torch.sigmoid(self.fc2(torch.relu(self.fc1(avg_pool))))
        return x * weights.unsqueeze(2)

class GatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        # GLU requires doubling the output channels of the convolution
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.se = SqueezeExcitation1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Apply padding mask before convolution to prevent noise bleeding
        x_masked = x * mask.float()
        
        out = self.conv(x_masked)
        
        # GLU Activation (Split channels in half: compute * gate)
        out, gate = out.chunk(2, dim=1)
        out = out * torch.sigmoid(gate)
        
        # Channel-wise feature recalibration
        out = self.se(out, mask)
        
        # Sequence-first for LayerNorm
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(out)
        
        # Re-apply strict mask after transformations
        out = (out + residual) * mask.float()
        return out

class TemporalTCNModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        frame_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        
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
        
        # Fusion dim: S2 (Rec_Max, Rec_Mean, Yr_Max, Yr_Mean, Diff_Max, Diff_Mean) + S1 equivalents
        # Plus absolute diffs. Masked max+mean doubles the pooled output size.
        feature_dim = (hidden_dim * 2 * 4) + (hidden_dim * 2 * 4)
        
        self.trunk = MLP(feature_dim, [256, 128], 64, dropout)
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
        
        # Retain raw logits for operational CUSUM filter tracking
        return logits

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
            return torch.zeros(1, proj.layers[-1].out_features * 2, device=device)

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
            
        return self._masked_dual_pooling(x, mask)

    def _masked_dual_pooling(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask.bool()
        
        # Max Pooling (ignores clouds)
        x_max = x.masked_fill(~mask_bool, -1e9)
        pooled_max = x_max.max(dim=2)[0]
        
        # Mean Pooling (ignores clouds)
        x_zero = x.masked_fill(~mask_bool, 0.0)
        valid_counts = mask_bool.sum(dim=2).clamp(min=1)
        pooled_mean = x_zero.sum(dim=2) / valid_counts
        
        # Handle 100% cloudy sequences safely
        all_masked = ~mask_bool.any(dim=2)
        pooled_max = pooled_max.masked_fill(all_masked, 0.0)
        pooled_mean = pooled_mean.masked_fill(all_masked, 0.0)
        
        return torch.cat([pooled_max, pooled_mean], dim=-1)

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
