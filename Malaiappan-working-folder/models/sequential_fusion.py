"""Sequential Multi-Sensor Spatial-Temporal Fusion Model (GRU + S1/S2)."""

import torch
from torch import nn
from forest_disturbance.data.constants import S2_BANDS
from forest_disturbance.models.layers import MLP


class SequentialFusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        frame_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.s2_frame_mlp = MLP(len(S2_BANDS) + 3, [], frame_dim, dropout)
        self.s1_frame_mlp = MLP(2, [], frame_dim, dropout)
        
        self.recent_gru = nn.GRU(
            input_size=frame_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.trunk = MLP(hidden_dim * 2, [64], 64, dropout)
        self.class_head = MLP(64, [], num_classes, dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        s2 = batch["inputs"]["s2"]
        s1 = batch["inputs"]["s1"] if "s1" in batch["inputs"] else None
        
        recent_feat = self.encode_series(s2["recent"], s1["recent"] if s1 else None)
        
        if s2["yearly"]:
            yearly_s2_dict = {
                key: torch.cat([year[key] for year in s2["yearly"]], dim=1)
                for key in ("values", "dates", "mask")
            }
            yearly_s1_dict = None
            if s1 and "yearly" in s1 and s1["yearly"]:
                yearly_s1_dict = {
                    key: torch.cat([year[key] for year in s1["yearly"]], dim=1)
                    for key in ("values", "dates", "mask")
                }
            yearly_feat = self.encode_series(yearly_s2_dict, yearly_s1_dict)
        else:
            yearly_feat = torch.zeros_like(recent_feat)
            
        combined = torch.cat((recent_feat, yearly_feat), dim=-1)
        logits = self.class_head(self.trunk(combined))
        
        if self.training:
            return logits
        
        probs = torch.softmax(logits, dim=-1)
        return torch.clamp(probs, 0.0, 1.0)

    def encode_series(self, s2_series: dict, s1_series: dict | None) -> torch.Tensor:
        s2_vals = s2_series["values"]
        
        # If zarr patch reader is used with image_size > 1, pool spatially. Otherwise index center pixel.
        if s2_vals.dim() == 5 and s2_vals.shape[-1] > 1:
            s2_pixels = s2_vals.mean(dim=(-2, -1))
        else:
            h2, w2 = s2_vals.shape[-2:]
            s2_pixels = s2_vals[:, :, :, h2 // 2, w2 // 2]
            
        s2_frames = torch.cat((s2_pixels, spectral_indices(s2_pixels)), dim=-1)
        s2_emb = self.s2_frame_mlp(s2_frames)
        
        if s1_series is not None:
            s1_vals = s1_series["values"]
            if s1_vals.dim() == 5 and s1_vals.shape[-1] > 1:
                s1_pixels = s1_vals.mean(dim=(-2, -1))
            else:
                h1, w1 = s1_vals.shape[-2:]
                s1_pixels = s1_vals[:, :, :, h1 // 2, w1 // 2]
                
            s1_emb = self.s1_frame_mlp(s1_pixels)
            min_t = min(s2_emb.shape[1], s1_emb.shape[1])
            frames = torch.cat((s2_emb[:, :min_t], s1_emb[:, :min_t]), dim=-1)
        else:
            frames = torch.cat((s2_emb, torch.zeros_like(s2_emb)), dim=-1)
            
        out, h_n = self.recent_gru(frames)
        return h_n[-1]


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
