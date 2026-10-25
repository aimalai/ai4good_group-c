"""Temporal Attention Network with Intelligent Semantic Pruning."""

import torch
from torch import nn
import math
from forest_disturbance.data.constants import S2_BANDS
from forest_disturbance.models.layers import MLP

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TemporalAttentionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        frame_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
        temperature: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        
        self.s2_proj = MLP(len(S2_BANDS) + 3, [], hidden_dim, dropout)
        self.s1_proj = MLP(2, [], hidden_dim, dropout)
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout,
            batch_first=True
        )
        self.s2_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.s1_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        fusion_dim = (hidden_dim * 4) + (hidden_dim * 4)
        self.trunk = MLP(fusion_dim, [128, 64], 64, dropout)
        self.class_head = nn.Linear(64, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        s2 = batch["inputs"]["s2"]
        s1 = batch["inputs"].get("s1", None)
        
        s2_rec = self.encode_sequence(s2["recent"], self.s2_proj, self.s2_transformer, is_s2=True)
        s2_year = self.encode_sequence(self._merge_yearly(s2["yearly"]), self.s2_proj, self.s2_transformer, is_s2=True)
        s2_diff = s2_rec - s2_year
        s2_features = torch.cat((s2_rec, s2_year, s2_diff, s2_diff.abs()), dim=-1)
        
        if s1 is not None:
            s1_rec = self.encode_sequence(s1["recent"], self.s1_proj, self.s1_transformer, is_s2=False)
            s1_year = self.encode_sequence(self._merge_yearly(s1["yearly"]), self.s1_proj, self.s1_transformer, is_s2=False)
            s1_diff = s1_rec - s1_year
            s1_features = torch.cat((s1_rec, s1_year, s1_diff, s1_diff.abs()), dim=-1)
        else:
            s1_features = torch.zeros_like(s2_features)
            
        combined = torch.cat((s2_features, s1_features), dim=-1)
        logits = self.class_head(self.trunk(combined))
        
        # Return scaled logits unconditionally. The evaluation framework handles the softmax.
        return logits / self.temperature

    def _merge_yearly(self, yearly_list: list) -> dict | None:
        if not yearly_list:
            return None
        return {
            key: torch.cat([year[key] for year in yearly_list], dim=1)
            for key in ("values", "dates", "mask")
        }

    def encode_sequence(self, series: dict | None, proj_layer: nn.Module, transformer: nn.Module, is_s2: bool) -> torch.Tensor:
        if not series or series["values"].shape[1] == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, transformer.layers[0].linear1.in_features, device=device)

        vals = series["values"]
        mask = series["mask"]
        h, w = vals.shape[-2:]
        pixels = vals[:, :, :, h // 2, w // 2]
        
        if is_s2:
            frames = torch.cat((pixels, spectral_indices(pixels)), dim=-1)
        else:
            frames = pixels
            
        # Protect against anomalous infinite inputs before projection
        frames = torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
            
        emb = proj_layer(frames)
        emb = self.pos_encoder(emb)
        
        key_padding_mask = ~mask.bool() 
        all_masked = key_padding_mask.all(dim=1, keepdim=True)
        
        # PREVENT NaN: If a sequence is 100% clouds, unmask the first token temporarily so softmax can compute safely.
        # We zero out these specific sequences entirely during the final pooling step below.
        safe_key_padding_mask = key_padding_mask.masked_fill(all_masked, False)
        
        out = transformer(emb, src_key_padding_mask=safe_key_padding_mask)
        
        # Apply strict mask before pooling: fill all cloudy/invalid tokens with -1e9 so they are ignored by max()
        strict_mask = key_padding_mask.unsqueeze(-1)
        out = out.masked_fill(strict_mask, -1e9)
        pooled = out.max(dim=1)[0]
        
        # Finally, force sequences that were entirely empty to true zeros
        return pooled.masked_fill(all_masked, 0.0)

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
