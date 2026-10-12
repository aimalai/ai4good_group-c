import torch
import torch.nn.functional as F
import zarr

from torchgeo.models import croma_base, CROMABase_Weights

DEVICE = "cuda"
STORE = "/zfs/ai4good/datasets/forest_disturbance/patches/0.zarr"

print("=== CROMA FOUNDATION MODEL SMOKE TEST ===")
print("GPU:", torch.cuda.get_device_name(0))

store = zarr.open(STORE, mode="r")

# First Sentinel-2 acquisition from one real DISFOR forest site.
b10 = torch.as_tensor(store["s2_10m"][0], dtype=torch.float32).unsqueeze(0)
b20 = torch.as_tensor(store["s2_20m"][0], dtype=torch.float32).unsqueeze(0)
b60 = torch.as_tensor(store["s2_60m"][0], dtype=torch.float32).unsqueeze(0)

def centre_crop(x, size):
    h, w = x.shape[-2:]
    top = (h - size) // 2
    left = (w - size) // 2
    return x[..., top:top+size, left:left+size]

# CROMA was pretrained at 120x120.
# Crop the same physical 1.2 km area at each Sentinel-2 resolution.
b10 = centre_crop(b10, 120)
b20 = centre_crop(b20, 60)
b60 = centre_crop(b60, 20)

b20 = F.interpolate(
    b20, size=(120, 120), mode="bilinear", align_corners=False
)
b60 = F.interpolate(
    b60, size=(120, 120), mode="bilinear", align_corners=False
)

# Reassemble DISFOR bands in wavelength order:
# B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
s2 = torch.cat(
    (
        b60[:, :1],
        b10[:, :3],
        b20[:, :3],
        b10[:, 3:],
        b20[:, 3:4],
        b60[:, 1:],
        b20[:, 4:],
    ),
    dim=1,
)

assert s2.shape == (1, 12, 120, 120), s2.shape

# CROMA-style per-band normalization.
mean = s2.mean(dim=(-2, -1), keepdim=True)
std = s2.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

s2 = ((s2 - (mean - 2 * std)) / (4 * std) * 255.0)
s2 = s2.clamp(0, 255).to(torch.uint8).float() / 255.0

print("Real DISFOR input:", tuple(s2.shape))
print("Input range:", float(s2.min()), "to", float(s2.max()))

print("\nLoading pretrained CROMA-base...")
model = croma_base(
    weights=CROMABase_Weights.CROMA_VIT,
    modalities=["optical"],
    image_size=120,
).to(DEVICE)

model.eval()

with torch.inference_mode():
    output = model(x_optical=s2.to(DEVICE))

embedding = output["optical_GAP"]

print("\nOutput keys:", list(output))
print("CROMA embedding:", tuple(embedding.shape))
print("Embedding finite:", bool(torch.isfinite(embedding).all()))
print("Embedding norm:", float(embedding.float().norm()))

assert embedding.shape == (1, 768)
assert torch.isfinite(embedding).all()

print("\n===================================")
print("CROMA FOUNDATION MODEL TEST PASSED")
print("===================================")
