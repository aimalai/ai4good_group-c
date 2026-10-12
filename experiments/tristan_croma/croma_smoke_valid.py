import torch
import torch.nn.functional as F
import zarr

from torchgeo.models import croma_base, CROMABase_Weights

STORE = "/zfs/ai4good/datasets/forest_disturbance/patches/0.zarr"
DEVICE = "cuda"

print("=== CROMA VALID-IMAGE TEST ===")
print("GPU:", torch.cuda.get_device_name(0))

store = zarr.open(STORE, mode="r")

def crop(x, size):
    h, w = x.shape[-2:]
    y = (h - size) // 2
    z = (w - size) // 2
    return x[..., y:y+size, z:z+size]

def load_s2(i):
    b10 = torch.as_tensor(store["s2_10m"][i], dtype=torch.float32).unsqueeze(0)
    b20 = torch.as_tensor(store["s2_20m"][i], dtype=torch.float32).unsqueeze(0)
    b60 = torch.as_tensor(store["s2_60m"][i], dtype=torch.float32).unsqueeze(0)

    b10 = crop(b10, 120)
    b20 = crop(b20, 60)
    b60 = crop(b60, 20)

    b20 = F.interpolate(b20, size=(120,120), mode="bilinear", align_corners=False)
    b60 = F.interpolate(b60, size=(120,120), mode="bilinear", align_corners=False)

    # B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
    return torch.cat(
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

# Find a genuinely non-empty / non-constant acquisition.
chosen = None

for i in range(min(200, store["s2_10m"].shape[0])):
    x = load_s2(i)

    nonzero = (x != 0).float().mean().item()
    band_std = x.flatten(2).std(dim=2)
    mean_std = band_std.mean().item()

    if nonzero > 0.90 and mean_std > 10:
        chosen = (i, x, nonzero, mean_std)
        break

if chosen is None:
    raise RuntimeError("Could not find a good Sentinel-2 frame in first 200 acquisitions")

idx, s2, nonzero, mean_std = chosen

print("Selected frame:", idx)
print("Raw shape:", tuple(s2.shape))
print("Raw range:", float(s2.min()), "to", float(s2.max()))
print("Nonzero fraction:", nonzero)
print("Mean spatial band std:", mean_std)

# Official CROMA-style normalization.
imgs = []

for channel in range(s2.shape[1]):
    band = s2[:, channel]

    mean = band.mean()
    std = band.std().clamp_min(1e-6)

    min_value = mean - 2 * std
    max_value = mean + 2 * std

    band = (band - min_value) / (max_value - min_value)
    band = torch.clamp(band, 0, 1).unsqueeze(1)

    imgs.append(band)

s2 = torch.cat(imgs, dim=1)

print("Normalized range:", float(s2.min()), "to", float(s2.max()))
print("Normalized std:", float(s2.std()))

assert s2.shape == (1, 12, 120, 120)
assert s2.std() > 0.01

print("\nLoading pretrained CROMA...")
model = croma_base(
    weights=CROMABase_Weights.CROMA_VIT,
    modalities=["optical"],
    image_size=120,
).to(DEVICE)

model.eval()

with torch.inference_mode():
    out = model(x_optical=s2.to(DEVICE))

embedding = out["optical_GAP"]

print("\nEmbedding:", tuple(embedding.shape))
print("Embedding finite:", bool(torch.isfinite(embedding).all()))
print("Embedding norm:", float(embedding.float().norm()))

assert embedding.shape == (1, 768)
assert torch.isfinite(embedding).all()

print("\n======================================")
print("CROMA REAL-IMAGE TEST PASSED")
print("======================================")
