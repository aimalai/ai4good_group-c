import time
import torch
import torch.nn.functional as F
import zarr

from torchgeo.models import croma_base, CROMABase_Weights

STORE = "/zfs/ai4good/datasets/forest_disturbance/patches/0.zarr"
DEVICE = "cuda"
N = 64

torch.set_float32_matmul_precision("high")

store = zarr.open(STORE, mode="r")

def crop(x, size):
    h, w = x.shape[-2:]
    y = (h - size) // 2
    z = (w - size) // 2
    return x[..., y:y+size, z:z+size]

def load_frame(i):
    b10 = torch.as_tensor(store["s2_10m"][i], dtype=torch.float32).unsqueeze(0)
    b20 = torch.as_tensor(store["s2_20m"][i], dtype=torch.float32).unsqueeze(0)
    b60 = torch.as_tensor(store["s2_60m"][i], dtype=torch.float32).unsqueeze(0)

    b10 = crop(b10, 120)
    b20 = crop(b20, 60)
    b60 = crop(b60, 20)

    b20 = F.interpolate(b20, (120,120), mode="bilinear", align_corners=False)
    b60 = F.interpolate(b60, (120,120), mode="bilinear", align_corners=False)

    x = torch.cat((
        b60[:, :1],
        b10[:, :3],
        b20[:, :3],
        b10[:, 3:],
        b20[:, 3:4],
        b60[:, 1:],
        b20[:, 4:],
    ), dim=1)

    # Skip empty/broken frames.
    if (x != 0).float().mean() < 0.9 or x.flatten(2).std(2).mean() < 10:
        return None

    # Same CROMA normalization as our successful smoke test.
    mean = x.mean(dim=(-2,-1), keepdim=True)
    std = x.std(dim=(-2,-1), keepdim=True).clamp_min(1e-6)
    x = ((x - (mean - 2*std)) / (4*std)).clamp(0, 1)

    return x

print("Collecting real DISFOR frames...")

frames = []
for i in range(store["s2_10m"].shape[0]):
    x = load_frame(i)
    if x is not None:
        frames.append(x)
    if len(frames) >= N:
        break

images = torch.cat(frames)
print("Loaded:", tuple(images.shape))

print("Loading pretrained CROMA...")

model = croma_base(
    weights=CROMABase_Weights.CROMA_VIT,
    modalities=["optical"],
    image_size=120,
).to(DEVICE).eval()

print("\n=== CROMA L40S THROUGHPUT ===")

for bs in [1, 4, 8, 16, 32]:
    if bs > len(images):
        continue

    try:
        x = images[:bs].to(DEVICE)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            for _ in range(3):
                model(x_optical=x)

            torch.cuda.synchronize()
            start = time.perf_counter()

            steps = 10
            for _ in range(steps):
                out = model(x_optical=x)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        rate = bs * steps / elapsed
        memory = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"batch={bs:2d} | "
            f"{rate:7.1f} images/s | "
            f"{1000/rate:7.2f} ms/image | "
            f"peak GPU={memory:.2f} GB | "
            f"embedding={tuple(out['optical_GAP'].shape)}"
        )

        del x, out

    except torch.cuda.OutOfMemoryError:
        print(f"batch={bs:2d} | OUT OF MEMORY")
        torch.cuda.empty_cache()

print("\n=== BENCHMARK FINISHED ===")
