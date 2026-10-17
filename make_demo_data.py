"""Create a tiny synthetic MRI-like dataset to verify the full training pipeline."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageFilter


def main(output: str, count: int, size: int, seed: int):
    rng = np.random.default_rng(seed); root = Path(output)
    (root / "images").mkdir(parents=True, exist_ok=True); (root / "masks").mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:size, :size]
    for i in range(count):
        liver = ((xx-size*.5)/(size*.34))**2 + ((yy-size*.53)/(size*.28))**2 < 1
        cx, cy = rng.uniform(.38, .65, 2) * size; radius = rng.uniform(.035, .09) * size
        tumor = ((xx-cx)**2 + (yy-cy)**2 < radius**2) & liver
        image = .10 + .42*liver + .24*tumor + rng.normal(0, .055, (size, size))
        image = np.clip(image, 0, 1)
        pil = Image.fromarray((image*255).astype("uint8")).filter(ImageFilter.GaussianBlur(.7))
        pil.save(root / "images" / f"demo_{i:03d}.png")
        Image.fromarray(tumor.astype("uint8")*255).save(root / "masks" / f"demo_{i:03d}.png")
    print(f"created {count} paired slices in {root}")


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output", default="demo_dataset"); p.add_argument("--count", type=int, default=40)
    p.add_argument("--size", type=int, default=128); p.add_argument("--seed", type=int, default=42); a=p.parse_args()
    main(a.output, a.count, a.size, a.seed)

