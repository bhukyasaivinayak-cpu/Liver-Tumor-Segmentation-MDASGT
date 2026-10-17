from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from preprocess import adaptive_pfwf


class PairedSliceDataset(Dataset):
    """Pairs images/<stem>.(png|jpg|npy) with masks/<stem>.png."""
    def __init__(self, root, size=256, augment=False):
        self.root, self.size, self.augment = Path(root), size, augment
        allowed = {".png", ".jpg", ".jpeg", ".npy"}
        self.images = sorted(p for p in (self.root / "images").iterdir() if p.suffix.lower() in allowed)
        if not self.images: raise ValueError("No slices found in dataset/images")

    def __len__(self): return len(self.images)

    def __getitem__(self, index):
        p = self.images[index]
        x = np.load(p) if p.suffix == ".npy" else np.asarray(Image.open(p).convert("L"))
        mask_path = self.root / "masks" / f"{p.stem}.png"
        if not mask_path.exists(): raise FileNotFoundError(mask_path)
        y = np.asarray(Image.open(mask_path).convert("L")) > 0
        x = np.asarray(Image.fromarray((adaptive_pfwf(x) * 255).astype("uint8")).resize((self.size, self.size)), np.float32) / 255
        y = np.asarray(Image.fromarray(y.astype("uint8") * 255).resize((self.size, self.size), Image.Resampling.NEAREST)) > 0
        if self.augment and np.random.rand() < .5: x, y = np.fliplr(x).copy(), np.fliplr(y).copy()
        return torch.from_numpy(x[None]).float(), torch.from_numpy(y[None]).float()

