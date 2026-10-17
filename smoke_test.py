import numpy as np
import torch
from model import MDASGT, combined_loss
from preprocess import adaptive_pfwf

x = np.random.default_rng(0).normal(.5, .1, (128, 128)).astype("float32")
assert adaptive_pfwf(x).shape == x.shape
model = MDASGT(base=16)
inp = torch.rand(2, 1, 128, 128); target = (torch.rand(2, 1, 128, 128) > .8).float()
out = model(inp); assert out.shape == target.shape
loss = combined_loss(out, target); loss.backward()
print(f"smoke test passed; output={tuple(out.shape)} loss={loss.item():.4f}")

