"""Adaptive probabilistic fused Wiener-like preprocessing for MRI slices."""
import numpy as np
from scipy.ndimage import uniform_filter


def normalize_mri(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo, hi = np.percentile(x[finite], [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def adaptive_pfwf(image: np.ndarray, threshold: float = .08) -> np.ndarray:
    """Implement the paper's adaptive 3x3/5x5 local Wiener + mean fusion.

    The publication does not provide executable probability sampling details;
    this deterministic uniform-weight fusion is the reproducible interpretation.
    """
    x = normalize_mri(image)
    noise_sigma = float(np.std(x - uniform_filter(x, 3)))
    size = 3 if noise_sigma < threshold else 5
    local_mean = uniform_filter(x, size)
    local_var = uniform_filter(x * x, size) - local_mean * local_mean
    noise_var = float(np.mean(local_var))
    wiener = local_mean + np.maximum(local_var - noise_var, 0) / np.maximum(local_var, 1e-8) * (x - local_mean)
    residual = np.abs(x - local_mean)
    confidence = np.clip(1 - residual / max(float(residual.mean()), 1e-6), 0, 1)
    return normalize_mri(confidence * wiener + (1 - confidence) * local_mean)

