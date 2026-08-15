"""Image preprocessing used by the released task models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .models import model_spec


RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _open_gray(path: str | Path, size_hw: tuple[int, int]) -> Image.Image:
    height, width = size_hw
    with Image.open(path) as image:
        return image.convert("L").resize((width, height), RESAMPLE_BILINEAR)


def percentile_normalize(array: np.ndarray, low: float, high: float) -> np.ndarray:
    lo, hi = np.percentile(array, [low, high])
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def preprocess_image(path: str | Path, model_id: str) -> torch.Tensor:
    """Return one image tensor in the training domain for ``model_id``."""
    spec = model_spec(model_id)
    image = _open_gray(path, tuple(spec["input_size"]))
    array = np.asarray(image, dtype=np.float32)
    if spec["task"] == "vein_pattern_segmentation":
        array = percentile_normalize(array, *spec["percentile_normalization"])
    elif spec["task"] == "finger_shape_segmentation":
        array = array / 255.0
    else:
        if spec["task"] == "roi_detection":
            minimum, maximum = float(array.min()), float(array.max())
            array = (array - minimum) / max(maximum - minimum, 1e-6)
        else:
            array = array / 255.0
        array = (array - 0.5) / 0.5
    return torch.from_numpy(array.copy()).float().unsqueeze(0)

