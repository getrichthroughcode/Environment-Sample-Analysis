from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

from .dataset import VAL_TRANSFORMS
from .utils import get_device


def predict(
    model: nn.Module,
    image_path: str | Path,
    class_names: list[str],
    device_backend: Optional[str] = None,
) -> tuple[str, float]:
    """Return the predicted class name and confidence for a single image.

    Args:
        model: Trained classification model.
        image_path: Path to the image file.
        class_names: Ordered list of class names matching the model output.
        device_backend: Explicit device string. Defaults to CUDA > CPU.

    Returns:
        ``(class_name, confidence)``
    """
    device = get_device(device_backend)
    model = model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORMS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    return class_names[pred_idx.item()], confidence.item()


def predict_batch(
    model: nn.Module,
    image_paths: list[str | Path],
    class_names: list[str],
    device_backend: Optional[str] = None,
) -> list[tuple[str, float]]:
    """Run :func:`predict` over a list of image paths."""
    return [predict(model, p, class_names, device_backend) for p in image_paths]
