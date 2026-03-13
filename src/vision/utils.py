from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def get_device(backend: Optional[str] = None) -> torch.device:
    """Return a torch device.

    Args:
        backend: Explicit backend string (``"cuda"``, ``"mps"``, ``"cpu"``…).
                 When *None*, CUDA is used if available, otherwise CPU.
    """
    if backend is not None:
        return torch.device(backend)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load a checkpoint into *model* (and optionally *optimizer*).

    Returns the saved epoch number.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
