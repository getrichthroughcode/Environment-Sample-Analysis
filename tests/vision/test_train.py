from pathlib import Path

import pytest
import torch
from PIL import Image

from src.vision.dataset import VAL_TRANSFORMS, PlanktonDataset, _TransformSubset
from src.vision.model import build_model
from src.vision.train import train


@pytest.fixture()
def tiny_loaders(tmp_path: Path):
    """Two-class, 4-image dataset split into 2-train / 2-val."""
    for cls in ("class_a", "class_b"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(2):
            img = Image.new("RGB", (64, 64))
            img.save(cls_dir / f"img_{i}.jpg")

    from torch.utils.data import DataLoader, Subset

    ds = PlanktonDataset(tmp_path)
    train_ds = _TransformSubset(Subset(ds, [0, 1]), VAL_TRANSFORMS)
    val_ds = _TransformSubset(Subset(ds, [2, 3]), VAL_TRANSFORMS)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)
    return train_loader, val_loader


def test_train_smoke(tiny_loaders, tmp_path):
    train_loader, val_loader = tiny_loaders
    model = build_model(num_classes=2, pretrained=False)
    trained = train(
        model,
        train_loader,
        val_loader,
        epochs=1,
        checkpoint_dir=tmp_path / "ckpt",
        device_backend="cpu",
    )
    assert (tmp_path / "ckpt" / "best.pt").exists()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = trained(dummy)
    assert out.shape == (1, 2)
