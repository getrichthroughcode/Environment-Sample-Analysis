from pathlib import Path

import pytest
import torch
from PIL import Image

from src.vision.dataset import (
    VAL_TRANSFORMS,
    PlanktonDataset,
    _TransformSubset,
    build_dataloaders,
)


@pytest.fixture()
def dummy_dataset_root(tmp_path: Path) -> Path:
    """Create a tiny dataset: 2 classes, 3 images each."""
    for cls in ("class_a", "class_b"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(i * 40, i * 40, i * 40))
            img.save(cls_dir / f"img_{i}.jpg")
    return tmp_path


def test_dataset_classes(dummy_dataset_root):
    ds = PlanktonDataset(dummy_dataset_root)
    assert ds.classes == ["class_a", "class_b"]
    assert ds.class_to_idx == {"class_a": 0, "class_b": 1}


def test_dataset_length(dummy_dataset_root):
    ds = PlanktonDataset(dummy_dataset_root)
    assert len(ds) == 6


def test_dataset_returns_pil_without_transform(dummy_dataset_root):
    ds = PlanktonDataset(dummy_dataset_root)
    image, label = ds[0]
    assert isinstance(image, Image.Image)
    assert label in (0, 1)


def test_dataset_applies_transform(dummy_dataset_root):
    ds = PlanktonDataset(dummy_dataset_root, transform=VAL_TRANSFORMS)
    tensor, label = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_transform_subset(dummy_dataset_root):
    from torch.utils.data import Subset

    ds = PlanktonDataset(dummy_dataset_root)
    subset = Subset(ds, indices=list(range(3)))
    wrapped = _TransformSubset(subset, VAL_TRANSFORMS)
    assert len(wrapped) == 3
    tensor, label = wrapped[0]
    assert isinstance(tensor, torch.Tensor)


def test_build_dataloaders(dummy_dataset_root):
    train_loader, val_loader, classes = build_dataloaders(
        dummy_dataset_root, val_split=0.5, batch_size=2, num_workers=0
    )
    assert classes == ["class_a", "class_b"]
    batch_images, batch_labels = next(iter(train_loader))
    assert batch_images.shape[1:] == (3, 224, 224)
    assert batch_labels.dtype == torch.int64
