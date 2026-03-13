from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

VAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PlanktonDataset(Dataset):
    """Dataset for the WHOI Plankton image collection.

    Expects the following directory layout::

        root/
            <class_a>/image1.jpg
            <class_a>/image2.jpg
            <class_b>/image1.jpg
            ...

    When *transform* is ``None`` the dataset returns raw PIL images so that
    a :class:`_TransformSubset` wrapper can apply split-specific augmentations
    after the train/val split.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        self.classes: list[str] = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }
        self.samples: list[tuple[Path, int]] = self._load_samples()

    def _load_samples(self) -> list[tuple[Path, int]]:
        samples = []
        for cls in self.classes:
            for img_path in (self.root / cls).glob("*.jpg"):
                samples.append((img_path, self.class_to_idx[cls]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class _TransformSubset(Dataset):
    """Wraps a :class:`~torch.utils.data.Subset` and applies a transform."""

    def __init__(self, subset: Subset, transform: Callable) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.subset[idx]
        return self.transform(image), label


def build_dataloaders(
    root: str | Path,
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train and validation :class:`~torch.utils.data.DataLoader`s.

    Returns:
        ``(train_loader, val_loader, class_names)``
    """
    base_dataset = PlanktonDataset(root, transform=None)

    n_val = int(len(base_dataset) * val_split)
    n_train = len(base_dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        base_dataset, [n_train, n_val], generator=generator
    )

    train_ds = _TransformSubset(train_subset, TRAIN_TRANSFORMS)
    val_ds = _TransformSubset(val_subset, VAL_TRANSFORMS)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, base_dataset.classes
