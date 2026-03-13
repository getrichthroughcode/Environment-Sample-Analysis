from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

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

    ds = PlanktonDataset(tmp_path)
    train_ds = _TransformSubset(Subset(ds, [0, 1]), VAL_TRANSFORMS)
    val_ds = _TransformSubset(Subset(ds, [2, 3]), VAL_TRANSFORMS)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)
    return train_loader, val_loader


def test_train_smoke(tiny_loaders, tmp_path):
    train_loader, val_loader = tiny_loaders
    model = build_model(num_classes=2, pretrained=False)

    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)

    with patch("src.vision.train.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = mock_run

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

    mock_mlflow.set_experiment.assert_called_once_with("plankton-vision")
    mock_mlflow.log_params.assert_called_once()
    assert mock_mlflow.log_metrics.call_count == 1
    mock_mlflow.log_metric.assert_called_once()
