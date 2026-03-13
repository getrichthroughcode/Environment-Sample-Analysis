from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow

from .utils import get_device, save_checkpoint


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-4,
    checkpoint_dir: str | Path = "checkpoints",
    device_backend: Optional[str] = None,
    experiment_name: str = "plankton-vision",
) -> nn.Module:
    """Train *model*, log to MLflow, and save the best checkpoint.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        epochs: Number of training epochs.
        lr: Learning rate for AdamW.
        checkpoint_dir: Directory where ``best.pt`` is saved.
        device_backend: Explicit device string (``"cuda"``, ``"mps"``, ``"cpu"``).
                        Defaults to CUDA when available, otherwise CPU.
        experiment_name: MLflow experiment name.

    Returns:
        The trained model (still on *device*).
    """
    device = get_device(device_backend)
    model = model.to(device)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "lr": lr,
                "device": str(device),
                "model_class": model.__class__.__name__,
            }
        )

        best_val_acc = -1.0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = _train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = checkpoint_dir / "best.pt"
                save_checkpoint(model, optimizer, epoch, ckpt_path)
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

        mlflow.log_metric("best_val_acc", best_val_acc)

    return model
