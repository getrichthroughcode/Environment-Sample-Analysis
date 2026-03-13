from __future__ import annotations

import timm
import torch.nn as nn


def build_model(
    num_classes: int,
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate a ViT model with a classification head for *num_classes*.

    Args:
        num_classes: Number of output classes.
        model_name: Any ``timm``-compatible model identifier.
        pretrained: Load ImageNet pre-trained weights when ``True``.
    """
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
