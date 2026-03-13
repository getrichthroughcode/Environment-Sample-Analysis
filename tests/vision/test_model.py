import torch

from src.vision.model import build_model


def test_build_model_output_shape():
    model = build_model(num_classes=10, pretrained=False)
    dummy = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 10)


def test_build_model_custom_classes():
    model = build_model(num_classes=103, pretrained=False)
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 103)
