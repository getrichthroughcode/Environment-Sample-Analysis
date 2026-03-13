import torch

from src.vision.utils import get_device


def test_get_device_default():
    device = get_device()
    assert isinstance(device, torch.device)
    # Default must be cuda if available, else cpu — never mps unless explicit.
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected


def test_get_device_explicit_cpu():
    device = get_device("cpu")
    assert device.type == "cpu"


def test_get_device_explicit_mps():
    device = get_device("mps")
    assert device.type == "mps"
