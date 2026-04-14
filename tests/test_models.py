import pytest
import torch

from src.models import build_model


pytest.importorskip("segmentation_models_pytorch")


@pytest.mark.parametrize("architecture", ["unet", "deeplabv3"])
def test_build_model_output_shape(architecture):
    """Model factory must keep the train/inference logits contract stable."""
    config = {
        "architecture": architecture,
        "encoder_name": "resnet18",
        "encoder_weights": None,
        "in_channels": 3,
        "num_classes": 5,
    }
    model = build_model(config)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))

    assert output.shape == (1, 5, 64, 64)


def test_build_model_unknown_architecture_raises():
    """Unknown architectures should fail before training starts."""
    config = {
        "architecture": "unknown",
        "encoder_name": "resnet18",
        "encoder_weights": None,
        "in_channels": 3,
        "num_classes": 5,
    }

    with pytest.raises(ValueError, match="Unknown model architecture"):
        build_model(config)
