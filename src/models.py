from typing import Any, Callable

import torch.nn as nn
import segmentation_models_pytorch as smp


ModelBuilder = Callable[[dict[str, Any]], nn.Module]


def _build_smp_model(config: dict[str, Any], model_class: type[nn.Module]) -> nn.Module:
    """Build a segmentation-models-pytorch model with raw logits output."""
    return model_class(
        encoder_name=config["encoder_name"],
        encoder_weights=config.get("encoder_weights"),
        in_channels=config.get("in_channels", 3),
        classes=config["num_classes"],
        activation=None,
    )


def build_unet(config: dict[str, Any]) -> nn.Module:
    """Build a UNet model."""
    return _build_smp_model(config, smp.Unet)


def build_deeplabv3(config: dict[str, Any]) -> nn.Module:
    """Build a DeepLabV3 model."""
    return _build_smp_model(config, smp.DeepLabV3)


def build_deeplabv3plus(config: dict[str, Any]) -> nn.Module:
    """Build a DeepLabV3+ model."""
    return _build_smp_model(config, smp.DeepLabV3Plus)


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "unet": build_unet,
    "deeplabv3": build_deeplabv3,
    "deeplabv3plus": build_deeplabv3plus,
    "deeplabv3+": build_deeplabv3plus,
}


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a segmentation model from a model config dictionary."""
    architecture = config["architecture"].lower()
    if architecture not in MODEL_REGISTRY:
        options = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model architecture '{architecture}'. "
            f"Available options: {options}"
        )
    return MODEL_REGISTRY[architecture](config)
