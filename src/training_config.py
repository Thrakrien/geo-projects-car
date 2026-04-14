import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    # Dados
    "train_txt": "data-segments/train_sample.txt",
    "val_txt": "data-segments/validation_sample.txt",
    "images_dir": "/data/integracar/replicate_article/satelite_images/",
    "masks_dir": "/data/integracar/replicate_article/masks_replicated_full/",

    # Patchify
    "use_patches": True,
    "image_size": 2048,
    "patch_size": 256,
    "patch_step": 32,
    "inference_stride": 32,

    # Modelo
    "architecture": "unet",
    "encoder_name": "efficientnet-b5",
    "encoder_weights": "imagenet",
    "in_channels": 3,
    "num_classes": 5,

    # Treinamento
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,

    # Otimizador
    "optimizer": "SGD",
    "scheduler": "MultiStepLR",
    "milestones": [25, 35, 45],
    "gamma": 0.1,

    # Loss
    "loss_function": "CrossEntropyLoss",
    "ignore_index": 5,
    "use_class_weights": True,

    # Logging
    "experiment_name": "replying-unet-icmbio-32-step",
    "use_wandb": False,

    # Sistema
    "num_workers": 4,
    "seed": 42,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file when PyYAML is available."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML configs. "
            "Install PyYAML or use a JSON config file."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the default training config and optionally override it from a file."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return resolve_config(config)

    path = Path(config_path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        overrides = _load_yaml(path)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            overrides = json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")

    config.update(overrides)
    return resolve_config(config)


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve derived patch stride values and validate experiment settings."""
    config = copy.deepcopy(config)
    patch_size = int(config["patch_size"])

    if "patch_overlap" in config:
        config["patch_step"] = _stride_from_overlap(
            patch_size,
            int(config["patch_overlap"]),
            "patch_overlap"
        )
    if "inference_overlap" in config:
        config["inference_stride"] = _stride_from_overlap(
            patch_size,
            int(config["inference_overlap"]),
            "inference_overlap"
        )

    _validate_patch_grid(
        image_size=int(config["image_size"]),
        patch_size=patch_size,
        step=int(config["patch_step"]),
        field_name="patch_step",
    )
    _validate_patch_grid(
        image_size=int(config["image_size"]),
        patch_size=patch_size,
        step=int(config["inference_stride"]),
        field_name="inference_stride",
    )
    _validate_splits(
        train_txt=Path(config["train_txt"]),
        val_txt=Path(config["val_txt"]),
    )
    return config


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the model-only config consumed by the model factory."""
    return {
        "architecture": config["architecture"],
        "encoder_name": config["encoder_name"],
        "encoder_weights": config.get("encoder_weights"),
        "in_channels": config.get("in_channels", 3),
        "num_classes": config["num_classes"],
    }


def _stride_from_overlap(
        patch_size: int,
        overlap: int,
        field_name: str) -> int:
    """Convert an overlap size in pixels to a stride."""
    if overlap < 0:
        raise ValueError(f"{field_name} must be >= 0")
    if overlap >= patch_size:
        raise ValueError(f"{field_name} must be smaller than patch_size")
    return patch_size - overlap


def _validate_patch_grid(
        image_size: int,
        patch_size: int,
        step: int,
        field_name: str) -> None:
    """Validate a spatial grid used for patch extraction or reconstruction."""
    if step <= 0:
        raise ValueError(f"{field_name} must be > 0")
    if patch_size > image_size:
        raise ValueError("patch_size cannot be greater than image_size")
    if step > patch_size:
        raise ValueError(f"{field_name} cannot be greater than patch_size")
    if (image_size - patch_size) % step != 0:
        raise ValueError(
            f"Inconsistent {field_name}: (image_size - patch_size) % "
            f"{field_name} != 0 ({image_size} - {patch_size}) % {step} != 0"
        )


def _validate_splits(train_txt: Path, val_txt: Path) -> None:
    """Prevent image-level overlap between train and validation splits."""
    if not train_txt.exists() or not val_txt.exists():
        return

    train_names = _read_split_names(train_txt)
    val_names = _read_split_names(val_txt)
    overlap = train_names & val_names
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            "Data leakage risk: train and validation splits share image names: "
            f"{examples}"
        )


def _read_split_names(path: Path) -> set[str]:
    """Read non-empty image names from a split file."""
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}
