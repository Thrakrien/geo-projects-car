import pytest

from src.training_config import DEFAULT_CONFIG, resolve_config


def test_resolve_config_converts_overlap_to_stride():
    """Overlap in pixels should be the easy way to tune reconstruction."""
    config = DEFAULT_CONFIG.copy()
    config.update({
        "patch_size": 512,
        "patch_overlap": 0,
        "inference_overlap": 256,
    })

    resolved = resolve_config(config)

    assert resolved["patch_step"] == 512
    assert resolved["inference_stride"] == 256


def test_resolve_config_rejects_inconsistent_patch_grid():
    """Patch settings must reconstruct the full image without spatial gaps."""
    config = DEFAULT_CONFIG.copy()
    config.update({
        "image_size": 2048,
        "patch_size": 512,
        "patch_step": 300,
    })

    with pytest.raises(ValueError, match="Inconsistent patch_step"):
        resolve_config(config)


def test_resolve_config_rejects_split_overlap(tmp_path):
    """Image-level overlap between train and validation is data leakage."""
    train_txt = tmp_path / "train.txt"
    val_txt = tmp_path / "val.txt"
    train_txt.write_text("a.png\nb.png\n", encoding="utf-8")
    val_txt.write_text("b.png\nc.png\n", encoding="utf-8")

    config = DEFAULT_CONFIG.copy()
    config.update({
        "train_txt": str(train_txt),
        "val_txt": str(val_txt),
    })

    with pytest.raises(ValueError, match="Data leakage risk"):
        resolve_config(config)
