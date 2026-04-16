from dataclasses import dataclass
import os
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ImageTransform = Optional[Callable[[Image.Image], torch.Tensor]]
ImageMaskPair = Tuple[Image.Image, Image.Image]


@dataclass(frozen=True)
class PatchGrid:
    """Patch grid parameters shared by image and mask extraction."""

    image_size: int
    patch_size: int
    step: int

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be > 0")
        if self.patch_size > self.image_size:
            raise ValueError("patch_size cannot be greater than image_size")
        if self.step > self.patch_size:
            raise ValueError("step cannot be greater than patch_size")
        if (self.image_size - self.patch_size) % self.step != 0:
            raise ValueError(
                f"Inconsistent patch grid: (image_size - patch_size) % step != 0 "
                f"({self.image_size} - {self.patch_size}) % {self.step} != 0"
            )

    @property
    def patches_per_row(self) -> int:
        """Return the number of patches in each spatial axis."""
        return ((self.image_size - self.patch_size) // self.step) + 1

    @property
    def patches_per_image(self) -> int:
        """Return the total number of patches generated from one image."""
        return self.patches_per_row ** 2


def read_image_names(txt_file: str) -> list[str]:
    """Read image names from a text split file."""
    with open(txt_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_image_and_mask(
        image_name: str,
        images_dir: str,
        masks_dir: str) -> ImageMaskPair:
    """Load one RGB image and its indexed mask by file name."""
    img_path = os.path.join(images_dir, image_name)
    mask_path = os.path.join(masks_dir, image_name)

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    return image, mask


def resize_image_and_mask(
        image: Image.Image,
        mask: Image.Image,
        image_size: int) -> ImageMaskPair:
    """Resize image and mask with interpolation modes that preserve labels."""
    expected_size = (image_size, image_size)
    if image.size != expected_size:
        image = image.resize(expected_size, Image.BILINEAR)
    if mask.size != expected_size:
        mask = mask.resize(expected_size, Image.NEAREST)
    return image, mask


def extract_patch_pair(
        image: Image.Image,
        mask: Image.Image,
        grid: PatchGrid,
        patch_idx: int) -> ImageMaskPair:
    """Extract aligned image and mask patches from the same grid index."""
    patch_row = patch_idx // grid.patches_per_row
    patch_col = patch_idx % grid.patches_per_row
    y_min = patch_row * grid.step
    x_min = patch_col * grid.step
    crop_box = (
        x_min,
        y_min,
        x_min + grid.patch_size,
        y_min + grid.patch_size,
    )

    return image.crop(crop_box), mask.crop(crop_box)


def apply_image_transform(
        image_patch: Image.Image,
        transform: ImageTransform) -> torch.Tensor:
    """Apply the configured image transform or fall back to ToTensor."""
    if transform:
        return transform(image_patch)
    return transforms.ToTensor()(image_patch)


class PatchifySegmentationDataset(Dataset):
    """Dataset for semantic segmentation with optional patch extraction."""

    def __init__(
            self,
            txt_file: str,
            images_dir: str,
            masks_dir: str,
            image_size: int = 1024,
            patch_size: int = 512,
            transform: ImageTransform = None,
            use_patches: bool = True,
            step: Optional[int] = None) -> None:
        """
        Args:
            txt_file: Path to the text file with image names.
            images_dir: Directory containing input images.
            masks_dir: Directory containing masks with matching file names.
            image_size: Expected square size for full images.
            patch_size: Square patch size.
            transform: Transform applied only to image patches.
            use_patches: If True, returns patches. If False, returns full images.
            step: Patch stride. Defaults to ``patch_size``.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transform
        self.use_patches = use_patches
        self.step = patch_size if step is None else step

        self.grid = PatchGrid(
            image_size=image_size,
            patch_size=patch_size,
            step=self.step
        )
        self.image_names = read_image_names(txt_file)
        self.patches_per_row = self.grid.patches_per_row
        self.patches_per_image = self.grid.patches_per_image

        if use_patches:
            self.total_patches = len(self.image_names) * self.patches_per_image
        else:
            self.total_patches = len(self.image_names)

        print(f"Dataset: {len(self.image_names)} imagens")
        if use_patches:
            print(
                f"Patches por imagem: {self.patches_per_image} "
                f"({self.patches_per_row}x{self.patches_per_row})"
            )
            print(f"Total de patches: {self.total_patches}")

    def __len__(self) -> int:
        return self.total_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_patches:
            img_idx = idx // self.patches_per_image
            patch_idx = idx % self.patches_per_image
            img_name = self.image_names[img_idx]
            image, mask = load_image_and_mask(
                img_name,
                self.images_dir,
                self.masks_dir
            )
            image, mask = resize_image_and_mask(image, mask, self.image_size)
            image_patch, mask_patch = extract_patch_pair(
                image,
                mask,
                self.grid,
                patch_idx
            )
        else:
            img_name = self.image_names[idx]
            image_patch, mask_patch = load_image_and_mask(
                img_name,
                self.images_dir,
                self.masks_dir
            )

        image_patch = apply_image_transform(image_patch, self.transform)
        mask_patch = torch.from_numpy(np.array(mask_patch)).long()
        return image_patch, mask_patch


def create_segmentation_dataloaders(
        config: dict,
        train_transform: ImageTransform,
        val_transform: ImageTransform
) -> Tuple[
        PatchifySegmentationDataset,
        PatchifySegmentationDataset,
        DataLoader,
        DataLoader]:
    """Create train/validation datasets and dataloaders from config."""
    train_dataset = PatchifySegmentationDataset(
        txt_file=config["train_txt"],
        images_dir=config["images_dir"],
        masks_dir=config["masks_dir"],
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        transform=train_transform,
        use_patches=config["use_patches"],
        step=config.get("patch_step", config["patch_size"])
    )

    val_dataset = PatchifySegmentationDataset(
        txt_file=config["val_txt"],
        images_dir=config["images_dir"],
        masks_dir=config["masks_dir"],
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        transform=val_transform,
        use_patches=config["use_patches"],
        step=config.get("patch_step", config["patch_size"])
    )

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    if config["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = config.get(
            "persistent_workers",
            True
        )
        loader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )

    return train_dataset, val_dataset, train_loader, val_loader


from src.inference import PatchifyInference  # noqa: E402
