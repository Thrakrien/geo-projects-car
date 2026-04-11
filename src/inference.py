import os
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


ImageTransform = Optional[Callable[[Image.Image], torch.Tensor]]


def apply_image_transform(image_patch: Image.Image, transform: ImageTransform) -> torch.Tensor:
    """Apply the configured image transform or fall back to ToTensor."""
    if transform:
        return transform(image_patch)
    return transforms.ToTensor()(image_patch)


def validate_window_grid(image_size: int, patch_size: int, step: int) -> None:
    """Validate a patch grid that is expected to cover the image without gaps."""
    if step <= 0:
        raise ValueError("step must be > 0")
    if patch_size > image_size:
        raise ValueError("patch_size cannot be greater than image_size")
    if step > patch_size:
        raise ValueError("step cannot be greater than patch_size")
    if (image_size - patch_size) % step != 0:
        raise ValueError(
            f"Inconsistent patch grid: (image_size - patch_size) % step != 0 "
            f"({image_size} - {patch_size}) % {step} != 0"
        )


class PatchifyInference:
    """Patch-based inference with optional overlap and logit aggregation."""

    def __init__(
            self,
            model,
            device,
            image_size: int = 1024,
            patch_size: int = 512,
            num_classes: int = 1,
            step: Optional[int] = None) -> None:
        self.model = model
        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.step = patch_size if step is None else step
        validate_window_grid(image_size, patch_size, self.step)

    @staticmethod
    def compute_window_positions(
            length: int,
            window_size: int,
            stride: int) -> list[int]:
        """Return starts that cover an axis completely without gaps."""
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if window_size > length:
            raise ValueError(
                f"window_size ({window_size}) cannot be greater than "
                f"image dimension ({length})"
            )
        if stride > window_size:
            raise ValueError("stride cannot be greater than window_size")

        positions = list(range(0, length - window_size + 1, stride))
        last_start = length - window_size
        if not positions or positions[-1] != last_start:
            positions.append(last_start)
        return positions

    def predict_image(
            self,
            image_path: str,
            transform: ImageTransform = None,
            window_size: Optional[int] = None,
            stride: Optional[int] = None,
            return_prob_map: bool = False,
            binary_classifier: bool = False):
        """
        Predict a full image by averaging logits over sliding windows.

        Args:
            image_path: Path to the input image.
            transform: Transform applied to each image window.
            window_size: Sliding window size. Defaults to ``patch_size``.
            stride: Sliding window stride. Defaults to ``step``.
            return_prob_map: If True, also return the probability map.
            binary_classifier: If True, expects a single-logit binary model.

        Returns:
            Predicted mask, or ``(mask, probability_map)`` when requested.
        """
        window_size = self.patch_size if window_size is None else window_size
        stride = self.step if stride is None else stride

        image = Image.open(image_path).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        image_np = np.array(image)
        height, width = image_np.shape[:2]
        ys = self.compute_window_positions(height, window_size, stride)
        xs = self.compute_window_positions(width, window_size, stride)

        if binary_classifier:
            logits_acc = np.zeros((1, height, width), dtype=np.float32)
        else:
            logits_acc = np.zeros((self.num_classes, height, width), dtype=np.float32)
        count_acc = np.zeros((height, width), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    patch = image_np[y:y + window_size, x:x + window_size]
                    patch_pil = Image.fromarray(patch.astype("uint8"))
                    patch_tensor = apply_image_transform(
                        patch_pil,
                        transform
                    ).unsqueeze(0).to(self.device)

                    patch_logits = (
                        self.model(patch_tensor)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                    )

                    if binary_classifier and patch_logits.ndim == 2:
                        patch_logits = patch_logits[None, :, :]

                    logits_acc[:, y:y + window_size, x:x + window_size] += patch_logits
                    count_acc[y:y + window_size, x:x + window_size] += 1.0

        mean_logits = logits_acc / np.clip(count_acc[None, :, :], 1e-6, None)
        if binary_classifier:
            prob_map = 1.0 / (1.0 + np.exp(-mean_logits[0]))
            prediction = (prob_map > 0.5).astype(np.uint8)
            if return_prob_map:
                return prediction, prob_map
            return prediction

        prediction = np.argmax(mean_logits, axis=0).astype(np.uint8)
        if return_prob_map:
            prob_map = torch.softmax(torch.from_numpy(mean_logits), dim=0).numpy()
            return prediction, prob_map
        return prediction

    def predict_batch(
            self,
            image_paths: list[str],
            transform: ImageTransform = None,
            save_dir: Optional[str] = None,
            window_size: Optional[int] = None,
            stride: Optional[int] = None,
            binary_classifier: bool = False) -> list[np.ndarray]:
        """Predict multiple images, optionally saving each mask to disk."""
        predictions = []

        for img_path in tqdm(image_paths, desc="Predicting images"):
            pred = self.predict_image(
                img_path,
                transform=transform,
                window_size=window_size,
                stride=stride,
                binary_classifier=binary_classifier
            )
            predictions.append(pred)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                img_name = os.path.basename(img_path)
                save_path = os.path.join(save_dir, img_name)
                Image.fromarray(pred.astype("uint8")).save(save_path)

        return predictions
