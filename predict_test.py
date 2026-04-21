"""Evaluate a semantic segmentation checkpoint on the test split.

This script reuses the project's existing config, preprocessing, dataset and
sliding-window inference utilities to evaluate predictions at full-image level.
The main evaluation unit is the reconstructed prediction for each full image,
which is then compared against the corresponding ground-truth mask.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from src.dataset import (
    load_image_and_mask,
    read_image_names,
    resize_image_and_mask,
)
from src.inference import PatchifyInference, apply_image_transform
from src.models import build_model
from src.preprocessing import PreProcessingImage
from src.training_config import load_config, model_config, resolve_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for test-time evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Run full-image test evaluation for a semantic segmentation "
            "checkpoint, with optional patch-based inference."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML/JSON config. If omitted, the script first tries to "
            "reuse the config stored inside the checkpoint."
        ),
    )
    parser.add_argument(
        "--test-txt",
        type=str,
        default=None,
        help="Text file with test image names, one per line.",
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default=None,
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--test-masks",
        type=str,
        default=None,
        help="Directory containing test masks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size used for patch inference inside each image.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override the number of valid classes.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=None,
        help="Override the ignore_index used during evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where evaluation artifacts will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit device, for example 'cuda', 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override the square evaluation image size.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Override the patch/window size used in inference.",
    )
    parser.add_argument(
        "--inference-stride",
        type=int,
        default=None,
        help="Override the sliding-window stride used in inference.",
    )
    parser.add_argument(
        "--use-patches",
        action="store_true",
        help="Force patch-based inference even if the config disables it.",
    )
    parser.add_argument(
        "--no-patches",
        action="store_true",
        help="Force full-image inference without sliding windows.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save reconstructed prediction masks for each test image.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    """Load a training checkpoint from disk."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint format is invalid: expected a dictionary.")
    return checkpoint


def build_runtime_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Build the effective runtime config with minimal overrides."""
    if args.config is not None:
        config = load_config(args.config)
    elif "config" in checkpoint:
        config = resolve_config(checkpoint["config"])
    else:
        config = load_config()

    if args.test_txt is not None:
        config["test_txt"] = args.test_txt
    elif "test_txt" not in config:
        default_test_txt = Path("data-segments/test.txt")
        if default_test_txt.exists():
            config["test_txt"] = str(default_test_txt)

    if args.test_images is not None:
        config["images_dir"] = args.test_images
    if args.test_masks is not None:
        config["masks_dir"] = args.test_masks
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_classes is not None:
        config["num_classes"] = args.num_classes
    if args.ignore_index is not None:
        config["ignore_index"] = args.ignore_index
    if args.image_size is not None:
        config["image_size"] = args.image_size
    if args.patch_size is not None:
        config["patch_size"] = args.patch_size
    if args.inference_stride is not None:
        config["inference_stride"] = args.inference_stride
    if args.use_patches:
        config["use_patches"] = True
    if args.no_patches:
        config["use_patches"] = False

    if "test_txt" not in config:
        raise ValueError(
            "Test split file is not defined. Provide --test-txt or store "
            "'test_txt' in the config/checkpoint."
        )
    if "images_dir" not in config or "masks_dir" not in config:
        raise ValueError(
            "Image/mask directories are not fully defined. Provide "
            "--test-images and --test-masks or store them in the config."
        )

    return resolve_config(config)


def get_device(device_arg: str | None) -> torch.device:
    """Resolve the target torch device."""
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate the model and load its trained weights."""
    cfg = model_config(config)
    cfg["encoder_weights"] = None
    model = build_model(cfg)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_test_dataset(config: dict[str, Any]) -> list[str]:
    """Load the list of image names from the configured test split."""
    test_txt = Path(config["test_txt"])
    if not test_txt.exists():
        raise FileNotFoundError(f"Test split file not found: {test_txt}")
    return read_image_names(str(test_txt))


def predict_full_image(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
) -> np.ndarray:
    """Run direct full-image inference and return the predicted class map."""
    image_tensor = apply_image_transform(image, transform).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return prediction.astype(np.int64)


def reconstruct_prediction(
    logits_accumulator: np.ndarray,
    count_accumulator: np.ndarray,
) -> np.ndarray:
    """Average aggregated logits and reconstruct the final class map."""
    mean_logits = logits_accumulator / np.clip(
        count_accumulator[None, :, :],
        a_min=1e-6,
        a_max=None,
    )
    return np.argmax(mean_logits, axis=0).astype(np.int64)


def predict_patches(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    batch_size: int,
    patch_size: int,
    stride: int,
    num_classes: int,
) -> np.ndarray:
    """Run batched sliding-window inference and reconstruct the full mask."""
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    ys = PatchifyInference.compute_window_positions(height, patch_size, stride)
    xs = PatchifyInference.compute_window_positions(width, patch_size, stride)

    logits_acc = np.zeros((num_classes, height, width), dtype=np.float32)
    count_acc = np.zeros((height, width), dtype=np.float32)

    batch_patches: list[torch.Tensor] = []
    batch_positions: list[tuple[int, int]] = []

    def flush_batch() -> None:
        if not batch_patches:
            return

        inputs = torch.stack(batch_patches).to(device)
        with torch.no_grad():
            patch_logits = model(inputs).detach().cpu().float().numpy()

        for logits, (y_coord, x_coord) in zip(patch_logits, batch_positions):
            logits_acc[
                :,
                y_coord:y_coord + patch_size,
                x_coord:x_coord + patch_size,
            ] += logits
            count_acc[
                y_coord:y_coord + patch_size,
                x_coord:x_coord + patch_size,
            ] += 1.0

        batch_patches.clear()
        batch_positions.clear()

    for y_coord in ys:
        for x_coord in xs:
            patch = image_np[
                y_coord:y_coord + patch_size,
                x_coord:x_coord + patch_size,
            ]
            patch_pil = Image.fromarray(patch.astype(np.uint8))
            patch_tensor = apply_image_transform(patch_pil, transform)
            batch_patches.append(patch_tensor)
            batch_positions.append((y_coord, x_coord))

            if len(batch_patches) >= batch_size:
                flush_batch()

    flush_batch()
    return reconstruct_prediction(logits_acc, count_acc)


def update_confusion_matrix(
    confusion_matrix: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    num_classes: int,
    ignore_index: int | None,
) -> np.ndarray:
    """Accumulate a confusion matrix using valid pixels only."""
    if target.shape != prediction.shape:
        raise ValueError(
            "Target and prediction shapes must match. "
            f"Got {target.shape} and {prediction.shape}."
        )

    target_flat = target.reshape(-1)
    prediction_flat = prediction.reshape(-1)

    valid_mask = (
        (target_flat >= 0)
        & (target_flat < num_classes)
        & (prediction_flat >= 0)
        & (prediction_flat < num_classes)
    )
    if ignore_index is not None:
        valid_mask &= target_flat != ignore_index

    if not np.any(valid_mask):
        return confusion_matrix

    encoded = (
        target_flat[valid_mask] * num_classes
        + prediction_flat[valid_mask]
    )
    bincount = np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    )
    confusion_matrix += bincount.reshape(num_classes, num_classes)
    return confusion_matrix


def compute_metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
) -> dict[str, Any]:
    """Compute per-class metrics and mIoU from the global confusion matrix."""
    total = confusion_matrix.sum()
    per_class_metrics: list[dict[str, Any]] = []
    iou_values: list[float] = []

    for class_index in range(confusion_matrix.shape[0]):
        tp = int(confusion_matrix[class_index, class_index])
        fp = int(confusion_matrix[:, class_index].sum() - tp)
        fn = int(confusion_matrix[class_index, :].sum() - tp)
        tn = int(total - tp - fp - fn)

        precision_den = tp + fp
        recall_den = tp + fn
        f1_den = (2 * tp) + fp + fn
        iou_den = tp + fp + fn

        precision = tp / precision_den if precision_den > 0 else float("nan")
        recall = tp / recall_den if recall_den > 0 else float("nan")
        f1_score = tp * 2 / f1_den if f1_den > 0 else float("nan")
        iou = tp / iou_den if iou_den > 0 else float("nan")

        if not np.isnan(iou):
            iou_values.append(iou)

        per_class_metrics.append(
            {
                "class_id": class_index,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "support": int(confusion_matrix[class_index, :].sum()),
                "predicted_pixels": int(confusion_matrix[:, class_index].sum()),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "iou": iou,
                "valid_for_miou": not np.isnan(iou),
            }
        )

    mean_iou = float(np.mean(iou_values)) if iou_values else float("nan")
    pixel_accuracy = (
        float(np.trace(confusion_matrix) / total)
        if total > 0
        else float("nan")
    )

    return {
        "confusion_matrix": confusion_matrix,
        "per_class": per_class_metrics,
        "summary": {
            "mean_iou": mean_iou,
            "pixel_accuracy": pixel_accuracy,
            "num_valid_pixels": int(total),
            "num_classes": int(confusion_matrix.shape[0]),
            "miou_averaging": (
                "Mean over classes with IoU denominator > 0 "
                "(classes with no support and no predictions are excluded)."
            ),
        },
    }


def save_confusion_matrix_csv(
    confusion_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Save the confusion matrix as a CSV file."""
    class_headers = [f"pred_{idx}" for idx in range(confusion_matrix.shape[1])]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_class", *class_headers])
        for class_index, row in enumerate(confusion_matrix):
            writer.writerow([f"true_{class_index}", *row.tolist()])


def save_class_metrics_csv(
    class_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save per-class metrics as a CSV file."""
    fieldnames = [
        "class_id",
        "tp",
        "fp",
        "fn",
        "tn",
        "support",
        "predicted_pixels",
        "precision",
        "recall",
        "f1_score",
        "iou",
        "valid_for_miou",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(class_metrics)


def save_summary_json(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the evaluation summary as JSON."""
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=4)


def save_prediction_mask(
    prediction: np.ndarray,
    image_name: str,
    output_dir: Path,
) -> None:
    """Save the predicted class map as an indexed image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_uint8 = prediction.astype(np.uint8)
    Image.fromarray(prediction_uint8, mode="L").save(output_dir / image_name)


def print_metrics(summary: dict[str, Any], class_metrics: list[dict[str, Any]]) -> None:
    """Print the evaluation summary in the terminal."""
    print("\n=== Test Evaluation Summary ===")
    print(f"mIoU: {summary['mean_iou']:.6f}")
    print(f"Pixel Accuracy: {summary['pixel_accuracy']:.6f}")
    print(f"Valid pixels: {summary['num_valid_pixels']}")

    print("\nPer-class metrics:")
    for class_result in class_metrics:
        precision = class_result["precision"]
        recall = class_result["recall"]
        f1_score = class_result["f1_score"]
        iou = class_result["iou"]

        precision_str = f"{precision:.6f}" if not np.isnan(precision) else "nan"
        recall_str = f"{recall:.6f}" if not np.isnan(recall) else "nan"
        f1_str = f"{f1_score:.6f}" if not np.isnan(f1_score) else "nan"
        iou_str = f"{iou:.6f}" if not np.isnan(iou) else "nan"

        print(
            f"Class {class_result['class_id']}: "
            f"IoU={iou_str}, "
            f"Precision={precision_str}, "
            f"Recall={recall_str}, "
            f"F1={f1_str}"
        )


def evaluate_test_set(
    model: torch.nn.Module,
    image_names: list[str],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    save_predictions: bool,
) -> dict[str, Any]:
    """Evaluate the full test split and save optional prediction artifacts."""
    num_classes = int(config["num_classes"])
    ignore_index = config.get("ignore_index")
    batch_size = int(config.get("batch_size", 1))
    patch_size = int(config["patch_size"])
    inference_stride = int(config["inference_stride"])
    image_size = int(config["image_size"])
    use_patches = bool(config.get("use_patches", True))
    transform = PreProcessingImage("validation").transformations()

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    predictions_dir = output_dir / "predictions"

    for image_name in tqdm(image_names, desc="Evaluating test images"):
        image, mask = load_image_and_mask(
            image_name=image_name,
            images_dir=config["images_dir"],
            masks_dir=config["masks_dir"],
        )
        image, mask = resize_image_and_mask(image, mask, image_size=image_size)
        target_mask = np.array(mask, dtype=np.int64)

        if use_patches:
            prediction = predict_patches(
                model=model,
                image=image,
                transform=transform,
                device=device,
                batch_size=batch_size,
                patch_size=patch_size,
                stride=inference_stride,
                num_classes=num_classes,
            )
        else:
            prediction = predict_full_image(
                model=model,
                image=image,
                transform=transform,
                device=device,
            )

        confusion_matrix = update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            target=target_mask,
            prediction=prediction,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        if save_predictions:
            save_prediction_mask(
                prediction=prediction,
                image_name=image_name,
                output_dir=predictions_dir,
            )

    return compute_metrics_from_confusion_matrix(confusion_matrix)


def save_metrics(
    metrics: dict[str, Any],
    config: dict[str, Any],
    checkpoint_path: str,
    output_dir: Path,
) -> None:
    """Persist evaluation artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    confusion_matrix = metrics["confusion_matrix"]
    per_class = metrics["per_class"]
    summary = dict(metrics["summary"])
    summary["checkpoint"] = checkpoint_path
    summary["config"] = config

    save_confusion_matrix_csv(
        confusion_matrix=confusion_matrix,
        output_path=output_dir / "confusion_matrix.csv",
    )
    save_class_metrics_csv(
        class_metrics=per_class,
        output_path=output_dir / "class_metrics.csv",
    )
    save_summary_json(
        summary=summary,
        output_path=output_dir / "summary.json",
    )


def main() -> None:
    """Entry point for checkpoint-based test evaluation."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = get_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    config = build_runtime_config(args, checkpoint)

    model = load_model(checkpoint, config, device)
    image_names = load_test_dataset(config)

    metrics = evaluate_test_set(
        model=model,
        image_names=image_names,
        config=config,
        device=device,
        output_dir=output_dir,
        save_predictions=args.save_predictions,
    )

    save_metrics(
        metrics=metrics,
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
    )
    print_metrics(metrics["summary"], metrics["per_class"])


if __name__ == "__main__":
    main()
