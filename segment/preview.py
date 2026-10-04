#!/usr/bin/env python3
"""Preview the four independent segmentation models on one image or a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from segment.dataset import STRUCTURES
from segment.predict import (
    TAGS,
    embed_half_mask,
    load_model,
    parse_thresholds,
    predict,
    resolve_structure_checkpoints,
    validate_threshold,
)
from utils import read_image, write_image, write_json


# OpenCV uses BGR colors for each anatomical structure.
COLORS = {
    "LI": (0, 0, 255),
    "LS": (0, 255, 0),
    "RS": (255, 0, 0),
    "RI": (0, 255, 255),
}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Image file or directory")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/segmentation_preview")
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional LI/LS/RI/RS overrides",
    )
    return parser.parse_args()


def make_overlay(image: np.ndarray, full_masks: dict[str, np.ndarray]) -> np.ndarray:
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    counts = np.stack([mask > 0 for mask in full_masks.values()]).sum(axis=0)
    regions = [((mask > 0) & (counts == 1), COLORS[tag]) for tag, mask in full_masks.items()]
    regions.append((counts > 1, (255, 0, 255)))
    for selected, color in regions:
        overlay[selected] = (
            0.55 * overlay[selected].astype(np.float32)
            + 0.45 * np.asarray(color, dtype=np.float32)
        ).astype(np.uint8)
    return overlay


def output_path(base: Path, relative_image: Path, suffix: str) -> Path:
    return base / relative_image.parent / f"{relative_image.stem}{suffix}.png"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    input_root = input_path.parent if input_path.is_file() else input_path
    images = (
        [input_path]
        if input_path.is_file() and input_path.suffix.lower() in SUPPORTED_EXTENSIONS
        else sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )
    if not images:
        raise FileNotFoundError(f"No supported images found under {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = parse_thresholds(args.thresholds) if args.thresholds else None
    checkpoint_paths, checkpoint_files = resolve_structure_checkpoints(args.checkpoint_dir)
    models: dict[str, torch.nn.Module] = {}
    thresholds: dict[str, float] = {}
    model_metadata: dict[str, dict[str, object]] = {}
    for structure in TAGS:
        path = checkpoint_paths[structure]
        model, saved_threshold, declared_structure, declared_side = load_model(path, device)
        expected_side = STRUCTURES[structure][0]
        if declared_structure not in (None, structure):
            raise ValueError(
                f"Checkpoint {path} declares {declared_structure!r}; expected {structure!r}"
            )
        if declared_side not in (None, expected_side):
            raise ValueError(
                f"Checkpoint {path} declares side={declared_side!r}; expected {expected_side!r}"
            )
        threshold = overrides[structure] if overrides else saved_threshold
        if threshold is None:
            raise ValueError(
                f"Checkpoint {path} has no validation threshold; pass --thresholds explicitly"
            )
        models[structure] = model
        thresholds[structure] = validate_threshold(threshold, structure)
        model_metadata[structure] = {
            **checkpoint_files[structure],
            "side": expected_side,
            "threshold": thresholds[structure],
        }

    output_root = Path(args.output_dir)
    completed = 0
    failures: list[dict[str, str]] = []
    image_results: list[dict[str, object]] = []
    for image_path in images:
        relative = image_path.relative_to(input_root)
        try:
            image = cv2.resize(
                read_image(image_path, cv2.IMREAD_GRAYSCALE),
                (512, 512),
                interpolation=cv2.INTER_AREA,
            )
            halves = {"left": image[:, :256], "right": image[:, 256:]}
            half_masks = {
                structure: (
                    predict(models[structure], halves[STRUCTURES[structure][0]], device)
                    > thresholds[structure]
                ).astype(np.uint8)
                for structure in TAGS
            }
            full_masks = {
                structure: embed_half_mask(mask, structure, image.shape)
                for structure, mask in half_masks.items()
            }
            for structure, mask in full_masks.items():
                write_image(
                    output_path(output_root / "masks", relative, f"_{structure}"),
                    mask * 255,
                )
                write_image(
                    output_path(
                        output_root / "overlay" / structure,
                        relative,
                        f"_{structure}_overlay",
                    ),
                    make_overlay(image, {structure: mask}),
                )
            write_image(
                output_path(output_root / "overlay", relative, "_all_overlay"),
                make_overlay(image, full_masks),
            )
            stack = np.stack(list(full_masks.values()))
            image_results.append(
                {
                    "image": relative.as_posix(),
                    "foreground_pixels": {
                        structure: int(mask.sum()) for structure, mask in full_masks.items()
                    },
                    "overlap_pixels": int((stack.sum(axis=0) > 1).sum()),
                }
            )
            completed += 1
            counts = ", ".join(
                f"{structure}={int(full_masks[structure].sum())}" for structure in TAGS
            )
            print(f"[{completed}/{len(images)}] {relative.as_posix()} {counts}")
        except Exception as error:  # continue so the report identifies every failed input
            failures.append({"image": relative.as_posix(), "error": str(error)})
            print(f"[FAILED] {relative.as_posix()}: {error}")

    write_json(
        output_root / "run_report.json",
        {
            "models": model_metadata,
            "threshold_source": "command_line_validation_override" if overrides else "checkpoint_validation",
            "total_images": len(images),
            "completed": completed,
            "failed": failures,
            "image_results": image_results,
            "overlay_colors": {
                "LI": "red",
                "LS": "green",
                "RS": "blue",
                "RI": "yellow",
                "overlap": "magenta",
            },
        },
    )
    if failures:
        raise RuntimeError(
            f"Preview failed for {len(failures)} of {len(images)} inputs; "
            f"see {output_root / 'run_report.json'}"
        )
    print(f"Preview outputs are in {output_root}")


if __name__ == "__main__":
    main()
