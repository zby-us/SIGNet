#!/usr/bin/env python3
"""Calibrate LI/LS/RI/RS probability thresholds on the frozen validation set."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from segment.dataset import StructureDataset
from segment.predict import TAGS, load_model, resolve_structure_checkpoints
from segment.train import validate
from utils import checkpoint_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help="Prepared dataset root containing the frozen val/ split",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-json", default="outputs/segmentation_thresholds.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    validation_root = Path(args.data_root) / "val"
    if not validation_root.is_dir():
        raise FileNotFoundError(f"Frozen validation split not found: {validation_root}")

    checkpoint_root = Path(args.checkpoint_dir)
    checkpoint_paths, _ = resolve_structure_checkpoints(checkpoint_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, object] = {}
    for structure in TAGS:
        checkpoint = checkpoint_paths[structure]
        model, _, declared_structure, declared_side = load_model(checkpoint, device)
        if declared_structure not in (None, structure):
            raise ValueError(
                f"Checkpoint {checkpoint} declares structure={declared_structure!r}; "
                f"expected {structure!r}"
            )
        expected_side = "left" if structure in {"LI", "LS"} else "right"
        if declared_side not in (None, expected_side):
            raise ValueError(
                f"Checkpoint {checkpoint} declares side={declared_side!r}; "
                f"expected {expected_side!r}"
            )
        dataset = StructureDataset(validation_root, structure, train=False)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        threshold, dice, iou, accuracy = validate(model, loader, device)
        results[structure] = {
            "threshold": round(float(threshold), 2),
            "dice": float(dice),
            "iou": float(iou),
            "accuracy": float(accuracy),
            "validation_masks": len(dataset),
            "checkpoint": checkpoint_metadata(checkpoint),
        }
        print(
            f"{structure}: threshold={threshold:.2f}, "
            f"Dice={dice:.6f}, IoU={iou:.6f}, n={len(dataset)}"
        )

    thresholds = ",".join(
        f"{structure}={results[structure]['threshold']:.2f}" for structure in TAGS
    )
    write_json(
        args.output_json,
        {
            "selection_split": "Dataset A frozen validation set",
            "search_grid": {"start": 0.01, "stop": 0.99, "step": 0.01},
            "selection_metric": "mean per-mask Dice",
            "results": results,
            "inference_argument": thresholds,
        },
    )
    print(f"Use for inference: --thresholds \"{thresholds}\"")
    print(f"Saved calibration record to {Path(args.output_json)}")


if __name__ == "__main__":
    main()
