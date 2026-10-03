#!/usr/bin/env python3
"""Run stage-2 inference and aggregate ordered slice predictions."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from model.aggregation import aggregate_case, aggregate_side
from model.signet import DualBranchGrader, GradingModelConfig, normalize_grading_state_dict
from model.validation import require_predicted_masks
from model.preprocessing import prepare_grading_inputs
from utils import checkpoint_metadata, read_image, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint-left", required=True)
    parser.add_argument("--checkpoint-right", required=True)
    parser.add_argument(
        "--slice-manifest",
        required=True,
        help="CSV with patient_id,case_id,slice_id,slice_order,image_path",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-consecutive", type=int, default=2)
    return parser.parse_args()


def load_model(
    path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, str, dict[str, str | bool | None]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    wrapped = isinstance(checkpoint, dict) and "model" in checkpoint
    state = checkpoint["model"] if wrapped else checkpoint
    config_values = dict(checkpoint.get("config", {})) if wrapped else {}
    checkpoint_architecture = checkpoint.get("architecture", "signet") if wrapped else "signet"
    checkpoint_config = GradingModelConfig(**config_values)
    # A complete grading checkpoint overwrites every parameter, so rebuilding
    # the architecture must not download ImageNet weights during inference.
    if checkpoint_architecture != "signet":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint_architecture}")
    model = DualBranchGrader(replace(checkpoint_config, imagenet_init=False))
    normalized = normalize_grading_state_dict(state)
    model.load_state_dict(normalized, strict=True)
    metadata = {
        "prior_mode": checkpoint.get("prior_mode") if wrapped else None,
        "checkpoint_type": checkpoint.get("checkpoint_type") if wrapped else None,
        "side": checkpoint.get("side") if wrapped else None,
        "imagenet_init": checkpoint_config.imagenet_init,
    }
    return model.to(device).eval(), checkpoint_architecture, metadata


def validate_side_checkpoints(left: str | Path, right: str | Path) -> dict[str, dict[str, str]]:
    """Reject accidental reuse of one grading checkpoint for both sides."""
    paths = {"left": Path(left), "right": Path(right)}
    metadata = {side: checkpoint_metadata(path) for side, path in paths.items()}
    if paths["left"].resolve() == paths["right"].resolve():
        raise ValueError("Left and right grading checkpoints must be different files")
    if metadata["left"]["sha256"] == metadata["right"]["sha256"]:
        raise ValueError(
            "Left and right grading checkpoints have identical SHA-256 hashes; "
            "left and right grading require separate models"
        )
    return metadata


def mask_for(roi_path: Path, roi_root: Path, mask_root: Path) -> Path:
    relative = roi_path.relative_to(roi_root)
    return (mask_root / relative.parent / f"{roi_path.stem}_mask.png")


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    roi_path: Path,
    mask_path: Path,
    device: torch.device,
) -> tuple[int, np.ndarray]:
    # 读取灰度 ROI 与对应预测掩膜，并统一缩放到 224x224。
    roi = read_image(roi_path, cv2.IMREAD_GRAYSCALE)
    mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
    roi_array, prior_array = prepare_grading_inputs(roi, mask)
    roi_tensor = torch.from_numpy(roi_array)[None].to(device)
    prior_tensor = torch.from_numpy(prior_array)[None].to(device)
    # 对原图和水平翻转图的 logits 取平均，得到切片级预测。
    logits, _ = model(roi_tensor, prior_tensor)
    flipped, _ = model(torch.flip(roi_tensor, [3]), torch.flip(prior_tensor, [3]))
    probability = torch.softmax((logits + flipped) / 2, dim=1)[0].cpu().numpy()
    return int(probability.argmax()), probability


def validate_grading_inputs(manifest_path: str | Path, roi_root: Path, mask_root: Path) -> pd.DataFrame:
    """Validate every manifest pair before loading any model weights."""
    for label, directory in (("ROI", roi_root), ("mask", mask_root)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    if not Path(manifest_path).is_file():
        raise FileNotFoundError(f"Slice manifest does not exist: {manifest_path}")
    manifest = pd.read_csv(
        manifest_path,
        dtype={"patient_id": str, "case_id": str, "slice_id": str, "image_path": str},
    )
    required = {"patient_id", "case_id", "slice_id", "slice_order", "image_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Slice manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Slice manifest is empty")
    if manifest[["patient_id", "case_id", "slice_id", "image_path"]].isna().any().any():
        raise ValueError("patient_id, case_id, slice_id and image_path cannot contain missing values")
    if manifest[["case_id", "slice_order"]].duplicated().any():
        raise ValueError("slice_order must be unique within each case")
    if manifest[["case_id", "slice_id"]].duplicated().any():
        raise ValueError("slice_id must be unique within each case")
    numeric_order = pd.to_numeric(manifest["slice_order"], errors="coerce")
    if not np.isfinite(numeric_order).all() or not np.equal(numeric_order, np.floor(numeric_order)).all():
        raise ValueError("slice_order must contain integers only")
    manifest = manifest.assign(slice_order=numeric_order.astype(int))
    if manifest[["case_id", "image_path"]].duplicated().any():
        raise ValueError("image_path must be unique within each case")
    for column in ("patient_id", "case_id", "slice_id", "image_path"):
        if manifest[column].str.strip().eq("").any():
            raise ValueError(f"{column} cannot contain blank values")
    safe_paths: list[Path] = []
    for value in manifest.image_path:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"image_path must be a safe relative path, got {value!r}")
        safe_paths.append(relative)
    manifest = manifest.assign(relative_image_path=safe_paths)
    # 只使用清单中的显式解剖顺序，绝不依赖文件名排序。
    manifest = manifest.sort_values(["case_id", "slice_order"]).reset_index(drop=True)
    if (manifest.groupby("case_id")["patient_id"].nunique() > 1).any():
        raise ValueError("Each case_id must belong to exactly one patient_id")
    missing_files = []
    for relative in manifest.relative_image_path:
        for side in ("left", "right"):
            roi = roi_root / relative.parent / f"{relative.stem}_{side}.png"
            mask = mask_for(roi, roi_root, mask_root)
            for kind, path in (("ROI", roi), ("mask", mask)):
                if not path.is_file():
                    missing_files.append(f"{kind}: {path}")
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} required ROI/mask files are missing. "
            "No slices will be skipped. Expected roi/<case>/<stem>_left.png and "
            "roi_masks/<case>/<stem>_left_mask.png (and right equivalents). "
            "Examples: " + "; ".join(missing_files[:10])
        )
    print(f"[INPUT] Validated {len(manifest)} slices, {manifest.case_id.nunique()} cases, "
          f"{2 * len(manifest)} bilateral ROI/mask pairs")
    return manifest


def main() -> None:
    args = parse_args()
    if args.minimum_consecutive != 2:
        raise ValueError("--minimum-consecutive must be 2 for this model configuration")
    roi_root, mask_root = Path(args.roi_dir), Path(args.mask_dir)
    manifest = validate_grading_inputs(args.slice_manifest, roi_root, mask_root)
    require_predicted_masks(mask_root.parent, "Testing")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    side_checkpoint_files = validate_side_checkpoints(
        args.checkpoint_left,
        args.checkpoint_right,
    )
    models: dict[str, torch.nn.Module] = {}
    architectures: dict[str, str] = {}
    checkpoint_metadata_by_side: dict[str, dict[str, object]] = {}
    for side, checkpoint_path in (
        ("left", args.checkpoint_left),
        ("right", args.checkpoint_right),
    ):
        model, architecture, metadata = load_model(checkpoint_path, device)
        if metadata["prior_mode"] not in (None, "normal"):
            raise ValueError(
                f"The {side} checkpoint was trained with prior_mode={metadata['prior_mode']!r}, "
                "but SIGNet inference requires the structural prior"
            )
        if metadata["side"] not in (None, side):
            raise ValueError(f"The {side} checkpoint declares side={metadata['side']!r}")
        models[side] = model
        architectures[side] = architecture
        checkpoint_metadata_by_side[side] = {
            **side_checkpoint_files[side],
            **metadata,
        }
    slice_rows = []
    grouped: dict[str, dict[str, object]] = {}
    for row in manifest.itertuples(index=False):
        relative_image = row.relative_image_path
        for side in ("left", "right"):
            roi_path = roi_root / relative_image.parent / f"{relative_image.stem}_{side}.png"
            mask_path = mask_for(roi_path, roi_root, mask_root)
            grade, probability = predict(models[side], roi_path, mask_path, device)
            cid = str(row.case_id)
            pid = str(row.patient_id)
            case_entry = grouped.setdefault(
                cid,
                {"patient_id": pid, "left": [], "right": [], "left_orders": [], "right_orders": []},
            )
            if case_entry["patient_id"] != pid:
                raise ValueError(f"case_id {cid} is associated with multiple patient_id values")
            case_entry[side].append(grade)
            case_entry[f"{side}_orders"].append(int(row.slice_order))
            slice_rows.append(
                {
                    "patient_id": pid,
                    "case_id": cid,
                    "slice_id": str(row.slice_id),
                    "slice_order": int(row.slice_order),
                    "image_path": relative_image.as_posix(),
                    "filename": roi_path.relative_to(roi_root).as_posix(),
                    "side": side,
                    "prediction": grade,
                    **{f"prob_{i}": float(probability[i]) for i in range(5)},
                }
            )
    case_rows = []
    for cid, sides in grouped.items():
        # 每侧先验证连续切片，再取左右两侧较高等级作为病例级结果。
        left = aggregate_side(sides["left"], args.minimum_consecutive, sides["left_orders"])
        right = aggregate_side(sides["right"], args.minimum_consecutive, sides["right_orders"])
        case_rows.append(
            {
                "patient_id": sides["patient_id"],
                "case_id": cid,
                "left_grade": left,
                "right_grade": right,
                "case_grade": aggregate_case(
                    sides["left"],
                    sides["right"],
                    args.minimum_consecutive,
                    sides["left_orders"],
                    sides["right_orders"],
                ),
            }
        )
    pd.DataFrame(slice_rows).to_csv(output_dir / "slice_predictions.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output_dir / "case_predictions.csv", index=False)
    write_json(
        output_dir / "inference_metadata.json",
        {
            "slice_manifest": str(Path(args.slice_manifest).resolve()),
            "ordering": "explicit slice_order from manifest",
            "prior_mode": "normal",
            "architectures": architectures,
            "minimum_consecutive": args.minimum_consecutive,
            "checkpoints": checkpoint_metadata_by_side,
        },
    )
    print(f"Wrote {len(slice_rows)} slice predictions and {len(case_rows)} case predictions to {output_dir}")


if __name__ == "__main__":
    main()
