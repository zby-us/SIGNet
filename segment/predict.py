#!/usr/bin/env python3
"""Run the four stage-1 models and export bilateral ROIs and union masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from segment.dataset import LEGACY_TAGS, LEGACY_TO_STRUCTURE, STRUCTURES
from segment.model import UnetVMamba, main_logits
from utils import checkpoint_metadata, read_image, write_image, write_json


TAGS = ("LI", "LS", "RI", "RS")


def normalize_structure_name(value: str) -> str:
    """Normalize paper or historical labels to LI/LS/RI/RS."""
    name = value.strip().upper()
    name = LEGACY_TO_STRUCTURE.get(name, name)
    if name not in TAGS:
        accepted = ", ".join((*TAGS, *LEGACY_TO_STRUCTURE))
        raise ValueError(f"Unknown structure label {value!r}; expected one of: {accepted}")
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional LI/LS/RI/RS validation-selected thresholds; otherwise use checkpoint metadata.",
    )
    return parser.parse_args()


def validate_threshold(value: float, structure: str) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"Threshold for {structure} must be strictly between 0 and 1")
    return value


def parse_thresholds(value: str) -> dict[str, float]:
    pairs = [item.split("=", 1) for item in value.split(",")]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("thresholds must use STRUCTURE=value pairs separated by commas")
    keys = [normalize_structure_name(key) for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("thresholds contain duplicate structure names")
    result = {
        key: validate_threshold(float(pair[1]), key)
        for key, pair in zip(keys, pairs)
    }
    if set(result) != set(TAGS):
        raise ValueError(f"Threshold structures must be exactly {list(TAGS)}, got {sorted(result)}")
    return result


def load_model(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, float | None, str | None, str | None]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(model_config, dict):
        raise ValueError(f"Checkpoint {path} has an invalid model_config")
    resolved_config = {"in_channels": 1, "num_classes": 2, "base_channels": 48, "drop_path_rate": 0.10}
    resolved_config.update(model_config)
    model = UnetVMamba(**resolved_config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "head_aux2.proj.weight",
        "head_aux2.proj.bias",
        "head_aux3.proj.weight",
        "head_aux3.proj.bias",
    }
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(
            f"Checkpoint architecture mismatch for {path}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    threshold = checkpoint.get("threshold") if isinstance(checkpoint, dict) else None
    structure = checkpoint.get("structure") if isinstance(checkpoint, dict) else None
    legacy_tag = checkpoint.get("tag") if isinstance(checkpoint, dict) else None
    declarations = [
        normalize_structure_name(str(value))
        for value in (structure, legacy_tag)
        if value is not None
    ]
    if len(set(declarations)) > 1:
        raise ValueError(
            f"Checkpoint {path} has conflicting structure={structure!r} and tag={legacy_tag!r}"
        )
    normalized = declarations[0] if declarations else None
    side = checkpoint.get("side") if isinstance(checkpoint, dict) else None
    return model.to(device).eval(), None if threshold is None else float(threshold), normalized, side


def resolve_structure_checkpoints(
    checkpoint_root: str | Path,
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    """Resolve all four files and reject accidental checkpoint reuse."""
    root = Path(checkpoint_root)
    paths: dict[str, Path] = {}
    metadata: dict[str, dict[str, str]] = {}
    for tag in TAGS:
        legacy = LEGACY_TAGS[tag]
        candidates = (
            root / f"unet_vmamba_best30_{tag}.pt",
            root / f"unet_vmamba_{tag}_best.pt",
            root / f"unet_vmamba_best30_{legacy}.pt",
            root / f"unet_vmamba_{legacy}_best.pt",
        )
        matches = [path for path in candidates if path.is_file()]
        if not matches:
            expected = ", ".join(path.name for path in candidates)
            raise FileNotFoundError(
                f"Checkpoint not found for {tag} under {root}; expected one of: {expected}"
            )
        if len(matches) > 1:
            hashes = {checkpoint_metadata(path)["sha256"] for path in matches}
            if len(hashes) > 1:
                names = ", ".join(path.name for path in matches)
                raise ValueError(
                    f"Multiple different checkpoints match {tag}: {names}. Keep only the intended file."
                )
        checkpoint = matches[0]
        paths[tag] = checkpoint
        metadata[tag] = checkpoint_metadata(checkpoint)

    by_hash: dict[str, list[str]] = {}
    for tag, item in metadata.items():
        by_hash.setdefault(item["sha256"], []).append(tag)
    duplicates = [tags for tags in by_hash.values() if len(tags) > 1]
    if duplicates:
        groups = "; ".join("/".join(tags) for tags in duplicates)
        raise ValueError(
            f"Segmentation checkpoints are byte-identical for {groups}; "
            "LI, LS, RI and RS require separate models"
        )
    return paths, metadata


@torch.no_grad()
def predict(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(device)
    logits = main_logits(model(tensor))
    return torch.softmax(logits, dim=1)[0, 1].cpu().numpy()


def crop(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Cannot construct a grading ROI from an empty predicted ilium-sacrum union")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # 使用髂骨与骶骨预测掩膜联合区域的最小外接矩形裁剪 ROI。
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def embed_half_mask(mask: np.ndarray, structure: str, full_shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Place a half-image prediction back into the full CT coordinate system."""
    height, width = full_shape
    if structure not in TAGS:
        raise ValueError(f"Unknown structure {structure}; choose from {TAGS}")
    if width % 2 or mask.shape != (height, width // 2):
        raise ValueError(
            f"Expected a {(height, width // 2)} half-image mask for a {full_shape} canvas, got {mask.shape}"
        )
    output = np.zeros(full_shape, dtype=mask.dtype)
    columns = slice(0, width // 2) if structure in {"LI", "LS"} else slice(width // 2, width)
    output[:, columns] = mask
    return output


def main() -> None:
    args = parse_args()
    input_root, output_root = Path(args.input_dir), Path(args.output_dir)
    image_root = input_root.parent if input_root.is_file() else input_root
    threshold_overrides = parse_thresholds(args.thresholds) if args.thresholds else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths, checkpoint_files = resolve_structure_checkpoints(args.checkpoint_dir)
    models: dict[str, torch.nn.Module] = {}
    thresholds: dict[str, float] = {}
    checkpoints: dict[str, dict[str, str]] = {}
    for tag in TAGS:
        # 分别加载 LI、LS、RI、RS 四个独立模型及验证集阈值。
        checkpoint = checkpoint_paths[tag]
        model, checkpoint_threshold, checkpoint_structure, checkpoint_side = load_model(checkpoint, device)
        if checkpoint_structure is not None and checkpoint_structure != tag:
            raise ValueError(
                f"Checkpoint {checkpoint} declares structure={checkpoint_structure!r}; expected {tag!r}. "
                "Use checkpoints with the LI/LS/RI/RS naming."
            )
        expected_side = STRUCTURES[tag][0]
        if checkpoint_side is not None and checkpoint_side != expected_side:
            raise ValueError(
                f"Checkpoint {checkpoint} declares side={checkpoint_side!r}; "
                f"expected {expected_side!r} for {tag}"
            )
        models[tag] = model
        checkpoints[tag] = checkpoint_files[tag]
        if threshold_overrides is not None:
            thresholds[tag] = threshold_overrides[tag]
        elif checkpoint_threshold is not None:
            thresholds[tag] = validate_threshold(checkpoint_threshold, tag)
        else:
            raise ValueError(
                f"Checkpoint {checkpoint} has no validation-selected threshold. "
                "Pass --thresholds with four pre-specified validation thresholds."
            )

    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = (
        [input_root]
        if input_root.is_file() and input_root.suffix.lower() in supported
        else sorted(path for path in input_root.rglob("*") if path.suffix.lower() in supported)
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {input_root}")
    qc_rows: list[dict[str, object]] = []
    for image_path in image_paths:
        try:
            image = read_image(image_path, cv2.IMREAD_GRAYSCALE)
        except FileNotFoundError:
            qc_rows.append({"image_path": image_path.relative_to(image_root).as_posix(), "status": "unreadable"})
            continue
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        # 沿解剖中线分割成左右两个 512x256 半幅图像。
        halves = {"left": image[:, :256], "right": image[:, 256:]}
        probabilities = {
            "LI": predict(models["LI"], halves["left"], device),
            "LS": predict(models["LS"], halves["left"], device),
            "RI": predict(models["RI"], halves["right"], device),
            "RS": predict(models["RS"], halves["right"], device),
        }
        masks = {tag: (probabilities[tag] > thresholds[tag]).astype(np.uint8) for tag in TAGS}
        areas = {tag: int(mask.sum()) for tag, mask in masks.items()}
        # 同侧髂骨与骶骨掩膜合并后定义左右关节 ROI。
        unions = {"left": np.maximum(masks["LI"], masks["LS"]), "right": np.maximum(masks["RI"], masks["RS"])}
        empty_sides = [side for side, union in unions.items() if not union.any()]
        if empty_sides:
            qc_rows.append(
                {
                    "image_path": image_path.relative_to(image_root).as_posix(),
                    "status": "empty_predicted_union",
                    "empty_sides": ",".join(empty_sides),
                    **{f"{tag}_area": areas[tag] for tag in TAGS},
                }
            )
            continue
        relative = image_path.parent.relative_to(image_root)
        roi_dir, mask_dir, part_dir = output_root / "roi" / relative, output_root / "roi_masks" / relative, output_root / "masks_4models" / relative
        for directory in (roi_dir, mask_dir, part_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for side in ("left", "right"):
            roi, roi_mask = crop(halves[side], unions[side])
            write_image(roi_dir / f"{image_path.stem}_{side}.png", roi)
            write_image(mask_dir / f"{image_path.stem}_{side}_mask.png", roi_mask * 255)
        for tag, mask in masks.items():
            # Restore the four half-image predictions to 512x512 coordinates.
            full_mask = embed_half_mask(mask, tag, image.shape)
            write_image(part_dir / f"{image_path.stem}_{tag}.png", full_mask * 255)
        qc_rows.append(
            {
                "image_path": image_path.relative_to(image_root).as_posix(),
                "status": "ok",
                **{f"{tag}_area": areas[tag] for tag in TAGS},
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(output_root / "segmentation_qc.csv", index=False)
    status_counts = pd.Series([row["status"] for row in qc_rows]).value_counts().to_dict()
    failed = sum(row.get("status") != "ok" for row in qc_rows)
    write_json(
        output_root / "inference_metadata.json",
        {
            "mask_source": "stage1_predictions",
            "thresholds": thresholds,
            "threshold_source": "command_line_validation_override" if threshold_overrides else "checkpoint_validation",
            "checkpoints": checkpoints,
            "processed": sum(row.get("status") == "ok" for row in qc_rows),
            "failed": failed,
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        },
    )
    if failed:
        raise RuntimeError(
            f"Segmentation failed for {failed} of {len(qc_rows)} inputs. "
            f"See {output_root / 'segmentation_qc.csv'}; grading was not started."
        )
    print(f"Processed {len(qc_rows)} inputs; outputs are in {output_root}")


if __name__ == "__main__":
    main()
