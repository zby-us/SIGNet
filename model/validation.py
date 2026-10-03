"""Validate segmentation-mask provenance for grading inputs."""

import json
import warnings
from pathlib import Path


def require_predicted_masks(split_root: Path, split_name: str) -> None:
    """Require stage-2 inputs generated from stage-1 predicted masks."""
    metadata_path = split_root / "inference_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} is required. Generate {split_name} ROIs and structural priors "
            "with segment.predict so they are derived from predicted masks."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("mask_source") != "stage1_predictions":
        raise ValueError(f"{split_name} grading inputs must be derived from stage-1 predicted masks")
    if metadata.get("threshold_source") not in {
        "checkpoint_validation",
        "command_line_validation_override",
    }:
        raise ValueError(f"{split_name} segmentation metadata does not contain validation-selected thresholds")
    if set(metadata.get("thresholds", {})) != {"LI", "LS", "RI", "RS"}:
        raise ValueError(f"{split_name} segmentation metadata must contain LI/LS/RI/RS thresholds")


def check_validation_mask_source(mask_dir: str) -> None:
    """Check available provenance without rejecting legacy prepared datasets."""
    metadata_path = Path(mask_dir).parent / "inference_metadata.json"
    if not metadata_path.is_file():
        warnings.warn(
            "Validation mask provenance is unavailable. Ensure validation ROIs and "
            "priors were generated from stage-1 predictions with validation-selected "
            "thresholds, as described in the paper. Legacy inputs remain supported.",
            UserWarning,
        )
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid validation metadata: {metadata_path}")
    if metadata.get("mask_source") != "stage1_predictions":
        raise ValueError("Validation inputs must use stage-1 predicted masks")
    if metadata.get("threshold_source") not in {
        "checkpoint_validation", "command_line_validation_override"
    }:
        raise ValueError("Validation metadata must declare validation-selected thresholds")
    thresholds = metadata.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {"LI", "LS", "RI", "RS"}:
        raise ValueError("Validation metadata must contain LI/LS/RI/RS thresholds")
