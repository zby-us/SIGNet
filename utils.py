"""用于固定随机性、记录运行环境并校验输出文件的通用工具。"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch without silently changing the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(0)
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["git_commit"] = None
    return metadata


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def checkpoint_metadata(path: str | Path) -> dict[str, str]:
    checkpoint = Path(path)
    return {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}


def read_image(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Read an image from paths that may contain non-ASCII characters."""
    source = Path(path)
    try:
        encoded = np.fromfile(source, dtype=np.uint8)
    except OSError as error:
        raise FileNotFoundError(source) from error
    image = cv2.imdecode(encoded, flags) if encoded.size else None
    if image is None:
        raise FileNotFoundError(f"Could not decode image: {source}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image reliably when a Windows path contains Unicode text."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extension = destination.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise OSError(f"Could not encode image for {destination}")
    encoded.tofile(destination)
