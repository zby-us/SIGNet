"""Dataset utilities for four independent half-image segmentation models."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import read_image


# Anatomical names: LI = left ilium; LS = left sacrum;
# RI = right ilium; RS = right sacrum.
# Each tuple is (stored-image half, anatomical side, bone).
# Mask paths: masks_4c/<anatomical_side>/<bone>/<case>/<stem>_<anatomical_side>_<bone>.png.
# Image halves follow the supplied dataset orientation.
STRUCTURES = {
    "LI": ("left", "left", "ilium"),
    "LS": ("left", "left", "sacrum"),
    "RI": ("right", "right", "ilium"),
    "RS": ("right", "right", "sacrum"),
}

def _read_gray(path: Path) -> np.ndarray:
    return read_image(path, cv2.IMREAD_GRAYSCALE)


def _resize_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.array(
        Image.fromarray(image).resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    )


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(mask).resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return (np.array(resized) > 0).astype(np.uint8)


def _bbox(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    height, width = mask.shape
    if len(xs) == 0:
        return None
    x0, x1 = max(0, int(xs.min()) - padding), min(width, int(xs.max()) + padding + 1)
    y0, y1 = max(0, int(ys.min()) - padding), min(height, int(ys.max()) + padding + 1)
    return x0, y0, x1, y1


def _jitter_bbox(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    expansion: tuple[float, float] = (1.1, 1.6),
    jitter: float = 0.12,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    height, width = shape
    box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)
    scale = random.uniform(*expansion)
    center_x = (x0 + x1) / 2 + random.uniform(-jitter, jitter) * box_w
    center_y = (y0 + y1) / 2 + random.uniform(-jitter, jitter) * box_h
    new_w, new_h = box_w * scale, box_h * scale
    new_x0 = max(0, int(round(center_x - new_w / 2)))
    new_y0 = max(0, int(round(center_y - new_h / 2)))
    new_x1 = min(width, int(round(center_x + new_w / 2)))
    new_y1 = min(height, int(round(center_y + new_h / 2)))
    new_x1 = min(width, max(new_x0 + 1, new_x1))
    new_y1 = min(height, max(new_y0 + 1, new_y1))
    return new_x0, new_y0, new_x1, new_y1


def _augment_image(image: np.ndarray) -> np.ndarray:
    """Apply stochastic intensity augmentation to a grayscale image."""
    value = image.astype(np.float32)
    value = value * random.uniform(0.80, 1.25) + random.uniform(-18.0, 18.0)
    value = np.clip(value, 0, 255) / 255.0
    value = np.power(value, random.uniform(0.85, 1.25)) * 255.0
    value = np.clip(value, 0, 255).astype(np.uint8)
    if random.random() < 0.12:
        kernel = random.choice((3, 5))
        value = cv2.GaussianBlur(value, (kernel, kernel), 0)
    noise_std = random.uniform(0.0, 6.0)
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, size=value.shape).astype(np.float32)
        value = np.clip(value.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return value


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return binary
    height, width = binary.shape
    image = binary * 255
    flood = image.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    return (cv2.bitwise_or(image, cv2.bitwise_not(flood)) > 0).astype(np.uint8)


def _augment_mask(mask: np.ndarray) -> np.ndarray:
    """Apply stochastic erosion or dilation to a binary mask."""
    binary = (mask > 0).astype(np.uint8)
    if not binary.any() or random.random() > 0.55:
        return binary
    kernel_size = random.choice((3, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if random.random() < 0.55:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    if random.random() < 0.35:
        binary = _fill_holes(binary)
    if random.random() < 0.25:
        binary = cv2.dilate(binary, kernel, iterations=1)
    return (binary > 0).astype(np.uint8)


def _rectangle(mask: np.ndarray, padding: int = 14) -> np.ndarray:
    rectangle = np.zeros_like(mask, dtype=np.uint8)
    box = _bbox(mask, padding=padding)
    if box is not None:
        x0, y0, x1, y1 = box
        rectangle[y0:y1, x0:x1] = 1
    return rectangle


class StructureDataset(Dataset):
    """Load one of LI/LS/RI/RS using anatomical side/bone mask directories."""

    def __init__(self, split_root: str | Path, structure: str, train: bool = False) -> None:
        if structure not in STRUCTURES:
            raise ValueError(f"Unknown structure {structure}; choose from {sorted(STRUCTURES)}")
        self.root = Path(split_root)
        self.structure = structure
        self.train = train
        self.full_shape = (512, 512)
        self.half_shape = (512, 256)
        self.images = sorted(
            path for path in (self.root / "images").rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        if not self.images:
            raise FileNotFoundError(f"No images found under {self.root / 'images'}")

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path(self, relative_image: Path, anatomical_side: str, bone: str) -> Path:
        """Resolve a mask without discarding the patient/case subdirectory.

        Preprocessed slices are stored as ``<case_id>/<slice>.png``.  Keeping
        that relative parent prevents identically named slices such as
        ``00000.png`` from different patients from colliding.
        """
        return (
            self.root
            / "masks_4c"
            / anatomical_side
            / bone
            / relative_image.parent
            / f"{relative_image.stem}_{anatomical_side}_{bone}.png"
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.images[index]
        relative_image = image_path.relative_to(self.root / "images")
        side, anatomical_side, bone = STRUCTURES[self.structure]
        other = "sacrum" if bone == "ilium" else "ilium"

        # 读取完整 512x512 CT 及当前结构、同侧伴随结构的参考掩膜。
        image = _resize_image(_read_gray(image_path), self.full_shape)
        target = _resize_mask(_read_gray(self._mask_path(relative_image, anatomical_side, bone)), self.full_shape)
        companion = _resize_mask(_read_gray(self._mask_path(relative_image, anatomical_side, other)), self.full_shape)
        union = np.maximum(target, companion)

        # 按解剖中线切成左、右两个 512x256 半幅图像。
        column = slice(0, 256) if side == "left" else slice(256, 512)
        image, target, union = image[:, column], target[:, column], union[:, column]
        if self.train:
            target = _augment_mask(target)
            union = _augment_mask(union)
        rectangle = _rectangle(union, padding=14)

        if self.train and random.random() < 0.90:
            image = _augment_image(image)

        to_image = lambda x: torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(0)
        output: dict[str, torch.Tensor | str] = {
            "image": to_image(image),
            "target": torch.from_numpy(target.astype(np.int64)),
            "rectangle": torch.from_numpy(rectangle.astype(np.float32)),
            "filename": relative_image.as_posix(),
        }
        if not self.train:
            return output

        rect_box = _bbox(rectangle)

        def view(box: tuple[int, int, int, int] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if box is None:
                return image.copy(), target.copy(), np.zeros_like(rectangle)
            vx0, vy0, vx1, vy1 = box
            return (
                _resize_image(image[vy0:vy1, vx0:vx1], self.half_shape),
                _resize_mask(target[vy0:vy1, vx0:vx1], self.half_shape),
                _resize_mask(rectangle[vy0:vy1, vx0:vx1], self.half_shape),
            )

        rect_image, rect_target, rect_rectangle = view(rect_box)
        roi_box = _jitter_bbox(rect_box, self.half_shape) if rect_box is not None else None
        roi_image, roi_target, roi_rectangle = view(roi_box)
        if random.random() < 0.80:
            rect_image = _augment_image(rect_image)
        if random.random() < 0.80:
            roi_image = _augment_image(roi_image)

        output.update(
            {
                "rect_image": to_image(rect_image),
                "rect_target": torch.from_numpy(rect_target.astype(np.int64)),
                "rect_rectangle": torch.from_numpy(rect_rectangle.astype(np.float32)),
                "roi_image": to_image(roi_image),
                "roi_target": torch.from_numpy(roi_target.astype(np.int64)),
                "roi_rectangle": torch.from_numpy(roi_rectangle.astype(np.float32)),
            }
        )
        return output
