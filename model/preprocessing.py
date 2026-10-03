"""Deterministic preprocessing matching the original validation recipe."""

import cv2
import numpy as np
from PIL import Image


def prepare_grading_inputs(roi, mask, image_size=224):
    if roi.ndim != 2 or mask.ndim != 2:
        raise ValueError("Expected grayscale ROI and mask")
    if roi.dtype != np.uint8 or mask.dtype != np.uint8:
        raise ValueError("Expected uint8 ROI and mask")
    if mask.shape != roi.shape:
        mask = np.array(Image.fromarray(mask).resize(
            (roi.shape[1], roi.shape[0]), Image.Resampling.NEAREST))
    binary = (mask > 0).astype(np.uint8)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3).astype(np.float32)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3).astype(np.float32)
    sdt = inside - outside
    # Preserve the original uint8 quantization before resizing.
    sdt = ((sdt - sdt.min()) / (sdt.max() - sdt.min() + 1e-6) * 255.0).astype(np.uint8)
    edge = cv2.morphologyEx(binary * 255, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    band = cv2.dilate((edge > 0).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) * 255
    four = np.stack([roi, mask, sdt, band], axis=-1)
    four = cv2.resize(four, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32) * 255.0
    scale = 1.0 / (np.array([0.5, 1.0, 1.0, 1.0], dtype=np.float32) * 255.0)
    four = (four.astype(np.float32) - mean) * scale
    channels = np.ascontiguousarray(four.transpose(2, 0, 1))
    return channels[:1], channels[1:]
