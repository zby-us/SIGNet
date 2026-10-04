#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SIGNet grading training: ROI intensity and structural-prior fusion
-------------------------------------------------------
输入：
- 分支A：ROI(1ch)
- 分支B：Mask(3ch) = [mask, SDT, band]

目的：
- 在已有 ROI + Mask 基础上，增加关节缘 band 通道
- 强化模型对“侵蚀边缘/皮质中断/局部凹陷”的学习

数据要求：
train_roi_dir : data/train/roi
train_mask_dir: data/train/roi_masks
val_roi_dir   : data/val/roi
val_mask_dir  : data/val/roi_masks
labels csv 中包含：filename,left,right

输出：
- outputs/grading_weights/dualfuse_band_left_best.pth
- outputs/grading_weights/dualfuse_band_right_best.pth
- outputs/grading_weights/dualfuse_band_left_final.pth
- outputs/grading_weights/dualfuse_band_right_final.pth
- outputs/grading_weights/cm_resnet18dualfuse_band/*.csv
"""

import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse
from model.validation import check_validation_mask_source
from pathlib import Path
import math
import random
import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ========================= 可选依赖 =========================
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from scipy.ndimage import distance_transform_edt as _edt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from model.attention import TripletAttention


# ========================= 配置 =========================
@dataclass
class Config:
    train_roi_dir: str = "data/train/roi"
    train_mask_dir: str = "data/train/roi_masks"
    val_roi_dir: str = "data/val/roi"
    val_mask_dir: str = "data/val/roi_masks"
    train_labels_csv: str = "data/train/labels.csv"
    val_labels_csv: str = "data/val/labels.csv"

    model_dir: str = "outputs/grading_weights"
    out_dir_eval: str = "outputs/grading_eval"

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    num_epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_classes: int = 5
    seed: int = 42

    warmup_epochs: int = 5

    use_focal: bool = True
    focal_gamma: float = 1.5
    label_smooth: float = 0.05

    use_weighted_sampler: bool = False
    use_cb_focal: bool = True
    cb_beta: float = 0.999

    use_tta_eval: bool = True

    use_swa: bool = True
    swa_start: int = 50

    ema_decay: float = 0.999

    head_type: str = "ce"  # "ce" / "ordinal"

    use_xattn: bool = True
    xattn_layers: int = 1
    xattn_heads: int = 8
    xattn_ffn: int = 1024
    xattn_dropout: float = 0.1
    xattn_activation: str = "gelu"

    feat_plugin: str = "eca"  # none / se / eca / cbam
    use_blurpool: bool = True
    blur_filt_size: int = 3

    act_fuse: str = "silu"
    act_head: str = "gelu"
    gate_hidden: int = 512
    gate_drop: float = 0.10

    aux12_w: float = 0.35
    consis_w: float = 0.08
    consis_T: float = 2.0

    mixup_p: float = 0.12
    cutmix_p: float = 0.22
    mix_alpha: float = 0.6

    rotate_limit: int = 10
    shift_limit: float = 0.05
    scale_limit: float = 0.10
    shear_limit: float = 5.0
    elastic_alpha: float = 20.0
    elastic_sigma: float = 4.0
    grid_distort: float = 0.05
    persp_scale: float = 0.05

    rrcrop_p: float = 0.30
    rrcrop_scale: tuple = (0.85, 1.00)
    rrcrop_ratio: tuple = (0.95, 1.05)

    enable_roi_intensity: bool = True
    roi_intensity_p: float = 0.70
    roi_gamma_range: Tuple[float, float] = (0.50, 1.25)
    roi_contrast_limit: float = 0.15
    roi_brightness_limit: float = 0.12
    roi_clahe_p: float = 0.30
    roi_noise_p: float = 0.30
    roi_blur_p: float = 0.25
    roi_unsharp_p: float = 0.25
    roi_gauss_noise_std: Tuple[float, float] = (2.0, 8.0)
    roi_blur_ks: Tuple[int, int] = (3, 5)
    roi_biasfield_p: float = 0.30
    roi_biasfield_amp: float = 0.20
    roi_coarsedrop_p: float = 0.25
    roi_coarsedrop_max_holes: int = 4
    roi_coarsedrop_size: Tuple[float, float] = (0.05, 0.12)
    roi_multnoise_p: float = 0.30
    roi_multnoise_std: Tuple[float, float] = (0.02, 0.08)
    roi_strip_p: float = 0.20
    roi_strip_ratio: tuple = (0.03, 0.12)

    enable_mask_morph: bool = True
    mask_morph_p: float = 0.35
    mask_morph_iters: Tuple[int, int] = (1, 2)
    mask_morph_ks: Tuple[int, int] = (3, 5)
    mask_dropout_p: float = 0.10

    band_mode: str = "edge_dilate"   # edge_dilate / ring
    band_edge_dilate_ks: int = 7
    band_ring_dilate_ks: int = 7
    band_ring_erode_ks: int = 5

    replace_backbone_relu: bool = False
    backbone_act: str = "silu"

    amp: bool = True
    channels_last: bool = True


cfg = Config()
cm_dir = os.path.join(cfg.model_dir, "cm_resnet18dualfuse_band")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")


# ========================= AMP工具 =========================
def autocast_context():
    if device.type == "cuda" and cfg.amp:
        return torch.amp.autocast(device_type="cuda")
    class DummyContext:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return DummyContext()


def build_grad_scaler():
    if device.type == "cuda" and cfg.amp:
        try:
            return torch.amp.GradScaler("cuda")
        except Exception:
            return torch.cuda.amp.GradScaler()
    return None


# ========================= 工具函数 =========================
def make_act(name: str):
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ["silu", "swish"]:
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    if name in ["leaky_relu", "lrelu"]:
        return nn.LeakyReLU(0.1, inplace=True)
    return nn.ReLU(inplace=True)


def replace_relu_(module: nn.Module, act_name: str):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, make_act(act_name))
        else:
            replace_relu_(child, act_name)


def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _bincount_safe(labels: List[int], num_classes: int) -> List[int]:
    bc = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes)
    return bc.astype(int).tolist()


# ========================= 评估表 =========================
def _macro_specificity_from_cm(cm: np.ndarray) -> float:
    K = cm.shape[0]
    specs = []
    for c in range(K):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        denom = TN + FP
        specs.append(TN / denom if denom > 0 else np.nan)
    return float(np.nanmean(np.array(specs, dtype=np.float64)))


def _safe_multiclass_auc(y_true, prob, K, avg="macro"):
    try:
        return float(
            roc_auc_score(
                np.asarray(y_true, dtype=np.int64),
                np.asarray(prob, dtype=np.float64),
                labels=list(range(K)),
                multi_class="ovr",
                average=avg
            )
        )
    except Exception:
        return np.nan


def compute_metrics_once(gts, preds, probs, K, allow_auc=True):
    gts = np.asarray(gts, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    cm = confusion_matrix(gts, preds, labels=list(range(K)))

    acc = float((preds == gts).mean())
    prec = float(precision_score(gts, preds, labels=list(range(K)), average="macro", zero_division=0))
    rec = float(recall_score(gts, preds, labels=list(range(K)), average="macro", zero_division=0))
    f1 = float(f1_score(gts, preds, labels=list(range(K)), average="macro", zero_division=0))
    spec = float(_macro_specificity_from_cm(cm))
    mcc = float(matthews_corrcoef(gts, preds))

    out = {
        "AUC": np.nan,
        "Accuracy": acc,
        "Recall": rec,
        "Specificity": spec,
        "Precision": prec,
        "F1": f1,
        "MCC": mcc,
    }

    if allow_auc and probs is not None:
        prob = np.asarray(probs, dtype=np.float64)
        if prob.ndim == 2 and prob.shape[1] == K:
            out["AUC"] = _safe_multiclass_auc(gts, prob, K, avg="macro")

    return out


def bootstrap_ci(gts, preds, probs, K, n_boot=500, seed=123, allow_auc=True):
    rng = np.random.default_rng(seed)
    N = len(gts)
    keys = ["AUC", "Accuracy", "Recall", "Specificity", "Precision", "F1", "MCC"]
    samples = {k: [] for k in keys}

    gts = np.asarray(gts)
    preds = np.asarray(preds)
    probs_arr = None if probs is None else np.asarray(probs)

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        g_b = gts[idx]
        p_b = preds[idx]
        pr_b = None if probs_arr is None else probs_arr[idx]
        m = compute_metrics_once(g_b, p_b, pr_b, K, allow_auc=allow_auc)
        for k in keys:
            if np.isfinite(m[k]):
                samples[k].append(m[k])

    ci = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=np.float64)
        if len(arr) < max(10, int(n_boot * 0.1)):
            ci[k] = (np.nan, np.nan)
        else:
            lo, hi = np.quantile(arr, [0.025, 0.975])
            ci[k] = (float(lo), float(hi))
    return ci


def _fmt(v, ci_pair):
    lo, hi = ci_pair
    if not np.isfinite(v):
        return "nan (nan–nan)"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{v:.3f} ({lo:.3f}–{hi:.3f})"
    return f"{v:.3f} (nan–nan)"


def print_eval_table(title: str, metrics: dict, ci: dict):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    rows = [
        ("AUC (95% CI)", "AUC"),
        ("Accuracy (95% CI)", "Accuracy"),
        ("Recall (95% CI)", "Recall"),
        ("Specificity (95% CI)", "Specificity"),
        ("Precision (95% CI)", "Precision"),
        ("F1-score (95% CI)", "F1"),
        ("MCC (95% CI)", "MCC"),
    ]
    for name, k in rows:
        print(f"{name:<22} {_fmt(metrics[k], ci[k])}")
    print("=" * 70 + "\n")


# ========================= ROI增强 =========================
def _roi_bias_field(roi: np.ndarray) -> np.ndarray:
    if random.random() >= cfg.roi_biasfield_p:
        return roi
    H, W = roi.shape
    yv, xv = np.mgrid[0:1:complex(0, H), 0:1:complex(0, W)]
    amp = cfg.roi_biasfield_amp
    f1x, f1y = random.uniform(0.6, 1.5), random.uniform(0.6, 1.5)
    f2x, f2y = random.uniform(0.6, 1.5), random.uniform(0.6, 1.5)
    ph1, ph2 = random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi)
    field = 1.0 + amp * (
        0.5 * np.sin(2 * np.pi * (f1x * xv + f1y * yv) + ph1)
        + 0.5 * np.sin(2 * np.pi * (f2x * xv + f2y * yv) + ph2)
    )
    out = roi.astype(np.float32) * field.astype(np.float32)
    return _clip_u8(out)


def _roi_coarse_dropout(roi: np.ndarray) -> np.ndarray:
    if random.random() >= cfg.roi_coarsedrop_p:
        return roi
    h, w = roi.shape
    out = roi.copy()
    for _ in range(random.randint(1, cfg.roi_coarsedrop_max_holes)):
        rh = int(random.uniform(*cfg.roi_coarsedrop_size) * h)
        rw = int(random.uniform(*cfg.roi_coarsedrop_size) * w)
        if rh <= 0 or rw <= 0:
            continue
        y0 = random.randint(0, max(0, h - rh))
        x0 = random.randint(0, max(0, w - rw))
        fill = random.randint(0, 255)
        out[y0:y0 + rh, x0:x0 + rw] = fill
    return out


def roi_intensity_aug(img: np.ndarray, **kwargs) -> np.ndarray:
    if not cfg.enable_roi_intensity or random.random() >= cfg.roi_intensity_p:
        return img

    x = img.copy()
    roi = x[..., 0].astype(np.float32)

    roi = _roi_bias_field(_clip_u8(roi)).astype(np.float32)

    if random.random() < 0.5:
        gamma = random.uniform(*cfg.roi_gamma_range)
        roi = 255.0 * np.power(np.clip(roi / 255.0, 0, 1), 1.0 / gamma)

    if random.random() < 0.5:
        alpha = 1.0 + random.uniform(-cfg.roi_contrast_limit, cfg.roi_contrast_limit)
        beta = 255.0 * random.uniform(-cfg.roi_brightness_limit, cfg.roi_brightness_limit)
        roi = alpha * roi + beta

    if random.random() < cfg.roi_clahe_p:
        if _HAS_CV2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi = clahe.apply(_clip_u8(roi))
        else:
            y = _clip_u8(roi)
            hist, _ = np.histogram(y.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf = (255 * (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-6)).astype(np.uint8)
            roi = cdf[y]

    if random.random() < cfg.roi_blur_p:
        k = random.choice([cfg.roi_blur_ks[0], cfg.roi_blur_ks[1]])
        if k % 2 == 0:
            k += 1
        if _HAS_CV2:
            blur = cv2.GaussianBlur(_clip_u8(roi), (k, k), 0)
        else:
            y = _clip_u8(roi).astype(np.float32)
            pad = np.pad(y, ((1, 1), (1, 1)), mode="reflect")
            blur = (
                pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:] +
                pad[1:-1, 0:-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
                pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
            ) / 9.0

        if random.random() < cfg.roi_unsharp_p:
            roi = np.clip(1.5 * _clip_u8(roi) - 0.5 * blur, 0, 255)
        else:
            roi = blur.astype(np.float32)

    if random.random() < cfg.roi_noise_p:
        std = random.uniform(*cfg.roi_gauss_noise_std)
        roi = roi + np.random.normal(0, std, size=roi.shape).astype(np.float32)

    if random.random() < cfg.roi_multnoise_p:
        stdmul = random.uniform(*cfg.roi_multnoise_std)
        m = 1.0 + np.random.normal(0.0, stdmul, size=roi.shape).astype(np.float32)
        roi = roi * np.clip(m, 0.5, 1.5)

    roi = _roi_coarse_dropout(_clip_u8(roi)).astype(np.float32)

    if random.random() < cfg.roi_strip_p:
        H, W = roi.shape
        h_ratio = random.uniform(*cfg.roi_strip_ratio)
        h = max(1, int(H * h_ratio))
        top = random.random() < 0.5
        y0 = 0 if top else (H - h)
        fill = random.randint(0, 30)
        roi[y0:y0 + h, :] = fill

    x[..., 0] = _clip_u8(roi)
    return x


# ========================= Mask 处理 =========================
def _morph_np_step(m: np.ndarray, op: str) -> np.ndarray:
    p = np.pad(m, ((1, 1), (1, 1)), "constant")
    neigh = [p[i:i + m.shape[0], j:j + m.shape[1]] for i in (0, 1, 2) for j in (0, 1, 2)]
    if op == "dilate":
        return np.maximum.reduce(neigh)
    return np.minimum.reduce(neigh)


def _get_cv_kernel(k: int):
    if k % 2 == 0:
        k += 1
    if _HAS_CV2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return None


def dilate_bin(mask_bin: np.ndarray, k: int) -> np.ndarray:
    m = (mask_bin > 0).astype(np.uint8)
    if _HAS_CV2:
        ker = _get_cv_kernel(k)
        out = cv2.dilate(m, ker, iterations=1)
        return (out > 0).astype(np.uint8)
    out = m.copy()
    steps = max(1, k // 2)
    for _ in range(steps):
        out = _morph_np_step(out, "dilate")
    return (out > 0).astype(np.uint8)


def erode_bin(mask_bin: np.ndarray, k: int) -> np.ndarray:
    m = (mask_bin > 0).astype(np.uint8)
    if _HAS_CV2:
        ker = _get_cv_kernel(k)
        out = cv2.erode(m, ker, iterations=1)
        return (out > 0).astype(np.uint8)
    out = m.copy()
    steps = max(1, k // 2)
    for _ in range(steps):
        out = _morph_np_step(out, "erode")
    return (out > 0).astype(np.uint8)


def jitter_mask_np(mask_np: np.ndarray) -> np.ndarray:
    if not cfg.enable_mask_morph or random.random() >= cfg.mask_morph_p:
        return (mask_np > 0).astype(np.uint8) * 255

    iters = random.randint(cfg.mask_morph_iters[0], cfg.mask_morph_iters[1])
    k = random.choice([cfg.mask_morph_ks[0], cfg.mask_morph_ks[1]])
    if k % 2 == 0:
        k += 1

    op = random.choice(["dilate", "erode"])
    m = (mask_np > 0).astype(np.uint8)

    if _HAS_CV2:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if op == "dilate":
            out = cv2.dilate(m, ker, iterations=iters)
        else:
            out = cv2.erode(m, ker, iterations=iters)
        return (out > 0).astype(np.uint8) * 255

    out = m.copy()
    for _ in range(iters):
        out = _morph_np_step(out, op)
    return (out > 0).astype(np.uint8) * 255


def _edge_from_mask_numpy(mask_np: np.ndarray) -> np.ndarray:
    m = (mask_np > 0).astype(np.uint8)
    p = np.pad(m, ((1, 1), (1, 1)), mode="constant")
    neigh = [p[i:i + m.shape[0], j:j + m.shape[1]] for i in (0, 1, 2) for j in (0, 1, 2)]
    dil = np.maximum.reduce(neigh)
    ero = np.minimum.reduce(neigh)
    edge = (dil.astype(np.int8) - ero.astype(np.int8))
    return (edge > 0).astype(np.uint8) * 255


def mask_to_edge(mask_np: np.ndarray, ksize: int = 3) -> np.ndarray:
    if _HAS_CV2:
        binm = (mask_np > 0).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        grad = cv2.morphologyEx(binm, cv2.MORPH_GRADIENT, k)
        return (grad > 0).astype(np.uint8) * 255
    return _edge_from_mask_numpy(mask_np)


def signed_distance_u8(mask_np: np.ndarray) -> np.uint8:
    binm = (mask_np > 0).astype(np.uint8)

    if _HAS_CV2:
        din = cv2.distanceTransform(binm, cv2.DIST_L2, 3).astype(np.float32)
        dout = cv2.distanceTransform(1 - binm, cv2.DIST_L2, 3).astype(np.float32)
    elif _HAS_SCIPY:
        din = _edt(binm).astype(np.float32)
        dout = _edt(1 - binm).astype(np.float32)
    else:
        return mask_to_edge(mask_np)

    sdt = din - dout
    sdt = (sdt - sdt.min()) / (sdt.max() - sdt.min() + 1e-6) * 255.0
    return sdt.astype(np.uint8)


def build_band_u8(mask_np: np.ndarray) -> np.ndarray:
    binm = (mask_np > 0).astype(np.uint8)
    if cfg.band_mode == "ring":
        d = dilate_bin(binm, cfg.band_ring_dilate_ks)
        e = erode_bin(binm, cfg.band_ring_erode_ks)
        band = ((d > 0) & (e == 0)).astype(np.uint8) * 255
        return band

    edge = (mask_to_edge(mask_np) > 0).astype(np.uint8)
    band = dilate_bin(edge, cfg.band_edge_dilate_ks)
    return (band > 0).astype(np.uint8) * 255


# ========================= Albumentations =========================
def get_transforms(img_size: int):
    INTER_LINEAR = cv2.INTER_LINEAR if _HAS_CV2 else 1
    INTER_NEAREST = cv2.INTER_NEAREST if _HAS_CV2 else 0
    BORDER_CONST = cv2.BORDER_CONSTANT if _HAS_CV2 else 0

    try:
        affine = A.Affine(
            scale=(1.0 - cfg.scale_limit, 1.0 + cfg.scale_limit),
            translate_percent={
                "x": (-cfg.shift_limit, cfg.shift_limit),
                "y": (-cfg.shift_limit, cfg.shift_limit),
            },
            rotate=(-cfg.rotate_limit, cfg.rotate_limit),
            shear={
                "x": (-cfg.shear_limit, cfg.shear_limit),
                "y": (-cfg.shear_limit, cfg.shear_limit),
            },
            interpolation=INTER_LINEAR,
            mask_interpolation=INTER_NEAREST,
            fill=0,
            fill_mask=0,
            p=0.8
        )
    except TypeError:
        affine = A.Affine(
            scale=(1.0 - cfg.scale_limit, 1.0 + cfg.scale_limit),
            translate_percent={
                "x": (-cfg.shift_limit, cfg.shift_limit),
                "y": (-cfg.shift_limit, cfg.shift_limit),
            },
            rotate=(-cfg.rotate_limit, cfg.rotate_limit),
            shear={
                "x": (-cfg.shear_limit, cfg.shear_limit),
                "y": (-cfg.shear_limit, cfg.shear_limit),
            },
            interpolation=INTER_LINEAR,
            mask_interpolation=INTER_NEAREST,
            cval=0,
            cval_mask=0,
            p=0.8
        )

    try:
        rrc = A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=cfg.rrcrop_scale,
            ratio=cfg.rrcrop_ratio,
            interpolation=INTER_LINEAR,
            p=cfg.rrcrop_p
        )
    except Exception:
        rrc = A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=cfg.rrcrop_scale,
            ratio=cfg.rrcrop_ratio,
            interpolation=INTER_LINEAR,
            p=cfg.rrcrop_p
        )

    train_tf = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            affine,
            A.ElasticTransform(
                alpha=cfg.elastic_alpha,
                sigma=cfg.elastic_sigma,
                interpolation=INTER_LINEAR,
                border_mode=BORDER_CONST,
                fill=0,
                fill_mask=0,
                p=0.15
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=cfg.grid_distort,
                border_mode=BORDER_CONST,
                p=0.20
            ),
            A.Perspective(
                scale=(0.0, cfg.persp_scale),
                keep_size=True,
                p=0.20
            ),
            rrc,
            A.Resize(img_size, img_size, interpolation=INTER_LINEAR),
            A.Lambda(image=roi_intensity_aug, p=1.0),
            A.Normalize(mean=[0.5, 0.0, 0.0, 0.0], std=[0.5, 1.0, 1.0, 1.0]),
            ToTensorV2(),
        ],
        is_check_shapes=False
    )

    eval_tf = A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=INTER_LINEAR),
            A.Normalize(mean=[0.5, 0.0, 0.0, 0.0], std=[0.5, 1.0, 1.0, 1.0]),
            ToTensorV2(),
        ],
        is_check_shapes=False
    )

    return train_tf, eval_tf


# ========================= 表构建 =========================


def build_side_table(labels_df: pd.DataFrame, side: str, roi_dir: str, mask_dir: str) -> pd.DataFrame:
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown side: {side}")
    missing_columns = {"filename", side} - set(labels_df.columns)
    if missing_columns:
        raise ValueError(f"Missing label columns: {sorted(missing_columns)}")
    if labels_df.empty:
        raise ValueError(f"Empty label table for {side}")
    rows = []
    errors = []
    for _, row in labels_df.iterrows():
        if pd.isna(row["filename"]) or not str(row["filename"]).strip():
            errors.append("Missing filename")
            continue
        stem_raw = str(row["filename"])
        stem = os.path.splitext(stem_raw)[0]

        roi_path = _first_existing([
            os.path.join(roi_dir, f"{stem}_{side}.png"),
            os.path.join(roi_dir, f"{stem}_{side}.jpg"),
            os.path.join(roi_dir, f"{stem}_{side}.jpeg"),
            os.path.join(roi_dir, f"{stem}.png"),
            os.path.join(roi_dir, f"{stem}.jpg"),
            os.path.join(roi_dir, f"{stem}.jpeg"),
        ])

        mask_path = _first_existing([
            os.path.join(mask_dir, f"{stem}_{side}_mask.png"),
            os.path.join(mask_dir, f"{stem}_{side}_mask.jpg"),
            os.path.join(mask_dir, f"{stem}_{side}_mask.jpeg"),
            os.path.join(mask_dir, f"{stem}_mask.png"),
            os.path.join(mask_dir, f"{stem}_mask.jpg"),
            os.path.join(mask_dir, f"{stem}_mask.jpeg"),
        ])

        if not roi_path or not mask_path or pd.isna(row[side]):
            missing = []
            if not roi_path:
                missing.append("ROI")
            if not mask_path:
                missing.append("mask")
            if pd.isna(row[side]):
                missing.append("label")
            errors.append(f"{stem_raw} ({side}): missing {', '.join(missing)}")
            continue
        try:
            label = float(row[side])
        except (TypeError, ValueError):
            label = float("nan")
        if not math.isfinite(label) or not label.is_integer() or not 0 <= label <= 4:
            errors.append(f"{stem_raw} ({side}): expected integer grade 0-4, got {row[side]!r}")
            continue
        if roi_path and mask_path:
            rows.append({
                "filename": stem,
                "roi_path": roi_path,
                "mask_path": mask_path,
                "label": int(label),
            })

    if errors:
        raise ValueError(
            f"Invalid {side} inputs ({len(errors)} rows); no samples were silently skipped.\n"
            + "\n".join(errors[:10])
        )
    return pd.DataFrame(rows)


# ========================= 注意力模块 =========================
class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // r), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // r), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sig(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


class CBAMBlock(nn.Module):
    def __init__(self, channels, r=16, spatial_kernel=7):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        mid = max(1, channels // r)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

        self.sconv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        ca = self.mlp(self.avg(x)) + self.mlp(self.max(x))
        x = x * self.sig(ca)
        s = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        x = x * self.sig(self.sconv(s))
        return x


def build_feat_plugin(kind: str, channels: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "se":
        return SEBlock(channels)
    if kind == "eca":
        return ECABlock(channels, 3)
    if kind == "cbam":
        return CBAMBlock(channels, 16, 7)
    return nn.Identity()


class BlurPool(nn.Module):
    def __init__(self, channels, filt_size=3, stride=2):
        super().__init__()
        assert filt_size in (3, 5)
        if filt_size == 3:
            a = torch.tensor([1., 2., 1.])
        else:
            a = torch.tensor([1., 4., 6., 4., 1.])
        k = (a[:, None] * a[None, :])
        k = k / k.sum()
        self.register_buffer("kernel", k[None, None, :, :].repeat(channels, 1, 1, 1))
        self.stride = stride
        self.pad = (filt_size - 1) // 2
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.pad, groups=self.groups)


# ========================= Dataset =========================
class DualFusionBandDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: A.Compose, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.tf = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]

        roi = Image.open(r["roi_path"]).convert("L")
        mask = Image.open(r["mask_path"]).convert("L")

        roi_np = np.array(roi)
        mask_np = np.array(mask)

        if roi_np.shape != mask_np.shape:
            mask_np = np.array(
                mask.resize((roi_np.shape[1], roi_np.shape[0]), resample=Image.NEAREST)
            )

        drop_sdt_flag = False
        drop_band_flag = False

        if self.is_train:
            mask_np = jitter_mask_np(mask_np)

            if random.random() < cfg.mask_dropout_p:
                mode = random.choice(["zero_mask", "thin_mask", "drop_sdt", "drop_band"])
                if mode == "zero_mask":
                    mask_np = np.zeros_like(mask_np, dtype=np.uint8)
                elif mode == "thin_mask":
                    m = (mask_np > 0).astype(np.uint8)
                    for _ in range(2):
                        m = _morph_np_step(m, "erode")
                    mask_np = (m > 0).astype(np.uint8) * 255
                elif mode == "drop_sdt":
                    drop_sdt_flag = True
                else:
                    drop_band_flag = True

        sdt = signed_distance_u8(mask_np)
        band = build_band_u8(mask_np)

        if drop_sdt_flag:
            sdt = np.zeros_like(sdt, dtype=np.uint8)
        if drop_band_flag:
            band = np.zeros_like(band, dtype=np.uint8)

        four = np.stack([roi_np, mask_np, sdt, band], axis=-1)
        t = self.tf(image=four)["image"]

        x_roi1 = t[0:1, ...]
        x_mask3 = t[1:4, ...]
        y = int(r["label"])

        return x_roi1, x_mask3, y


# ========================= 模型 =========================
class ResNet18_1ch(nn.Module):
    def __init__(self, imagenet_init=True):
        super().__init__()
        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_init else None
        )

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if imagenet_init:
            w = base.conv1.weight.data.mean(dim=1, keepdim=True)
            self.conv1.weight.data = w.clone()
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.blur = BlurPool(64, cfg.blur_filt_size, 2) if cfg.use_blurpool else None

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.ta1 = TripletAttention()
        self.ta2 = TripletAttention()
        self.ta3 = TripletAttention()

        self.p1 = build_feat_plugin(cfg.feat_plugin, 64)
        self.p2 = build_feat_plugin(cfg.feat_plugin, 128)
        self.p3 = build_feat_plugin(cfg.feat_plugin, 256)
        self.p4 = build_feat_plugin(cfg.feat_plugin, 512)

        self.avgpool = base.avgpool

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.blur(x) if self.blur is not None else self.maxpool(x)

        x = self.ta1(self.layer1(x)); x = self.p1(x)
        x = self.ta2(self.layer2(x)); x = self.p2(x)
        x = self.ta3(self.layer3(x)); x = self.p3(x)
        x = self.layer4(x); x = self.p4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNet18_3ch(nn.Module):
    def __init__(self, imagenet_init=True):
        super().__init__()
        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_init else None
        )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if imagenet_init:
            self.conv1.weight.data = base.conv1.weight.data.clone()
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.blur = BlurPool(64, cfg.blur_filt_size, 2) if cfg.use_blurpool else None

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.ta1 = TripletAttention()
        self.ta2 = TripletAttention()
        self.ta3 = TripletAttention()

        self.p1 = build_feat_plugin(cfg.feat_plugin, 64)
        self.p2 = build_feat_plugin(cfg.feat_plugin, 128)
        self.p3 = build_feat_plugin(cfg.feat_plugin, 256)
        self.p4 = build_feat_plugin(cfg.feat_plugin, 512)

        self.avgpool = base.avgpool

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.blur(x) if self.blur is not None else self.maxpool(x)

        x = self.ta1(self.layer1(x)); x = self.p1(x)
        x = self.ta2(self.layer2(x)); x = self.p2(x)
        x = self.ta3(self.layer3(x)); x = self.p3(x)
        x = self.layer4(x); x = self.p4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class DualFuseResNet18Band(nn.Module):
    def __init__(self, num_classes=5, imagenet_init=True):
        super().__init__()
        self.num_classes = int(num_classes)

        self.branch_roi1 = ResNet18_1ch(imagenet_init=imagenet_init)
        self.branch_msk3 = ResNet18_3ch(imagenet_init=imagenet_init)

        if cfg.replace_backbone_relu:
            replace_relu_(self.branch_roi1, cfg.backbone_act)
            replace_relu_(self.branch_msk3, cfg.backbone_act)

        self.proj_a = nn.Linear(512, 512)
        self.proj_b = nn.Linear(512, 512)

        self.use_xattn = cfg.use_xattn
        if self.use_xattn:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=512,
                nhead=cfg.xattn_heads,
                dim_feedforward=cfg.xattn_ffn,
                dropout=cfg.xattn_dropout,
                activation=cfg.xattn_activation,
                batch_first=True,
                norm_first=True,
            )
            self.xattn = nn.TransformerEncoder(enc_layer, num_layers=cfg.xattn_layers)

        act_fuse = make_act(cfg.act_fuse)
        act_head = make_act(cfg.act_head)

        self.gate = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, cfg.gate_hidden),
            act_fuse,
            nn.Dropout(cfg.gate_drop),
            nn.Linear(cfg.gate_hidden, 512),
            nn.Sigmoid(),
        )

        self.fuse_norm = nn.LayerNorm(512)

        if cfg.head_type == "ordinal":
            self.head_is_ordinal = True
            self.cls = nn.Linear(512, self.num_classes - 1)
        else:
            self.head_is_ordinal = False
            self.cls = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, 512),
                act_head,
                nn.Dropout(0.35),
                nn.Linear(512, self.num_classes),
            )

        self.aux12 = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            act_head,
            nn.Dropout(0.25),
            nn.Linear(256, 2),
        )

        for m in [self.proj_a, self.proj_b]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_roi1, x_mask3):
        fa = self.branch_roi1(x_roi1)
        fb = self.branch_msk3(x_mask3)

        fa_p = self.proj_a(fa)
        fb_p = self.proj_b(fb)

        if self.use_xattn:
            tokens = torch.stack([fa_p, fb_p], dim=1)
            tokens = self.xattn(tokens)
            fa_p, fb_p = tokens[:, 0], tokens[:, 1]

        g = self.gate(torch.cat([fa_p, fb_p], dim=1))
        fused = g * fa_p + (1.0 - g) * fb_p
        fused = self.fuse_norm(fused)

        logits_main = self.cls(fused)
        logits_aux12 = self.aux12(fused)
        return logits_main, logits_aux12


# ========================= 损失 =========================
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        num_classes: int = 5,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits, target):
        with torch.no_grad():
            true = torch.zeros((target.size(0), self.num_classes), device=logits.device)
            true.fill_(self.label_smoothing / (self.num_classes - 1))
            true.scatter_(1, target.view(-1, 1), 1.0 - self.label_smoothing)

        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        ce = -(true * logp).sum(dim=1)
        pt = (true * p).sum(dim=1).clamp_min(1e-6)
        loss = ((1 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            loss = self.alpha[target] * loss

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def load_shadow(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])


def save_ema_state_dict(model: nn.Module, ema: EMA, path: str, backup: dict):
    ema.load_shadow(model)
    torch.save(model.state_dict(), path)
    print(f" ↳ [EMA] 已保存: {path}")

    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(backup[n])


def _reset_bn_stats(module: nn.Module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean.zero_()
        module.running_var.fill_(1)
        module.num_batches_tracked.zero_()


@torch.no_grad()
def update_bn_dualfuse(loader: DataLoader, model: nn.Module, device: torch.device):
    was_training = model.training
    model.train()
    model.apply(_reset_bn_stats)

    for x_roi1, x_msk3, _ in loader:
        x_roi1 = x_roi1.to(device, memory_format=torch.channels_last)
        x_msk3 = x_msk3.to(device, memory_format=torch.channels_last)
        _ = model(x_roi1, x_msk3)

    model.train(was_training)


# ========================= 采样与权重 =========================
def compute_class_stats(df: pd.DataFrame, num_classes: int):
    labels = df["label"].values.tolist()
    counts = np.zeros(num_classes, dtype=np.int64)
    for y in labels:
        counts[y] += 1

    counts = np.maximum(counts, 1)
    sample_weights = np.array([1.0 / counts[y] for y in labels], dtype=np.float32)
    class_weights = counts.sum() / (counts.astype(np.float32) * num_classes)
    return sample_weights, torch.tensor(class_weights, dtype=torch.float32)


def cb_alpha_from_counts(counts: np.ndarray, beta: float, K: int) -> torch.Tensor:
    eff = 1.0 - np.power(beta, counts)
    alpha = (1.0 - beta) / np.clip(eff, 1e-8, None)
    alpha = alpha / alpha.sum() * K
    return torch.tensor(alpha, dtype=torch.float32, device=device)


# ========================= MixUp / CutMix =========================
def _rand_bbox(W: int, H: int, lam: float):
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))

    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_mix_batch(x_roi1: torch.Tensor, x_msk3: torch.Tensor, y: torch.Tensor):
    B, _, H, W = x_roi1.shape
    if B < 2:
        return x_roi1, x_msk3, y, None, 1.0, None

    r = random.random()

    if r < cfg.mixup_p:
        lam = np.random.beta(cfg.mix_alpha, cfg.mix_alpha)
        perm = torch.randperm(B, device=x_roi1.device)
        xr = lam * x_roi1 + (1 - lam) * x_roi1[perm]
        xm = lam * x_msk3 + (1 - lam) * x_msk3[perm]
        return xr, xm, y, y[perm], float(lam), "mixup"

    if r < cfg.mixup_p + cfg.cutmix_p:
        lam = np.random.beta(cfg.mix_alpha, cfg.mix_alpha)
        perm = torch.randperm(B, device=x_roi1.device)
        x1, y1, x2, y2 = _rand_bbox(W, H, lam)

        xr = x_roi1.clone()
        xm = x_msk3.clone()

        xr[:, :, y1:y2, x1:x2] = x_roi1[perm, :, y1:y2, x1:x2]
        xm[:, :, y1:y2, x1:x2] = x_msk3[perm, :, y1:y2, x1:x2]

        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
        return xr, xm, y, y[perm], float(lam), "cutmix"

    return x_roi1, x_msk3, y, None, 1.0, None


# ========================= eval =========================
def _unpack_main_logits(out):
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def eval_loader(model: nn.Module, dl: DataLoader, device: torch.device, tta: bool = False):
    model.eval()

    correct = 0
    total = 0
    preds_all, gts_all, probs_all = [], [], []

    for x_roi1, x_msk3, labels in dl:
        x_roi1 = x_roi1.to(device, memory_format=torch.channels_last)
        x_msk3 = x_msk3.to(device, memory_format=torch.channels_last)
        labels = labels.to(device).long()

        if not tta:
            logits = _unpack_main_logits(model(x_roi1, x_msk3))
        else:
            out1 = model(x_roi1, x_msk3)
            out2 = model(torch.flip(x_roi1, dims=[3]), torch.flip(x_msk3, dims=[3]))
            logits = (_unpack_main_logits(out1) + _unpack_main_logits(out2)) / 2.0

        if cfg.head_type == "ordinal":
            ordinal_prob = torch.sigmoid(logits)
            pred = (ordinal_prob > 0.5).sum(dim=1)
            probs_all.extend(ordinal_prob.cpu().numpy().tolist())
        else:
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            probs_all.extend(prob.cpu().numpy().tolist())

        correct += (pred == labels).sum().item()
        total += labels.size(0)

        preds_all.extend(pred.cpu().tolist())
        gts_all.extend(labels.cpu().tolist())

    acc = correct / max(1, total)
    return acc, preds_all, gts_all, probs_all


# ========================= 训练 =========================
def _seed_worker(worker_id: int):
    worker_seed = cfg.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_side(side: str, cfg: Config):
    assert side in ["left", "right"]
    check_validation_mask_source(cfg.val_mask_dir)

    df_tr = pd.read_csv(cfg.train_labels_csv)
    df_va = pd.read_csv(cfg.val_labels_csv)

    tr_tbl = build_side_table(df_tr, side, cfg.train_roi_dir, cfg.train_mask_dir)
    va_tbl = build_side_table(df_va, side, cfg.val_roi_dir, cfg.val_mask_dir)

    print(f"\n===== [{side}] 数据量: train={len(tr_tbl)} val={len(va_tbl)} =====")
    print(
        f"[{side}] 类分布 | train={_bincount_safe(tr_tbl['label'].tolist(), cfg.num_classes)} "
        f"val={_bincount_safe(va_tbl['label'].tolist(), cfg.num_classes)}"
    )

    if len(tr_tbl) == 0 or len(va_tbl) == 0:
        print(f"[{side}] 数据为空，跳过。")
        return

    tr_tf, ev_tf = get_transforms(cfg.img_size)

    ds_tr = DualFusionBandDataset(tr_tbl, transform=tr_tf, is_train=True)
    ds_va = DualFusionBandDataset(va_tbl, transform=ev_tf, is_train=False)
    ds_bn = DualFusionBandDataset(tr_tbl, transform=ev_tf, is_train=False)

    sample_w, class_w = compute_class_stats(tr_tbl, cfg.num_classes)

    dl_tr_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )

    if cfg.use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w),
            num_samples=len(sample_w),
            replacement=True,
        )
        dl_tr = DataLoader(ds_tr, sampler=sampler, shuffle=False, **dl_tr_kwargs)
        alpha_for_loss = None
    else:
        dl_tr = DataLoader(ds_tr, shuffle=True, **dl_tr_kwargs)
        alpha_for_loss = class_w.to(device)

    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    dl_bn = DataLoader(
        ds_bn,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = DualFuseResNet18Band(num_classes=cfg.num_classes, imagenet_init=True).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if cfg.use_cb_focal:
        counts = np.bincount(tr_tbl["label"].values, minlength=cfg.num_classes)
        alpha_for_loss = cb_alpha_from_counts(counts, beta=cfg.cb_beta, K=cfg.num_classes)
        if cfg.use_weighted_sampler:
            warnings.warn("同时使用 CB-Focal 与 WeightedSampler，可能双重加权。")

    if cfg.head_type == "ordinal":
        def ordinal_loss(logits, target):
            B = target.size(0)
            Km1 = logits.size(1)
            thr = torch.arange(Km1, device=logits.device).unsqueeze(0).expand(B, -1)
            T = (target.unsqueeze(1) > thr).float()
            return F.binary_cross_entropy_with_logits(logits, T)
        criterion = ordinal_loss
    else:
        if cfg.use_focal:
            criterion = FocalLoss(
                alpha=alpha_for_loss,
                gamma=cfg.focal_gamma,
                label_smoothing=cfg.label_smooth,
                num_classes=cfg.num_classes,
            )
        else:
            criterion = lambda logits, target: F.cross_entropy(
                logits,
                target,
                weight=alpha_for_loss,
                label_smoothing=cfg.label_smooth,
            )

    aux_criterion = FocalLoss(gamma=2.0, num_classes=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, (cfg.num_epochs - cfg.warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = build_grad_scaler()

    ema = EMA(model, decay=cfg.ema_decay)

    swa_model = None
    swa_scheduler = None
    if cfg.use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg.lr * 0.5)

    # Save the first evaluated model even when validation accuracy is zero.
    best_acc = -1.0

    ckpt_best = os.path.join(cfg.model_dir, f"dualfuse_band_{side}_best.pth")
    ckpt_final = os.path.join(cfg.model_dir, f"dualfuse_band_{side}_final.pth")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        n_seen = 0

        for x_roi1, x_msk3, labels in dl_tr:
            x_roi1 = x_roi1.to(device, non_blocking=True)
            x_msk3 = x_msk3.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            if cfg.channels_last:
                x_roi1 = x_roi1.to(memory_format=torch.channels_last)
                x_msk3 = x_msk3.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                x_roi1_m, x_msk3_m, y_main, y2, lam, _ = apply_mix_batch(x_roi1, x_msk3, labels)
                logits, logits_aux12 = model(x_roi1_m, x_msk3_m)

                aux12_w = cfg.aux12_w
                if cfg.warmup_epochs > 0:
                    aux12_w = cfg.aux12_w * min(1.0, epoch / float(cfg.warmup_epochs))

                if y2 is None:
                    loss = criterion(logits, y_main)

                    if aux12_w > 0.0:
                        mask_12 = (y_main == 1) | (y_main == 2)
                        if mask_12.any():
                            y12 = torch.where(
                                y_main[mask_12] == 1,
                                torch.zeros_like(y_main[mask_12]),
                                torch.ones_like(y_main[mask_12]),
                            )
                            aux_loss = aux_criterion(logits_aux12[mask_12], y12)
                            loss = loss + aux12_w * aux_loss
                else:
                    loss = lam * criterion(logits, y_main) + (1.0 - lam) * criterion(logits, y2)

                    if aux12_w > 0.0:
                        parts = []

                        mask_12_a = (y_main == 1) | (y_main == 2)
                        if mask_12_a.any():
                            ya = torch.where(
                                y_main[mask_12_a] == 1,
                                torch.zeros_like(y_main[mask_12_a]),
                                torch.ones_like(y_main[mask_12_a]),
                            )
                            parts.append(lam * aux_criterion(logits_aux12[mask_12_a], ya))

                        mask_12_b = (y2 == 1) | (y2 == 2)
                        if mask_12_b.any():
                            yb = torch.where(
                                y2[mask_12_b] == 1,
                                torch.zeros_like(y2[mask_12_b]),
                                torch.ones_like(y2[mask_12_b]),
                            )
                            parts.append((1.0 - lam) * aux_criterion(logits_aux12[mask_12_b], yb))

                        if len(parts) > 0:
                            loss = loss + aux12_w * torch.stack(parts).sum()

                if cfg.consis_w > 0:
                    out_flip = model(
                        torch.flip(x_roi1_m, dims=[3]),
                        torch.flip(x_msk3_m, dims=[3]),
                    )
                    logits_flip = _unpack_main_logits(out_flip)

                    if cfg.head_type == "ordinal":
                        p = torch.sigmoid(logits / cfg.consis_T)
                        ql = torch.log(torch.sigmoid(logits_flip / cfg.consis_T).clamp_min(1e-6))
                        consis = F.kl_div(ql, p, reduction="batchmean") * (cfg.consis_T ** 2)
                    else:
                        p = F.softmax(logits / cfg.consis_T, dim=1)
                        logq = F.log_softmax(logits_flip / cfg.consis_T, dim=1)
                        consis = F.kl_div(logq, p, reduction="batchmean") * (cfg.consis_T ** 2)

                    loss = loss + cfg.consis_w * consis

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            ema.update(model)

            bs = labels.size(0)
            tr_loss_sum += loss.item() * bs
            n_seen += bs

        if cfg.use_swa and epoch >= cfg.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        tr_loss = tr_loss_sum / max(1, n_seen)

        backup = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        ema.load_shadow(model)
        va_acc, va_preds, va_gts, va_probs = eval_loader(model, dl_va, device, tta=cfg.use_tta_eval)

        if epoch % 10 == 0 or epoch == 1:
            cm = confusion_matrix(va_gts, va_preds, labels=list(range(cfg.num_classes)))
            np.savetxt(
                os.path.join(cm_dir, f"cm_resnet18dualfuse_band_{side}_e{epoch:03d}.csv"),
                cm,
                fmt="%d",
                delimiter=",",
            )
            print(f"[{side}] Ep{epoch:03d} | Val CM saved.")

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{side}] Ep{epoch:03d} | lr={lr_now:.2e} | tr_loss={tr_loss:.4f} | val_acc={va_acc:.4f}")

        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[n])

        if va_acc > best_acc:
            best_acc = va_acc
            save_ema_state_dict(model, ema, ckpt_best, backup)
            print(f" ↳ [{side}] best 更新: val_acc={va_acc:.4f}")

    torch.save(model.state_dict(), ckpt_final)
    print(f"[{side}] Final 已保存: {ckpt_final}")

    # SWA parameters are maintained during training; only EMA best is used for testing.

    state = torch.load(ckpt_best, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"[{side}] reload best | missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    va_acc, va_preds, va_gts, va_probs = eval_loader(model, dl_va, device, tta=cfg.use_tta_eval)
    allow_auc = (cfg.head_type == "ce")

    metrics = compute_metrics_once(
        va_gts, va_preds, va_probs, cfg.num_classes, allow_auc=allow_auc
    )
    ci = bootstrap_ci(
        va_gts, va_preds, va_probs, cfg.num_classes,
        n_boot=300, seed=cfg.seed + 7, allow_auc=allow_auc
    )

    print_eval_table(f"[{side}] VAL Metrics", metrics, ci)

    rep = classification_report(
        va_gts, va_preds, labels=list(range(cfg.num_classes)), digits=4
    )
    with open(os.path.join(cm_dir, f"report_resnet18dualfuse_band_{side}_val.txt"), "w", encoding="utf-8") as f:
        f.write(rep)


# ========================= main =========================
def main():
    global cm_dir
    parser = argparse.ArgumentParser(description="Original SIGNet grading training recipe")
    parser.add_argument("--data-root", required=True, help="Contains train/ and val/ with roi, roi_masks and labels.csv")
    parser.add_argument("--output-dir", default="outputs/grading_weights")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    for split in ("train", "val"):
        setattr(cfg, f"{split}_roi_dir", str(data_root / split / "roi"))
        setattr(cfg, f"{split}_mask_dir", str(data_root / split / "roi_masks"))
        setattr(cfg, f"{split}_labels_csv", str(data_root / split / "labels.csv"))
    # Validate both sides before spending time training either network.
    check_validation_mask_source(cfg.val_mask_dir)
    for split in ("train", "val"):
        labels = pd.read_csv(getattr(cfg, f"{split}_labels_csv"))
        for side in ("left", "right"):
            build_side_table(labels, side, getattr(cfg, f"{split}_roi_dir"),
                             getattr(cfg, f"{split}_mask_dir"))
    cfg.model_dir = args.output_dir
    cfg.out_dir_eval = str(Path(args.output_dir) / "evaluation")
    cm_dir = str(Path(args.output_dir) / "cm_resnet18dualfuse_band")
    for directory in (cfg.model_dir, cfg.out_dir_eval, cm_dir):
        os.makedirs(directory, exist_ok=True)
    print("Device:", device)
    print(
        f"[CFG] feat_plugin={cfg.feat_plugin} | blurpool={cfg.use_blurpool}({cfg.blur_filt_size}) "
        f"| xattn={cfg.use_xattn} | band_mode={cfg.band_mode}"
    )

    train_one_side("left", cfg)
    train_one_side("right", cfg)

    print("\n✅ 训练完成")
    print("✅ best/final 权重目录:", cfg.model_dir)
    print("✅ 混淆矩阵目录:", cm_dir)


if __name__ == "__main__":
    main()
