"""Loss functions for sacroiliac structure segmentation."""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment.model import main_logits


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)[:, 1]
        truth = target.float()
        tp = (prob * truth).sum(dim=(1, 2))
        fp = (prob * (1.0 - truth)).sum(dim=(1, 2))
        fn = ((1.0 - prob) * truth).sum(dim=(1, 2))
        score = tp / (tp + self.alpha * fn + self.beta * fp + 1e-6)
        return ((1.0 - score) ** self.gamma).mean()


class RegionLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.25, tversky_weight: float = 0.75) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.register_buffer("class_weights", torch.tensor([1.0, 1.2], dtype=torch.float32))
        self.tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long(), weight=self.class_weights)
        return self.ce_weight * ce + self.tversky_weight * self.tversky(logits, target)


class EdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _edge(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        return grad / (grad.amax(dim=(2, 3), keepdim=True) + 1e-6)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE is unsafe under CUDA autocast.  Compute the complete edge term in
        # fp32 while leaving the surrounding training step under BF16 autocast.
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            prob = torch.softmax(logits.float(), dim=1)[:, 1:2]
            truth = target.float().unsqueeze(1)
            return F.binary_cross_entropy(
                self._edge(prob).clamp(0, 1), self._edge(truth).clamp(0, 1)
            )


def outside_penalty(logits: torch.Tensor, rectangle: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1]
    return (prob * (1.0 - rectangle.float()).clamp(0, 1)).mean()


class SegmentationObjective(nn.Module):
    """Combine full-image, rectangular-view, and perturbed-ROI losses."""

    def __init__(
        self,
        rect_probability: float = 0.5,
        roi_probability: float = 0.5,
        rect_weight: float = 0.6,
        roi_weight: float = 0.6,
        edge_weight: float = 0.12,
        outside_weight: float = 0.20,
    ) -> None:
        super().__init__()
        self.region = RegionLoss()
        self.edge = EdgeLoss()
        self.rect_probability = rect_probability
        self.roi_probability = roi_probability
        self.rect_weight = rect_weight
        self.roi_weight = roi_weight
        self.edge_weight = edge_weight
        self.outside_weight = outside_weight

    def forward(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target: torch.Tensor,
        rectangle: torch.Tensor,
        rect_image: Optional[torch.Tensor] = None,
        rect_target: Optional[torch.Tensor] = None,
        rect_rectangle: Optional[torch.Tensor] = None,
        roi_image: Optional[torch.Tensor] = None,
        roi_target: Optional[torch.Tensor] = None,
        roi_rectangle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The experiment loss applies the region and edge terms to every
        # sampled view, while the outside penalty is accumulated across the
        # full image and whichever auxiliary views are active.
        logits = main_logits(model(image))
        loss = self.region(logits, target) + self.edge_weight * self.edge(logits, target)
        outside = outside_penalty(logits, rectangle)

        if rect_image is not None and rect_target is not None and random.random() < self.rect_probability:
            # 以 0.5 概率启用 14 像素扩展矩形视图。
            rect_logits = main_logits(model(rect_image))
            rect_loss = self.region(rect_logits, rect_target) + self.edge_weight * self.edge(rect_logits, rect_target)
            loss = loss + self.rect_weight * rect_loss
            if rect_rectangle is not None:
                outside = outside + outside_penalty(rect_logits, rect_rectangle)

        if roi_image is not None and roi_target is not None and random.random() < self.roi_probability:
            # 以 0.5 概率启用随机扩展并抖动的 ROI 视图。
            roi_logits = main_logits(model(roi_image))
            roi_loss = self.region(roi_logits, roi_target) + self.edge_weight * self.edge(roi_logits, roi_target)
            loss = loss + self.roi_weight * roi_loss
            if roi_rectangle is not None:
                outside = outside + outside_penalty(roi_logits, roi_rectangle)

        return loss + self.outside_weight * outside
