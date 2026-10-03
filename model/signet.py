"""SIGNet 双分支骶髂炎分级网络。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .attention import TripletAttention


def normalize_grading_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map saved checkpoint names to the current module names.

    The mapping changes names only.  Strict loading after this conversion still
    verifies that every parameter and tensor shape matches the released model.
    """
    prefixes = (
        ("branch_roi1.", "roi_branch."),
        ("branch_msk3.", "prior_branch."),
        ("backbone.", "roi_branch."),
        ("proj_a.", "roi_projection."),
        ("proj_b.", "prior_projection."),
        ("xattn.", "interaction."),
        ("fuse_norm.", "fusion_norm."),
        ("cls.", "classifier."),
        ("aux12.", "auxiliary_1_vs_2."),
    )
    components = (
        (".blur.", ".blurpool."),
        (".ta1.", ".triplet1."),
        (".ta2.", ".triplet2."),
        (".ta3.", ".triplet3."),
        (".p1.", ".eca1."),
        (".p2.", ".eca2."),
        (".p3.", ".eca3."),
        (".p4.", ".eca4."),
    )
    normalized: dict[str, torch.Tensor] = {}
    for original_key, value in state.items():
        key = original_key[7:] if original_key.startswith("module.") else original_key
        for source, target in prefixes:
            if key.startswith(source):
                key = target + key[len(source):]
                break
        for source, target in components:
            key = key.replace(source, target)
        normalized[key] = value
    return normalized


@dataclass(frozen=True)
class GradingModelConfig:
    num_classes: int = 5
    embedding_dim: int = 512
    transformer_layers: int = 1
    transformer_heads: int = 8
    transformer_ffn: int = 1024
    transformer_dropout: float = 0.1
    gate_hidden: int = 512
    gate_dropout: float = 0.1
    classifier_dropout: float = 0.35
    auxiliary_hidden: int = 256
    auxiliary_dropout: float = 0.25
    # Training can enable ImageNet initialization. Inference uses the complete
    # SIGNet checkpoint and does not download backbone weights.
    imagenet_init: bool = False


class ECABlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.pool(x).squeeze(-1).transpose(-1, -2)
        weight = torch.sigmoid(self.conv(weight).transpose(-1, -2).unsqueeze(-1))
        return x * weight


class BlurPool(nn.Module):
    def __init__(self, channels: int, stride: int = 2) -> None:
        super().__init__()
        coeff = torch.tensor([1.0, 2.0, 1.0])
        kernel = coeff[:, None] * coeff[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=x.shape[1])


class ResNet18Branch(nn.Module):
    def __init__(self, input_channels: int, imagenet_init: bool = False) -> None:
        super().__init__()
        if input_channels not in (1, 3):
            raise ValueError("ResNet18Branch supports one or three input channels")
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_init else None
        base = models.resnet18(weights=weights)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if imagenet_init:
            pretrained = base.conv1.weight.detach()
            self.conv1.weight.data.copy_(
                pretrained.mean(dim=1, keepdim=True) if input_channels == 1 else pretrained
            )
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.bn1 = base.bn1
        self.relu = base.relu
        self.blurpool = BlurPool(64)
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.triplet1, self.triplet2, self.triplet3 = TripletAttention(), TripletAttention(), TripletAttention()
        self.eca1, self.eca2 = ECABlock(64), ECABlock(128)
        self.eca3, self.eca4 = ECABlock(256), ECABlock(512)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 以 BlurPool 替代原始最大池化，降低下采样造成的位移敏感性。
        x = self.blurpool(self.relu(self.bn1(self.conv1(x))))
        # 前三个残差阶段同时使用 Triplet Attention 和 ECA，第四阶段仅使用 ECA。
        x = self.eca1(self.triplet1(self.layer1(x)))
        x = self.eca2(self.triplet2(self.layer2(x)))
        x = self.eca3(self.triplet3(self.layer3(x)))
        x = self.eca4(self.layer4(x))
        return self.pool(x).flatten(1)


class DualBranchGrader(nn.Module):
    def __init__(self, config: GradingModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or GradingModelConfig()
        dim = self.config.embedding_dim
        # 两个结构相同但参数独立的 ResNet18 分别处理灰度 ROI 和三通道结构先验。
        self.roi_branch = ResNet18Branch(1, self.config.imagenet_init)
        self.prior_branch = ResNet18Branch(3, self.config.imagenet_init)
        self.roi_projection = nn.Linear(512, dim)
        self.prior_projection = nn.Linear(512, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=self.config.transformer_heads,
            dim_feedforward=self.config.transformer_ffn,
            dropout=self.config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(encoder_layer, self.config.transformer_layers)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, self.config.gate_hidden),
            nn.SiLU(),
            nn.Dropout(self.config.gate_dropout),
            nn.Linear(self.config.gate_hidden, dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(dim, self.config.num_classes),
        )
        self.auxiliary_1_vs_2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.config.auxiliary_hidden),
            nn.GELU(),
            nn.Dropout(self.config.auxiliary_dropout),
            nn.Linear(self.config.auxiliary_hidden, 2),
        )

    def forward(self, roi: torch.Tensor, structural_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        roi_feature = self.roi_projection(self.roi_branch(roi))
        prior_feature = self.prior_projection(self.prior_branch(structural_prior))

        # 将两路 512 维特征视为两个 token，通过单层 Transformer 双向交互。
        tokens = self.interaction(torch.stack([roi_feature, prior_feature], dim=1))
        roi_feature, prior_feature = tokens[:, 0], tokens[:, 1]

        # 通过逐特征门控融合 ROI 和结构先验特征。
        gate = self.gate(torch.cat([roi_feature, prior_feature], dim=1))
        fused = self.fusion_norm(gate * roi_feature + (1.0 - gate) * prior_feature)

        # 主分类头预测 0-4 级，辅助头仅用于加强 1 级与 2 级的区分。
        return self.classifier(fused), self.auxiliary_1_vs_2(fused)

