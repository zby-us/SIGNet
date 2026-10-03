"""Triplet Attention 特征增强模块。"""

import torch
import torch.nn as nn


class TripletAttention(nn.Module):
    """在 H-W、C-H 和 C-W 三个方向复用同一个空间注意力门。

    三个分支共享卷积与 BN 参数，输出取平均。
    """

    def __init__(self, no_spatial=False):
        super().__init__()
        self.no_spatial = no_spatial
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(1)

    def _spatial_attention(self, x):
        pooled = torch.cat(
            (torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]),
            dim=1,
        )
        weight = torch.sigmoid(self.bn1(self.conv1(pooled)))
        return x * weight

    def forward(self, x):
        spatial = self._spatial_attention(x)
        channel_height = self._spatial_attention(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        channel_width = self._spatial_attention(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        if self.no_spatial:
            return 0.5 * (channel_height + channel_width)
        return (spatial + channel_height + channel_width) / 3.0
