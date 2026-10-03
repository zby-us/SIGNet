"""Direction-aware U-Net used by the four segmentation models.

The two auxiliary heads are retained because they are present in the original
training checkpoints.  The experiment loss and inference pipeline use the
first (main) output only.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# 基础卷积模块

def conv3x3(in_ch, out_ch, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)

def conv1x1(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for tensors shaped (B, C, H, W)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps    = eps

    def forward(self, x):
        mu  = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x   = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

class DropPath(nn.Module):
    """Per-sample stochastic depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if (self.drop_prob == 0.0) or (not self.training):
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


# 水平与垂直方向上的轻量级上下文混合模块

class VMamba2D(nn.Module):
    """Preserve (B, C, H, W) while mixing horizontal and vertical context.

    A pointwise projection creates content and gate branches.  Depthwise
    convolution extracts local context, directional average pooling supplies
    horizontal and vertical context, and learned scalar maps weight the two
    directions before gated residual projection.
    """
    def __init__(self, dim, expand=2.0, d_conv=5, drop_path=0.0):
        super().__init__()
        inner = int(dim * expand)

        self.norm = LayerNorm2d(dim)
        self.proj_in = conv1x1(dim, inner * 2, bias=True)  # content and gate branches
        self.dwconv  = nn.Conv2d(inner, inner, kernel_size=d_conv,
                                 padding=d_conv // 2, groups=inner, bias=True)
        self.act     = nn.SiLU()
        self.alpha_h = nn.Parameter(torch.zeros(1, inner, 1, 1))
        self.alpha_w = nn.Parameter(torch.zeros(1, inner, 1, 1))
        self.proj_out = conv1x1(inner, dim, bias=True)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        uv = self.proj_in(x)
        u, v = torch.chunk(uv, 2, dim=1)  # [B,inner,H,W] x2

        # Local spatial context.
        u = self.dwconv(u)

        # 分别聚合水平和垂直方向的局部上下文。
        h_win = 7
        w_win = 7
        s_h = F.avg_pool2d(u, kernel_size=(1, h_win), stride=1, padding=(0, h_win // 2))
        s_w = F.avg_pool2d(u, kernel_size=(w_win, 1), stride=1, padding=(w_win // 2, 0))
        s   = torch.sigmoid(self.alpha_h) * s_h + torch.sigmoid(self.alpha_w) * s_w

        y = s * torch.sigmoid(v)
        y = self.act(y)
        y = self.proj_out(y)
        return identity + self.drop_path(y)


# U-Net 编码器和解码器模块

class DoubleConv(nn.Module):
    """(Conv3x3-BN-SiLU) x2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            conv3x3(out_ch, out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """Downsample, apply two convolutions, and mix directional context."""
    def __init__(self, in_ch, out_ch, dp=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
        self.vm   = VMamba2D(out_ch, expand=2.0, d_conv=5, drop_path=dp)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.vm(x)
        return x

class Up(nn.Module):
    """Upsample a decoder tensor, concatenate its skip, and refine it."""

    def __init__(self, decoder_ch, skip_ch, out_ch, dp=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(decoder_ch + skip_ch, out_ch)
        self.vm   = VMamba2D(out_ch, expand=2.0, d_conv=5, drop_path=dp)

    def forward(self, x, skip):
        x = self.up(x)
        # Align odd input sizes without changing the skip tensor.
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY or diffX:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        x = self.vm(x)
        return x

class OutHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.proj = conv1x1(in_ch, num_classes, bias=True)
    def forward(self, x):
        return self.proj(x)


# 方向感知 U-Net

class UnetVMamba(nn.Module):
    """四层 U-Net，并在每个尺度加入方向感知模块。"""
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        drop_path_rate: float = 0.10,
    ):
        super().__init__()
        ch1 = base_channels
        ch2 = ch1 * 2
        ch3 = ch2 * 2
        ch4 = ch3 * 2
        ch5 = ch4 * 2

        # Increase stochastic depth gradually in deeper blocks.
        dps = torch.linspace(0.0, drop_path_rate, steps=10).tolist()

        # 编码器：逐级提取局部和方向上下文特征。
        self.enc1_conv = DoubleConv(in_channels, ch1)
        self.enc1_vm   = VMamba2D(ch1, expand=2.0, d_conv=5, drop_path=dps[0])

        self.down2 = Down(ch1, ch2, dp=dps[2])
        self.down3 = Down(ch2, ch3, dp=dps[4])
        self.down4 = Down(ch3, ch4, dp=dps[6])

        # 瓶颈层与原始论文模型保持在 H/8 x W/8；此处不再下采样。
        self.bott_conv = DoubleConv(ch4, ch5)
        self.bott_vm   = VMamba2D(ch5, expand=2.0, d_conv=5, drop_path=dps[7])

        # 解码器：上采样后与同尺度编码特征拼接。
        self.up4 = Up(ch5, ch4, ch4, dp=dps[7])
        self.up3 = Up(ch4, ch3, ch3, dp=dps[6])
        self.up2 = Up(ch3, ch2, ch2, dp=dps[5])
        self.up1 = Up(ch2, ch1, ch1, dp=dps[4])

        # 使用 1x1 卷积将解码特征映射为分割类别。
        self.head_main = OutHead(ch1, num_classes)
        self.head_aux2 = OutHead(ch2, num_classes)
        self.head_aux3 = OutHead(ch3, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        height, width = x.shape[-2:]
        # 编码路径。
        e1 = self.enc1_vm(self.enc1_conv(x))   # [B,ch1,H,W]
        e2 = self.down2(e1)                    # [B,ch2,H/2,W/2]
        e3 = self.down3(e2)                    # [B,ch3,H/4,W/4]
        e4 = self.down4(e3)                    # [B,ch4,H/8,W/8]

        # 瓶颈层。原始训练网络在这里没有第四次池化。
        b = self.bott_vm(self.bott_conv(e4))            # [B,ch5,H/8,W/8]

        # 解码路径。
        d4 = self.up4(b,  e4)                  # [B,ch4,H/8,W/8]
        d3 = self.up3(d4, e3)                  # [B,ch3,H/4,W/4]
        d2 = self.up2(d3, e2)                  # [B,ch2,H/2,W/2]
        d1 = self.up1(d2, e1)                  # [B,ch1,H,  W]

        # Auxiliary heads are checkpoint-compatible but are not supervised by
        # the released training recipe.  Keep their historical return order.
        out_main = self.head_main(d1)
        out_aux2 = F.interpolate(
            self.head_aux2(d2), size=(height, width), mode="bilinear", align_corners=True
        )
        out_aux3 = F.interpolate(
            self.head_aux3(d3), size=(height, width), mode="bilinear", align_corners=True
        )
        return out_main, out_aux2, out_aux3


def main_logits(output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    """Return the primary segmentation logits from either supported API."""
    return output[0] if isinstance(output, (tuple, list)) else output
