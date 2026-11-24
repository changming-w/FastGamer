"""Global feature encoder (Sec. 3.1).

A small strided-convolution encoder that turns the 256x256 SDR thumbnail
``I_thumb`` into a single global feature vector ``F_thumb``. Every image-adaptive
operator (scale head, bilateral-grid generator, 3D-LUT generator, neural
modulator) is conditioned on this vector, so the heavy neural computation is
confined to the low-resolution thumbnail (the high-resolution path is
network-free).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """3x3 strided conv + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class GlobalEncoder(nn.Module):
    """Thumbnail -> global feature vector F_thumb.

    Args:
        in_nc:    number of input channels (3, RGB).
        widths:   channel widths of the strided stages.
        feat_dim: dimensionality of the output global feature F_thumb.
    """

    def __init__(self, in_nc: int = 3, widths=(16, 32, 48, 64, 64), feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        chs = [in_nc, *widths]
        self.blocks = nn.ModuleList(
            [ConvBlock(chs[i], chs[i + 1], stride=2) for i in range(len(widths))]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(widths[-1], feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, thumb: torch.Tensor) -> torch.Tensor:
        x = thumb
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).flatten(1)          # (B, widths[-1])
        return self.proj(x)                  # (B, feat_dim)
