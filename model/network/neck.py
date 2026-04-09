from __future__ import annotations

from collections import OrderedDict
from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class RefinementBlock(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )


class LightweightBiFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList([ConvBlock(ch, out_channels) for ch in in_channels])
        self.top_down = nn.ModuleList([RefinementBlock(out_channels) for _ in in_channels])
        self.bottom_up = nn.ModuleList([RefinementBlock(out_channels) for _ in in_channels])
        self.out_channels = out_channels

    def forward(self, features: List[torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        p3, p4, p5 = [layer(feature) for layer, feature in zip(self.lateral, features)]
        p4 = self.top_down[1](p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
        p3 = self.top_down[0](p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest"))
        p4 = self.bottom_up[1](p4 + F.max_pool2d(p3, kernel_size=2, stride=2))
        p5 = self.bottom_up[2](p5 + F.max_pool2d(p4, kernel_size=2, stride=2))
        return OrderedDict({"0": p3, "1": p4, "2": p5})

