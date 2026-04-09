from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class SmallObjectRefineHead(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.blocks = nn.ModuleDict(
            {
                "0": nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ),
                "1": nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ),
                "2": nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ),
            }
        )

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        refined = OrderedDict()
        for key, feat in features.items():
            refined[key] = self.blocks[key](feat)
        return refined

