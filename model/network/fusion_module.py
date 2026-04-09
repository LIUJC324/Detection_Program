from __future__ import annotations

import torch
from torch import nn


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.rgb_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.thermal_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.attn = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feat_rgb: torch.Tensor, feat_thermal: torch.Tensor) -> torch.Tensor:
        rgb = self.rgb_proj(feat_rgb)
        thermal = self.thermal_proj(feat_thermal)
        fused_context = torch.cat([rgb, thermal, torch.abs(rgb - thermal)], dim=1)
        attention = self.attn(fused_context)
        fused = attention * rgb + (1.0 - attention) * thermal
        return self.out_proj(fused)

