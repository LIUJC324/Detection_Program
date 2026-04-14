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


class ReliabilityAwareFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.rgb_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.thermal_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.spatial_quality = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
        )
        self.channel_quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1),
        )
        self.conflict_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conflict_residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feat_rgb: torch.Tensor, feat_thermal: torch.Tensor) -> torch.Tensor:
        rgb = self.rgb_proj(feat_rgb)
        thermal = self.thermal_proj(feat_thermal)
        conflict = torch.abs(rgb - thermal)
        fused_context = torch.cat([rgb, thermal, conflict], dim=1)

        batch_size, channels = rgb.shape[:2]
        spatial_logits = self.spatial_quality(fused_context).unsqueeze(2)
        channel_logits = self.channel_quality(fused_context).view(batch_size, 2, channels, 1, 1)
        modality_weights = torch.softmax(spatial_logits + channel_logits, dim=1)
        fused = modality_weights[:, 0] * rgb + modality_weights[:, 1] * thermal

        fused = fused + self.conflict_residual(conflict * self.conflict_gate(conflict))
        return self.out_proj(fused)


def build_fusion_module(channels: int, fusion_type: str = "cross_attention") -> nn.Module:
    normalized_name = str(fusion_type or "cross_attention").strip().lower()
    if normalized_name in {"cross_attention", "attention", "baseline"}:
        return CrossModalAttentionFusion(channels)
    if normalized_name in {"reliability_aware", "quality_gated", "conflict_aware"}:
        return ReliabilityAwareFusion(channels)
    raise ValueError(f"Unsupported fusion module: {fusion_type}")
