from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from torchvision.models import resnet18

from .fusion_module import build_fusion_module


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.use_skip = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_skip:
            out = out + x
        return out


class LightweightBranch(nn.Module):
    def __init__(self, in_channels: int = 3, channels: List[int] | None = None):
        super().__init__()
        channels = channels or [32, 64, 96]
        c1, c2, c3 = channels
        self.stem = ConvBNAct(in_channels, c1, kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(c1, c1, stride=1),
            DepthwiseSeparableBlock(c1, c1, stride=1),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(c1, c2, stride=2),
            DepthwiseSeparableBlock(c2, c2, stride=1),
            DepthwiseSeparableBlock(c2, c2, stride=1),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(c2, c3, stride=2),
            DepthwiseSeparableBlock(c3, c3, stride=1),
            DepthwiseSeparableBlock(c3, c3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]


def _extract_state_dict(payload: dict) -> dict:
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def _clean_state_dict_prefixes(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        normalized = key
        for prefix in ("module.", "backbone.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        cleaned[normalized] = value
    return cleaned


class ResNet18Branch(nn.Module):
    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        backbone = resnet18(weights=None)
        if checkpoint_path:
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(f"Pretrained backbone checkpoint not found: {checkpoint_path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = _clean_state_dict_prefixes(_extract_state_dict(payload))
            backbone.load_state_dict(state_dict, strict=False)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]


class RGBTFeatureExtractor(nn.Module):
    def __init__(self, channels: List[int] | None = None, fusion_type: str = "cross_attention"):
        super().__init__()
        channels = channels or [32, 64, 96]
        self.out_channels_per_stage = channels
        self.needs_fcos_scale_alignment = True
        self.rgb_branch = LightweightBranch(3, channels)
        self.thermal_branch = LightweightBranch(3, channels)
        self.fusions = nn.ModuleList([build_fusion_module(c, fusion_type=fusion_type) for c in channels])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[1] != 6:
            raise ValueError(f"Expected a 6-channel tensor, got {x.shape}.")
        rgb = x[:, :3]
        thermal = x[:, 3:]
        rgb_feats = self.rgb_branch(rgb)
        thermal_feats = self.thermal_branch(thermal)
        fused = [fusion(r, t) for fusion, r, t in zip(self.fusions, rgb_feats, thermal_feats)]
        return fused


class RGBTResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained_backbone_path: Optional[str] = None, fusion_type: str = "cross_attention"):
        super().__init__()
        channels = [128, 256, 512]
        self.out_channels_per_stage = channels
        self.needs_fcos_scale_alignment = False
        self.rgb_branch = ResNet18Branch(pretrained_backbone_path)
        self.thermal_branch = ResNet18Branch(pretrained_backbone_path)
        self.fusions = nn.ModuleList([build_fusion_module(c, fusion_type=fusion_type) for c in channels])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[1] != 6:
            raise ValueError(f"Expected a 6-channel tensor, got {x.shape}.")
        rgb = x[:, :3]
        thermal = x[:, 3:]
        rgb_feats = self.rgb_branch(rgb)
        thermal_feats = self.thermal_branch(thermal)
        return [fusion(r, t) for fusion, r, t in zip(self.fusions, rgb_feats, thermal_feats)]


def build_feature_extractor(
    backbone_name: str = "lightweight",
    channels: Optional[List[int]] = None,
    pretrained_backbone_path: Optional[str] = None,
    fusion_type: str = "cross_attention",
) -> nn.Module:
    normalized_name = str(backbone_name or "lightweight").strip().lower()
    if normalized_name == "resnet18_twin":
        return RGBTResNet18FeatureExtractor(
            pretrained_backbone_path=pretrained_backbone_path or None,
            fusion_type=fusion_type,
        )
    return RGBTFeatureExtractor(channels, fusion_type=fusion_type)
