from __future__ import annotations

import re

import torch
from torch import nn


class WeakModalityDropout(nn.Module):
    """Mildly degrades one modality during training so the fused model does not overfit a single branch."""

    def __init__(self, p: float = 0.25, min_scale: float = 0.35, max_scale: float = 0.75, noise_std: float = 0.02):
        super().__init__()
        self.p = float(p)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.noise_std = float(noise_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x

        batch_size = x.shape[0]
        device = x.device
        out = x.clone()

        apply_mask = (torch.rand(batch_size, 1, 1, 1, device=device) < self.p).to(dtype=out.dtype)
        rgb_selected = (torch.rand(batch_size, 1, 1, 1, device=device) < 0.5).to(dtype=out.dtype)
        thermal_selected = 1.0 - rgb_selected
        scales = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(self.min_scale, self.max_scale)

        rgb_factor = 1.0 - apply_mask * rgb_selected * (1.0 - scales)
        thermal_factor = 1.0 - apply_mask * thermal_selected * (1.0 - scales)
        out[:, :3] = out[:, :3] * rgb_factor
        out[:, 3:] = out[:, 3:] * thermal_factor

        if self.noise_std > 0.0:
            rgb_noise = torch.randn_like(out[:, :3]) * self.noise_std
            thermal_noise = torch.randn_like(out[:, 3:]) * self.noise_std
            out[:, :3] = out[:, :3] + rgb_noise * apply_mask * rgb_selected
            out[:, 3:] = out[:, 3:] + thermal_noise * apply_mask * thermal_selected

        return out.clamp_(0.0, 1.0)


class ReliabilityAwareStemGate(nn.Module):
    """Learn RGB/Thermal reliability weights before the first stem convolution."""

    def __init__(self, hidden_channels: int = 8):
        super().__init__()
        hidden = max(2, int(hidden_channels))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(6, hidden, kernel_size=1, stride=1, padding=0)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc2(self.act(self.fc1(self.pool(x))))
        gates = 0.5 + torch.sigmoid(logits)
        rgb_gate = gates[:, 0:1]
        thermal_gate = gates[:, 1:2]
        rgb = x[:, :3] * rgb_gate
        thermal = x[:, 3:] * thermal_gate
        return torch.cat([rgb, thermal], dim=1)


class SmallTargetStemBlock(nn.Module):
    """A lightweight residual detail enhancer placed before the first spatial downsampling."""

    def __init__(self, channels: int = 6, expansion: int = 2, res_scale: float = 0.5):
        super().__init__()
        c = int(channels)
        hidden = max(c, c * int(expansion))
        self.dw = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False)
        self.dw_bn = nn.BatchNorm2d(c)
        self.pw1 = nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw1_bn = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw2_bn = nn.BatchNorm2d(c)
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw_bn(self.dw(x))
        y = self.act(self.pw1_bn(self.pw1(y)))
        y = self.pw2_bn(self.pw2(y))
        return x + y * self.res_scale


def register_rgbt6_yolo_modules() -> None:
    import ultralytics.nn.tasks as tasks

    tasks.WeakModalityDropout = WeakModalityDropout
    tasks.ReliabilityAwareStemGate = ReliabilityAwareStemGate
    tasks.SmallTargetStemBlock = SmallTargetStemBlock


def remap_model_key_for_stem_shift(key: str, index_shift: int) -> str | None:
    if index_shift <= 0:
        return key
    match = re.match(r"model\.(\d+)(\..+)$", key)
    if not match:
        return key
    index = int(match.group(1))
    if index < index_shift:
        return None
    return f"model.{index - index_shift}{match.group(2)}"
