from __future__ import annotations

from typing import Dict, List

import torch


def count_small_objects(targets: List[Dict[str, torch.Tensor]], small_object_area: float = 1024.0) -> float:
    total = 0
    small = 0
    for target in targets:
        boxes = target["boxes"]
        if boxes.numel() == 0:
            continue
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        total += areas.numel()
        small += (areas <= small_object_area).sum().item()
    if total == 0:
        return 0.0
    return small / total


class SmallObjectLossAggregator:
    def __init__(
        self,
        small_object_area: float = 1024.0,
        small_object_boost: float = 0.4,
        regression_weight: float = 1.0,
        centerness_weight: float = 1.0,
    ) -> None:
        self.small_object_area = small_object_area
        self.small_object_boost = small_object_boost
        self.regression_weight = regression_weight
        self.centerness_weight = centerness_weight

    def __call__(self, loss_dict: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if not loss_dict:
            return torch.tensor(0.0)
        small_ratio = count_small_objects(targets, self.small_object_area)
        cls_weight = 1.0 + self.small_object_boost * small_ratio
        cls_loss = loss_dict.get("classification", 0.0)
        reg_loss = loss_dict.get("bbox_regression", 0.0)
        ctr_loss = loss_dict.get("bbox_ctrness", 0.0)
        if not torch.is_tensor(cls_loss):
            cls_loss = torch.tensor(float(cls_loss))
        if not torch.is_tensor(reg_loss):
            reg_loss = torch.tensor(float(reg_loss), device=cls_loss.device)
        if not torch.is_tensor(ctr_loss):
            ctr_loss = torch.tensor(float(ctr_loss), device=cls_loss.device)
        return cls_weight * cls_loss + self.regression_weight * reg_loss + self.centerness_weight * ctr_loss

