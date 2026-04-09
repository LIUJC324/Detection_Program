from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator

from data.preprocess import stack_modalities
from model.network.backbone import build_feature_extractor
from model.network.head import SmallObjectRefineHead
from model.network.neck import LightweightBiFPN


class RGBTBackboneWithNeck(nn.Module):
    def __init__(
        self,
        backbone_channels: Optional[List[int]] = None,
        fpn_out_channels: int = 128,
        backbone_name: str = "lightweight",
        pretrained_backbone_path: str = "",
    ):
        super().__init__()
        backbone_channels = backbone_channels or [32, 64, 96]
        self.feature_extractor = build_feature_extractor(
            backbone_name=backbone_name,
            channels=backbone_channels,
            pretrained_backbone_path=pretrained_backbone_path or None,
        )
        stage_channels = getattr(self.feature_extractor, "out_channels_per_stage", backbone_channels)
        self.needs_fcos_scale_alignment = bool(getattr(self.feature_extractor, "needs_fcos_scale_alignment", False))
        self.neck = LightweightBiFPN(stage_channels, fpn_out_channels)
        self.refine_head = SmallObjectRefineHead(fpn_out_channels)
        self.out_channels = fpn_out_channels

    @staticmethod
    def _align_fcos_feature_scales(features: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        # FCOS matching and box coding assume a detection pyramid similar to strides 8/16/32.
        # The lightweight RGB-T backbone emits finer maps (roughly strides 2/4/8), which leads
        # to dense point-like responses and poor regression. Downsampling the refined pyramid by
        # 4x keeps the learned feature channels intact while restoring the expected scale layout.
        return OrderedDict(
            {
                key: F.max_pool2d(feature, kernel_size=4, stride=4)
                for key, feature in features.items()
            }
        )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        pyramid = self.neck(features)
        refined = self.refine_head(pyramid)
        if self.needs_fcos_scale_alignment:
            return self._align_fcos_feature_scales(refined)
        return refined


class RGBTDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = 640,
        backbone_channels: Optional[List[int]] = None,
        fpn_out_channels: int = 128,
        backbone_name: str = "lightweight",
        pretrained_backbone_path: str = "",
        score_thresh: float = 0.2,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
    ) -> None:
        super().__init__()
        backbone = RGBTBackboneWithNeck(
            backbone_channels,
            fpn_out_channels,
            backbone_name=backbone_name,
            pretrained_backbone_path=pretrained_backbone_path,
        )
        anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,)),
            aspect_ratios=((1.0,), (1.0,), (1.0,)),
        )
        self.detector = FCOS(
            backbone=backbone,
            num_classes=num_classes,
            min_size=input_size,
            max_size=input_size,
            image_mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],
            image_std=[0.229, 0.224, 0.225, 0.25, 0.25, 0.25],
            anchor_generator=anchor_generator,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )

    def forward(
        self,
        rgb_images: List[torch.Tensor],
        thermal_images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        if len(rgb_images) != len(thermal_images):
            raise ValueError("RGB and thermal batches must have identical length.")
        images = [stack_modalities(rgb, thermal) for rgb, thermal in zip(rgb_images, thermal_images)]
        return self.detector(images, targets)


class SingleBatchExportWrapper(nn.Module):
    def __init__(self, model: RGBTDetector):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        rgb = [item[:3] for item in x]
        thermal = [item[3:] for item in x]
        outputs = self.model(rgb, thermal)
        max_detections = self.model.detector.detections_per_img
        boxes = []
        scores = []
        labels = []
        for output in outputs:
            padded_boxes = torch.zeros((max_detections, 4), device=x.device, dtype=output["boxes"].dtype)
            padded_scores = torch.zeros((max_detections,), device=x.device, dtype=output["scores"].dtype)
            padded_labels = torch.full((max_detections,), -1, device=x.device, dtype=output["labels"].dtype)
            count = min(output["boxes"].shape[0], max_detections)
            padded_boxes[:count] = output["boxes"][:count]
            padded_scores[:count] = output["scores"][:count]
            padded_labels[:count] = output["labels"][:count]
            boxes.append(padded_boxes)
            scores.append(padded_scores)
            labels.append(padded_labels)
        return torch.stack(boxes), torch.stack(scores), torch.stack(labels)


def build_model(config: Dict) -> RGBTDetector:
    model_cfg = config.get("model", config)
    return RGBTDetector(
        num_classes=model_cfg["num_classes"],
        input_size=model_cfg.get("input_size", 640),
        backbone_channels=model_cfg.get("backbone_channels", [32, 64, 96]),
        fpn_out_channels=model_cfg.get("fpn_out_channels", 128),
        backbone_name=model_cfg.get("backbone_name", "lightweight"),
        pretrained_backbone_path=model_cfg.get("pretrained_backbone_path", ""),
        score_thresh=model_cfg.get("score_thresh", 0.2),
        nms_thresh=model_cfg.get("nms_thresh", 0.5),
        detections_per_img=model_cfg.get("detections_per_img", 100),
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    scaler,
    epoch: int,
    config: Dict,
    path: str | Path,
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
    }
    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = "cpu",
) -> Dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
