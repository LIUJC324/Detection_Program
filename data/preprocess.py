from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def read_image(path: str | Path, mode: str = "rgb") -> np.ndarray:
    image = Image.open(path)
    if mode == "rgb":
        image = image.convert("RGB")
    elif mode == "thermal":
        image = image.convert("L")
        image = Image.merge("RGB", (image, image, image))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return np.array(image, copy=True)


def decode_image_bytes(data: bytes, mode: str = "rgb") -> np.ndarray:
    image = Image.open(io.BytesIO(data))
    if mode == "rgb":
        image = image.convert("RGB")
    elif mode == "thermal":
        image = image.convert("L")
        image = Image.merge("RGB", (image, image, image))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return np.array(image, copy=True)


def ensure_same_size(rgb: np.ndarray, thermal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] == thermal.shape[:2]:
        return rgb, thermal
    resized = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return rgb, resized


def clip_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep]


def build_resize_meta(
    src_h: int,
    src_w: int,
    target_h: int,
    target_w: int,
    scale_x: float,
    scale_y: float,
    pad_left: int = 0,
    pad_top: int = 0,
) -> Dict[str, float]:
    resized_h = max(1, min(target_h, int(round(src_h * scale_y))))
    resized_w = max(1, min(target_w, int(round(src_w * scale_x))))
    return {
        "orig_height": int(src_h),
        "orig_width": int(src_w),
        "target_height": int(target_h),
        "target_width": int(target_w),
        "resized_height": int(resized_h),
        "resized_width": int(resized_w),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
    }


def resize_pair(
    rgb: np.ndarray,
    thermal: np.ndarray,
    boxes: np.ndarray,
    size: Tuple[int, int],
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    target_h, target_w = size
    src_h, src_w = rgb.shape[:2]
    rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    thermal_resized = cv2.resize(thermal, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resize_meta = build_resize_meta(
        src_h=src_h,
        src_w=src_w,
        target_h=target_h,
        target_w=target_w,
        scale_x=target_w / max(src_w, 1),
        scale_y=target_h / max(src_h, 1),
    )
    if boxes.size == 0:
        empty_boxes = boxes.reshape(0, 4)
        if return_meta:
            return rgb_resized, thermal_resized, empty_boxes, resize_meta
        return rgb_resized, thermal_resized, empty_boxes
    scale_x = resize_meta["scale_x"]
    scale_y = resize_meta["scale_y"]
    resized_boxes = boxes.copy().astype(np.float32)
    resized_boxes[:, [0, 2]] *= scale_x
    resized_boxes[:, [1, 3]] *= scale_y
    if return_meta:
        return rgb_resized, thermal_resized, resized_boxes, resize_meta
    return rgb_resized, thermal_resized, resized_boxes


def letterbox_resize_pair(
    rgb: np.ndarray,
    thermal: np.ndarray,
    boxes: np.ndarray,
    size: Tuple[int, int],
    pad_value: int = 114,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    target_h, target_w = size
    src_h, src_w = rgb.shape[:2]
    scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
    resized_w = max(1, min(target_w, int(round(src_w * scale))))
    resized_h = max(1, min(target_h, int(round(src_h * scale))))

    rgb_resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    thermal_resized = cv2.resize(thermal, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    border_value = (pad_value, pad_value, pad_value)
    rgb_letterboxed = cv2.copyMakeBorder(
        rgb_resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )
    thermal_letterboxed = cv2.copyMakeBorder(
        thermal_resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )

    resize_meta = build_resize_meta(
        src_h=src_h,
        src_w=src_w,
        target_h=target_h,
        target_w=target_w,
        scale_x=resized_w / max(src_w, 1),
        scale_y=resized_h / max(src_h, 1),
        pad_left=pad_left,
        pad_top=pad_top,
    )

    if boxes.size == 0:
        return rgb_letterboxed, thermal_letterboxed, boxes.reshape(0, 4), resize_meta

    resized_boxes = boxes.copy().astype(np.float32)
    resized_boxes[:, [0, 2]] *= resize_meta["scale_x"]
    resized_boxes[:, [1, 3]] *= resize_meta["scale_y"]
    resized_boxes[:, [0, 2]] += pad_left
    resized_boxes[:, [1, 3]] += pad_top
    resized_boxes = clip_boxes(resized_boxes, target_h, target_w)
    return rgb_letterboxed, thermal_letterboxed, resized_boxes, resize_meta


def crop_pair(
    rgb: np.ndarray,
    thermal: np.ndarray,
    boxes: np.ndarray,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cropped_rgb = rgb[top : top + crop_h, left : left + crop_w]
    cropped_thermal = thermal[top : top + crop_h, left : left + crop_w]
    if boxes.size == 0:
        return cropped_rgb, cropped_thermal, boxes.reshape(0, 4)
    cropped_boxes = boxes.copy().astype(np.float32)
    cropped_boxes[:, [0, 2]] -= left
    cropped_boxes[:, [1, 3]] -= top
    cropped_boxes = clip_boxes(cropped_boxes, crop_h, crop_w)
    return cropped_rgb, cropped_thermal, cropped_boxes


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous().float() / 255.0
    return tensor


def restore_boxes_to_original_size(
    boxes: torch.Tensor,
    resize_meta: Dict[str, float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        empty_keep = torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
        return boxes.reshape(0, 4), empty_keep
    if resize_meta is None:
        keep = torch.ones((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
        return boxes, keep

    restored = boxes.clone().to(dtype=torch.float32)
    restored[:, [0, 2]] -= float(resize_meta.get("pad_left", 0))
    restored[:, [1, 3]] -= float(resize_meta.get("pad_top", 0))
    restored[:, [0, 2]] /= max(float(resize_meta.get("scale_x", 1.0)), 1e-6)
    restored[:, [1, 3]] /= max(float(resize_meta.get("scale_y", 1.0)), 1e-6)
    restored[:, [0, 2]] = restored[:, [0, 2]].clamp(0, max(int(resize_meta.get("orig_width", 1)) - 1, 0))
    restored[:, [1, 3]] = restored[:, [1, 3]].clamp(0, max(int(resize_meta.get("orig_height", 1)) - 1, 0))
    keep = (restored[:, 2] > restored[:, 0]) & (restored[:, 3] > restored[:, 1])
    return restored[keep].to(dtype=boxes.dtype), keep


def normalize_tensor(tensor: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    mean_tensor = torch.as_tensor(list(mean), dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.as_tensor(list(std), dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def stack_modalities(rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
    if rgb.shape[-2:] != thermal.shape[-2:]:
        raise ValueError("RGB and thermal tensors must share spatial shape before stacking.")
    return torch.cat([rgb, thermal], dim=0)
