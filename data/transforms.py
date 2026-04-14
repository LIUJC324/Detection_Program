from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List

import cv2
import numpy as np
import torch

from .preprocess import image_to_tensor, letterbox_resize_pair, resize_pair


Sample = Dict[str, object]


class Compose:
    def __init__(self, transforms: List[Callable[[Sample], Sample]]):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        rgb = np.ascontiguousarray(sample["rgb"][:, ::-1])
        thermal = np.ascontiguousarray(sample["thermal"][:, ::-1])
        boxes = sample["targets"]["boxes"].copy()
        width = rgb.shape[1]
        if boxes.size > 0:
            x1 = width - boxes[:, 2]
            x2 = width - boxes[:, 0]
            boxes[:, 0], boxes[:, 2] = x1, x2
        sample["rgb"] = rgb
        sample["thermal"] = thermal
        sample["targets"]["boxes"] = boxes
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.0):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        rgb = np.ascontiguousarray(sample["rgb"][::-1, :])
        thermal = np.ascontiguousarray(sample["thermal"][::-1, :])
        boxes = sample["targets"]["boxes"].copy()
        height = rgb.shape[0]
        if boxes.size > 0:
            y1 = height - boxes[:, 3]
            y2 = height - boxes[:, 1]
            boxes[:, 1], boxes[:, 3] = y1, y2
        sample["rgb"] = rgb
        sample["thermal"] = thermal
        sample["targets"]["boxes"] = boxes
        return sample


class RandomCrop:
    def __init__(self, p: float = 0.3, min_scale: float = 0.7):
        self.p = p
        self.min_scale = min_scale

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        rgb = sample["rgb"]
        thermal = sample["thermal"]
        boxes = sample["targets"]["boxes"].copy()
        labels = sample["targets"]["labels"].copy()
        height, width = rgb.shape[:2]
        crop_h = random.randint(int(height * self.min_scale), height)
        crop_w = random.randint(int(width * self.min_scale), width)
        top = random.randint(0, height - crop_h) if height > crop_h else 0
        left = random.randint(0, width - crop_w) if width > crop_w else 0

        rgb = rgb[top : top + crop_h, left : left + crop_w]
        thermal = thermal[top : top + crop_h, left : left + crop_w]
        if boxes.size > 0:
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, crop_w - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, crop_h - 1)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]
        sample["rgb"] = np.ascontiguousarray(rgb)
        sample["thermal"] = np.ascontiguousarray(thermal)
        sample["targets"]["boxes"] = boxes.reshape(-1, 4)
        sample["targets"]["labels"] = labels
        return sample


class Resize:
    def __init__(self, size: int, mode: str = "stretch", pad_value: int = 114):
        self.size = size
        self.mode = mode
        self.pad_value = pad_value

    def __call__(self, sample: Sample) -> Sample:
        original_size = sample["targets"]["orig_size"]
        if self.mode == "none":
            rgb = np.ascontiguousarray(sample["rgb"])
            thermal = np.ascontiguousarray(sample["thermal"])
            boxes = sample["targets"]["boxes"].copy().astype(np.float32).reshape(-1, 4)
            resize_meta = None
        elif self.mode == "letterbox":
            rgb, thermal, boxes, resize_meta = letterbox_resize_pair(
                sample["rgb"],
                sample["thermal"],
                sample["targets"]["boxes"],
                (self.size, self.size),
                pad_value=self.pad_value,
            )
        else:
            rgb, thermal, boxes, resize_meta = resize_pair(
                sample["rgb"],
                sample["thermal"],
                sample["targets"]["boxes"],
                (self.size, self.size),
                return_meta=True,
            )
        sample["rgb"] = rgb
        sample["thermal"] = thermal
        sample["targets"]["boxes"] = boxes
        sample["targets"]["orig_size"] = original_size
        sample["targets"]["resized_size"] = np.asarray([rgb.shape[0], rgb.shape[1]], dtype=np.int64)
        if resize_meta is not None:
            sample["targets"]["resize_meta"] = resize_meta
        return sample


class ColorJitterRGB:
    def __init__(
        self,
        p: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.15,
    ):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        rgb = sample["rgb"].astype(np.float32) / 255.0
        brightness = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation = 1.0 + random.uniform(-self.saturation, self.saturation)
        rgb = np.clip(rgb * brightness, 0.0, 1.0)
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray = np.repeat(gray[..., None], 3, axis=2)
        rgb = np.clip((rgb - 0.5) * contrast + 0.5, 0.0, 1.0)
        rgb = np.clip(gray + (rgb - gray) * saturation, 0.0, 1.0)
        sample["rgb"] = (rgb * 255.0).astype(np.uint8)
        return sample


class RandomLowLightRGB:
    def __init__(
        self,
        p: float = 0.0,
        gamma_min: float = 1.4,
        gamma_max: float = 2.2,
    ):
        self.p = p
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        rgb = sample["rgb"].astype(np.float32) / 255.0
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)
        sample["rgb"] = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        return sample


class RandomWeakModality:
    def __init__(
        self,
        p: float = 0.0,
        rgb_primary_prob: float = 0.7,
        min_scale: float = 0.05,
        max_scale: float = 0.35,
        blur_prob: float = 0.5,
        noise_std: float = 0.03,
    ):
        self.p = p
        self.rgb_primary_prob = rgb_primary_prob
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.blur_prob = blur_prob
        self.noise_std = noise_std

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        key = "rgb" if random.random() < self.rgb_primary_prob else "thermal"
        image = sample[key].astype(np.float32) / 255.0
        image = image * random.uniform(self.min_scale, self.max_scale)
        if random.random() < self.blur_prob:
            image = cv2.GaussianBlur(image, (5, 5), 0)
        if self.noise_std > 0.0:
            noise = np.random.normal(0.0, self.noise_std, image.shape).astype(np.float32)
            image = image + noise
        sample[key] = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        return sample


class RandomMotionBlurPair:
    def __init__(self, p: float = 0.0, kernel_sizes: tuple[int, ...] = (3, 5, 7)):
        self.p = p
        self.kernel_sizes = tuple(size for size in kernel_sizes if size >= 3 and size % 2 == 1) or (3, 5, 7)

    @staticmethod
    def _build_kernel(size: int, mode: str) -> np.ndarray:
        kernel = np.zeros((size, size), dtype=np.float32)
        if mode == "horizontal":
            kernel[size // 2, :] = 1.0
        elif mode == "vertical":
            kernel[:, size // 2] = 1.0
        elif mode == "diag_down":
            np.fill_diagonal(kernel, 1.0)
        else:
            np.fill_diagonal(np.fliplr(kernel), 1.0)
        kernel /= max(kernel.sum(), 1.0)
        return kernel

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        size = random.choice(self.kernel_sizes)
        mode = random.choice(("horizontal", "vertical", "diag_down", "diag_up"))
        kernel = self._build_kernel(size, mode)
        sample["rgb"] = cv2.filter2D(sample["rgb"], -1, kernel)
        sample["thermal"] = cv2.filter2D(sample["thermal"], -1, kernel)
        return sample


class ToTensor:
    def __call__(self, sample: Sample) -> Sample:
        sample["rgb"] = image_to_tensor(sample["rgb"])
        sample["thermal"] = image_to_tensor(sample["thermal"])
        boxes = sample["targets"]["boxes"]
        labels = sample["targets"]["labels"]
        sample["targets"]["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        sample["targets"]["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        sample["targets"]["image_id"] = torch.as_tensor(sample["targets"]["image_id"], dtype=torch.int64)
        sample["targets"]["orig_size"] = torch.as_tensor(sample["targets"]["orig_size"], dtype=torch.int64)
        if "resized_size" in sample["targets"]:
            sample["targets"]["resized_size"] = torch.as_tensor(sample["targets"]["resized_size"], dtype=torch.int64)
        return sample


@dataclass
class TransformConfig:
    image_size: int = 640
    resize_mode: str = "stretch"
    letterbox_pad_value: int = 114
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    random_crop_prob: float = 0.3
    crop_min_scale: float = 0.7
    color_jitter_prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.15
    lowlight_aug_prob: float = 0.0
    lowlight_gamma_min: float = 1.4
    lowlight_gamma_max: float = 2.2
    weak_modality_prob: float = 0.0
    weak_rgb_primary_prob: float = 0.7
    weak_modality_min_scale: float = 0.05
    weak_modality_max_scale: float = 0.35
    weak_modality_blur_prob: float = 0.5
    weak_modality_noise_std: float = 0.03
    motion_blur_prob: float = 0.0
    motion_blur_kernel_sizes: tuple[int, ...] = (3, 5, 7)


def build_train_transforms(config: TransformConfig) -> Compose:
    return Compose(
        [
            RandomHorizontalFlip(config.horizontal_flip_prob),
            RandomVerticalFlip(config.vertical_flip_prob),
            RandomCrop(config.random_crop_prob, config.crop_min_scale),
            Resize(config.image_size, mode=config.resize_mode, pad_value=config.letterbox_pad_value),
            ColorJitterRGB(
                p=config.color_jitter_prob,
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
            ),
            RandomLowLightRGB(
                p=config.lowlight_aug_prob,
                gamma_min=config.lowlight_gamma_min,
                gamma_max=config.lowlight_gamma_max,
            ),
            RandomWeakModality(
                p=config.weak_modality_prob,
                rgb_primary_prob=config.weak_rgb_primary_prob,
                min_scale=config.weak_modality_min_scale,
                max_scale=config.weak_modality_max_scale,
                blur_prob=config.weak_modality_blur_prob,
                noise_std=config.weak_modality_noise_std,
            ),
            RandomMotionBlurPair(
                p=config.motion_blur_prob,
                kernel_sizes=config.motion_blur_kernel_sizes,
            ),
            ToTensor(),
        ]
    )


def build_val_transforms(config: TransformConfig) -> Compose:
    return Compose(
        [
            Resize(config.image_size, mode=config.resize_mode, pad_value=config.letterbox_pad_value),
            ToTensor(),
        ]
    )
