from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml

from data.preprocess import (
    decode_image_bytes,
    ensure_same_size,
    image_to_tensor,
    letterbox_resize_pair,
    resize_pair,
    restore_boxes_to_original_size,
)
from model.detector import build_model, load_checkpoint
from service.utils.inference_engine import TorchInferenceEngine


def merge_nested_dict(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


class Predictor:
    def __init__(
        self,
        model,
        class_mapping: Dict[str, str],
        device: torch.device,
        requested_device: str,
        cuda_available: bool,
        result_min_confidence: float = 0.0,
        result_merge_distance: float = 0.0,
        result_merge_degenerate_size: float = 2.0,
        result_duplicate_iou: float = 0.0,
        result_containment_ratio: float = 0.0,
        result_duplicate_center_distance: float = 0.0,
        result_box_expand_ratio: float = 0.0,
        result_box_expand_min_pixels: float = 0.0,
        result_box_pad_left: float = 0.0,
        result_box_pad_top: float = 0.0,
        result_box_pad_right: float = 0.0,
        result_box_pad_bottom: float = 0.0,
        model_version: str = "rgbt_detector_v1",
        input_size: int = 640,
        preprocess_mode: str = "stretch",
        letterbox_pad_value: int = 114,
        enable_rgb_lowlight_enhance: bool = False,
        rgb_lowlight_luma_threshold: float = 70.0,
        rgb_lowlight_gamma: float = 1.15,
        rgb_lowlight_clahe_clip_limit: float = 2.0,
        rgb_lowlight_confidence_scale: float = 0.75,
    ):
        self.model = model
        self.class_mapping = class_mapping
        self.device = device
        self.requested_device = requested_device
        self.cuda_available = cuda_available
        self.result_min_confidence = result_min_confidence
        self.result_merge_distance = result_merge_distance
        self.result_merge_degenerate_size = result_merge_degenerate_size
        self.result_duplicate_iou = result_duplicate_iou
        self.result_containment_ratio = result_containment_ratio
        self.result_duplicate_center_distance = result_duplicate_center_distance
        self.result_box_expand_ratio = result_box_expand_ratio
        self.result_box_expand_min_pixels = result_box_expand_min_pixels
        self.result_box_pad_left = result_box_pad_left
        self.result_box_pad_top = result_box_pad_top
        self.result_box_pad_right = result_box_pad_right
        self.result_box_pad_bottom = result_box_pad_bottom
        self.model_version = model_version
        self.input_size = input_size
        self.preprocess_mode = preprocess_mode
        self.letterbox_pad_value = int(letterbox_pad_value)
        self.enable_rgb_lowlight_enhance = bool(enable_rgb_lowlight_enhance)
        self.rgb_lowlight_luma_threshold = float(rgb_lowlight_luma_threshold)
        self.rgb_lowlight_gamma = float(rgb_lowlight_gamma)
        self.rgb_lowlight_clahe_clip_limit = float(rgb_lowlight_clahe_clip_limit)
        self.rgb_lowlight_confidence_scale = float(rgb_lowlight_confidence_scale)
        self.engine = TorchInferenceEngine(model, device)
        self.last_rgb_image = None
        self._last_resize_meta = None
        self._last_lowlight_applied = False

    @classmethod
    def from_deploy_config(cls, config_path: str | Path) -> "Predictor":
        with open(config_path, "r", encoding="utf-8") as fp:
            deploy_cfg = yaml.safe_load(fp)["service"]

        project_root = Path(config_path).resolve().parents[1]
        class_mapping_path = Path(deploy_cfg["class_mapping_path"])
        if not class_mapping_path.is_absolute():
            class_mapping_path = (project_root / class_mapping_path).resolve()
        with open(class_mapping_path, "r", encoding="utf-8") as fp:
            class_mapping = json.load(fp)

        model_path = Path(deploy_cfg["model_path"])
        if not model_path.is_absolute():
            model_path = (project_root / model_path).resolve()
        default_config_path = project_root / "configs" / "default.yaml"
        with open(default_config_path, "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)

        requested_device = deploy_cfg.get("device", "cuda")
        cuda_available = torch.cuda.is_available()
        device = torch.device(requested_device if requested_device == "cpu" or cuda_available else "cpu")
        result_min_confidence = float(deploy_cfg.get("result_min_confidence", 0.0))
        result_merge_distance = float(deploy_cfg.get("result_merge_distance", 0.0))
        result_merge_degenerate_size = float(deploy_cfg.get("result_merge_degenerate_size", 2.0))
        result_duplicate_iou = float(deploy_cfg.get("result_duplicate_iou", 0.0))
        result_containment_ratio = float(deploy_cfg.get("result_containment_ratio", 0.0))
        result_duplicate_center_distance = float(deploy_cfg.get("result_duplicate_center_distance", 0.0))
        result_box_expand_ratio = float(deploy_cfg.get("result_box_expand_ratio", 0.0))
        result_box_expand_min_pixels = float(deploy_cfg.get("result_box_expand_min_pixels", 0.0))
        result_box_pad_left = float(deploy_cfg.get("result_box_pad_left", 0.0))
        result_box_pad_top = float(deploy_cfg.get("result_box_pad_top", 0.0))
        result_box_pad_right = float(deploy_cfg.get("result_box_pad_right", 0.0))
        result_box_pad_bottom = float(deploy_cfg.get("result_box_pad_bottom", 0.0))
        checkpoint = None
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            checkpoint_config = checkpoint.get("config")
            if checkpoint_config:
                config = merge_nested_dict(config, checkpoint_config)
                if "resize_mode" not in checkpoint_config.get("dataset", {}):
                    config.setdefault("dataset", {})["resize_mode"] = "stretch"
        model_cfg = config.setdefault("model", {})
        for key in ("input_size", "score_thresh", "nms_thresh", "detections_per_img"):
            if key in deploy_cfg and deploy_cfg[key] is not None:
                model_cfg[key] = deploy_cfg[key]
        model = build_model(config).to(device)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        return cls(
            model=model,
            class_mapping=class_mapping,
            device=device,
            requested_device=requested_device,
            cuda_available=cuda_available,
            result_min_confidence=result_min_confidence,
            result_merge_distance=result_merge_distance,
            result_merge_degenerate_size=result_merge_degenerate_size,
            result_duplicate_iou=result_duplicate_iou,
            result_containment_ratio=result_containment_ratio,
            result_duplicate_center_distance=result_duplicate_center_distance,
            result_box_expand_ratio=result_box_expand_ratio,
            result_box_expand_min_pixels=result_box_expand_min_pixels,
            result_box_pad_left=result_box_pad_left,
            result_box_pad_top=result_box_pad_top,
            result_box_pad_right=result_box_pad_right,
            result_box_pad_bottom=result_box_pad_bottom,
            model_version=model_path.stem if model_path.exists() else "untrained",
            input_size=config["model"].get("input_size", 640),
            preprocess_mode=config.get("dataset", {}).get("resize_mode", "stretch"),
            letterbox_pad_value=config.get("dataset", {}).get("letterbox_pad_value", 114),
            enable_rgb_lowlight_enhance=bool(deploy_cfg.get("enable_rgb_lowlight_enhance", False)),
            rgb_lowlight_luma_threshold=float(deploy_cfg.get("rgb_lowlight_luma_threshold", 70.0)),
            rgb_lowlight_gamma=float(deploy_cfg.get("rgb_lowlight_gamma", 1.15)),
            rgb_lowlight_clahe_clip_limit=float(deploy_cfg.get("rgb_lowlight_clahe_clip_limit", 2.0)),
            rgb_lowlight_confidence_scale=float(deploy_cfg.get("rgb_lowlight_confidence_scale", 0.75)),
        )

    def _prepare_tensors(self, rgb_bytes: bytes, thermal_bytes: bytes):
        rgb_image = decode_image_bytes(rgb_bytes, mode="rgb")
        thermal_image = decode_image_bytes(thermal_bytes, mode="thermal")
        return self._prepare_arrays(rgb_image, thermal_image)

    def _prepare_arrays(self, rgb_image: np.ndarray, thermal_image: np.ndarray):
        rgb_image = self._ensure_rgb_array(rgb_image)
        thermal_image = self._ensure_thermal_array(thermal_image)
        rgb_image, thermal_image = ensure_same_size(rgb_image, thermal_image)
        self.last_rgb_image = rgb_image
        prepared_rgb = self._maybe_enhance_lowlight_rgb(rgb_image)
        prepared_thermal = thermal_image
        resize_meta = None
        if self.preprocess_mode == "stretch":
            prepared_rgb, prepared_thermal, _, resize_meta = resize_pair(
                rgb_image,
                thermal_image,
                np.zeros((0, 4), dtype=np.float32),
                (self.input_size, self.input_size),
                return_meta=True,
            )
        elif self.preprocess_mode == "letterbox":
            prepared_rgb, prepared_thermal, _, resize_meta = letterbox_resize_pair(
                rgb_image,
                thermal_image,
                np.zeros((0, 4), dtype=np.float32),
                (self.input_size, self.input_size),
                pad_value=self.letterbox_pad_value,
            )
        elif self.preprocess_mode != "none":
            raise ValueError(f"Unsupported predictor preprocess mode: {self.preprocess_mode}")
        self._last_resize_meta = resize_meta
        rgb_tensor = image_to_tensor(prepared_rgb)
        thermal_tensor = image_to_tensor(prepared_thermal)
        return rgb_image, rgb_tensor, thermal_tensor

    def _maybe_enhance_lowlight_rgb(self, image_rgb: np.ndarray) -> np.ndarray:
        self._last_lowlight_applied = False
        if not self.enable_rgb_lowlight_enhance:
            return image_rgb
        luma = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).mean()
        if luma >= self.rgb_lowlight_luma_threshold:
            return image_rgb
        self._last_lowlight_applied = True
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.rgb_lowlight_clahe_clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
        gamma = max(self.rgb_lowlight_gamma, 1e-6)
        table = np.array([((idx / 255.0) ** (1.0 / gamma)) * 255.0 for idx in range(256)]).clip(0, 255).astype(np.uint8)
        return cv2.LUT(enhanced, table)

    @staticmethod
    def _ensure_rgb_array(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1)
        if image.ndim == 3 and image.shape[2] == 3:
            return np.array(image, copy=True)
        raise ValueError(f"Unsupported RGB image shape: {image.shape}")

    @staticmethod
    def _ensure_thermal_array(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported thermal image shape: {image.shape}")
        return np.stack([gray, gray, gray], axis=-1)

    def predict(
        self,
        rgb_bytes: bytes,
        thermal_bytes: bytes,
        request_id: Optional[str] = None,
    ) -> Dict:
        rgb_image, rgb_tensor, thermal_tensor = self._prepare_tensors(rgb_bytes, thermal_bytes)
        return self._predict_from_tensors(rgb_image, rgb_tensor, thermal_tensor, request_id=request_id)

    def predict_arrays(
        self,
        rgb_image: np.ndarray,
        thermal_image: np.ndarray,
        request_id: Optional[str] = None,
    ) -> Dict:
        rgb_image, rgb_tensor, thermal_tensor = self._prepare_arrays(rgb_image, thermal_image)
        return self._predict_from_tensors(rgb_image, rgb_tensor, thermal_tensor, request_id=request_id)

    def _predict_from_tensors(
        self,
        rgb_image: np.ndarray,
        rgb_tensor: torch.Tensor,
        thermal_tensor: torch.Tensor,
        request_id: Optional[str] = None,
    ) -> Dict:
        start = time.perf_counter()
        output = self.engine.predict(rgb_tensor, thermal_tensor)
        output = self._restore_output_boxes(output)
        elapsed = time.perf_counter() - start

        raw_detections = self._build_detections(output)
        raw_detections = self._expand_detections_to_image(raw_detections, rgb_image.shape[:2])
        detections = self._postprocess_detections(raw_detections)

        return {
            "request_id": request_id or uuid.uuid4().hex,
            "detections": detections,
            "raw_detections": raw_detections,
            "inference_time": round(elapsed, 4),
            "model_input_size": [self.input_size, self.input_size],
            "image_size": [int(rgb_image.shape[1]), int(rgb_image.shape[0])],
            "model_version": self.model_version,
        }

    def _restore_output_boxes(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._last_resize_meta is None or "boxes" not in output:
            return output
        restored_boxes, keep = restore_boxes_to_original_size(output["boxes"], self._last_resize_meta)
        restored_output = dict(output)
        restored_output["boxes"] = restored_boxes
        for key in ("scores", "labels"):
            if key in restored_output:
                restored_output[key] = restored_output[key][keep]
        return restored_output

    def _build_detections(self, output: Dict) -> List[Dict]:
        detections = []
        boxes = output.get("boxes", torch.zeros((0, 4)))
        scores = output.get("scores", torch.zeros((0,)))
        labels = output.get("labels", torch.zeros((0,), dtype=torch.int64))
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            detections.append(
                {
                    "bbox": [round(float(v), 2) for v in box],
                    "confidence": round(float(score), 4),
                    "class_id": int(label),
                    "class_name": self.class_mapping.get(str(label), f"class_{label}"),
                }
            )
        return detections

    def _expand_detections_to_image(self, detections: List[Dict], image_hw: tuple[int, int]) -> List[Dict]:
        if (
            self.result_box_expand_ratio <= 0
            and self.result_box_expand_min_pixels <= 0
            and self.result_box_pad_left == 0
            and self.result_box_pad_top == 0
            and self.result_box_pad_right == 0
            and self.result_box_pad_bottom == 0
        ):
            return detections
        image_h, image_w = image_hw
        expanded: List[Dict] = []
        for item in detections:
            clone = dict(item)
            x1, y1, x2, y2 = [float(v) for v in clone["bbox"]]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            expand_x = max(width * self.result_box_expand_ratio * 0.5, self.result_box_expand_min_pixels)
            expand_y = max(height * self.result_box_expand_ratio * 0.5, self.result_box_expand_min_pixels)
            x1 = max(0.0, x1 - expand_x - self.result_box_pad_left)
            y1 = max(0.0, y1 - expand_y - self.result_box_pad_top)
            x2 = min(float(max(image_w - 1, 0)), x2 + expand_x + self.result_box_pad_right)
            y2 = min(float(max(image_h - 1, 0)), y2 + expand_y + self.result_box_pad_bottom)
            clone["bbox"] = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            expanded.append(clone)
        return expanded

    def _postprocess_detections(self, detections: List[Dict]) -> List[Dict]:
        filtered = [item for item in detections if item["confidence"] >= self.result_min_confidence]
        if not filtered and self._last_lowlight_applied:
            relaxed_threshold = self.result_min_confidence * self.rgb_lowlight_confidence_scale
            filtered = [item for item in detections if item["confidence"] >= relaxed_threshold]

        kept: List[Dict] = []
        for item in sorted(filtered, key=lambda det: det["confidence"], reverse=True):
            merge_index = self._find_merge_target_index(item, kept)
            if merge_index is not None:
                kept[merge_index] = self._merge_duplicate_detection(kept[merge_index], item)
                continue
            if not self._is_redundant_detection(item, kept):
                kept.append(item)
        return kept

    def _find_merge_target_index(self, item: Dict, kept: List[Dict]) -> Optional[int]:
        if self.result_merge_distance <= 0 and self.result_duplicate_iou <= 0 and self.result_containment_ratio <= 0:
            return None
        for idx, existing in enumerate(kept):
            if item["class_id"] != existing["class_id"]:
                continue
            if self._is_overlap_duplicate(item, existing):
                return idx
        return None

    def _merge_duplicate_detection(self, existing: Dict, incoming: Dict) -> Dict:
        existing_area = self._box_area(existing["bbox"])
        incoming_area = self._box_area(incoming["bbox"])
        if incoming_area > existing_area:
            preferred = dict(incoming)
            preferred["confidence"] = round(max(float(existing["confidence"]), float(incoming["confidence"])), 4)
            return preferred
        merged = dict(existing)
        merged["confidence"] = round(max(float(existing["confidence"]), float(incoming["confidence"])), 4)
        return merged

    def _is_redundant_detection(self, item: Dict, kept: List[Dict]) -> bool:
        if self.result_merge_distance <= 0 and self.result_duplicate_iou <= 0 and self.result_containment_ratio <= 0:
            return False
        x1, y1, x2, y2 = item["bbox"]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        item_is_degenerate = self._is_degenerate_box(item["bbox"])
        for existing in kept:
            if item["class_id"] != existing["class_id"]:
                continue
            if item_is_degenerate and self._is_overlap_duplicate(item, existing):
                return True
        return False

    def _is_overlap_duplicate(self, item: Dict, existing: Dict) -> bool:
        x1, y1, x2, y2 = item["bbox"]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        existing_is_degenerate = self._is_degenerate_box(existing["bbox"])
        ex1, ey1, ex2, ey2 = existing["bbox"]
        existing_center_x = (ex1 + ex2) / 2.0
        existing_center_y = (ey1 + ey2) / 2.0
        center_distance_ok = (
            self.result_duplicate_center_distance <= 0
            or (
                abs(center_x - existing_center_x) <= self.result_duplicate_center_distance
                and abs(center_y - existing_center_y) <= self.result_duplicate_center_distance
            )
        )

        item_is_degenerate = self._is_degenerate_box(item["bbox"])
        if item_is_degenerate or existing_is_degenerate:
            return (
                self.result_merge_distance > 0
                and abs(center_x - existing_center_x) <= self.result_merge_distance
                and abs(center_y - existing_center_y) <= self.result_merge_distance
            )

        intersection = self._intersection_area(item["bbox"], existing["bbox"])
        if intersection <= 0:
            return False
        item_area = self._box_area(item["bbox"])
        existing_area = self._box_area(existing["bbox"])
        union = item_area + existing_area - intersection
        iou = intersection / max(union, 1e-6)
        containment = intersection / max(min(item_area, existing_area), 1e-6)
        if self.result_duplicate_iou > 0 and iou >= self.result_duplicate_iou:
            return True
        if self.result_containment_ratio > 0 and containment >= self.result_containment_ratio and center_distance_ok:
            return True
        return False

    def _is_degenerate_box(self, bbox: List[float]) -> bool:
        x1, y1, x2, y2 = bbox
        return (x2 - x1) <= self.result_merge_degenerate_size or (y2 - y1) <= self.result_merge_degenerate_size

    @staticmethod
    def _box_area(bbox: List[float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _intersection_area(box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        return max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    def health(self) -> Dict:
        return {
            "status": "ok",
            "model_loaded": self.model is not None,
            "model_version": self.model_version,
            "requested_device": self.requested_device,
            "runtime_device": self.device.type,
            "cuda_available": self.cuda_available,
        }

    def model_info(self) -> Dict:
        return {
            "model_name": "RGBT FCOS Detector",
            "model_version": self.model_version,
            "input_size": [self.input_size, self.input_size],
            "num_classes": len(self.class_mapping),
            "class_mapping": self.class_mapping,
            "backend": "PyTorch",
            "requested_device": self.requested_device,
            "runtime_device": self.device.type,
            "cuda_available": self.cuda_available,
        }
