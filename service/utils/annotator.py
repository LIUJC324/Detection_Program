from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


DEFAULT_CLASS_COLORS = {
    "car": (0, 200, 255),
    "truck": (56, 189, 248),
    "bus": (16, 185, 129),
    "van": (249, 115, 22),
    "freight_car": (244, 63, 94),
}


@dataclass
class AnnotatorConfig:
    annotation_mode: str = "rectangle"
    min_confidence: float = 0.0
    line_thickness: int = 2
    font_scale: float = 0.5
    draw_degenerate_as_point: bool = True
    point_radius: int = 5
    show_angle: bool = False


class DetectionAnnotator:
    def __init__(
        self,
        class_colors: Dict[str, Tuple[int, int, int]] | None = None,
        config: AnnotatorConfig | None = None,
    ) -> None:
        self.class_colors = class_colors or DEFAULT_CLASS_COLORS
        self.config = config or AnnotatorConfig()

    def annotate(self, image_rgb: np.ndarray, detections: Iterable[Dict]) -> np.ndarray:
        canvas = cv2.cvtColor(np.array(image_rgb, copy=True), cv2.COLOR_RGB2BGR)
        for det in detections:
            confidence = float(det.get("confidence", det.get("score", 0.0)))
            if confidence < self.config.min_confidence:
                continue
            polygon = self._normalize_polygon(det.get("polygon") or det.get("points"))
            class_name = str(det.get("class_name", det.get("tag", "object")))
            color = self.class_colors.get(class_name, (0, 255, 255))
            angle = det.get("angle")
            angle_deg = None if angle is None else float(angle)

            if self.config.annotation_mode == "polygon" and polygon is not None:
                points = np.array(
                    [[int(round(float(x))), int(round(float(y))) ] for x, y in polygon],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=self.config.line_thickness)
                anchor_x = int(np.min(points[:, 0, 0]))
                anchor_y = int(max(16, np.min(points[:, 0, 1]) - 8))
                self._draw_label(canvas, class_name, confidence, anchor_x, anchor_y, color, angle_deg)
                continue

            bbox = det.get("bbox") or [
                det.get("x1", 0.0),
                det.get("y1", 0.0),
                det.get("x2", 0.0),
                det.get("y2", 0.0),
            ]
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]

            if self.config.annotation_mode == "point" or (self.config.draw_degenerate_as_point and self._is_degenerate_box(x1, y1, x2, y2)):
                center_x = int(round((x1 + x2) / 2.0))
                center_y = int(round((y1 + y2) / 2.0))
                cv2.circle(canvas, (center_x, center_y), self.config.point_radius, color, thickness=-1)
                self._draw_label(canvas, class_name, confidence, center_x, max(16, center_y - 8), color, angle_deg)
                continue

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.config.line_thickness)
            self._draw_label(canvas, class_name, confidence, x1, max(16, y1 - 8), color, angle_deg)
        return canvas

    def _draw_label(
        self,
        image_bgr: np.ndarray,
        class_name: str,
        confidence: float,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        angle_deg: float | None = None,
    ) -> None:
        text = f"{class_name}:{confidence:.2f}"
        if self.config.show_angle and angle_deg is not None:
            text = f"{text}@{angle_deg:.1f}deg"
        cv2.putText(
            image_bgr,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _is_degenerate_box(x1: int, y1: int, x2: int, y2: int) -> bool:
        return (x2 - x1) <= 1 or (y2 - y1) <= 1

    @staticmethod
    def _normalize_polygon(value) -> list[tuple[float, float]] | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            return None
        if all(isinstance(item, (int, float)) for item in value) and len(value) % 2 == 0:
            points = []
            for index in range(0, len(value), 2):
                points.append((float(value[index]), float(value[index + 1])))
            return points
        normalized = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                return None
            normalized.append((float(item[0]), float(item[1])))
        return normalized
