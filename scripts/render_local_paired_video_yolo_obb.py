from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.streaming.session_manager import _split_pair_frame
from service.utils import AnnotatorConfig, DetectionAnnotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local paired RGB-T video inference with a YOLO-OBB model and render an angle-aware preview."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "yolo_obb_runs" / "official_rgb_trueobb_stage1_stable_v1" / "weights" / "best.pt"),
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "demo_video" / "dronevehicle_rgb_thermal_side_by_side_hq_12fps_600f.mp4"),
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "local_preview" / "annotated_preview_yolo_obb_angle_20260413.mp4"),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "local_preview" / "annotated_preview_yolo_obb_angle_20260413.json"),
    )
    parser.add_argument("--pair-layout", type=str, default="side_by_side_h")
    parser.add_argument("--rgb-position", type=str, default="left")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", help="Use cpu or 0/cuda for GPU.")
    return parser.parse_args()


def create_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def build_detections(result) -> list[dict]:
    detections: list[dict] = []
    if result.obb is None or len(result.obb) == 0:
        return detections

    polygons = result.obb.xyxyxyxy.tolist()
    xywhr = result.obb.xywhr.tolist()
    scores = result.obb.conf.tolist()
    classes = result.obb.cls.tolist()

    names = result.names
    for polygon, obb, score, class_id in zip(polygons, xywhr, scores, classes):
        class_idx = int(class_id)
        angle_rad = float(obb[4])
        detections.append(
            {
                "class_id": class_idx,
                "class_name": str(names[class_idx]),
                "confidence": round(float(score), 4),
                "angle": round(angle_rad * 180.0 / math.pi, 3),
                "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon],
            }
        )
    return detections


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(str(Path(args.model).resolve()))
    capture = cv2.VideoCapture(str(Path(args.video).resolve()))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open input video: {args.video}")

    sample_every = max(1, int(args.sample_every))
    max_frames = max(0, int(args.max_frames))
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    output_fps = max(1.0, fps / sample_every)
    writer = None
    frame_index = 0
    rendered_frames = 0
    inferenced_frames = 0
    empty_detection_frames = 0
    cached_detections: list[dict] = []
    latency_ms: list[float] = []

    annotator = DetectionAnnotator(
        config=AnnotatorConfig(
            annotation_mode="polygon",
            min_confidence=float(args.conf),
            line_thickness=2,
            font_scale=0.55,
            show_angle=True,
        )
    )

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index += 1
            if max_frames > 0 and frame_index > max_frames:
                break
            if (frame_index - 1) % sample_every != 0:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb_frame, _ = _split_pair_frame(frame_rgb, args.pair_layout, args.rgb_position)
            start = cv2.getTickCount()
            results = model.predict(
                source=rgb_frame,
                verbose=False,
                imgsz=int(args.imgsz),
                conf=float(args.conf),
                iou=float(args.iou),
                device=args.device,
            )
            elapsed_ms = (cv2.getTickCount() - start) * 1000.0 / cv2.getTickFrequency()
            detections = build_detections(results[0])
            inferenced_frames += 1
            if not detections:
                empty_detection_frames += 1
            cached_detections = detections
            latency_ms.append(elapsed_ms)

            annotated = annotator.annotate(rgb_frame, cached_detections)
            if writer is None:
                writer = create_writer(
                    Path(args.output_video),
                    output_fps,
                    (annotated.shape[1], annotated.shape[0]),
                )
            writer.write(annotated)
            rendered_frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    summary = {
        "task": "yolo_obb_angle_preview",
        "model_path": str(Path(args.model).resolve()),
        "input_video": str(Path(args.video).resolve()),
        "output_video": str(Path(args.output_video).resolve()),
        "pair_layout": args.pair_layout,
        "rgb_position": args.rgb_position,
        "device": args.device,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "sample_every": sample_every,
        "output_fps": round(float(output_fps), 3),
        "rendered_frames": rendered_frames,
        "inferenced_frames": inferenced_frames,
        "empty_detection_frames": empty_detection_frames,
        "empty_detection_ratio": round(empty_detection_frames / max(inferenced_frames, 1), 6),
        "avg_model_latency_ms": round(sum(latency_ms) / max(len(latency_ms), 1), 3),
        "max_model_latency_ms": round(max(latency_ms, default=0.0), 3),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
