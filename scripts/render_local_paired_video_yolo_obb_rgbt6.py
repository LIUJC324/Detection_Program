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
from model.network.rgbt6_yolo_modules import register_rgbt6_yolo_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local paired RGB-T video inference with a 6-channel YOLO-OBB model and render a preview."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "yolo_obb_runs"
            / "official_rgbt6_full_official_speedup_v3"
            / "weights"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "demo_video" / "dronevehicle_rgb_thermal_side_by_side.mp4"),
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "local_preview" / "annotated_preview_yolo_obb_rgbt6_final.mp4"),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "local_preview" / "annotated_preview_yolo_obb_rgbt6_final.json"),
    )
    parser.add_argument("--pair-layout", type=str, default="side_by_side_h")
    parser.add_argument("--rgb-position", type=str, default="left")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def create_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
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


def combine_frames(
    rgb_annotated_bgr: np.ndarray,
    thermal_annotated_bgr: np.ndarray,
    pair_layout: str,
    rgb_position: str,
) -> np.ndarray:
    if pair_layout == "side_by_side_h":
        frames = [rgb_annotated_bgr, thermal_annotated_bgr] if rgb_position == "left" else [thermal_annotated_bgr, rgb_annotated_bgr]
        return np.hstack(frames)
    if pair_layout == "stacked_v":
        frames = [rgb_annotated_bgr, thermal_annotated_bgr] if rgb_position == "top" else [thermal_annotated_bgr, rgb_annotated_bgr]
        return np.vstack(frames)
    raise ValueError(f"Unsupported pair_layout: {pair_layout}")


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    register_rgbt6_yolo_modules()
    model_path = Path(args.model).resolve()
    video_path = Path(args.video).resolve()
    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open input video: {video_path}")

    sample_every = max(1, int(args.sample_every))
    max_frames = max(0, int(args.max_frames))
    fps = capture.get(cv2.CAP_PROP_FPS) or 8.0
    output_fps = max(1.0, fps / sample_every)
    writer = None
    frame_index = 0
    rendered_frames = 0
    inferenced_frames = 0
    empty_detection_frames = 0
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
            rgb_frame, thermal_frame = _split_pair_frame(frame_rgb, args.pair_layout, args.rgb_position)
            stacked = np.concatenate([rgb_frame, thermal_frame], axis=2)

            start = cv2.getTickCount()
            results = model.predict(
                source=stacked,
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
            latency_ms.append(elapsed_ms)

            rgb_annotated = annotator.annotate(rgb_frame, detections)
            thermal_annotated = annotator.annotate(thermal_frame, detections)
            combined = combine_frames(rgb_annotated, thermal_annotated, args.pair_layout, args.rgb_position)

            if writer is None:
                writer = create_writer(Path(args.output_video), output_fps, (combined.shape[1], combined.shape[0]))
            writer.write(combined)
            rendered_frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    summary = {
        "task": "yolo_obb_rgbt6_preview",
        "model_path": str(model_path),
        "input_video": str(video_path),
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
