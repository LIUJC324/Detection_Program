from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.core.predictor import Predictor
from service.streaming.session_manager import _split_pair_frame
from service.utils import AnnotatorConfig, DetectionAnnotator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local paired RGB-T video inference and render an annotated preview video for local inspection."
    )
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "deploy_stable.yaml"))
    parser.add_argument("--video", type=str, required=True, help="Local side-by-side or stacked RGB-T video path.")
    parser.add_argument(
        "--output-video",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "local_preview" / "annotated_preview.mp4"),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "local_preview" / "annotated_preview_summary.json"),
    )
    parser.add_argument("--pair-layout", type=str, default="side_by_side_h")
    parser.add_argument("--rgb-position", type=str, default="left")
    parser.add_argument("--sample-every", type=int, default=3, help="Run inference every N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["sampled", "blank_on_skip", "hold_last"],
        default="blank_on_skip",
        help="sampled: only output inferred frames; blank_on_skip: output all frames but only draw on inferred frames; hold_last: reuse the last detections on skipped frames.",
    )
    parser.add_argument(
        "--enhance-rgb-lowlight",
        action="store_true",
        help="Apply lightweight CLAHE+gamma enhancement on the RGB branch before inference for local dark-scene inspection.",
    )
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


def enhance_lowlight_rgb(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    gamma = 1.15
    table = np.array([((idx / 255.0) ** (1.0 / gamma)) * 255.0 for idx in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(enhanced, table)


def main():
    args = parse_args()
    predictor = Predictor.from_deploy_config(args.config)
    annotator = DetectionAnnotator(
        config=AnnotatorConfig(annotation_mode="rectangle", min_confidence=float(args.min_confidence))
    )

    capture = cv2.VideoCapture(str(Path(args.video).resolve()))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open input video: {args.video}")

    sample_every = max(1, int(args.sample_every))
    max_frames = max(0, int(args.max_frames))
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    output_fps = max(1.0, fps / sample_every) if args.render_mode == "sampled" else fps
    writer = None
    frame_index = 0
    rendered_frames = 0
    inferenced_frames = 0
    empty_detection_frames = 0
    cached_detections = []
    latency_ms = []

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index += 1
            if max_frames > 0 and frame_index > max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            should_infer = (frame_index - 1) % sample_every == 0
            detections_for_render = cached_detections if args.render_mode == "hold_last" else []
            if should_infer:
                rgb_frame, thermal_frame = _split_pair_frame(frame_rgb, args.pair_layout, args.rgb_position)
                rgb_for_model = enhance_lowlight_rgb(rgb_frame) if args.enhance_rgb_lowlight else rgb_frame
                result = predictor.predict_arrays(rgb_for_model, thermal_frame, request_id=f"local_preview:{frame_index}")
                cached_detections = result["detections"]
                latency_ms.append(float(result["inference_time"]) * 1000.0)
                inferenced_frames += 1
                if not cached_detections:
                    empty_detection_frames += 1
                detections_for_render = cached_detections

            if args.render_mode == "sampled" and not should_infer:
                continue

            rgb_frame, _ = _split_pair_frame(frame_rgb, args.pair_layout, args.rgb_position)
            annotated = annotator.annotate(rgb_frame, detections_for_render)
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
        "input_video": str(Path(args.video).resolve()),
        "output_video": str(Path(args.output_video).resolve()),
        "pair_layout": args.pair_layout,
        "rgb_position": args.rgb_position,
        "sample_every": sample_every,
        "render_mode": args.render_mode,
        "enhance_rgb_lowlight": bool(args.enhance_rgb_lowlight),
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
