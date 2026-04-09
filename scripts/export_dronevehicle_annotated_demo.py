from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CLASS_MAPPING = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "van",
    4: "freight_car",
}

RAW_NAME_TO_CLASS = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "van",
    "freight car": "freight_car",
    "feright car": "freight_car",
    "feright_car": "freight_car",
}

CLASS_TO_ID = {value: key for key, value in CLASS_MAPPING.items()}
BOX_COLORS = {
    "car": (0, 200, 255),
    "truck": (56, 189, 248),
    "bus": (16, 185, 129),
    "van": (249, 115, 22),
    "freight_car": (244, 63, 94),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render DroneVehicle ground-truth annotations into prerecorded demo videos."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "raw" / "dronevehicle" / "val_unpack" / "val"),
        help="Root containing valimg/valimgr/vallabel/vallabelr.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "demo_video"),
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-count", type=int, default=150)
    parser.add_argument("--start-number", type=int, default=1)
    parser.add_argument(
        "--prefix",
        type=str,
        default="dronevehicle_gt",
        help="Output video prefix.",
    )
    return parser.parse_args()


def normalize_class_name(raw_name: str) -> str:
    normalized = RAW_NAME_TO_CLASS.get(raw_name.strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported raw class name: {raw_name}")
    return normalized


def polygon_to_bbox(polygon: List[List[int]]) -> List[float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def parse_annotation(xml_path: Path) -> List[Dict]:
    tree = ET.parse(xml_path)
    objects = []
    for obj in tree.findall(".//object"):
        raw_name = (obj.findtext("name") or "").strip()
        class_name = normalize_class_name(raw_name)
        polygon_node = obj.find("polygon")
        if polygon_node is None:
            continue
        polygon = []
        for idx in range(1, 5):
            x = int(float(polygon_node.findtext(f"x{idx}", "0")))
            y = int(float(polygon_node.findtext(f"y{idx}", "0")))
            polygon.append([x, y])
        objects.append(
            {
                "class_name": class_name,
                "class_id": CLASS_TO_ID[class_name],
                "bbox": polygon_to_bbox(polygon),
            }
        )
    return objects


def draw_detections(image_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
    canvas = np.array(image_bgr, copy=True)
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        color = BOX_COLORS.get(det["class_name"], (255, 255, 0))
        label = f'{det["class_name"]}:{det["class_id"]}'
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def create_writer(path: Path, fps: int, frame_size: tuple[int, int]) -> cv2.VideoWriter:
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


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()

    rgb_root = source_root / "valimg"
    thermal_root = source_root / "valimgr"
    rgb_ann_root = source_root / "vallabel"
    thermal_ann_root = source_root / "vallabelr"

    rgb_video_path = output_root / f"{args.prefix}_rgb_annotated.mp4"
    thermal_video_path = output_root / f"{args.prefix}_thermal_annotated.mp4"
    side_by_side_path = output_root / f"{args.prefix}_side_by_side_annotated.mp4"
    manifest_path = output_root / f"{args.prefix}_manifest.json"

    rgb_writer = None
    thermal_writer = None
    side_by_side_writer = None
    rendered_frames = 0
    rendered_ids: List[str] = []

    try:
        for sample_number in range(args.start_number, args.start_number + args.frame_count):
            sample_id = f"{sample_number:05d}"
            rgb_image_path = rgb_root / f"{sample_id}.jpg"
            thermal_image_path = thermal_root / f"{sample_id}.jpg"
            rgb_xml_path = rgb_ann_root / f"{sample_id}.xml"
            thermal_xml_path = thermal_ann_root / f"{sample_id}.xml"

            required = [rgb_image_path, thermal_image_path, rgb_xml_path, thermal_xml_path]
            if not all(path.exists() for path in required):
                continue

            rgb_image = cv2.imread(str(rgb_image_path), cv2.IMREAD_COLOR)
            thermal_image = cv2.imread(str(thermal_image_path), cv2.IMREAD_COLOR)
            if rgb_image is None or thermal_image is None:
                continue

            rgb_overlay = draw_detections(rgb_image, parse_annotation(rgb_xml_path))
            thermal_overlay = draw_detections(thermal_image, parse_annotation(thermal_xml_path))
            side_by_side = np.concatenate([rgb_overlay, thermal_overlay], axis=1)

            if rgb_writer is None:
                rgb_writer = create_writer(rgb_video_path, args.fps, (rgb_overlay.shape[1], rgb_overlay.shape[0]))
                thermal_writer = create_writer(
                    thermal_video_path,
                    args.fps,
                    (thermal_overlay.shape[1], thermal_overlay.shape[0]),
                )
                side_by_side_writer = create_writer(
                    side_by_side_path,
                    args.fps,
                    (side_by_side.shape[1], side_by_side.shape[0]),
                )

            rgb_writer.write(rgb_overlay)
            thermal_writer.write(thermal_overlay)
            side_by_side_writer.write(side_by_side)
            rendered_frames += 1
            rendered_ids.append(sample_id)
    finally:
        if rgb_writer is not None:
            rgb_writer.release()
        if thermal_writer is not None:
            thermal_writer.release()
        if side_by_side_writer is not None:
            side_by_side_writer.release()

    manifest = {
        "dataset": "DroneVehicle",
        "source_root": str(source_root),
        "fps": args.fps,
        "requested_frame_count": args.frame_count,
        "rendered_frames": rendered_frames,
        "start_number": args.start_number,
        "frame_ids": rendered_ids,
        "videos": {
            "rgb_annotated": str(rgb_video_path),
            "thermal_annotated": str(thermal_video_path),
            "side_by_side_annotated": str(side_by_side_path),
        },
        "notes": [
            "These videos are rendered from raw DroneVehicle ground-truth annotations.",
            "Use them for prerecorded frontend demo playback when online model callback latency is too high.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
