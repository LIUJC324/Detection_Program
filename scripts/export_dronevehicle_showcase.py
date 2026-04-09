from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


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
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export one DroneVehicle sample for frontend/backend showcase.")
    parser.add_argument(
        "--source-root",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "raw" / "dronevehicle" / "val_unpack" / "val"),
        help="Root containing valimg/valimgr/vallabel/vallabelr.",
    )
    parser.add_argument("--sample-id", type=str, default="00001")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "showcase"),
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


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
                "raw_class_name": raw_name,
                "class_name": class_name,
                "class_id": CLASS_TO_ID[class_name],
                "bbox": polygon_to_bbox(polygon),
                "polygon": polygon,
            }
        )
    return objects


def draw_overlay(image_path: Path, objects: List[Dict], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(18)

    for item in objects:
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        color = BOX_COLORS.get(item["class_name"], (255, 255, 0))
        label = f'{item["class_name"]}:{item["class_id"]}'
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_x = x1
        text_y = max(0, y1 - (text_box[3] - text_box[1]) - 6)
        draw.rectangle(
            (text_x - 2, text_y - 2, text_x + (text_box[2] - text_box[0]) + 4, text_y + (text_box[3] - text_box[1]) + 4),
            fill=(15, 23, 42),
        )
        draw.text((text_x, text_y), label, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def export_sample(source_root: Path, sample_id: str, output_root: Path) -> Path:
    rgb_image = source_root / "valimg" / f"{sample_id}.jpg"
    thermal_image = source_root / "valimgr" / f"{sample_id}.jpg"
    rgb_label = source_root / "vallabel" / f"{sample_id}.xml"
    thermal_label = source_root / "vallabelr" / f"{sample_id}.xml"

    required = [rgb_image, thermal_image, rgb_label, thermal_label]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sample files: {missing}")

    rgb_objects = parse_annotation(rgb_label)
    thermal_objects = parse_annotation(thermal_label)

    sample_dir = output_root / f"dronevehicle_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    copy_file(rgb_image, sample_dir / "rgb.jpg")
    copy_file(thermal_image, sample_dir / "thermal.jpg")
    copy_file(rgb_label, sample_dir / "rgb_raw.xml")
    copy_file(thermal_label, sample_dir / "thermal_raw.xml")

    draw_overlay(rgb_image, rgb_objects, sample_dir / "rgb_overlay.jpg")
    draw_overlay(thermal_image, thermal_objects, sample_dir / "thermal_overlay.jpg")

    rgb_payload = {
        "sample_id": sample_id,
        "modality": "rgb",
        "image_file": "rgb.jpg",
        "annotation_file": "rgb_raw.xml",
        "objects": rgb_objects,
    }
    thermal_payload = {
        "sample_id": sample_id,
        "modality": "thermal",
        "image_file": "thermal.jpg",
        "annotation_file": "thermal_raw.xml",
        "objects": thermal_objects,
    }
    manifest = {
        "dataset": "DroneVehicle",
        "sample_id": sample_id,
        "source_root": str(source_root),
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "notes": [
            "DroneVehicle raw annotations are modality-specific.",
            "This showcase package is for communication/demo, not a direct training sample for the current dataset.py.",
            "Current service API shape stays the same; only label values changed to DroneVehicle classes.",
        ],
        "files": {
            "rgb_image": "rgb.jpg",
            "thermal_image": "thermal.jpg",
            "rgb_overlay": "rgb_overlay.jpg",
            "thermal_overlay": "thermal_overlay.jpg",
            "rgb_annotations": "rgb_annotations.json",
            "thermal_annotations": "thermal_annotations.json",
        },
    }

    with (sample_dir / "rgb_annotations.json").open("w", encoding="utf-8") as fp:
        json.dump(rgb_payload, fp, ensure_ascii=False, indent=2)
    with (sample_dir / "thermal_annotations.json").open("w", encoding="utf-8") as fp:
        json.dump(thermal_payload, fp, ensure_ascii=False, indent=2)
    with (sample_dir / "showcase_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    return sample_dir


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    sample_dir = export_sample(source_root, args.sample_id, output_root)
    print(sample_dir)


if __name__ == "__main__":
    main()
