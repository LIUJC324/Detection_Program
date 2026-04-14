from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RAW_NAME_TO_CLASS = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "van",
    "feright": "freight_car",
    "freight car": "freight_car",
    "feright car": "freight_car",
    "feright_car": "freight_car",
}

CLASS_MAPPING = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "van",
    4: "freight_car",
}

CLASS_TO_ID = {value: key for key, value in CLASS_MAPPING.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DroneVehicle raw xml/image files into the local dronevehicle_like format.")
    parser.add_argument(
        "--source-root",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "raw" / "dronevehicle" / "val_unpack" / "val"),
        help="Root containing valimg/valimgr/vallabel/vallabelr.",
    )
    parser.add_argument(
        "--train-source-root",
        type=str,
        default="",
        help="Optional official train root. When set together with --val-source-root, official split is preserved.",
    )
    parser.add_argument(
        "--val-source-root",
        type=str,
        default="",
        help="Optional official validation root. When set together with --train-source-root, official split is preserved.",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "dronevehicle_like_refined"),
        help="Output dataset root matching configs/default.yaml.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--annotation-source",
        type=str,
        default="merged_union",
        choices=["rgb", "thermal", "merged_union"],
        help="Which modality annotation to use as training target.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="How to materialize image files in the target dataset.",
    )
    parser.add_argument(
        "--clear-target",
        action="store_true",
        help="Remove existing target-root before generating.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=245,
        help="Pixels whose RGB channels are all >= threshold are treated as white border.",
    )
    parser.add_argument(
        "--merge-max-distance",
        type=float,
        default=32.0,
        help="Only merge RGB/Thermal boxes when center distance is within this threshold in pixels.",
    )
    parser.add_argument(
        "--bbox-expand-ratio",
        type=float,
        default=0.0,
        help="Expand each target bbox by this ratio around its center after crop.",
    )
    parser.add_argument(
        "--bbox-expand-min-pixels",
        type=float,
        default=0.0,
        help="Minimum expansion applied to each bbox side in pixels after crop.",
    )
    return parser.parse_args()


def normalize_class_name(raw_name: str) -> str:
    normalized = RAW_NAME_TO_CLASS.get(raw_name.strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported raw class name: {raw_name}")
    return normalized


def polygon_to_bbox(polygon: Sequence[Sequence[int]]) -> List[float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def parse_annotation(xml_path: Path) -> List[Dict]:
    tree = ET.parse(xml_path)
    objects = []
    for obj in tree.findall(".//object"):
        raw_name = (obj.findtext("name") or "").strip()
        try:
            class_name = normalize_class_name(raw_name)
        except ValueError:
            continue
        polygon_node = obj.find("polygon")
        if polygon_node is None:
            continue
        polygon = []
        for idx in range(1, 5):
            x = int(float(polygon_node.findtext(f"x{idx}", "0")))
            y = int(float(polygon_node.findtext(f"y{idx}", "0")))
            polygon.append([x, y])
        bbox = polygon_to_bbox(polygon)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        objects.append(
            {
                "bbox": bbox,
                "class_id": CLASS_TO_ID[class_name],
                "class_name": class_name,
                "polygon": polygon,
            }
        )
    return objects


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def detect_content_box(image: Image.Image, white_threshold: int) -> Tuple[int, int, int, int]:
    array = np.asarray(image, dtype=np.uint8)
    mask = np.min(array, axis=2) < white_threshold
    if not mask.any():
        return 0, 0, image.width, image.height
    ys, xs = np.where(mask)
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return left, top, right, bottom


def pair_content_box(rgb_image: Image.Image, thermal_image: Image.Image, white_threshold: int) -> Tuple[int, int, int, int]:
    rgb_box = detect_content_box(rgb_image, white_threshold)
    thermal_box = detect_content_box(thermal_image, white_threshold)
    left = min(rgb_box[0], thermal_box[0])
    top = min(rgb_box[1], thermal_box[1])
    right = max(rgb_box[2], thermal_box[2])
    bottom = max(rgb_box[3], thermal_box[3])
    return left, top, right, bottom


def crop_objects(objects: Sequence[Dict], crop_box: Tuple[int, int, int, int]) -> List[Dict]:
    left, top, right, bottom = crop_box
    crop_w = right - left
    crop_h = bottom - top
    cropped: List[Dict] = []
    for item in objects:
        x1, y1, x2, y2 = item["bbox"]
        x1 = max(0.0, min(float(crop_w - 1), x1 - left))
        y1 = max(0.0, min(float(crop_h - 1), y1 - top))
        x2 = max(0.0, min(float(crop_w - 1), x2 - left))
        y2 = max(0.0, min(float(crop_h - 1), y2 - top))
        if x2 <= x1 or y2 <= y1:
            continue
        cropped.append(
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": item["class_id"],
                "class_name": item["class_name"],
            }
        )
    return cropped


def expand_objects(
    objects: Sequence[Dict],
    image_size: Tuple[int, int],
    expand_ratio: float,
    min_pixels: float,
) -> List[Dict]:
    if expand_ratio <= 0.0 and min_pixels <= 0.0:
        return [dict(item) for item in objects]
    image_w, image_h = image_size
    expanded: List[Dict] = []
    for item in objects:
        x1, y1, x2, y2 = [float(v) for v in item["bbox"]]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        expand_x = max(width * expand_ratio * 0.5, min_pixels)
        expand_y = max(height * expand_ratio * 0.5, min_pixels)
        x1 = max(0.0, x1 - expand_x)
        y1 = max(0.0, y1 - expand_y)
        x2 = min(float(max(image_w - 1, 0)), x2 + expand_x)
        y2 = min(float(max(image_h - 1, 0)), y2 + expand_y)
        if x2 <= x1 or y2 <= y1:
            continue
        clone = dict(item)
        clone["bbox"] = [x1, y1, x2, y2]
        expanded.append(clone)
    return expanded


def bbox_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def union_bbox(box_a: Sequence[float], box_b: Sequence[float]) -> List[float]:
    return [
        float(min(box_a[0], box_b[0])),
        float(min(box_a[1], box_b[1])),
        float(max(box_a[2], box_b[2])),
        float(max(box_a[3], box_b[3])),
    ]


def merge_annotations(primary_objects: Sequence[Dict], secondary_objects: Sequence[Dict], max_distance: float) -> Tuple[List[Dict], Dict[str, int]]:
    merged: List[Dict] = []
    used_secondary: set[int] = set()
    merged_count = 0

    for primary in primary_objects:
        px, py = bbox_center(primary["bbox"])
        best_idx = None
        best_distance = None
        for idx, secondary in enumerate(secondary_objects):
            if idx in used_secondary or secondary["class_id"] != primary["class_id"]:
                continue
            sx, sy = bbox_center(secondary["bbox"])
            distance = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_idx = idx

        bbox = list(primary["bbox"])
        if best_idx is not None and best_distance is not None and best_distance <= max_distance:
            used_secondary.add(best_idx)
            bbox = union_bbox(primary["bbox"], secondary_objects[best_idx]["bbox"])
            merged_count += 1

        merged.append(
            {
                "bbox": bbox,
                "class_id": primary["class_id"],
                "class_name": primary["class_name"],
            }
        )

    stats = {
        "primary_objects": len(primary_objects),
        "secondary_objects": len(secondary_objects),
        "merged_objects": merged_count,
        "unmatched_primary_objects": len(primary_objects) - merged_count,
        "unused_secondary_objects": len(secondary_objects) - len(used_secondary),
    }
    return merged, stats


def link_or_copy(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def resolve_source_layout(source_root: Path) -> Tuple[Path, Path, Path, Path]:
    candidates = [
        ("valimg", "valimgr", "vallabel", "vallabelr"),
        ("trainimg", "trainimgr", "trainlabel", "trainlabelr"),
    ]
    for rgb_dir, thermal_dir, rgb_ann_dir, thermal_ann_dir in candidates:
        rgb_root = source_root / rgb_dir
        thermal_root = source_root / thermal_dir
        rgb_ann_root = source_root / rgb_ann_dir
        thermal_ann_root = source_root / thermal_ann_dir
        if rgb_root.exists() and thermal_root.exists() and rgb_ann_root.exists() and thermal_ann_root.exists():
            return rgb_root, thermal_root, rgb_ann_root, thermal_ann_root
    raise FileNotFoundError(
        f"Unsupported DroneVehicle source layout under {source_root}. "
        "Expected valimg/valimgr/vallabel/vallabelr or trainimg/trainimgr/trainlabel/trainlabelr."
    )


def collect_samples(source_root: Path) -> List[Dict]:
    rgb_root, thermal_root, rgb_ann_root, thermal_ann_root = resolve_source_layout(source_root)

    samples = []
    for rgb_path in sorted(rgb_root.glob("*.jpg")):
        sample_id = rgb_path.stem
        thermal_path = thermal_root / f"{sample_id}.jpg"
        rgb_ann = rgb_ann_root / f"{sample_id}.xml"
        thermal_ann = thermal_ann_root / f"{sample_id}.xml"
        if not thermal_path.exists() or not rgb_ann.exists() or not thermal_ann.exists():
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "rgb_path": rgb_path,
                "thermal_path": thermal_path,
                "rgb_annotation_path": rgb_ann,
                "thermal_annotation_path": thermal_ann,
            }
        )
    return samples


def ensure_structure(target_root: Path) -> None:
    for split in ("train", "val"):
        for folder in ("rgb", "thermal", "annotations"):
            (target_root / folder / split).mkdir(parents=True, exist_ok=True)


def write_annotation(target_root: Path, split: str, sample_id: str, objects: Sequence[Dict]) -> None:
    payload = {
        "image_id": sample_id,
        "objects": [
            {
                "bbox": obj["bbox"],
                "class_id": obj["class_id"],
                "class_name": obj["class_name"],
            }
            for obj in objects
        ],
    }
    annotation_path = target_root / "annotations" / split / f"{sample_id}.json"
    with annotation_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def save_processed_image(image: Image.Image, src_path: Path, dst_path: Path, copy_mode: str, transformed: bool) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    if not transformed and copy_mode == "symlink":
        dst_path.symlink_to(src_path.resolve())
        return
    if not transformed and copy_mode == "copy":
        shutil.copy2(src_path, dst_path)
        return
    image.save(dst_path, quality=95, subsampling=0)


def materialize_sample(
    target_root: Path,
    split: str,
    sample: Dict,
    copy_mode: str,
    annotation_source: str,
    white_threshold: int,
    merge_max_distance: float,
    bbox_expand_ratio: float,
    bbox_expand_min_pixels: float,
    aggregate_stats: Dict[str, float],
) -> bool:
    try:
        rgb_image = load_rgb_image(sample["rgb_path"])
        thermal_image = load_rgb_image(sample["thermal_path"])
    except OSError:
        aggregate_stats["skipped_corrupt_samples"] += 1
        return False
    crop_box = pair_content_box(rgb_image, thermal_image, white_threshold)
    transformed = crop_box != (0, 0, rgb_image.width, rgb_image.height)
    if transformed:
        rgb_image = rgb_image.crop(crop_box)
        thermal_image = thermal_image.crop(crop_box)

    rgb_objects = crop_objects(parse_annotation(sample["rgb_annotation_path"]), crop_box)
    thermal_objects = crop_objects(parse_annotation(sample["thermal_annotation_path"]), crop_box)

    if annotation_source == "rgb":
        objects = rgb_objects
    elif annotation_source == "thermal":
        objects = thermal_objects
    else:
        objects, merge_stats = merge_annotations(rgb_objects, thermal_objects, max_distance=merge_max_distance)
        aggregate_stats["merged_objects"] += merge_stats["merged_objects"]
        aggregate_stats["unmatched_primary_objects"] += merge_stats["unmatched_primary_objects"]
        aggregate_stats["unused_secondary_objects"] += merge_stats["unused_secondary_objects"]

    objects = expand_objects(
        objects,
        image_size=(rgb_image.width, rgb_image.height),
        expand_ratio=bbox_expand_ratio,
        min_pixels=bbox_expand_min_pixels,
    )

    aggregate_stats["samples"] += 1
    aggregate_stats["cropped_samples"] += int(transformed)
    aggregate_stats["rgb_objects"] += len(rgb_objects)
    aggregate_stats["thermal_objects"] += len(thermal_objects)
    aggregate_stats["final_objects"] += len(objects)
    aggregate_stats["crop_left_total"] += crop_box[0]
    aggregate_stats["crop_top_total"] += crop_box[1]
    aggregate_stats["crop_right_margin_total"] += sample["image_size"][0] - crop_box[2]
    aggregate_stats["crop_bottom_margin_total"] += sample["image_size"][1] - crop_box[3]

    rgb_target = target_root / "rgb" / split / f"{sample['sample_id']}_rgb.jpg"
    thermal_target = target_root / "thermal" / split / f"{sample['sample_id']}_thermal.jpg"
    save_processed_image(rgb_image, sample["rgb_path"], rgb_target, copy_mode, transformed)
    save_processed_image(thermal_image, sample["thermal_path"], thermal_target, copy_mode, transformed)
    write_annotation(target_root, split, sample["sample_id"], objects)
    return True


def write_summary(
    target_root: Path,
    samples_train: Sequence[Dict],
    samples_val: Sequence[Dict],
    annotation_source: str,
    copy_mode: str,
    white_threshold: int,
    merge_max_distance: float,
    bbox_expand_ratio: float,
    bbox_expand_min_pixels: float,
    aggregate_stats: Dict[str, float],
    split_strategy: str,
) -> None:
    samples_count = max(int(aggregate_stats.get("samples", 0)), 1)
    summary = {
        "dataset": "DroneVehicle",
        "annotation_source": annotation_source,
        "copy_mode": copy_mode,
        "num_train": len(samples_train),
        "num_val": len(samples_val),
        "split_strategy": split_strategy,
        "white_threshold": white_threshold,
        "merge_max_distance": merge_max_distance,
        "bbox_expand_ratio": bbox_expand_ratio,
        "bbox_expand_min_pixels": bbox_expand_min_pixels,
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "processing": {
            "cropped_samples": int(aggregate_stats.get("cropped_samples", 0)),
            "average_crop_left": round(aggregate_stats.get("crop_left_total", 0.0) / samples_count, 2),
            "average_crop_top": round(aggregate_stats.get("crop_top_total", 0.0) / samples_count, 2),
            "average_crop_right_margin": round(aggregate_stats.get("crop_right_margin_total", 0.0) / samples_count, 2),
            "average_crop_bottom_margin": round(aggregate_stats.get("crop_bottom_margin_total", 0.0) / samples_count, 2),
            "rgb_objects": int(aggregate_stats.get("rgb_objects", 0)),
            "thermal_objects": int(aggregate_stats.get("thermal_objects", 0)),
            "final_objects": int(aggregate_stats.get("final_objects", 0)),
            "merged_objects": int(aggregate_stats.get("merged_objects", 0)),
            "unmatched_primary_objects": int(aggregate_stats.get("unmatched_primary_objects", 0)),
            "unused_secondary_objects": int(aggregate_stats.get("unused_secondary_objects", 0)),
            "skipped_corrupt_samples": int(aggregate_stats.get("skipped_corrupt_samples", 0)),
        },
        "notes": (
            [
                "This dataset was generated from raw DroneVehicle train/validation files using the official split.",
                "Images are cropped to remove the fixed white border before training.",
                "Merged annotations expand RGB boxes with nearby thermal boxes so a single supervision target better covers both modalities.",
            ]
            if split_strategy == "official_split"
            else [
                "This dataset was generated from raw DroneVehicle validation files available locally.",
                "Images are cropped to remove the fixed white border before training.",
                "Merged annotations expand RGB boxes with nearby thermal boxes so a single supervision target better covers both modalities.",
            ]
        ),
    }
    with (target_root / "dataset_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    train_source_root = Path(args.train_source_root).resolve() if args.train_source_root else None
    val_source_root = Path(args.val_source_root).resolve() if args.val_source_root else None
    target_root = Path(args.target_root).resolve()

    if args.clear_target and target_root.exists():
        shutil.rmtree(target_root)

    ensure_structure(target_root)
    if bool(train_source_root) != bool(val_source_root):
        raise ValueError("--train-source-root and --val-source-root must be provided together.")

    if train_source_root and val_source_root:
        samples_train = collect_samples(train_source_root)
        samples_val = collect_samples(val_source_root)
        if not samples_train:
            raise FileNotFoundError(f"No aligned DroneVehicle training samples found under {train_source_root}")
        if not samples_val:
            raise FileNotFoundError(f"No aligned DroneVehicle validation samples found under {val_source_root}")
        split_strategy = "official_split"
    else:
        samples = collect_samples(source_root)
        if not samples:
            raise FileNotFoundError(f"No aligned DroneVehicle samples found under {source_root}")
        rng = random.Random(args.seed)
        rng.shuffle(samples)
        split_index = max(1, min(len(samples) - 1, int(len(samples) * args.train_ratio)))
        samples_train = sorted(samples[:split_index], key=lambda item: item["sample_id"])
        samples_val = sorted(samples[split_index:], key=lambda item: item["sample_id"])
        split_strategy = "random_split_from_single_source"

    for sample in [*samples_train, *samples_val]:
        with Image.open(sample["rgb_path"]) as image:
            sample["image_size"] = image.size

    aggregate_stats: Dict[str, float] = {
        "samples": 0,
        "cropped_samples": 0,
        "rgb_objects": 0,
        "thermal_objects": 0,
        "final_objects": 0,
        "merged_objects": 0,
        "unmatched_primary_objects": 0,
        "unused_secondary_objects": 0,
        "skipped_corrupt_samples": 0,
        "crop_left_total": 0.0,
        "crop_top_total": 0.0,
        "crop_right_margin_total": 0.0,
        "crop_bottom_margin_total": 0.0,
    }

    for sample in samples_train:
        materialize_sample(
            target_root,
            "train",
            sample,
            args.copy_mode,
            args.annotation_source,
            args.white_threshold,
            args.merge_max_distance,
            args.bbox_expand_ratio,
            args.bbox_expand_min_pixels,
            aggregate_stats,
        )
    for sample in samples_val:
        materialize_sample(
            target_root,
            "val",
            sample,
            args.copy_mode,
            args.annotation_source,
            args.white_threshold,
            args.merge_max_distance,
            args.bbox_expand_ratio,
            args.bbox_expand_min_pixels,
            aggregate_stats,
        )

    write_summary(
        target_root,
        samples_train,
        samples_val,
        args.annotation_source,
        args.copy_mode,
        args.white_threshold,
        args.merge_max_distance,
        args.bbox_expand_ratio,
        args.bbox_expand_min_pixels,
        aggregate_stats,
        split_strategy,
    )

    print(
        json.dumps(
            {
                "target_root": str(target_root),
                "annotation_source": args.annotation_source,
                "copy_mode": args.copy_mode,
                "split_strategy": split_strategy,
                "num_total": len(samples_train) + len(samples_val),
                "num_train": len(samples_train),
                "num_val": len(samples_val),
                "bbox_expand_ratio": args.bbox_expand_ratio,
                "bbox_expand_min_pixels": args.bbox_expand_min_pixels,
                "cropped_samples": int(aggregate_stats["cropped_samples"]),
                "merged_objects": int(aggregate_stats["merged_objects"]),
                "skipped_corrupt_samples": int(aggregate_stats["skipped_corrupt_samples"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
