from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an RGB-only YOLO-OBB dataset from existing JSON HBB annotations.")
    parser.add_argument("--source-root", type=Path, required=True, help="Source dataset root, e.g. datasets/dronevehicle_like_rgb_expand_refined")
    parser.add_argument("--target-root", type=Path, required=True, help="Target YOLO-OBB dataset root")
    parser.add_argument("--class-mapping-path", type=Path, default=Path("/home/liujuncheng/rgbt_uav_detection/class_mapping.json"))
    parser.add_argument("--rgb-dir", type=str, default="rgb")
    parser.add_argument("--annotation-dir", type=str, default="annotations")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--rgb-suffix", type=str, default="_rgb")
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    return parser.parse_args()


def load_class_mapping(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return {str(k): str(v) for k, v in payload.items()}


def find_rgb_image(rgb_root: Path, sample_id: str, rgb_suffix: str) -> Path:
    candidates = []
    for ext in IMAGE_EXTS:
        candidates.append(rgb_root / f"{sample_id}{rgb_suffix}{ext}")
        candidates.append(rgb_root / f"{sample_id}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"RGB image not found for sample_id={sample_id} under {rgb_root}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize_image(src: Path, dst: Path, link_mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "symlink":
        os.symlink(src, dst)
        return
    if link_mode == "hardlink":
        os.link(src, dst)
        return
    dst.write_bytes(src.read_bytes())


def bbox_to_yolo_obb_line(class_id: int, bbox: Iterable[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {bbox}")
    points = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]
    normalized = []
    for x, y in points:
        normalized.extend([x / max(width, 1), y / max(height, 1)])
    return " ".join([str(int(class_id))] + [f"{value:.6f}" for value in normalized])


def convert_annotation(annotation_path: Path, target_label_path: Path, width: int, height: int) -> int:
    with annotation_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    objects = payload.get("objects", payload.get("annotations", []))
    lines = []
    for item in objects:
        bbox = item.get("bbox", item.get("box"))
        class_id = item.get("class_id", item.get("category_id"))
        if bbox is None or class_id is None:
            continue
        try:
            lines.append(bbox_to_yolo_obb_line(int(class_id), bbox, width, height))
        except ValueError:
            continue
    ensure_parent(target_label_path)
    target_label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_dataset_yaml(target_root: Path, class_mapping: Dict[str, str]) -> Path:
    yaml_lines = [
        f"path: {target_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for class_id, class_name in sorted(class_mapping.items(), key=lambda item: int(item[0])):
        yaml_lines.append(f"  {int(class_id)}: {class_name}")
    yaml_path = target_root / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    class_mapping_path = args.class_mapping_path.resolve()
    class_mapping = load_class_mapping(class_mapping_path)

    summary: Dict[str, object] = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "task": "yolo_obb_rgb",
        "geometry": "pseudo_obb_from_hbb",
        "link_mode": args.link_mode,
        "rgb_suffix": args.rgb_suffix,
        "splits": {},
        "class_mapping": class_mapping,
        "notes": [
            "This dataset converts horizontal boxes to axis-aligned 4-point OBB labels.",
            "The generated labels are suitable for OBB pipeline smoke tests, not true angle supervision.",
        ],
    }

    for split in args.splits:
        rgb_root = source_root / args.rgb_dir / split
        ann_root = source_root / args.annotation_dir / split
        target_img_root = target_root / "images" / split
        target_label_root = target_root / "labels" / split
        target_img_root.mkdir(parents=True, exist_ok=True)
        target_label_root.mkdir(parents=True, exist_ok=True)

        image_count = 0
        object_count = 0
        skipped = 0
        for annotation_path in sorted(ann_root.glob("*.json")):
            sample_id = annotation_path.stem
            try:
                rgb_image_path = find_rgb_image(rgb_root, sample_id, args.rgb_suffix)
            except FileNotFoundError:
                skipped += 1
                continue

            from PIL import Image

            with Image.open(rgb_image_path) as image:
                width, height = image.size

            target_image_path = target_img_root / rgb_image_path.name
            target_label_path = target_label_root / f"{target_image_path.stem}.txt"
            materialize_image(rgb_image_path, target_image_path, args.link_mode)
            object_count += convert_annotation(annotation_path, target_label_path, width, height)
            image_count += 1

        summary["splits"][split] = {
            "images": image_count,
            "objects": object_count,
            "skipped_missing_rgb": skipped,
        }

    dataset_yaml_path = write_dataset_yaml(target_root, class_mapping)
    summary["dataset_yaml"] = str(dataset_yaml_path)
    (target_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"YOLO-OBB RGB dataset created at {target_root}")
    print(f"Dataset YAML: {dataset_yaml_path}")


if __name__ == "__main__":
    main()
