from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a LabelMe-compatible true OBB annotation pack from priority candidate lists.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--val-count", type=int, default=100)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def load_pseudo_polygon(label_path: Path, class_id: int, target_index: int) -> list[list[float]]:
    lines = [line.strip().split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matched = [line for line in lines if int(float(line[0])) == class_id]
    if not matched:
        return []
    index = min(target_index, len(matched) - 1)
    values = [float(v) for v in matched[index][1:9]]
    polygon = []
    for i in range(0, len(values), 2):
        polygon.append([values[i], values[i + 1]])
    return polygon


def denormalize_polygon(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    if not points:
        return []
    return [[round(p[0] * width, 2), round(p[1] * height, 2)] for p in points]


def write_labelme_json(image_path: Path, target_json_path: Path, sample: dict) -> None:
    with Image.open(image_path) as image:
        width, height = image.size

    shapes = []
    class_hist = sample.get("class_hist", {})
    # We seed one polygon per class occurrence order, using pseudo-OBB points as editable initialization.
    offset_per_class: dict[int, int] = {}
    for class_id_str, count in sorted(class_hist.items(), key=lambda item: int(item[0])):
        class_id = int(class_id_str)
        for _ in range(int(count)):
            index = offset_per_class.get(class_id, 0)
            offset_per_class[class_id] = index + 1
            pseudo_polygon = load_pseudo_polygon(Path(sample["label_path"]), class_id, index)
            points = denormalize_polygon(pseudo_polygon, width, height)
            if not points:
                continue
            shapes.append(
                {
                    "label": str(class_id),
                    "points": points,
                    "group_id": None,
                    "description": "pseudo_obb_seed_edit_to_true_obb",
                    "shape_type": "polygon",
                    "flags": {},
                }
            )

    payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    target_json_path.parent.mkdir(parents=True, exist_ok=True)
    target_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_split(candidates: list[dict], output_root: Path, split: str, count: int) -> list[dict]:
    exported = []
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    for sample in candidates[:count]:
        image_path = Path(sample["image_path"])
        image_target = split_root / image_path.name
        json_target = split_root / f"{image_path.stem}.json"
        ensure_symlink(image_path, image_target)
        write_labelme_json(image_target, json_target, sample)
        exported.append(
            {
                "sample_id": sample["sample_id"],
                "image": str(image_target),
                "labelme_json": str(json_target),
                "priority_score": sample["priority_score"],
                "num_objects": sample["num_objects"],
                "elongated_objects": sample["elongated_objects"],
                "edge_objects": sample["edge_objects"],
            }
        )
    return exported


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_candidates = load_json(dataset_root / "priority_candidates_train.json")
    val_candidates = load_json(dataset_root / "priority_candidates_val.json")

    exported_train = export_split(train_candidates, output_root, "train", args.train_count)
    exported_val = export_split(val_candidates, output_root, "val", args.val_count)

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "format": "labelme_polygon_seed_pack",
        "train_count": len(exported_train),
        "val_count": len(exported_val),
        "notes": [
            "Image files are symlinked into the pack.",
            "LabelMe JSON files are initialized from pseudo OBB polygons and must be manually corrected to true OBB.",
            "The label field currently stores class_id; rename to class_name in the annotation tool if preferred.",
        ],
        "train": exported_train,
        "val": exported_val,
    }
    (output_root / "pack_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LabelMe seed pack exported to {output_root}")


if __name__ == "__main__":
    main()
