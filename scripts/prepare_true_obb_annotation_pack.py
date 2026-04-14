from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ELONGATED_CLASS_IDS = {1, 2, 4}  # truck, bus, freight_car


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a true-OBB annotation pack from the official pseudo-OBB dataset.")
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--train-topk", type=int, default=2000)
    parser.add_argument("--val-topk", type=int, default=800)
    return parser.parse_args()


def resolve_source_json(source_root: Path, split: str, sample_stem: str) -> Path:
    annotation_path = source_root / "annotations" / split / f"{sample_stem.replace('_rgb', '')}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing source annotation: {annotation_path}")
    return annotation_path


def link_tree(src_root: Path, dst_root: Path) -> None:
    for path in sorted(src_root.rglob("*")):
        relative = path.relative_to(src_root)
        target = dst_root / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(path, target)


def copy_tree(src_root: Path, dst_root: Path) -> None:
    for path in sorted(src_root.rglob("*")):
        relative = path.relative_to(src_root)
        target = dst_root / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def analyze_annotation(annotation_path: Path, image_path: Path) -> Tuple[Dict, float]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    objects = payload.get("objects", payload.get("annotations", []))
    with Image.open(image_path) as image:
        width, height = image.size

    elongated_count = 0
    edge_count = 0
    crowd_count = len(objects)
    avg_aspect_ratio = 1.0
    aspect_ratios: List[float] = []
    class_hist: Dict[str, int] = {}

    for item in objects:
        bbox = item.get("bbox", item.get("box"))
        class_id = int(item.get("class_id", item.get("category_id", -1)))
        if bbox is None or class_id < 0:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        aspect = max(w / h, h / w)
        aspect_ratios.append(aspect)
        if class_id in ELONGATED_CLASS_IDS:
            elongated_count += 1
        if x1 <= 5 or y1 <= 5 or x2 >= (width - 5) or y2 >= (height - 5):
            edge_count += 1
        class_hist[str(class_id)] = class_hist.get(str(class_id), 0) + 1

    if aspect_ratios:
        avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)

    # Heuristic priority score for true OBB correction.
    score = (
        elongated_count * 4.0
        + edge_count * 1.5
        + max(crowd_count - 3, 0) * 0.5
        + max(avg_aspect_ratio - 1.5, 0.0) * 2.0
    )

    return {
        "num_objects": crowd_count,
        "elongated_objects": elongated_count,
        "edge_objects": edge_count,
        "avg_aspect_ratio": round(avg_aspect_ratio, 4),
        "class_hist": class_hist,
    }, round(score, 4)


def write_dataset_yaml(target_root: Path) -> None:
    yaml_path = target_root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {target_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: car",
                "  1: truck",
                "  2: bus",
                "  3: van",
                "  4: freight_car",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    pseudo_root = args.pseudo_root.resolve()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    # images remain symlinked, labels copied for future manual replacement
    link_tree(pseudo_root / "images", target_root / "images")
    copy_tree(pseudo_root / "labels", target_root / "labels")
    write_dataset_yaml(target_root)

    manifest: Dict[str, List[Dict]] = {"train": [], "val": []}
    split_candidates: Dict[str, List[Dict]] = {"train": [], "val": []}

    for split in ("train", "val"):
        label_dir = target_root / "labels" / split
        image_dir = target_root / "images" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            sample_stem = label_path.stem
            image_path = None
            for ext in IMAGE_EXTS:
                candidate = image_dir / f"{sample_stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            source_annotation_path = resolve_source_json(source_root, split, sample_stem)
            analysis, priority_score = analyze_annotation(source_annotation_path, image_path)
            item = {
                "sample_id": sample_stem,
                "split": split,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "source_annotation_path": str(source_annotation_path),
                "label_status": "pseudo_obb_pending_trueobb",
                "priority_score": priority_score,
                **analysis,
            }
            manifest[split].append(item)
            split_candidates[split].append(item)

    split_candidates["train"].sort(key=lambda item: item["priority_score"], reverse=True)
    split_candidates["val"].sort(key=lambda item: item["priority_score"], reverse=True)

    annotation_manifest = {
        "dataset_root": str(target_root),
        "source_pseudo_root": str(pseudo_root),
        "source_hbb_root": str(source_root),
        "label_status_meaning": {
            "pseudo_obb_pending_trueobb": "Axis-aligned rectangle points exist and need manual true OBB correction.",
        },
        "splits": {
            "train": {
                "images": len(manifest["train"]),
                "priority_topk": args.train_topk,
            },
            "val": {
                "images": len(manifest["val"]),
                "priority_topk": args.val_topk,
            },
        },
        "items": manifest,
    }
    (target_root / "annotation_manifest.json").write_text(
        json.dumps(annotation_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (target_root / "priority_candidates_train.json").write_text(
        json.dumps(split_candidates["train"][: args.train_topk], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (target_root / "priority_candidates_val.json").write_text(
        json.dumps(split_candidates["val"][: args.val_topk], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "dataset_root": str(target_root),
        "source_pseudo_root": str(pseudo_root),
        "source_hbb_root": str(source_root),
        "task": "true_obb_annotation_pack",
        "label_seed": "pseudo_obb",
        "train_images": len(manifest["train"]),
        "val_images": len(manifest["val"]),
        "train_priority_topk": args.train_topk,
        "val_priority_topk": args.val_topk,
        "notes": [
            "Images are symlinked from the pseudo-OBB official dataset.",
            "Labels are copied so manual true-OBB correction can overwrite them in-place.",
            "priority_candidates_* files rank samples that should be annotated first.",
        ],
    }
    (target_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared true OBB annotation pack at {target_root}")


if __name__ == "__main__":
    main()
