from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine pseudo OBB labels with a trained YOLO-OBB model.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_label_line(line: str) -> dict:
    values = line.strip().split()
    class_id = int(float(values[0]))
    coords = [float(v) for v in values[1:9]]
    polygon = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
    return {"class_id": class_id, "polygon": polygon}


def polygon_to_xyxy(points: list[list[float]], width: int, height: int) -> list[float]:
    xs = [p[0] * width for p in points]
    ys = [p[1] * height for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def normalize_polygon(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    return [[float(x) / max(width, 1), float(y) / max(height, 1)] for x, y in points]


def serialize_label(class_id: int, polygon: list[list[float]]) -> str:
    flat = []
    for x, y in polygon:
        flat.extend([x, y])
    return " ".join([str(class_id)] + [f"{value:.6f}" for value in flat])


def refine_file(label_path: Path, image_path: Path, model, conf: float, iou: float, match_iou: float, imgsz: int, device: str) -> dict:
    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [parse_label_line(line) for line in lines]
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = model.predict(source=frame_rgb, verbose=False, imgsz=imgsz, conf=conf, iou=iou, device=device)[0]

    predictions = []
    if result.obb is not None:
        polygons = result.obb.xyxyxyxy.tolist()
        classes = result.obb.cls.tolist()
        scores = result.obb.conf.tolist()
        for polygon, class_id, score in zip(polygons, classes, scores):
            predictions.append(
                {
                    "class_id": int(class_id),
                    "score": float(score),
                    "polygon_abs": [[float(x), float(y)] for x, y in polygon],
                    "bbox_abs": [
                        min(float(x) for x, _ in polygon),
                        min(float(y) for _, y in polygon),
                        max(float(x) for x, _ in polygon),
                        max(float(y) for _, y in polygon),
                    ],
                    "used": False,
                }
            )

    updated = []
    matched = 0
    for label in labels:
        src_bbox = polygon_to_xyxy(label["polygon"], width, height)
        best_index = None
        best_score = -1.0
        for index, pred in enumerate(predictions):
            if pred["used"] or pred["class_id"] != label["class_id"]:
                continue
            overlap = box_iou(src_bbox, pred["bbox_abs"])
            if overlap >= match_iou and overlap > best_score:
                best_score = overlap
                best_index = index
        if best_index is not None:
            predictions[best_index]["used"] = True
            polygon = normalize_polygon(predictions[best_index]["polygon_abs"], width, height)
            matched += 1
        else:
            polygon = label["polygon"]
        updated.append(serialize_label(label["class_id"], polygon))

    label_path.write_text("\n".join(updated) + ("\n" if updated else ""), encoding="utf-8")
    return {
        "label_path": str(label_path),
        "image_path": str(image_path),
        "original_count": len(labels),
        "prediction_count": len(predictions),
        "matched_count": matched,
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    train_candidates = load_json(dataset_root / "priority_candidates_train.json")[: args.train_count]
    val_candidates = load_json(dataset_root / "priority_candidates_val.json")[: args.val_count]

    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    backup_root = dataset_root / "labels_pseudo_backup"
    backup_root.mkdir(parents=True, exist_ok=True)

    summary = {"train": [], "val": []}
    for split, candidates in (("train", train_candidates), ("val", val_candidates)):
        for sample in candidates:
            label_path = Path(sample["label_path"])
            image_path = Path(sample["image_path"])
            backup_path = backup_root / split / label_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                backup_path.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")
            result = refine_file(
                label_path=label_path,
                image_path=image_path,
                model=model,
                conf=float(args.conf),
                iou=float(args.iou),
                match_iou=float(args.match_iou),
                imgsz=int(args.imgsz),
                device=args.device,
            )
            summary[split].append(result)

    output_path = dataset_root / "model_refine_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved model-assisted refinement summary to {output_path}")


if __name__ == "__main__":
    main()
