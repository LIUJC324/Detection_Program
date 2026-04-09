from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import RGBTTargetDataset, rgbt_collate_fn
from data.transforms import TransformConfig, build_val_transforms
from model.detector import build_model
from scripts.eval_utils import evaluate_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the RGB-T detector.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "outputs" / "evaluation.json"))
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--diagnostic-thresholds", type=str, default="0.05,0.10,0.20,0.30,0.50")
    return parser.parse_args()


def parse_thresholds(raw_value: str) -> List[float]:
    if not raw_value.strip():
        return []
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def merge_nested_dict(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_named_class_metrics(
    per_class_stats: Dict[int, Dict[str, float]],
    class_mapping: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    named_metrics: Dict[str, Dict[str, float]] = {}
    for class_id, stats in per_class_stats.items():
        class_name = class_mapping.get(str(class_id), f"class_{class_id}")
        named_metrics[class_name] = {
            "class_id": class_id,
            "ap50": round(float(stats["ap50"]), 6),
            "num_gt": int(stats["num_gt"]),
            "num_predictions": int(stats["num_predictions"]),
            "tp": int(stats["tp"]),
            "fp": int(stats["fp"]),
            "fn": int(stats["fn"]),
            "avg_score": round(float(stats["avg_score"]), 6),
        }
    return named_metrics


def to_eval_target(target: Dict) -> Dict[str, torch.Tensor]:
    return {
        "boxes": target["boxes"].detach().cpu(),
        "labels": target["labels"].detach().cpu(),
    }


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config:
        config = merge_nested_dict(config, checkpoint_config)
        if "resize_mode" not in checkpoint_config.get("dataset", {}):
            config.setdefault("dataset", {})["resize_mode"] = "stretch"

    dataset_cfg = config["dataset"]
    transform_cfg = TransformConfig(
        image_size=dataset_cfg["image_size"],
        resize_mode=dataset_cfg.get("resize_mode", "stretch"),
        letterbox_pad_value=dataset_cfg.get("letterbox_pad_value", 114),
    )
    diagnostic_thresholds = parse_thresholds(args.diagnostic_thresholds)
    score_threshold = args.score_threshold if args.score_threshold is not None else float(config["model"].get("score_thresh", 0.2))
    if diagnostic_thresholds:
        score_threshold = min(score_threshold, min(diagnostic_thresholds))
    config["model"]["score_thresh"] = score_threshold

    class_mapping_path = resolve_path(dataset_cfg["class_mapping_path"])
    with class_mapping_path.open("r", encoding="utf-8") as fp:
        class_mapping = json.load(fp)

    dataset = RGBTTargetDataset(
        root=resolve_path(dataset_cfg["root"]),
        split=dataset_cfg["split_val"],
        rgb_dir=dataset_cfg["rgb_dir"],
        thermal_dir=dataset_cfg["thermal_dir"],
        annotation_dir=dataset_cfg["annotation_dir"],
        rgb_suffix=dataset_cfg["rgb_suffix"],
        thermal_suffix=dataset_cfg["thermal_suffix"],
        transform=build_val_transforms(transform_cfg),
        allow_empty_annotations=dataset_cfg.get("allow_empty_annotations", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, config["train"]["batch_size"] // 2),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else config["train"]["num_workers"],
        pin_memory=True,
        collate_fn=rgbt_collate_fn,
    )

    requested_device = args.device or config["train"].get("device", "cuda")
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            rgb = [tensor.to(device) for tensor in batch["rgb"]]
            thermal = [tensor.to(device) for tensor in batch["thermal"]]
            outputs = model(rgb, thermal)
            predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
            targets.extend([to_eval_target(target) for target in batch["targets"]])

    metrics = evaluate_predictions(
        predictions=predictions,
        targets=targets,
        num_classes=config["model"]["num_classes"],
        small_object_area=config["train"]["small_object_area"],
        score_thresholds=diagnostic_thresholds,
    )
    per_class_named = build_named_class_metrics(metrics.per_class_stats, class_mapping)
    payload = {
        "model_score_threshold": score_threshold,
        "diagnostic_thresholds": diagnostic_thresholds,
        "map50": metrics.map50,
        "recall50": metrics.recall50,
        "small_recall50": metrics.small_recall50,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "per_class_ap50": {class_name: stats["ap50"] for class_name, stats in per_class_named.items()},
        "per_class_stats": per_class_named,
        "threshold_sweep": {
            f"{threshold:.2f}": {
                "map50": round(float(values["map50"]), 6),
                "recall50": round(float(values["recall50"]), 6),
                "small_recall50": round(float(values["small_recall50"]), 6),
                "true_positives": int(values["true_positives"]),
                "false_positives": int(values["false_positives"]),
                "false_negatives": int(values["false_negatives"]),
            }
            for threshold, values in metrics.threshold_metrics.items()
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
