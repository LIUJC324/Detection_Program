from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import RGBTTargetDataset, rgbt_collate_fn
from data.transforms import TransformConfig, build_train_transforms, build_val_transforms
from model.detector import build_model, load_checkpoint, save_checkpoint
from model.loss import SmallObjectLossAggregator
from scripts.eval_utils import evaluate_predictions


def supports_live_progress(stream) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RGB-T UAV detection model.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    return config


def resolve_path(project_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_dir(base_output_dir: str) -> Path:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, allow_unicode=True, sort_keys=False)


def append_csv(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(message.rstrip() + "\n")


def load_epoch_metrics(path: Path) -> list[Dict]:
    if not path.exists():
        return []
    rows: list[Dict] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if not row:
                continue
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "map50": float(row["map50"]),
                    "recall50": float(row["recall50"]),
                    "small_recall50": float(row["small_recall50"]),
                    "lr": float(row["lr"]),
                }
            )
    return rows


def load_resume_history(output_dir: str | Path, current_run_dir: Path, completed_epochs: int) -> list[Dict]:
    history_by_epoch: Dict[int, Dict] = {}
    for run_dir in sorted(Path(output_dir).glob("run_*")):
        if run_dir.resolve() == current_run_dir.resolve():
            continue
        for row in load_epoch_metrics(run_dir / "epoch_metrics.csv"):
            epoch = int(row["epoch"])
            if epoch > completed_epochs:
                continue
            history_by_epoch[epoch] = row
    return [history_by_epoch[epoch] for epoch in sorted(history_by_epoch)]


def is_resume_history_compatible(checkpoint_config: Dict | None, current_config: Dict) -> bool:
    if not checkpoint_config:
        return True
    checkpoint_dataset = checkpoint_config.get("dataset", {})
    current_dataset = current_config.get("dataset", {})
    keys = (
        "root",
        "split_train",
        "split_val",
        "rgb_dir",
        "thermal_dir",
        "annotation_dir",
        "rgb_suffix",
        "thermal_suffix",
        "num_classes",
        "image_size",
        "resize_mode",
        "letterbox_pad_value",
    )
    return all(str(checkpoint_dataset.get(key)) == str(current_dataset.get(key)) for key in keys)


def tensor_or_array_to_image(image) -> np.ndarray:
    if torch.is_tensor(image):
        array = image.detach().cpu().permute(1, 2, 0).numpy()
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        return array
    return np.array(image, copy=True)


def draw_boxes(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, class_mapping: Dict[str, str]) -> np.ndarray:
    canvas = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        class_name = class_mapping.get(str(int(label)), f"class_{int(label)}")
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            canvas,
            class_name,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def save_dataset_previews(train_dataset: RGBTTargetDataset, val_dataset: RGBTTargetDataset, class_mapping: Dict[str, str], run_dir: Path, count: int = 3) -> None:
    preview_dir = run_dir / "dataset_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    def export_samples(dataset: RGBTTargetDataset, split_name: str) -> None:
        sample_count = min(count, len(dataset))
        for idx in range(sample_count):
            sample = dataset[idx]
            rgb = tensor_or_array_to_image(sample["rgb"])
            thermal = tensor_or_array_to_image(sample["thermal"])
            boxes = sample["targets"]["boxes"]
            labels = sample["targets"]["labels"]
            if torch.is_tensor(boxes):
                boxes = boxes.detach().cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().numpy()
            rgb_vis = draw_boxes(rgb, boxes, labels, class_mapping)
            thermal_vis = draw_boxes(thermal, boxes, labels, class_mapping)
            merged = np.concatenate([rgb_vis, thermal_vis], axis=1)
            out_path = preview_dir / f"{split_name}_{idx:02d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))

    export_samples(train_dataset, "train")
    export_samples(val_dataset, "val")


def build_dataset_stats(dataset: RGBTTargetDataset, class_mapping: Dict[str, str]) -> Dict:
    counts = {name: 0 for name in class_mapping.values()}
    total_boxes = 0
    empty_annotations = 0
    for sample in dataset.samples:
        annotation = dataset._load_annotation(sample["annotation_path"])
        labels = annotation["labels"]
        if len(labels) == 0:
            empty_annotations += 1
        for label in labels.tolist():
            counts[class_mapping.get(str(int(label)), f"class_{int(label)}")] = counts.get(
                class_mapping.get(str(int(label)), f"class_{int(label)}"),
                0,
            ) + 1
        total_boxes += int(len(labels))
    return {
        "num_samples": len(dataset),
        "total_boxes": total_boxes,
        "empty_annotations": empty_annotations,
        "boxes_per_class": counts,
    }


def save_dataset_summary(train_dataset: RGBTTargetDataset, val_dataset: RGBTTargetDataset, class_mapping_path: str, run_dir: Path) -> Dict:
    with open(class_mapping_path, "r", encoding="utf-8") as fp:
        class_mapping = json.load(fp)

    summary = {
        "class_mapping": class_mapping,
        "train": build_dataset_stats(train_dataset, class_mapping),
        "val": build_dataset_stats(val_dataset, class_mapping),
    }
    with (run_dir / "dataset_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    save_dataset_previews(train_dataset, val_dataset, class_mapping, run_dir)
    return summary


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def save_interrupt_state(
    run_dir: Path,
    checkpoint_dir: str | Path,
    run_checkpoint_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    resume_epoch: int,
    current_epoch: int,
    current_phase: str,
    current_step: int,
    global_step: int,
    optimizer_step: int,
    config: Dict,
) -> Dict:
    interrupt_state = {
        "interrupted": True,
        "current_epoch": current_epoch + 1 if current_epoch >= 0 else 0,
        "phase": current_phase,
        "step_in_phase": current_step,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "resume_epoch_index": resume_epoch,
        "resume_next_epoch": max(resume_epoch + 2, 1),
    }
    save_json(run_dir / "interrupt_state.json", interrupt_state)

    interrupt_path = Path(checkpoint_dir) / "interrupt_last.pt"
    run_interrupt_path = run_checkpoint_dir / "interrupt_last.pt"
    save_checkpoint(model, optimizer, scheduler, scaler, resume_epoch, config, interrupt_path)
    save_checkpoint(model, optimizer, scheduler, scaler, resume_epoch, config, run_interrupt_path)
    return {
        "interrupt_path": str(interrupt_path),
        "run_interrupt_path": str(run_interrupt_path),
        "resume_next_epoch": int(interrupt_state["resume_next_epoch"]),
    }


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Dict):
    total_epochs = max(int(train_cfg["epochs"]), 1)
    warmup_epochs = max(int(train_cfg.get("warmup_epochs", 0)), 0)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(train_cfg.get("warmup_start_factor", 0.2)),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)


def advance_scheduler_for_resume(scheduler, completed_epochs: int) -> None:
    if scheduler is None or completed_epochs <= 0:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()` was not necessary.*")
        for _ in range(completed_epochs):
            scheduler.step()


def step_scheduler(scheduler) -> None:
    if scheduler is None:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()` was not necessary.*")
        scheduler.step()


def build_class_balanced_sampler(
    dataset: RGBTTargetDataset,
    num_classes: int,
    power: float = 0.5,
):
    class_counts = {cls_id: 0 for cls_id in range(num_classes)}
    sample_labels: list[list[int]] = []
    for sample in dataset.samples:
        labels = dataset._load_annotation(sample["annotation_path"])["labels"]
        labels_list = [int(label) for label in labels.tolist()]
        sample_labels.append(sorted(set(labels_list)))
        for label in labels_list:
            class_counts[label] = class_counts.get(label, 0) + 1

    max_count = max((count for count in class_counts.values() if count > 0), default=1)
    class_weights = {
        cls_id: ((max_count / count) ** power) if count > 0 else 1.0
        for cls_id, count in class_counts.items()
    }
    sample_weights = []
    for labels in sample_labels:
        if not labels:
            sample_weights.append(1.0)
            continue
        weight = sum(class_weights[label] for label in labels) / len(labels)
        sample_weights.append(max(1.0, float(weight)))

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    sampler_summary = {
        "enabled": True,
        "power": power,
        "class_counts": class_counts,
        "class_weights": {str(key): round(float(value), 6) for key, value in class_weights.items()},
        "sample_weight_min": round(float(min(sample_weights, default=1.0)), 6),
        "sample_weight_mean": round(float(sum(sample_weights) / max(len(sample_weights), 1)), 6),
        "sample_weight_max": round(float(max(sample_weights, default=1.0)), 6),
    }
    return sampler, sampler_summary


def plot_training_curves(epoch_history: list[Dict], run_dir: Path) -> None:
    if not epoch_history:
        return
    epochs = [item["epoch"] for item in epoch_history]
    train_losses = [item["train_loss"] for item in epoch_history]
    map50 = [item["map50"] for item in epoch_history]
    recall50 = [item["recall50"] for item in epoch_history]
    small_recall50 = [item["small_recall50"] for item in epoch_history]
    lr_values = [item["lr"] for item in epoch_history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, train_losses, marker="o")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, map50, marker="o", label="mAP@0.5")
    axes[0, 1].plot(epochs, recall50, marker="o", label="Recall@0.5")
    axes[0, 1].plot(epochs, small_recall50, marker="o", label="Small Recall@0.5")
    axes[0, 1].set_title("Validation Metrics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, lr_values, marker="o", color="tab:green")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"epochs_logged: {len(epoch_history)}",
                f"best_map50: {max(map50):.4f}",
                f"last_train_loss: {train_losses[-1]:.4f}",
                f"last_recall50: {recall50[-1]:.4f}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def move_targets_to_device(targets, device):
    moved = []
    for target in targets:
        moved.append(
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
                "image_id": target["image_id"].to(device),
                "orig_size": target["orig_size"].to(device),
            }
        )
    return moved


def build_dataloaders(config: Dict):
    dataset_cfg = config["dataset"]
    augment_cfg = config["augment"]
    train_cfg = config["train"]
    transform_cfg = TransformConfig(
        image_size=dataset_cfg["image_size"],
        resize_mode=dataset_cfg.get("resize_mode", "stretch"),
        letterbox_pad_value=dataset_cfg.get("letterbox_pad_value", 114),
        horizontal_flip_prob=augment_cfg["horizontal_flip_prob"],
        vertical_flip_prob=augment_cfg["vertical_flip_prob"],
        random_crop_prob=augment_cfg["random_crop_prob"],
        crop_min_scale=augment_cfg["crop_min_scale"],
        color_jitter_prob=augment_cfg["color_jitter_prob"],
        brightness=augment_cfg["brightness"],
        contrast=augment_cfg["contrast"],
        saturation=augment_cfg["saturation"],
        lowlight_aug_prob=augment_cfg.get("lowlight_aug_prob", 0.0),
        lowlight_gamma_min=augment_cfg.get("lowlight_gamma_min", 1.4),
        lowlight_gamma_max=augment_cfg.get("lowlight_gamma_max", 2.2),
        weak_modality_prob=augment_cfg.get("weak_modality_prob", 0.0),
        weak_rgb_primary_prob=augment_cfg.get("weak_rgb_primary_prob", 0.7),
        weak_modality_min_scale=augment_cfg.get("weak_modality_min_scale", 0.05),
        weak_modality_max_scale=augment_cfg.get("weak_modality_max_scale", 0.35),
        weak_modality_blur_prob=augment_cfg.get("weak_modality_blur_prob", 0.5),
        weak_modality_noise_std=augment_cfg.get("weak_modality_noise_std", 0.03),
        motion_blur_prob=augment_cfg.get("motion_blur_prob", 0.0),
        motion_blur_kernel_sizes=tuple(augment_cfg.get("motion_blur_kernel_sizes", [3, 5, 7])),
    )

    train_dataset = RGBTTargetDataset(
        root=dataset_cfg["root"],
        split=dataset_cfg["split_train"],
        rgb_dir=dataset_cfg["rgb_dir"],
        thermal_dir=dataset_cfg["thermal_dir"],
        annotation_dir=dataset_cfg["annotation_dir"],
        rgb_suffix=dataset_cfg["rgb_suffix"],
        thermal_suffix=dataset_cfg["thermal_suffix"],
        transform=build_train_transforms(transform_cfg),
        allow_empty_annotations=dataset_cfg.get("allow_empty_annotations", False),
    )
    val_dataset = RGBTTargetDataset(
        root=dataset_cfg["root"],
        split=dataset_cfg["split_val"],
        rgb_dir=dataset_cfg["rgb_dir"],
        thermal_dir=dataset_cfg["thermal_dir"],
        annotation_dir=dataset_cfg["annotation_dir"],
        rgb_suffix=dataset_cfg["rgb_suffix"],
        thermal_suffix=dataset_cfg["thermal_suffix"],
        transform=build_val_transforms(transform_cfg),
        allow_empty_annotations=dataset_cfg.get("allow_empty_annotations", False),
    )

    train_sampler = None
    sampler_summary = {
        "enabled": False,
    }
    if train_cfg.get("balance_sampling", False):
        train_sampler, sampler_summary = build_class_balanced_sampler(
            train_dataset,
            num_classes=config["model"]["num_classes"],
            power=float(train_cfg.get("class_balance_power", 0.5)),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=rgbt_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, train_cfg["batch_size"] // 2),
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=rgbt_collate_fn,
    )
    return train_loader, val_loader, sampler_summary


def validate(model, data_loader, device, num_classes: int, small_object_area: float, show_progress: bool = True):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", leave=False, disable=not show_progress):
            rgb = [tensor.to(device) for tensor in batch["rgb"]]
            thermal = [tensor.to(device) for tensor in batch["thermal"]]
            targets = move_targets_to_device(batch["targets"], device)
            outputs = model(rgb, thermal)
            all_predictions.extend([{k: v.detach().cpu() for k, v in out.items()} for out in outputs])
            all_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
    metrics = evaluate_predictions(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=num_classes,
        iou_threshold=0.5,
        small_object_area=small_object_area,
    )
    model.train()
    return metrics


def resolve_checkpoint_arg(cli_value: str, config_value: str, project_root: Path) -> str:
    value = cli_value or config_value or ""
    if not value:
        return ""
    return resolve_path(project_root, value)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.num_workers is not None:
        config.setdefault("train", {})["num_workers"] = args.num_workers
    config["dataset"]["root"] = resolve_path(PROJECT_ROOT, config["dataset"]["root"])
    config["dataset"]["class_mapping_path"] = resolve_path(PROJECT_ROOT, config["dataset"]["class_mapping_path"])
    config["train"]["output_dir"] = resolve_path(PROJECT_ROOT, config["train"]["output_dir"])
    config["train"]["checkpoint_dir"] = resolve_path(PROJECT_ROOT, config["train"]["checkpoint_dir"])
    Path(config["train"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["train"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir(config["train"]["output_dir"])
    run_checkpoint_dir = run_dir / "checkpoints"
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(run_dir / "config_resolved.yaml", config)
    log_file = run_dir / "run.log"

    set_seed(config.get("seed", 42))

    requested_device = args.device or config["train"].get("device", "cuda")
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")

    train_loader, val_loader, sampler_summary = build_dataloaders(config)
    save_dataset_summary(train_loader.dataset, val_loader.dataset, config["dataset"]["class_mapping_path"], run_dir)
    save_json(run_dir / "sampler_summary.json", sampler_summary)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, config["train"])
    scaler = torch.amp.GradScaler(enabled=config["train"].get("amp", True) and device.type == "cuda")
    aggregator = SmallObjectLossAggregator(
        small_object_area=config["train"]["small_object_area"],
        small_object_boost=config["train"]["small_object_boost"],
    )
    accumulate_steps = max(1, int(config["train"].get("accumulate_steps", 1)))
    effective_batch_size = int(config["train"]["batch_size"]) * accumulate_steps

    start_epoch = 0
    completed_epochs = 0
    checkpoint = None
    init_checkpoint = resolve_checkpoint_arg(
        args.init_checkpoint,
        config.get("train", {}).get("init_checkpoint", ""),
        PROJECT_ROOT,
    )
    resume_history_compatible = True
    if args.resume:
        checkpoint = load_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        completed_epochs = start_epoch
        resume_history_compatible = is_resume_history_compatible(checkpoint.get("config"), config)
        if checkpoint.get("scheduler_state_dict") is None:
            advance_scheduler_for_resume(scheduler, completed_epochs)
    elif init_checkpoint:
        checkpoint = load_checkpoint(model, init_checkpoint, map_location=device, strict=False)
        log_line(log_file, f"init_from={init_checkpoint} weights_only_init=true")
        missing_keys = checkpoint.get("_load_state_dict_missing_keys", [])
        unexpected_keys = checkpoint.get("_load_state_dict_unexpected_keys", [])
        if missing_keys or unexpected_keys:
            log_line(log_file, f"init_missing_keys={missing_keys}")
            log_line(log_file, f"init_unexpected_keys={unexpected_keys}")

    epoch_history = []
    best_map50 = 0.0
    if resume_history_compatible:
        epoch_history = load_resume_history(config["train"]["output_dir"], run_dir, completed_epochs)
        best_map50 = max((item["map50"] for item in epoch_history), default=0.0)
    global_step = len(train_loader) * completed_epochs
    optimizer_step = math.ceil(len(train_loader) / accumulate_steps) * completed_epochs
    show_live_progress = supports_live_progress(sys.stdout)
    log_interval = max(1, config["train"].get("log_interval", 10))
    current_epoch = start_epoch - 1
    current_step = 0
    current_phase = "idle"
    epoch_in_progress = False
    log_line(log_file, f"run_dir={run_dir}")
    log_line(log_file, f"device={device}")
    log_line(log_file, f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)}")
    log_line(log_file, f"accumulate_steps={accumulate_steps} effective_batch_size={effective_batch_size}")
    log_line(
        log_file,
        f"warmup_epochs={int(config['train'].get('warmup_epochs', 0))} warmup_start_factor={float(config['train'].get('warmup_start_factor', 0.2)):.4f}",
    )
    log_line(log_file, f"class_balance_sampling={bool(sampler_summary.get('enabled', False))}")
    log_line(log_file, f"show_live_progress={show_live_progress}")
    if args.resume:
        log_line(
            log_file,
            f"resume_from={args.resume} start_epoch={start_epoch + 1} restored_history_epochs={len(epoch_history)} best_map50={best_map50:.6f}",
        )
        log_line(
            log_file,
            f"scheduler_state_restored={checkpoint.get('scheduler_state_dict') is not None}",
        )
        if not resume_history_compatible:
            log_line(log_file, "resume_history_reset=true reason=dataset_config_changed")
    try:
        for epoch in range(start_epoch, config["train"]["epochs"]):
            current_epoch = epoch
            current_step = 0
            current_phase = "train"
            epoch_in_progress = True
            model.train()
            epoch_loss = 0.0
            epoch_total = len(train_loader)
            optimizer.zero_grad(set_to_none=True)
            if not show_live_progress:
                print(
                    f"epoch={epoch + 1} phase=train status=started total_steps={epoch_total}",
                    flush=True,
                )
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config['train']['epochs']}",
                disable=not show_live_progress,
            )
            for step, batch in enumerate(progress, start=1):
                current_step = step
                rgb = [tensor.to(device) for tensor in batch["rgb"]]
                thermal = [tensor.to(device) for tensor in batch["thermal"]]
                targets = move_targets_to_device(batch["targets"], device)

                with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    loss_dict = model(rgb, thermal, targets)
                    total_loss = aggregator(loss_dict, targets)
                loss_to_backprop = total_loss / accumulate_steps
                scaler.scale(loss_to_backprop).backward()

                should_step = step % accumulate_steps == 0 or step == epoch_total
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                epoch_loss += float(total_loss.item())
                avg_loss = epoch_loss / step
                global_step += 1
                if show_live_progress:
                    progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
                if step == 1 or step % log_interval == 0 or step == epoch_total:
                    step_row = {
                        "epoch": epoch + 1,
                        "step": step,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "accumulate_steps": accumulate_steps,
                        "optimizer_updated": int(should_step),
                        "loss": round(float(total_loss.item()), 6),
                        "avg_loss": round(float(avg_loss), 6),
                        "lr": round(float(optimizer.param_groups[0]["lr"]), 10),
                    }
                    append_csv(run_dir / "train_steps.csv", step_row)
                    append_jsonl(run_dir / "train_steps.jsonl", step_row)
                    if not show_live_progress:
                        print(
                            f"epoch={epoch + 1} phase=train step={step}/{epoch_total} "
                            f"global_step={global_step} optimizer_step={optimizer_step} "
                            f"loss={step_row['loss']:.4f} "
                            f"avg_loss={step_row['avg_loss']:.4f} lr={step_row['lr']:.6f}",
                            flush=True,
                        )

            step_scheduler(scheduler)
            current_phase = "validate"
            current_step = 0
            if not show_live_progress:
                print(
                    f"epoch={epoch + 1} phase=validate status=started batches={len(val_loader)}",
                    flush=True,
                )

            metrics = validate(
                model,
                val_loader,
                device,
                num_classes=config["model"]["num_classes"],
                small_object_area=config["train"]["small_object_area"],
                show_progress=show_live_progress,
            )
            if not show_live_progress:
                print(
                    f"epoch={epoch + 1} phase=validate status=completed "
                    f"map50={metrics.map50:.4f} recall50={metrics.recall50:.4f} "
                    f"small_recall50={metrics.small_recall50:.4f} "
                    f"tp={metrics.true_positives} fp={metrics.false_positives} fn={metrics.false_negatives}",
                    flush=True,
                )

            current_phase = "checkpoint"
            latest_path = Path(config["train"]["checkpoint_dir"]) / "last.pt"
            run_latest_path = run_checkpoint_dir / "last.pt"
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, config, latest_path)
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, config, run_latest_path)
            if metrics.map50 >= best_map50:
                best_map50 = metrics.map50
                best_path = Path(config["train"]["checkpoint_dir"]) / "best.pt"
                run_best_path = run_checkpoint_dir / "best.pt"
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, config, best_path)
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, config, run_best_path)

            epoch_row = {
                "epoch": epoch + 1,
                "train_loss": round(epoch_loss / max(len(train_loader), 1), 6),
                "map50": round(metrics.map50, 6),
                "recall50": round(metrics.recall50, 6),
                "small_recall50": round(metrics.small_recall50, 6),
                "true_positives": int(metrics.true_positives),
                "false_positives": int(metrics.false_positives),
                "false_negatives": int(metrics.false_negatives),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 10),
            }
            epoch_history.append(epoch_row)
            append_csv(run_dir / "epoch_metrics.csv", epoch_row)
            append_jsonl(run_dir / "epoch_metrics.jsonl", epoch_row)
            with (run_dir / "latest_metrics.json").open("w", encoding="utf-8") as fp:
                json.dump(epoch_row, fp, ensure_ascii=False, indent=2)
            plot_training_curves(epoch_history, run_dir)

            summary_line = (
                f"epoch={epoch + 1} "
                f"train_loss={epoch_row['train_loss']:.4f} "
                f"map50={epoch_row['map50']:.4f} "
                f"recall50={epoch_row['recall50']:.4f} "
                f"small_recall50={epoch_row['small_recall50']:.4f} "
                f"tp={epoch_row['true_positives']} "
                f"fp={epoch_row['false_positives']} "
                f"fn={epoch_row['false_negatives']} "
                f"lr={epoch_row['lr']:.6f}"
            )
            log_line(log_file, summary_line)
            print(summary_line, flush=True)
            current_phase = "idle"
            current_step = 0
            epoch_in_progress = False
    except KeyboardInterrupt:
        resume_epoch = current_epoch - 1 if epoch_in_progress else current_epoch
        interrupt_artifacts = save_interrupt_state(
            run_dir=run_dir,
            checkpoint_dir=config["train"]["checkpoint_dir"],
            run_checkpoint_dir=run_checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resume_epoch=resume_epoch,
            current_epoch=current_epoch,
            current_phase=current_phase,
            current_step=current_step,
            global_step=global_step,
            optimizer_step=optimizer_step,
            config=config,
        )
        interrupt_line = (
            f"interrupted=true phase={current_phase} "
            f"epoch_in_progress={current_epoch + 1 if current_epoch >= 0 else 0} "
            f"step_in_progress={current_step} "
            f"resume_next_epoch={interrupt_artifacts['resume_next_epoch']}"
        )
        log_line(log_file, interrupt_line)
        log_line(log_file, f"interrupt_checkpoint={interrupt_artifacts['interrupt_path']}")
        log_line(log_file, f"interrupt_run_checkpoint={interrupt_artifacts['run_interrupt_path']}")
        print(interrupt_line, flush=True)
        print(f"interrupt_checkpoint={interrupt_artifacts['interrupt_path']}", flush=True)


if __name__ == "__main__":
    main()
