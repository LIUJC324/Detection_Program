from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO-OBB experiment from YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    from ultralytics import YOLO

    model_name = cfg["model"]
    train_cfg = cfg.get("train", {})
    model = YOLO(model_name)
    results = model.train(
        data=cfg["data"],
        epochs=int(train_cfg.get("epochs", 50)),
        imgsz=int(train_cfg.get("imgsz", 640)),
        batch=int(train_cfg.get("batch", 16)),
        device=train_cfg.get("device", 0),
        workers=int(train_cfg.get("workers", 8)),
        project=train_cfg.get("project", "outputs/yolo_obb_runs"),
        name=train_cfg.get("name", config_path.stem),
        pretrained=bool(train_cfg.get("pretrained", True)),
        optimizer=train_cfg.get("optimizer", "auto"),
        lr0=float(train_cfg.get("lr0", 0.01)),
        cos_lr=bool(train_cfg.get("cos_lr", True)),
        patience=int(train_cfg.get("patience", 30)),
        degrees=float(train_cfg.get("degrees", 0.0)),
        scale=float(train_cfg.get("scale", 0.5)),
        fliplr=float(train_cfg.get("fliplr", 0.5)),
        mosaic=float(train_cfg.get("mosaic", 1.0)),
        mixup=float(train_cfg.get("mixup", 0.0)),
        close_mosaic=int(train_cfg.get("close_mosaic", 10)),
        save=bool(train_cfg.get("save", True)),
        val=bool(train_cfg.get("val", True)),
        amp=bool(train_cfg.get("amp", True)),
        seed=int(train_cfg.get("seed", 42)),
        exist_ok=bool(train_cfg.get("exist_ok", True)),
        plots=bool(train_cfg.get("plots", False)),
        resume=bool(train_cfg.get("resume", False)),
        verbose=True,
    )
    print(results)


if __name__ == "__main__":
    main()
