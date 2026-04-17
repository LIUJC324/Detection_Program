from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.network.rgbt6_yolo_modules import register_rgbt6_yolo_modules, remap_model_key_for_stem_shift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO-OBB experiment from YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def ensure_obb_checkpoint_metadata(path: str | Path) -> None:
    ckpt_path = Path(path)
    if ckpt_path.suffix != ".pt" or not ckpt_path.exists():
        return

    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    changed = False

    train_args = ckpt.get("train_args")
    if isinstance(train_args, dict) and train_args.get("task") != "obb":
        train_args["task"] = "obb"
        changed = True

    for key in ("model", "ema"):
        model_obj = ckpt.get(key)
        if model_obj is None:
            continue
        if getattr(model_obj, "task", None) != "obb":
            model_obj.task = "obb"
            changed = True
        model_args = getattr(model_obj, "args", None)
        if isinstance(model_args, dict) and model_args.get("task") != "obb":
            model_args["task"] = "obb"
            changed = True

    if changed:
        torch.save(ckpt, ckpt_path)
        print(f"patched_checkpoint_task=obb path={ckpt_path}")


def _build_expanded_first_conv(src_weight, thermal_scale: float):
    import torch

    thermal_seed = src_weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * float(thermal_scale)
    return torch.cat([src_weight, thermal_seed], dim=1)


def prepare_shifted_pretrained_checkpoint(
    source_path: str | Path,
    model_yaml: str | Path,
    index_shift: int,
    thermal_scale: float = 0.0,
) -> Path:
    import torch
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version
    from ultralytics.nn.tasks import OBBModel

    register_rgbt6_yolo_modules()
    source_path = Path(source_path).resolve()
    model_yaml = Path(model_yaml).resolve()
    output_path = source_path.with_name(f"{source_path.stem}_{model_yaml.stem}_shift{index_shift}.pt")

    if output_path.exists():
        ensure_obb_checkpoint_metadata(output_path)
        return output_path

    src = YOLO(str(source_path), task="obb")
    dst_model = OBBModel(cfg=str(model_yaml), ch=6, nc=5, verbose=False)
    src_state = src.model.state_dict()
    dst_state = dst_model.state_dict()
    merged = {}

    for key, dst_tensor in dst_state.items():
        src_key = remap_model_key_for_stem_shift(key, int(index_shift))
        if src_key is None or src_key not in src_state:
            merged[key] = dst_tensor
            continue
        src_tensor = src_state[src_key]
        if (
            src_tensor.ndim == 4
            and dst_tensor.ndim == 4
            and src_tensor.shape[1] == 3
            and dst_tensor.shape[1] == 6
            and key.endswith(".conv.weight")
        ):
            merged[key] = _build_expanded_first_conv(src_tensor, thermal_scale).to(dtype=dst_tensor.dtype)
        elif src_tensor.shape == dst_tensor.shape:
            merged[key] = src_tensor.to(dtype=dst_tensor.dtype)
        else:
            merged[key] = dst_tensor

    dst_model.load_state_dict(merged, strict=False)
    dst_model.args = {
        "task": "obb",
        "model": str(model_yaml),
        "imgsz": 640,
        "channels": 6,
        "nc": 5,
    }
    dst_model.task = "obb"
    ckpt = {
        "model": dst_model.half(),
        "train_args": {"task": "obb", "model": str(model_yaml)},
        "date": "prepared_by_train_yolo_obb",
        "version": ultralytics_version,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    torch.save(ckpt, output_path)
    print(f"prepared_shifted_pretrained={output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    from ultralytics import YOLO
    register_rgbt6_yolo_modules()

    model_name = cfg["model"]
    train_cfg = cfg.get("train", {})
    resume = train_cfg.get("resume", False)
    cache = train_cfg.get("cache", False)
    pretrained = train_cfg.get("pretrained", True)
    pretrained_index_shift = int(train_cfg.get("pretrained_index_shift", 0))
    ensure_obb_checkpoint_metadata(model_name)
    if isinstance(resume, (str, Path)):
        ensure_obb_checkpoint_metadata(resume)
    if pretrained_index_shift > 0 and isinstance(pretrained, (str, Path)) and str(model_name).endswith((".yaml", ".yml")):
        pretrained = str(prepare_shifted_pretrained_checkpoint(pretrained, model_name, pretrained_index_shift))
    if isinstance(pretrained, (str, Path)):
        ensure_obb_checkpoint_metadata(pretrained)
    model = YOLO(model_name)
    # Force OBB task even when checkpoint metadata was saved incorrectly as detect.
    model.task = "obb"
    if hasattr(model, "overrides") and isinstance(model.overrides, dict):
        model.overrides["task"] = "obb"
    if hasattr(model, "model"):
        model.model.task = "obb"
        if hasattr(model.model, "args") and isinstance(model.model.args, dict):
            model.model.args["task"] = "obb"
    results = model.train(
        data=cfg["data"],
        task="obb",
        epochs=int(train_cfg.get("epochs", 50)),
        imgsz=int(train_cfg.get("imgsz", 640)),
        batch=int(train_cfg.get("batch", 16)),
        device=train_cfg.get("device", 0),
        workers=int(train_cfg.get("workers", 8)),
        project=train_cfg.get("project", "outputs/yolo_obb_runs"),
        name=train_cfg.get("name", config_path.stem),
        pretrained=pretrained,
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
        cache=cache,
        resume=resume,
        verbose=True,
    )
    print(results)


if __name__ == "__main__":
    main()
