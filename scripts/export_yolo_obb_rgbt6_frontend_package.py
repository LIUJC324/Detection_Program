from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version
from ultralytics.nn.tasks import OBBModel

from model.network.rgbt6_yolo_modules import register_rgbt6_yolo_modules, remap_model_key_for_stem_shift


DEFAULT_SOURCE = PROJECT_ROOT / "yolo11n-obb.pt"
DEFAULT_MODEL_YAML = PROJECT_ROOT / "configs" / "yolo11_obb_rgbt6.yaml"
DEFAULT_INIT_PT = PROJECT_ROOT / "weights" / "yolo11_obb_rgbt6_fastdemo_init.pt"
DEFAULT_ONNX = PROJECT_ROOT / "weights" / "yolo11_obb_rgbt6_fastdemo.onnx"
DEFAULT_CONFIG_SRC = PROJECT_ROOT / "docs" / "integration" / "frontend_model_config_yolo_obb_rgbt6_20260416.json"
DEFAULT_CONFIG_DST = PROJECT_ROOT / "weights" / "frontend_model_config_yolo_obb_rgbt6_fastdemo.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export minimal 6-channel YOLO-OBB frontend package.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-yaml", type=Path, default=DEFAULT_MODEL_YAML)
    parser.add_argument("--init-pt", type=Path, default=DEFAULT_INIT_PT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--config-src", type=Path, default=DEFAULT_CONFIG_SRC)
    parser.add_argument("--config-dst", type=Path, default=DEFAULT_CONFIG_DST)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--thermal-scale",
        type=float,
        default=0.0,
        help="Thermal branch init scale. 0.0 keeps current demo behavior close to RGB-only YOLO-OBB while accepting 6ch input.",
    )
    return parser.parse_args()


def build_expanded_first_conv(src_weight: torch.Tensor, thermal_scale: float) -> torch.Tensor:
    thermal_seed = src_weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * float(thermal_scale)
    return torch.cat([src_weight, thermal_seed], dim=1)


def build_rgbt6_model(source: Path, model_yaml: Path, thermal_scale: float) -> OBBModel:
    register_rgbt6_yolo_modules()
    src = YOLO(str(source.resolve()), task="obb")
    dst = OBBModel(cfg=str(model_yaml.resolve()), ch=6, nc=5, verbose=False)

    src_state = src.model.state_dict()
    dst_state = dst.state_dict()
    merged = {}

    for key, dst_tensor in dst_state.items():
        src_key = remap_model_key_for_stem_shift(key, 0)
        if src_key not in src_state:
            merged[key] = dst_tensor
            continue
        src_tensor = src_state[src_key]
        if src_tensor.shape[1] == 3 and dst_tensor.shape[1] == 6 and key.endswith(".conv.weight"):
            merged[key] = build_expanded_first_conv(src_tensor, thermal_scale).to(dtype=dst_tensor.dtype)
        elif src_tensor.shape == dst_tensor.shape:
            merged[key] = src_tensor.to(dtype=dst_tensor.dtype)
        else:
            merged[key] = dst_tensor

    dst.load_state_dict(merged, strict=False)
    dst.eval()
    return dst


def save_init_checkpoint(model: OBBModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.args = {
        "task": "obb",
        "model": str(DEFAULT_MODEL_YAML.resolve()),
        "imgsz": 640,
        "channels": 6,
        "nc": 5,
    }
    model.task = "obb"
    ckpt = {
        "model": deepcopy(model).half(),
        "train_args": {"task": "obb", "model": str(DEFAULT_MODEL_YAML.resolve())},
        "date": datetime.now().isoformat(),
        "version": ultralytics_version,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    torch.save(ckpt, path)


def export_onnx(model: OBBModel, onnx_path: Path, imgsz: int, opset: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 6, imgsz, imgsz, dtype=torch.float32)

    class SingleOutputWrapper(nn.Module):
        def __init__(self, inner: OBBModel):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)[0]

    wrapper = SingleOutputWrapper(model).eval()
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        opset_version=int(opset),
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=None,
        do_constant_folding=True,
    )


def write_config(config_src: Path, config_dst: Path, onnx_path: Path, init_pt: Path) -> None:
    cfg = json.loads(config_src.read_text(encoding="utf-8"))
    cfg["model_name"] = "yolo_obb_rgbt6_frontend_fastdemo"
    cfg["model_version"] = "exported_fastdemo_20260416"
    cfg["onnx_path"] = str(onnx_path.resolve())
    cfg["source_checkpoint"] = str(init_pt.resolve())
    cfg["notes"]["current_strategy"] = "fast 6ch compatibility export; thermal channels are neutralized first so frontend can switch to 6ch immediately without waiting for full RGB-T OBB training"
    cfg["notes"]["important"] = "this version accepts RGB+Thermal 6ch input, but thermal branch is currently compatibility-initialized and should be fine-tuned later"
    config_dst.parent.mkdir(parents=True, exist_ok=True)
    config_dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Source OBB checkpoint not found: {args.source}")
    if not args.model_yaml.exists():
        raise FileNotFoundError(f"6-channel model YAML not found: {args.model_yaml}")
    if not args.config_src.exists():
        raise FileNotFoundError(f"Frontend config template not found: {args.config_src}")

    model = build_rgbt6_model(args.source, args.model_yaml, args.thermal_scale)
    save_init_checkpoint(model, args.init_pt.resolve())
    export_onnx(model, args.onnx.resolve(), int(args.imgsz), int(args.opset))
    write_config(args.config_src.resolve(), args.config_dst.resolve(), args.onnx.resolve(), args.init_pt.resolve())

    print(f"first_conv={tuple(model.model[0].conv.weight.shape)}")
    print(f"init_pt={args.init_pt.resolve()}")
    print(f"onnx={args.onnx.resolve()}")
    print(f"config={args.config_dst.resolve()}")


if __name__ == "__main__":
    main()
