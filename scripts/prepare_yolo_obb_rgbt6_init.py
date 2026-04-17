from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version
from ultralytics.nn.tasks import OBBModel

from model.network.rgbt6_yolo_modules import register_rgbt6_yolo_modules, remap_model_key_for_stem_shift


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_YAML = PROJECT_ROOT / "configs" / "yolo11_obb_rgbt6.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "weights" / "yolo11_obb_rgbt6_init.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a minimal 6-channel YOLO-OBB checkpoint by expanding the first conv from a 3-channel source."
    )
    parser.add_argument("--source", type=str, required=True, help="Source 3-channel YOLO-OBB weights (.pt or model name).")
    parser.add_argument("--model-yaml", type=Path, default=DEFAULT_MODEL_YAML, help="6-channel model YAML.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output checkpoint path.")
    parser.add_argument(
        "--thermal-scale",
        type=float,
        default=0.0,
        help="Scale factor applied to the thermal branch initialization copied from RGB mean weights. Use 0.0 to neutralize thermal channels for fast 6ch compatibility.",
    )
    parser.add_argument("--task", type=str, default="obb")
    parser.add_argument(
        "--index-shift",
        type=int,
        default=0,
        help="Remap destination model.N.* keys to source model.(N-index_shift).* for architectures that prepend extra stem modules.",
    )
    return parser.parse_args()


def build_expanded_first_conv(src_weight: torch.Tensor, thermal_scale: float) -> torch.Tensor:
    if src_weight.ndim != 4 or src_weight.shape[1] != 3:
        raise ValueError(f"Expected first conv weight with shape [out, 3, k, k], got {tuple(src_weight.shape)}")
    thermal_seed = src_weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * float(thermal_scale)
    expanded = torch.cat([src_weight, thermal_seed], dim=1)
    return expanded


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    register_rgbt6_yolo_modules()
    src = YOLO(args.source, task=args.task)
    dst_model = OBBModel(cfg=str(args.model_yaml.resolve()), ch=6, nc=5, verbose=False)

    src_state = src.model.state_dict()
    dst_state = dst_model.state_dict()
    merged = {}
    loaded = 0
    skipped: list[str] = []

    for key, dst_tensor in dst_state.items():
        src_key = remap_model_key_for_stem_shift(key, int(args.index_shift))
        if src_key is None or src_key not in src_state:
            merged[key] = dst_tensor
            skipped.append(key)
            continue
        src_tensor = src_state[src_key]
        if src_tensor.shape[1] == 3 and dst_tensor.shape[1] == 6 and key.endswith(".conv.weight"):
            merged[key] = build_expanded_first_conv(src_tensor, args.thermal_scale).to(dtype=dst_tensor.dtype)
            loaded += 1
            continue
        if src_tensor.shape == dst_tensor.shape:
            merged[key] = src_tensor.to(dtype=dst_tensor.dtype)
            loaded += 1
        else:
            merged[key] = dst_tensor
            skipped.append(key)

    dst_model.load_state_dict(merged, strict=False)
    dst_model.args = {
        "task": "obb",
        "model": str(args.model_yaml.resolve()),
        "imgsz": 640,
        "channels": 6,
        "nc": 5,
    }
    dst_model.task = "obb"

    ckpt = {
        "model": dst_model.half(),
        "train_args": {"task": "obb", "model": str(args.model_yaml.resolve())},
        "date": datetime.now().isoformat(),
        "version": ultralytics_version,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    torch.save(ckpt, output_path)

    print(f"source={args.source}")
    print(f"model_yaml={args.model_yaml.resolve()}")
    print(f"output={output_path}")
    print(f"loaded_keys={loaded}")
    print(f"skipped_keys={len(skipped)}")
    if skipped:
        print("first_skipped=", skipped[:10])


if __name__ == "__main__":
    main()
