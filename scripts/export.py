from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.detector import SingleBatchExportWrapper, build_model, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Export RGB-T detector to TorchScript or ONNX.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", type=str, choices=["torchscript", "onnx", "both"], default="both")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    export_cfg = config["export"]
    model = build_model(config).to(args.device)
    load_checkpoint(model, args.checkpoint, map_location=args.device)
    model.eval()
    wrapper = SingleBatchExportWrapper(model).to(args.device)

    input_size = config["model"].get("input_size", 640)
    dummy = torch.randn(1, 6, input_size, input_size, device=args.device)
    if args.format in {"torchscript", "both"}:
        ts_path = Path(export_cfg["torchscript_path"])
        if not ts_path.is_absolute():
            ts_path = (PROJECT_ROOT / ts_path).resolve()
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        traced = torch.jit.trace(wrapper, dummy, strict=False)
        traced.save(str(ts_path))
        print(f"TorchScript exported to {ts_path}")

    if args.format in {"onnx", "both"}:
        onnx_path = Path(export_cfg["onnx_path"])
        if not onnx_path.is_absolute():
            onnx_path = (PROJECT_ROOT / onnx_path).resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        dynamic_axes = None
        if export_cfg.get("dynamic_axes", True):
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},
            }
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            opset_version=export_cfg.get("opset", 17),
            input_names=["images"],
            output_names=["boxes", "scores", "labels"],
            dynamic_axes=dynamic_axes,
        )
        print(f"ONNX exported to {onnx_path}")


if __name__ == "__main__":
    main()

