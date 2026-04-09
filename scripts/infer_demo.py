from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.core.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Run RGB-T image pair inference and save visualization.")
    parser.add_argument("--rgb", type=str, required=True)
    parser.add_argument("--thermal", type=str, required=True)
    parser.add_argument("--deploy-config", type=str, default=str(PROJECT_ROOT / "configs" / "deploy.yaml"))
    parser.add_argument("--output-image", type=str, default=str(PROJECT_ROOT / "outputs" / "demo_result.jpg"))
    parser.add_argument("--output-json", type=str, default=str(PROJECT_ROOT / "outputs" / "demo_result.json"))
    return parser.parse_args()


def draw_detections(image: np.ndarray, detections):
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f'{det["class_name"]}:{det["confidence"]:.2f}'
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(canvas, label, (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return canvas


def main():
    args = parse_args()
    predictor = Predictor.from_deploy_config(args.deploy_config)

    with open(args.rgb, "rb") as fp:
        rgb_data = fp.read()
    with open(args.thermal, "rb") as fp:
        thermal_data = fp.read()

    result = predictor.predict(rgb_data, thermal_data)
    rgb_image = predictor.last_rgb_image
    vis = draw_detections(rgb_image, result["detections"])

    output_image = Path(args.output_image)
    output_json = Path(args.output_json)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), vis)
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

