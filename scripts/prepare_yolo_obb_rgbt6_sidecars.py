from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocess import read_image, ensure_same_size

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "yolo_obb_official_rgb_trueobb_v1"
DEFAULT_RGBT_ROOT = PROJECT_ROOT / "datasets" / "dronevehicle_like_refined"
DEFAULT_SUMMARY = PROJECT_ROOT / "datasets" / "yolo_obb_official_rgb_trueobb_v1" / "rgbt6_sidecar_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 6-channel RGB-T .npy sidecars next to YOLO-OBB RGB images for minimal Ultralytics 6ch training."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--rgbt-root", type=Path, default=DEFAULT_RGBT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_sidecar(rgb_path: Path, thermal_path: Path, overwrite: bool) -> bool:
    npy_path = rgb_path.with_suffix(".npy")
    if npy_path.exists() and not overwrite:
        return False

    rgb = read_image(rgb_path, mode="rgb")
    thermal = read_image(thermal_path, mode="thermal")
    rgb, thermal = ensure_same_size(rgb, thermal)
    stacked = np.concatenate([rgb, thermal], axis=2).astype(np.uint8)
    np.save(npy_path, stacked, allow_pickle=False)
    return True


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    rgbt_root = args.rgbt_root.resolve()
    summary_path = args.summary_path.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    missing_thermal: list[str] = []

    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        thermal_dir = rgbt_root / "thermal" / split
        if not image_dir.exists():
            continue
        for rgb_path in sorted(image_dir.glob("*_rgb.jpg")):
            stem = rgb_path.stem.replace("_rgb", "")
            thermal_path = thermal_dir / f"{stem}_thermal.jpg"
            if not thermal_path.exists():
                missing_thermal.append(str(rgb_path))
                continue
            changed = build_sidecar(rgb_path, thermal_path, overwrite=args.overwrite)
            if changed:
                written += 1
            else:
                skipped += 1

    summary = {
        "dataset_root": str(dataset_root),
        "rgbt_root": str(rgbt_root),
        "written_sidecars": written,
        "skipped_existing": skipped,
        "missing_thermal_count": len(missing_thermal),
        "missing_thermal_examples": missing_thermal[:20],
        "channel_layout": "rgb3_then_thermal3",
        "dtype": "uint8",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
