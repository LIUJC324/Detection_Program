from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocess import ensure_same_size, read_image

DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "datasets" / "yolo_obb_official_rgb_trueobb_v1"
DEFAULT_RGBT_ROOT = PROJECT_ROOT / "datasets" / "dronevehicle_like_refined"
DEFAULT_TARGET_ROOT = PROJECT_ROOT / "datasets" / "yolo_obb_official_rgbt6_matched_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a matched RGB-T 6-channel OBB subset with jpg symlinks and npy sidecars.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--rgbt-root", type=Path, default=DEFAULT_RGBT_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def save_sidecar(rgb_path: Path, thermal_path: Path, npy_path: Path) -> None:
    rgb = read_image(rgb_path, mode="rgb")
    thermal = read_image(thermal_path, mode="thermal")
    rgb, thermal = ensure_same_size(rgb, thermal)
    stacked = np.concatenate([rgb, thermal], axis=2).astype(np.uint8)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, stacked, allow_pickle=False)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    rgbt_root = args.rgbt_root.resolve()
    target_root = args.target_root.resolve()

    written = {"train": 0, "val": 0}
    missing = {"train": 0, "val": 0}

    for split in ("train", "val"):
        src_img_dir = source_root / "images" / split
        src_label_dir = source_root / "labels" / split
        thermal_dir = rgbt_root / "thermal" / split
        dst_img_dir = target_root / "images" / split
        dst_label_dir = target_root / "labels" / split

        for rgb_path in sorted(src_img_dir.glob("*_rgb.jpg")):
            stem = rgb_path.stem.replace("_rgb", "")
            thermal_path = thermal_dir / f"{stem}_thermal.jpg"
            label_path = src_label_dir / f"{rgb_path.stem}.txt"
            if not thermal_path.exists() or not label_path.exists():
                missing[split] += 1
                continue

            dst_rgb_path = dst_img_dir / rgb_path.name
            dst_label_path = dst_label_dir / label_path.name
            dst_npy_path = dst_rgb_path.with_suffix(".npy")

            if args.overwrite or not dst_rgb_path.exists():
                symlink_force(rgb_path, dst_rgb_path)
            if args.overwrite or not dst_label_path.exists():
                symlink_force(label_path, dst_label_path)
            if args.overwrite or not dst_npy_path.exists():
                save_sidecar(rgb_path, thermal_path, dst_npy_path)

            written[split] += 1

    dataset_yaml = target_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {target_root}",
                "train: images/train",
                "val: images/val",
                "channels: 6",
                "names:",
                "  0: car",
                "  1: truck",
                "  2: bus",
                "  3: van",
                "  4: freight_car",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "source_root": str(source_root),
        "rgbt_root": str(rgbt_root),
        "target_root": str(target_root),
        "written": written,
        "missing": missing,
        "dataset_yaml": str(dataset_yaml),
        "channel_layout": "rgb3_then_thermal3",
    }
    (target_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
