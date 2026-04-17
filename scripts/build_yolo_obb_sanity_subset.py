from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small YOLO-OBB sanity subset with symlinked assets.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--train-ids", nargs="+", required=True, help="Sample stems without _rgb suffix, e.g. 00001")
    parser.add_argument("--val-ids", nargs="+", required=True, help="Sample stems without _rgb suffix, e.g. 00001")
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--link-npy", action="store_true", help="Link .npy sidecars when present.")
    return parser.parse_args()


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def link_sample(source_root: Path, target_root: Path, split: str, stem: str, link_npy: bool) -> None:
    image_dir = source_root / "images" / split
    label_dir = source_root / "labels" / split
    dst_image_dir = target_root / "images" / split
    dst_label_dir = target_root / "labels" / split

    base = f"{stem}_rgb"
    jpg_path = image_dir / f"{base}.jpg"
    npy_path = image_dir / f"{base}.npy"
    txt_path = label_dir / f"{base}.txt"

    if not txt_path.exists():
        raise FileNotFoundError(f"Missing label: {txt_path}")
    if not jpg_path.exists() and not npy_path.exists():
        raise FileNotFoundError(f"Missing image assets for {base} under {image_dir}")

    if jpg_path.exists():
        symlink_force(jpg_path, dst_image_dir / jpg_path.name)
    if link_npy and npy_path.exists():
        symlink_force(npy_path, dst_image_dir / npy_path.name)
    symlink_force(txt_path, dst_label_dir / txt_path.name)


def write_dataset_yaml(source_root: Path, target_root: Path, channels: int | None) -> None:
    source_yaml = source_root / "dataset.yaml"
    lines = source_yaml.read_text(encoding="utf-8").splitlines()
    output: list[str] = [f"path: {target_root}"]
    for line in lines:
        if line.startswith("path:"):
            continue
        if line.startswith("train:"):
            output.append("train: images/train")
            continue
        if line.startswith("val:"):
            output.append("val: images/val")
            continue
        if line.startswith("channels:"):
            if channels is None:
                output.append(line)
            continue
        output.append(line)
    if channels is not None and not any(line.startswith("channels:") for line in output):
        output.insert(3, f"channels: {channels}")
    (target_root / "dataset.yaml").write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for stem in args.train_ids:
        link_sample(source_root, target_root, "train", stem, args.link_npy)
    for stem in args.val_ids:
        link_sample(source_root, target_root, "val", stem, args.link_npy)

    write_dataset_yaml(source_root, target_root, args.channels)

    summary = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "train_ids": args.train_ids,
        "val_ids": args.val_ids,
        "channels": args.channels,
        "link_npy": args.link_npy,
    }
    (target_root / "sanity_subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
