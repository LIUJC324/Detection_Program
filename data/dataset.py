from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from .preprocess import ensure_same_size, read_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class RGBTTargetDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        rgb_dir: str = "rgb",
        thermal_dir: str = "thermal",
        annotation_dir: str = "annotations",
        rgb_suffix: str = "_rgb",
        thermal_suffix: str = "_thermal",
        transform=None,
        allow_empty_annotations: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.rgb_root = self.root / rgb_dir / split
        self.thermal_root = self.root / thermal_dir / split
        self.annotation_root = self.root / annotation_dir / split
        self.rgb_suffix = rgb_suffix
        self.thermal_suffix = thermal_suffix
        self.transform = transform
        self.allow_empty_annotations = allow_empty_annotations
        self.samples = self._build_index()

        if not self.samples:
            raise FileNotFoundError(
                f"No aligned RGB-T samples found under {self.rgb_root} and {self.thermal_root}."
            )

    def _base_stem(self, path: Path) -> str:
        stem = path.stem
        if self.rgb_suffix and stem.endswith(self.rgb_suffix):
            return stem[: -len(self.rgb_suffix)]
        return stem

    def _find_thermal(self, base_stem: str) -> Optional[Path]:
        candidates = []
        for ext in IMAGE_EXTS:
            candidates.append(self.thermal_root / f"{base_stem}{self.thermal_suffix}{ext}")
            candidates.append(self.thermal_root / f"{base_stem}{ext}")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _build_index(self) -> List[Dict[str, Path]]:
        samples: List[Dict[str, Path]] = []
        for rgb_path in sorted(self.rgb_root.glob("*")):
            if rgb_path.suffix.lower() not in IMAGE_EXTS:
                continue
            base_stem = self._base_stem(rgb_path)
            thermal_path = self._find_thermal(base_stem)
            annotation_path = self.annotation_root / f"{base_stem}.json"
            if thermal_path is None:
                continue
            if not annotation_path.exists() and not self.allow_empty_annotations:
                continue
            samples.append(
                {
                    "rgb_path": rgb_path,
                    "thermal_path": thermal_path,
                    "annotation_path": annotation_path,
                    "sample_id": base_stem,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_annotation(self, path: Path) -> Dict[str, np.ndarray]:
        if not path.exists():
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
            }
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        objects = payload.get("objects", payload.get("annotations", []))
        boxes = []
        labels = []
        for item in objects:
            bbox = item.get("bbox", item.get("box"))
            class_id = item.get("class_id", item.get("category_id"))
            if bbox is None or class_id is None:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(int(class_id))
        return {
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(labels, dtype=np.int64),
        }

    def __getitem__(self, index: int):
        sample_info = self.samples[index]
        rgb = read_image(sample_info["rgb_path"], mode="rgb")
        thermal = read_image(sample_info["thermal_path"], mode="thermal")
        rgb, thermal = ensure_same_size(rgb, thermal)
        targets = self._load_annotation(sample_info["annotation_path"])
        targets["image_id"] = index
        targets["sample_id"] = sample_info["sample_id"]
        targets["orig_size"] = np.asarray([rgb.shape[0], rgb.shape[1]], dtype=np.int64)
        sample = {"rgb": rgb, "thermal": thermal, "targets": targets}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def rgbt_collate_fn(batch):
    rgbs = [item["rgb"] for item in batch]
    thermals = [item["thermal"] for item in batch]
    targets = [item["targets"] for item in batch]
    return {"rgb": rgbs, "thermal": thermals, "targets": targets}
