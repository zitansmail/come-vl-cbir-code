"""
Dataset loading for CBIR feature extraction.

Supports two layouts:

1. A generic image-classification-style folder tree::

    dataset/
    |-- class_1/
    |   |-- image_001.jpg
    |   `-- image_002.jpg
    `-- class_2/
        |-- image_003.jpg
        `-- image_004.jpg

2. A manifest CSV with columns ``image_path,label,class_name,split``
   (``label``/``class_name``/``split`` optional; ``label`` is derived from
   ``class_name`` if omitted).

Each sample returns ``(patches, image_path, label, class_name)`` where
``patches`` is the deterministic, patchified SigLIP-style tensor produced by
``come_cbir.feature_extractor._siglip_preprocess_single_crop`` -- the same
function the online (PIL-input) path of ``CoMECbirFeatureExtractor`` uses,
so dataset-based and single-image extraction are guaranteed to preprocess
identically (see docs/cbir_architecture_analysis.md section 9 for what this
preprocessing does and does not reproduce from the official pipeline).
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from come_cbir.feature_extractor import _siglip_preprocess_single_crop

logger = logging.getLogger("come_cbir.datasets")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Sample:
    image_path: str
    label: int
    class_name: str
    split: Optional[str] = None


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def scan_folder_dataset(root: Path) -> List[Sample]:
    """Discover samples from a `root/class_name/*.jpg` folder tree, sorted for determinism."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(
            f"No class subdirectories found under {root}. Expected layout: "
            f"{root}/<class_name>/<image files>"
        )

    class_to_label = {d.name: i for i, d in enumerate(class_dirs)}
    samples: List[Sample] = []
    for class_dir in class_dirs:
        image_paths = sorted(p for p in class_dir.iterdir() if p.is_file() and _is_image_file(p))
        for image_path in image_paths:
            samples.append(
                Sample(
                    image_path=str(image_path),
                    label=class_to_label[class_dir.name],
                    class_name=class_dir.name,
                )
            )
    if not samples:
        raise ValueError(f"No image files with extensions {IMAGE_EXTENSIONS} found under {root}")
    return samples


def load_manifest_dataset(manifest_csv: Path) -> List[Sample]:
    """Load samples from a manifest CSV: image_path,label,class_name,split."""
    manifest_csv = Path(manifest_csv)
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    samples: List[Sample] = []
    class_name_to_label: Dict[str, int] = {}
    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError(
                f"Manifest CSV {manifest_csv} must have an 'image_path' column, "
                f"found columns: {reader.fieldnames}"
            )
        for row_idx, row in enumerate(reader):
            image_path = row["image_path"].strip()
            if not image_path:
                raise ValueError(f"Empty image_path in manifest row {row_idx}")
            class_name = (row.get("class_name") or "").strip() or "unknown"
            raw_label = (row.get("label") or "").strip()
            if raw_label:
                label = int(raw_label)
            else:
                label = class_name_to_label.setdefault(class_name, len(class_name_to_label))
            split = (row.get("split") or "").strip() or None
            samples.append(Sample(image_path=image_path, label=label, class_name=class_name, split=split))
    if not samples:
        raise ValueError(f"Manifest CSV {manifest_csv} contains no rows")
    return samples


class CBIRImageDataset(Dataset):
    """
    Deterministic image dataset for CBIR feature extraction.

    Parameters
    ----------
    root:
        Folder-tree dataset root (mutually exclusive with ``manifest_csv``).
    manifest_csv:
        Manifest CSV path (mutually exclusive with ``root``).
    image_size, patch_size:
        Must match the ``CoMECbirFeatureExtractor`` these features will be
        fed to.
    skip_corrupted:
        If False (default), a corrupted/unreadable image raises immediately
        (fail fast). If True, the image is skipped but the failure is never
        silent: it is logged at WARNING level and recorded in
        ``self.skipped`` for the caller to report.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        manifest_csv: Optional[str] = None,
        image_size: int = 384,
        patch_size: int = 16,
        skip_corrupted: bool = False,
    ):
        if (root is None) == (manifest_csv is None):
            raise ValueError("Exactly one of `root` or `manifest_csv` must be provided")

        if manifest_csv is not None:
            self.samples = load_manifest_dataset(Path(manifest_csv))
        else:
            self.samples = scan_folder_dataset(Path(root))

        self.image_size = image_size
        self.patch_size = patch_size
        self.skip_corrupted = skip_corrupted
        self.skipped: List[str] = []

        class_names = sorted({s.class_name for s in self.samples})
        self.class_names = class_names

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, image_path: str) -> Optional[Image.Image]:
        try:
            with Image.open(image_path) as img:
                img.load()
                return img.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            if self.skip_corrupted:
                logger.warning("Skipping corrupted/unreadable image %s: %s", image_path, exc)
                self.skipped.append(image_path)
                return None
            raise RuntimeError(f"Failed to load image {image_path}: {exc}") from exc

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, str, int, str]]:
        sample = self.samples[index]
        image = self._load_rgb(sample.image_path)
        if image is None:
            return None  # filtered by collate_fn; recorded in self.skipped, not silently dropped
        patches = _siglip_preprocess_single_crop(
            image, output_size=(self.image_size, self.image_size), patch_size=self.patch_size
        )
        return torch.from_numpy(patches), sample.image_path, sample.label, sample.class_name


def cbir_collate_fn(batch: List[Optional[Tuple[torch.Tensor, str, int, str]]]):
    """Drops skipped (None) samples -- the skip itself is already logged/recorded, not silent."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    patches, paths, labels, class_names = zip(*batch)
    return torch.stack(patches, dim=0), list(paths), list(labels), list(class_names)
