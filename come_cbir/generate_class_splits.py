"""
CLI: regenerate class-disjoint train/validation/test splits for the unseen-class
protocol, using the SAME `three_way_class_split` function already used by
ARP-NP training and validated by its own test suite -- this does not
reimplement split logic, it only drives the existing function and writes a
fully self-describing JSON per seed.

    python -m come_cbir.generate_class_splits \
        --labels /content/outputs/corel1k_v2/siglip/labels.npy \
        --output-dir outputs/corel1k_v2/splits \
        --seeds 0 1 2

Each output file `split_seed_<N>.json` contains everything needed for exact
reproducibility and for direct consumption by validate_graph_fusion.py's
--unseen-splits: seed, val/test class fractions, the full class list, train/
val/test classes, train/val/test indices, per-subset image counts, and the
total dataset size.

Every generated split is verified before being written:
  - train/val/test classes are pairwise disjoint
  - every test index's label is actually a test class (and same for val/train)
  - no duplicate indices within or across the three index sets
  - all indices are within [0, n)
No split is silently repaired -- any violation raises ValueError.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from come_cbir.retrieval_adapter import three_way_class_split
from come_cbir.utils import setup_logging


def verify_split(split: Dict, labels: np.ndarray) -> None:
    train_classes, val_classes, test_classes = (
        set(split["train_classes"]), set(split["val_classes"]), set(split["test_classes"]),
    )
    if train_classes & val_classes or train_classes & test_classes or val_classes & test_classes:
        raise ValueError("Split classes are not pairwise disjoint")

    n = len(labels)
    all_indices = np.concatenate([split["train_indices"], split["val_indices"], split["test_indices"]])
    if len(all_indices) != len(set(all_indices.tolist())):
        raise ValueError("Duplicate indices found across train/val/test index sets")
    if all_indices.size and (all_indices.min() < 0 or all_indices.max() >= n):
        raise ValueError(f"Indices out of bounds [0, {n})")

    for name, indices, classes in (
        ("train", split["train_indices"], train_classes),
        ("val", split["val_indices"], val_classes),
        ("test", split["test_indices"], test_classes),
    ):
        bad = [int(i) for i in indices if int(labels[i]) not in classes]
        if bad:
            raise ValueError(f"{name}_indices contains indices whose label is not in {name}_classes: {bad[:5]}...")


def generate_split(labels: np.ndarray, seed: int, val_class_fraction: float, test_class_fraction: float) -> Dict:
    split = three_way_class_split(
        labels, val_class_fraction=val_class_fraction, test_class_fraction=test_class_fraction, seed=seed,
    )
    verify_split(split, labels)

    payload = {
        "seed": seed,
        "config": {
            "val_class_fraction": val_class_fraction, "test_class_fraction": test_class_fraction,
            "split_function": "come_cbir.retrieval_adapter.three_way_class_split",
        },
        "dataset_size": int(len(labels)),
        "all_classes": sorted(int(c) for c in np.unique(labels)),
        "train_classes": [int(c) for c in split["train_classes"]],
        "val_classes": [int(c) for c in split["val_classes"]],
        "test_classes": [int(c) for c in split["test_classes"]],
        "train_indices": [int(i) for i in split["train_indices"]],
        "val_indices": [int(i) for i in split["val_indices"]],
        "test_indices": [int(i) for i in split["test_indices"]],
        "n_train_images": int(len(split["train_indices"])),
        "n_val_images": int(len(split["val_indices"])),
        "n_test_images": int(len(split["test_indices"])),
    }
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val-class-fraction", type=float, default=0.2)
    parser.add_argument("--test-class-fraction", type=float, default=0.2)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()

    labels = np.load(args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        payload = generate_split(labels, seed, args.val_class_fraction, args.test_class_fraction)
        out_path = output_dir / f"split_seed_{seed}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "seed=%d: train=%d classes/%d images, val=%d classes/%d images, test=%d classes/%d images -> %s",
            seed, len(payload["train_classes"]), payload["n_train_images"],
            len(payload["val_classes"]), payload["n_val_images"],
            len(payload["test_classes"]), payload["n_test_images"], out_path,
        )


if __name__ == "__main__":
    main()
