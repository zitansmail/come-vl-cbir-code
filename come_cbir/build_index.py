"""
CLI: build a nearest-neighbor index over saved CBIR features.

    python -m come_cbir.build_index \\
        --features outputs/features.npy \\
        --backend faiss-flat \\
        --output outputs/index.faiss

    python -m come_cbir.build_index \\
        --features outputs/features.npy \\
        --backend annoy \\
        --annoy-trees 20 \\
        --output outputs/index.ann

Requires a paths.json next to --features (as written by extract_features.py)
so the row<->image-path mapping can be saved alongside the index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from come_cbir.indexing import SUPPORTED_BACKENDS, build_index, save_paths
from come_cbir.utils import setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--paths", type=str, default=None,
                         help="paths.json to associate with the index (defaults to <features_dir>/paths.json)")
    parser.add_argument("--backend", type=str, required=True, choices=list(SUPPORTED_BACKENDS))
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--ivf-nlist", type=int, default=100)
    parser.add_argument("--ivf-nprobe", type=int, default=8)
    parser.add_argument("--annoy-trees", type=int, default=20)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()

    features_path = Path(args.features)
    features = np.load(features_path).astype(np.float32)

    paths_path = Path(args.paths) if args.paths else features_path.parent / "paths.json"
    if not paths_path.exists():
        raise FileNotFoundError(
            f"Could not find paths file {paths_path}. Pass --paths explicitly or run "
            f"extract_features.py first (it always writes paths.json next to features.npy)."
        )
    with open(paths_path) as f:
        paths = json.load(f)
    if len(paths) != features.shape[0]:
        raise ValueError(
            f"paths.json has {len(paths)} entries but features has {features.shape[0]} rows"
        )

    kwargs = {}
    if args.backend == "faiss-hnsw":
        kwargs = dict(m=args.hnsw_m, ef_construction=args.hnsw_ef_construction, ef_search=args.hnsw_ef_search)
    elif args.backend == "faiss-ivf":
        kwargs = dict(nlist=args.ivf_nlist, nprobe=args.ivf_nprobe)
    elif args.backend == "annoy":
        kwargs = dict(n_trees=args.annoy_trees)

    logger.info("Building %s index over %d vectors (dim=%d)", args.backend, *features.shape)
    index = build_index(args.backend, features, **kwargs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(output_path))
    save_paths(str(output_path), paths)
    logger.info("Saved index to %s (+ %s.paths.json)", output_path, output_path)


if __name__ == "__main__":
    main()
