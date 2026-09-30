"""
CLI: fit PCA on a (database/train) feature set and apply it, with L2
re-normalization after projection.

    python -m come_cbir.apply_pca \\
        --features outputs/features.npy \\
        --dimension 256 \\
        --output outputs/features_pca256.npy

To avoid query/test leakage, pass --fit-features pointing at a *different*,
database-only feature file when the array you want to transform includes
query/test rows (see docs/CBIR.md "PCA and data leakage"). If --fit-features
is omitted, --features is used for both fitting and transforming, which is
only correct when --features itself is exactly the database/train split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from come_cbir.dimensionality import PCAReducer
from come_cbir.utils import setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=str, required=True, help="Features to transform (.npy)")
    parser.add_argument("--fit-features", type=str, default=None,
                         help="Database/train-only features to fit PCA on (defaults to --features)")
    parser.add_argument("--dimension", type=int, required=True, help="Target PCA dimension")
    parser.add_argument("--output", type=str, required=True, help="Where to write transformed features (.npy)")
    parser.add_argument("--model-output", type=str, default=None,
                         help="Where to save the fitted PCA model (defaults to <output>.pca.pkl)")
    parser.add_argument("--incremental", action="store_true", help="Use IncrementalPCA for large datasets")
    parser.add_argument("--batch-size", type=int, default=4096, help="IncrementalPCA batch size")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()

    features = np.load(args.features).astype(np.float32)
    fit_features = np.load(args.fit_features).astype(np.float32) if args.fit_features else features
    if args.fit_features is None:
        logger.warning(
            "No --fit-features given: fitting PCA directly on --features. Only do this if "
            "--features is exactly your database/train split -- fitting on query or test "
            "rows leaks their statistics into the representation used to score them."
        )

    reducer = PCAReducer(
        n_components=args.dimension, incremental=args.incremental, batch_size=args.batch_size
    ).fit(fit_features)

    transformed = reducer.transform(features)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, transformed)

    model_output = args.model_output or str(output_path.with_suffix(".pca.pkl"))
    reducer.save(model_output)

    logger.info(
        "Wrote %s (%s) using PCA model %s", output_path, transformed.shape, model_output
    )


if __name__ == "__main__":
    main()
