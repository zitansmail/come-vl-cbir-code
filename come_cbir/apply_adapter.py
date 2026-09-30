"""
CLI: apply a trained retrieval adapter (see train_adapter.py) to a saved
feature array, producing a compact, adapted feature array -- mirrors
apply_pca.py's fit/transform split, since build_index.py needs a
materialized .npy to build an index over.

    python -m come_cbir.apply_adapter \\
        --features outputs/siglip/features.npy \\
        --adapter outputs/siglip/adapter/adapter.pt \\
        --output outputs/siglip/features_adapted.npy

For a leakage-free before/after comparison, apply this to the *same* feature
array the adapter's split.json marks as holdout, then evaluate only that
subset for both the original and adapted runs -- see
docs/retrieval_adapter.md and evaluate.py's --holdout-indices.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from come_cbir.retrieval_adapter import AdaptiveRetrievalProjection, apply_adapter
from come_cbir.utils import setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=str, required=True, help="Features to transform (.npy)")
    parser.add_argument("--adapter", type=str, required=True, help="Trained adapter checkpoint (adapter.pt)")
    parser.add_argument("--output", type=str, required=True, help="Where to write adapted features (.npy)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()

    features = np.load(args.features).astype(np.float32)
    adapter = AdaptiveRetrievalProjection.load(args.adapter, map_location=args.device)

    if features.shape[1] != adapter.config.input_dim:
        raise ValueError(
            f"Feature dimension {features.shape[1]} does not match adapter's expected "
            f"input_dim={adapter.config.input_dim} -- wrong adapter for this descriptor mode?"
        )

    adapted = apply_adapter(features, adapter, batch_size=args.batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, adapted)

    logger.info(
        "Wrote %s: %s -> %s (adapter=%s)", output_path, features.shape, adapted.shape, args.adapter
    )


if __name__ == "__main__":
    main()
