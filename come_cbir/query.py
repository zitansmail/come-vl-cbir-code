"""
CLI: query a built index with a single image and (optionally) render a
result grid.

    python -m come_cbir.query \\
        --image examples/query.jpg \\
        --checkpoint /models/come-vl \\
        --descriptor-mode come_fused \\
        --pooling mean \\
        --index outputs/index.faiss \\
        --metadata outputs/metadata.json \\
        --top-k 10 \\
        --output outputs/query_result.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from come_cbir.dimensionality import PCAReducer
from come_cbir.feature_extractor import CoMECbirFeatureExtractor
from come_cbir.indexing import SUPPORTED_BACKENDS, load_index, load_paths
from come_cbir.retrieval_adapter import AdaptiveRetrievalProjection, apply_adapter
from come_cbir.utils import setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--descriptor-mode", type=str, default="come_fused",
                         choices=["siglip", "dino", "concat", "come_fused"])
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls", "gem", "attention"])
    parser.add_argument("--gem-p", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--index-backend", type=str, default="faiss-flat", choices=list(SUPPORTED_BACKENDS))
    parser.add_argument("--pca-model", type=str, default=None, help="Fitted PCA model (.pkl) to apply, if any")
    parser.add_argument("--adapter-checkpoint", type=str, default=None,
                         help="Trained come_cbir.retrieval_adapter checkpoint (adapter.pt) to apply to the "
                              "query descriptor -- the index searched must have been built from features "
                              "put through the same adapter (see apply_adapter.py)")
    parser.add_argument("--metadata", type=str, default=None, help="metadata.json from extract_features.py")
    parser.add_argument("--labels", type=str, default=None, help="labels.npy aligned with the index's paths")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=str, default=None, help="Optional result-grid image path (.png)")
    parser.add_argument(
        "--query-label", type=int, default=None,
        help="Ground-truth class label for the query image, used to color each result's border green "
             "(same class as query) or red (different class) in the result grid. Auto-detected from "
             "--labels when the query image is itself one of the indexed paths (the leave-one-out case "
             "this project uses); only needs to be passed explicitly for a query image outside the index.",
    )
    parser.add_argument(
        "--low-memory", action="store_true",
        help="Load the checkpoint with a much lower peak-RAM footprint -- see "
             "come_cbir/checkpoint_loading.py and extract_features.py --help for details.",
    )
    return parser


def run_query(
    image_path: str,
    extractor: CoMECbirFeatureExtractor,
    index,
    index_paths: List[str],
    top_k: int,
    pca: Optional[PCAReducer] = None,
    labels: Optional[np.ndarray] = None,
    adapter: Optional[AdaptiveRetrievalProjection] = None,
) -> Dict:
    image = Image.open(image_path).convert("RGB")

    t0 = time.perf_counter()
    descriptor = extractor.encode_images([image]).cpu().numpy()
    extract_time = time.perf_counter() - t0

    if pca is not None:
        descriptor = pca.transform(descriptor)

    if adapter is not None:
        if descriptor.shape[1] != adapter.config.input_dim:
            raise ValueError(
                f"Query descriptor dimension {descriptor.shape[1]} does not match adapter's expected "
                f"input_dim={adapter.config.input_dim} -- wrong adapter for this descriptor mode?"
            )
        descriptor = apply_adapter(descriptor, adapter)

    # Search for one extra result in case the query image is itself in the database.
    t0 = time.perf_counter()
    result = index.search(descriptor, min(top_k + 1, len(index_paths)))
    search_time = time.perf_counter() - t0

    query_abspath = str(Path(image_path).resolve())
    ranked = []
    rank = 0
    for idx, score in zip(result.indices[0], result.scores[0]):
        if idx < 0:
            continue
        candidate_path = index_paths[idx]
        if str(Path(candidate_path).resolve()) == query_abspath:
            continue  # exclude the query image itself if it exists in the database
        rank += 1
        ranked.append({
            "rank": rank,
            "path": candidate_path,
            "score": float(score),
            "label": int(labels[idx]) if labels is not None else None,
        })
        if rank >= top_k:
            break

    return {
        "query_path": image_path,
        "results": ranked,
        "extraction_time_seconds": extract_time,
        "search_time_seconds": search_time,
    }


def render_result_grid(
    query_path: str, results: List[Dict], output_path: str, query_label: Optional[int] = None
) -> None:
    """
    Query image on its own row at the top, results below in a grid. Each
    result's border is green if its label matches ``query_label`` (same
    class as the query), red if it doesn't, gray if the label -- either the
    query's or that result's -- isn't known.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    CORRECT_COLOR = "#2ecc71"
    WRONG_COLOR = "#e74c3c"
    UNKNOWN_COLOR = "#999999"

    n = len(results)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(3 * cols, 3 * rows + 3.4))
    gs = GridSpec(rows + 1, cols, height_ratios=[2.4] + [1] * rows, figure=fig)

    query_ax = fig.add_subplot(gs[0, :])
    query_img = Image.open(query_path).convert("RGB")
    query_ax.imshow(query_img)
    query_ax.set_title("Query Image", fontsize=15, fontweight="bold")
    query_ax.set_xticks([])
    query_ax.set_yticks([])

    for i, item in enumerate(results):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(gs[row + 1, col])
        try:
            img = Image.open(item["path"]).convert("RGB")
            ax.imshow(img)
        except (OSError, FileNotFoundError):
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

        result_label = item.get("label")
        if query_label is not None and result_label is not None:
            color = CORRECT_COLOR if int(result_label) == int(query_label) else WRONG_COLOR
        else:
            color = UNKNOWN_COLOR
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.5)
            spine.set_visible(True)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()

    extractor = CoMECbirFeatureExtractor(
        model_name_or_path=args.checkpoint,
        descriptor_mode=args.descriptor_mode,
        pooling=args.pooling,
        device=args.device,
        dtype=args.dtype,
        gem_p=args.gem_p,
        low_memory=args.low_memory,
    )

    index = load_index(args.index_backend, args.index)
    index_paths = load_paths(args.index)

    pca = PCAReducer.load(args.pca_model) if args.pca_model else None
    labels = np.load(args.labels) if args.labels else None
    adapter = AdaptiveRetrievalProjection.load(args.adapter_checkpoint) if args.adapter_checkpoint else None

    if args.metadata:
        with open(args.metadata) as f:
            metadata = json.load(f)
        if metadata.get("descriptor_mode") != args.descriptor_mode or metadata.get("pooling") != args.pooling:
            logger.warning(
                "Query descriptor_mode/pooling (%s/%s) does not match metadata.json (%s/%s) -- "
                "results will not be meaningfully comparable to the indexed database.",
                args.descriptor_mode, args.pooling, metadata.get("descriptor_mode"), metadata.get("pooling"),
            )

    result = run_query(args.image, extractor, index, index_paths, args.top_k, pca=pca, labels=labels, adapter=adapter)

    print(json.dumps(result, indent=2))

    if args.output:
        query_label = args.query_label
        if query_label is None and labels is not None:
            query_abspath = str(Path(args.image).resolve())
            for idx, candidate_path in enumerate(index_paths):
                if str(Path(candidate_path).resolve()) == query_abspath:
                    query_label = int(labels[idx])
                    break
            if query_label is None:
                logger.warning(
                    "Could not auto-detect the query image's label (it is not one of the indexed "
                    "--labels paths) -- result borders will be gray instead of green/red. Pass "
                    "--query-label explicitly if you know it.",
                )
        render_result_grid(args.image, result["results"], args.output, query_label=query_label)
        logger.info("Wrote result grid to %s", args.output)


if __name__ == "__main__":
    main()
