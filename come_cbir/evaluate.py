"""
Retrieval evaluation metrics and CLI for class-based CBIR datasets.

Two images are considered relevant to each other if they share the same
class label. Evaluation defaults to leave-one-out: every database image is
also used as a query, with the query itself excluded from its own results.

Exact cosine search (faiss-flat / IndexFlatIP) is used as the primary
descriptor-quality measure so PCA/pooling/descriptor-mode comparisons are
not confounded by approximate-index recall loss. Approximate-index quality
(HNSW/IVF/Annoy) should be evaluated separately by pointing --index-backend
at one of those backends -- the CLI reports which backend produced a given
metrics.json in metadata so the two are never silently mixed.

    python -m come_cbir.evaluate \\
        --features outputs/features.npy \\
        --labels outputs/labels.npy \\
        --paths outputs/paths.json \\
        --top-k 10 \\
        --output-dir outputs/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from come_cbir.indexing import SUPPORTED_BACKENDS, build_index
from come_cbir.utils import setup_logging

logger = None  # set in main()


# ----------------------------------------------------------------------
# Per-query metrics
# ----------------------------------------------------------------------


def precision_at_k(relevance: np.ndarray, k: int) -> float:
    """relevance: (n_retrieved,) binary array, ranked. Fraction relevant among top-k."""
    if k <= 0:
        return 0.0
    top = relevance[:k]
    return float(top.sum()) / float(min(k, len(relevance))) if len(relevance) > 0 else 0.0


def recall_at_k(relevance: np.ndarray, k: int, num_relevant_total: int) -> float:
    if num_relevant_total <= 0:
        return 0.0
    top = relevance[:k]
    return float(top.sum()) / float(num_relevant_total)


def average_precision_at_k(relevance: np.ndarray, k: int, num_relevant_total: int) -> float:
    """AP@k: mean of precision@i for each relevant hit i<=k, normalized by min(num_relevant_total, k)."""
    if num_relevant_total <= 0:
        return 0.0
    top = relevance[:k]
    hits = 0
    precisions = []
    for i, rel in enumerate(top, start=1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    denom = min(num_relevant_total, k)
    return float(sum(precisions) / denom) if denom > 0 else 0.0


def full_average_precision(relevance: np.ndarray, num_relevant_total: int) -> float:
    """Standard (unclamped) average precision over the full ranked list."""
    if num_relevant_total <= 0:
        return 0.0
    hits = 0
    precisions = []
    for i, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    return float(sum(precisions) / num_relevant_total) if precisions else 0.0


def reciprocal_rank(relevance: np.ndarray) -> float:
    hits = np.nonzero(relevance)[0]
    if len(hits) == 0:
        return 0.0
    return 1.0 / float(hits[0] + 1)


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Binary-relevance NDCG@k (gain=relevance, log2 discount)."""
    top = relevance[:k]
    if len(top) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float(np.sum(top * discounts))
    ideal = np.sort(top)[::-1]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------------------
# Full evaluation run
# ----------------------------------------------------------------------


def evaluate_retrieval(
    features: np.ndarray,
    labels: np.ndarray,
    paths: Sequence[str],
    top_k: int = 10,
    index_backend: str = "faiss-flat",
    compute_full_map: bool = True,
    **index_kwargs,
) -> Dict:
    """
    Leave-one-out class-based retrieval evaluation. Returns a dict with
    aggregate metrics, per-query records, and timing/memory diagnostics.
    """
    n = features.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to evaluate retrieval")

    build_start = time.perf_counter()
    index = build_index(index_backend, features, **index_kwargs)
    build_time = time.perf_counter() - build_start

    # +1 so we can drop the self-match (leave-one-out) and still return top_k results;
    # for full mAP we search the whole database.
    search_k = min(n, top_k + 1)
    full_k = n if compute_full_map else search_k

    search_start = time.perf_counter()
    result_topk = index.search(features, search_k)
    search_time_topk = time.perf_counter() - search_start

    full_result = None
    full_search_time = 0.0
    if compute_full_map:
        full_start = time.perf_counter()
        full_result = index.search(features, full_k)
        full_search_time = time.perf_counter() - full_start

    per_query_records: List[Dict] = []
    precisions_1, precisions_5, precisions_10 = [], [], []
    recalls_10 = []
    maps_10 = []
    full_maps = []
    mrrs = []
    ndcgs_10 = []
    # Metrics computed at the *actual* requested --top-k, in addition to the fixed
    # @1/@5/@10 checkpoints above. Those checkpoints stay hardcoded so already-reported
    # results (e.g. the Corel-1K/Corel-10K tables) remain reproducible bit-for-bit
    # regardless of --top-k; these dynamically-named fields (precision_at_{top_k}, etc.)
    # are what actually respond to --top-k, since passing --top-k 20 previously silently
    # produced identical output to --top-k 10 -- every metric above was hardcoded to a
    # fixed rank, never using the top_k variable at all.
    precisions_topk, recalls_topk, maps_topk, ndcgs_topk = [], [], [], []

    for qi in range(n):
        query_label = labels[qi]
        num_relevant_total = int(np.sum(labels == query_label)) - 1  # exclude self
        if num_relevant_total <= 0:
            continue  # no other same-class images -- cannot evaluate this query meaningfully

        row_indices = result_topk.indices[qi]
        row_indices = row_indices[row_indices != qi][:top_k]
        relevance = (labels[row_indices] == query_label).astype(np.int64)

        p1 = precision_at_k(relevance, 1)
        p5 = precision_at_k(relevance, 5)
        p10 = precision_at_k(relevance, 10)
        r10 = recall_at_k(relevance, 10, num_relevant_total)
        ap10 = average_precision_at_k(relevance, 10, num_relevant_total)
        rr = reciprocal_rank(relevance)
        ndcg10 = ndcg_at_k(relevance, 10)

        p_topk = precision_at_k(relevance, top_k)
        r_topk = recall_at_k(relevance, top_k, num_relevant_total)
        ap_topk = average_precision_at_k(relevance, top_k, num_relevant_total)
        ndcg_topk = ndcg_at_k(relevance, top_k)

        full_ap = None
        if compute_full_map and full_result is not None:
            full_row = full_result.indices[qi]
            full_row = full_row[full_row != qi]
            full_relevance = (labels[full_row] == query_label).astype(np.int64)
            full_ap = full_average_precision(full_relevance, num_relevant_total)
            full_maps.append(full_ap)

        precisions_1.append(p1)
        precisions_5.append(p5)
        precisions_10.append(p10)
        recalls_10.append(r10)
        maps_10.append(ap10)
        mrrs.append(rr)
        ndcgs_10.append(ndcg10)
        precisions_topk.append(p_topk)
        recalls_topk.append(r_topk)
        maps_topk.append(ap_topk)
        ndcgs_topk.append(ndcg_topk)

        per_query_records.append({
            "query_index": qi,
            "query_path": paths[qi],
            "query_label": int(query_label),
            "num_relevant_total": num_relevant_total,
            "precision_at_1": p1,
            "precision_at_5": p5,
            "precision_at_10": p10,
            "recall_at_10": r10,
            "average_precision_at_10": ap10,
            "full_average_precision": full_ap,
            "reciprocal_rank": rr,
            "ndcg_at_10": ndcg10,
            f"precision_at_{top_k}": p_topk,
            f"recall_at_{top_k}": r_topk,
            f"average_precision_at_{top_k}": ap_topk,
            f"ndcg_at_{top_k}": ndcg_topk,
        })

    descriptor_bytes = int(features.nbytes)
    metrics = {
        "num_queries_evaluated": len(per_query_records),
        "num_database_images": n,
        "requested_top_k": top_k,
        "precision_at_1": float(np.mean(precisions_1)) if precisions_1 else None,
        "precision_at_5": float(np.mean(precisions_5)) if precisions_5 else None,
        "precision_at_10": float(np.mean(precisions_10)) if precisions_10 else None,
        "recall_at_10": float(np.mean(recalls_10)) if recalls_10 else None,
        "mAP_at_10": float(np.mean(maps_10)) if maps_10 else None,
        "mAP_full": float(np.mean(full_maps)) if full_maps else None,
        "mean_reciprocal_rank": float(np.mean(mrrs)) if mrrs else None,
        "ndcg_at_10": float(np.mean(ndcgs_10)) if ndcgs_10 else None,
        f"precision_at_{top_k}": float(np.mean(precisions_topk)) if precisions_topk else None,
        f"recall_at_{top_k}": float(np.mean(recalls_topk)) if recalls_topk else None,
        f"mAP_at_{top_k}": float(np.mean(maps_topk)) if maps_topk else None,
        f"ndcg_at_{top_k}": float(np.mean(ndcgs_topk)) if ndcgs_topk else None,
        "index_backend": index_backend,
        "index_build_time_seconds": build_time,
        "query_latency_seconds_per_query_top_k": search_time_topk / n,
        "index_search_latency_seconds_total_top_k": search_time_topk,
        "descriptor_memory_bytes": descriptor_bytes,
        "descriptor_memory_mb": descriptor_bytes / (1024 ** 2),
        "descriptor_dim": int(features.shape[1]),
    }
    if compute_full_map:
        metrics["full_map_search_latency_seconds_total"] = full_search_time

    return {"metrics": metrics, "per_query": per_query_records}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--paths", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--index-backend", type=str, default="faiss-flat", choices=list(SUPPORTED_BACKENDS))
    parser.add_argument("--no-full-map", action="store_true",
                         help="Skip full (unclamped) mAP -- useful for very large databases")
    parser.add_argument("--adapter-checkpoint", type=str, default=None,
                         help="Trained come_cbir.retrieval_adapter checkpoint (adapter.pt) to apply "
                              "before evaluation -- see train_adapter.py/docs/retrieval_adapter.md")
    parser.add_argument("--holdout-indices", type=str, default=None,
                         help="A split.json (from train_adapter.py) to restrict evaluation to the "
                              "holdout subset only -- required for a leakage-free original-vs-adapted "
                              "comparison; pass the same file for both the original-descriptor and "
                              "adapted-descriptor runs")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    global logger
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()

    features = np.load(args.features).astype(np.float32)
    labels = np.load(args.labels)
    with open(args.paths) as f:
        paths = json.load(f)

    if not (len(paths) == len(labels) == features.shape[0]):
        raise ValueError(
            f"Mismatched lengths: features={features.shape[0]}, labels={len(labels)}, paths={len(paths)}"
        )

    if args.holdout_indices:
        from come_cbir.retrieval_adapter import load_split

        _, holdout_idx = load_split(args.holdout_indices)
        features = features[holdout_idx]
        labels = labels[holdout_idx]
        paths = [paths[i] for i in holdout_idx]
        logger.info(
            "Restricted evaluation to %d holdout samples from %s (leakage-free comparison subset)",
            len(holdout_idx), args.holdout_indices,
        )

    if args.adapter_checkpoint:
        from come_cbir.retrieval_adapter import AdaptiveRetrievalProjection, apply_adapter

        adapter = AdaptiveRetrievalProjection.load(args.adapter_checkpoint)
        if features.shape[1] != adapter.config.input_dim:
            raise ValueError(
                f"Feature dimension {features.shape[1]} does not match adapter's expected "
                f"input_dim={adapter.config.input_dim} -- wrong adapter for this descriptor mode?"
            )
        original_dim = features.shape[1]
        features = apply_adapter(features, adapter)
        logger.info(
            "Applied adapter %s: %d -> %d dimensions", args.adapter_checkpoint, original_dim, features.shape[1]
        )

    logger.info(
        "Evaluating %d descriptors (dim=%d) with backend=%s, top_k=%d",
        features.shape[0], features.shape[1], args.index_backend, args.top_k,
    )
    result = evaluate_retrieval(
        features, labels, paths, top_k=args.top_k, index_backend=args.index_backend,
        compute_full_map=not args.no_full_map,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(result["metrics"], f, indent=2)

    per_query = result["per_query"]
    if per_query:
        with open(output_dir / "per_query_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_query[0].keys()))
            writer.writeheader()
            writer.writerows(per_query)

    _write_summary_md(output_dir / "summary.md", result["metrics"], args)
    logger.info("Wrote metrics.json, per_query_metrics.csv, summary.md to %s", output_dir)


def _write_summary_md(path: Path, metrics: Dict, args: argparse.Namespace) -> None:
    lines = [
        "# CBIR Retrieval Evaluation Summary",
        "",
        f"- Features: `{args.features}`",
        f"- Index backend (primary, exact unless stated otherwise): `{args.index_backend}`",
        f"- Top-K: {args.top_k}",
        f"- Database size: {metrics['num_database_images']}",
        f"- Queries evaluated (leave-one-out, classes with >=2 members): {metrics['num_queries_evaluated']}",
        f"- Adapter applied: `{args.adapter_checkpoint}`" if args.adapter_checkpoint else "- Adapter applied: none (original descriptors)",
        f"- Restricted to holdout subset: `{args.holdout_indices}`" if args.holdout_indices else "- Restricted to holdout subset: no (full dataset)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    base_keys = [
        "precision_at_1", "precision_at_5", "precision_at_10", "recall_at_10",
        "mAP_at_10", "mAP_full", "mean_reciprocal_rank", "ndcg_at_10",
    ]
    # Only real, additional rows if --top-k isn't one of the fixed checkpoints already
    # printed above (e.g. --top-k 20 adds precision_at_20/recall_at_20/mAP_at_20/ndcg_at_20;
    # --top-k 10, the default, would just repeat the @10 rows already listed and is skipped).
    topk_keys = [
        f"precision_at_{args.top_k}", f"recall_at_{args.top_k}",
        f"mAP_at_{args.top_k}", f"ndcg_at_{args.top_k}",
    ]
    tail_keys = [
        "query_latency_seconds_per_query_top_k", "index_search_latency_seconds_total_top_k",
        "descriptor_memory_mb", "descriptor_dim",
    ]
    seen = set()
    ordered_keys = []
    for key in base_keys + topk_keys + tail_keys:
        if key not in seen:
            seen.add(key)
            ordered_keys.append(key)

    for key in ordered_keys:
        value = metrics.get(key)
        if value is not None:
            lines.append(f"| {key} | {value:.6g} |" if isinstance(value, float) else f"| {key} | {value} |")
    if args.index_backend != "faiss-flat":
        lines.append("")
        lines.append(
            "**Note:** this run used an approximate index "
            f"(`{args.index_backend}`); compare against a `faiss-flat` run of the same "
            "features before attributing metric differences to the descriptor itself."
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
