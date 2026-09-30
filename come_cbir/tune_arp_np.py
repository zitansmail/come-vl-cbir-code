"""
CLI: validation-only hyperparameter sweep for ARP-NP, on already-extracted,
frozen CoME-VL descriptors.

    python -m come_cbir.tune_arp_np \
        --features outputs/come_fused/features.npy \
        --labels outputs/come_fused/labels.npy \
        --paths outputs/come_fused/paths.json \
        --output-dir outputs/arp_np_tuning/come_fused \
        --seeds 0 1 2

Sweeps ``lambda_np`` x ``neighbors_k`` x weight-normalization x graph-mode
(directed/symmetric). For each seed, ``three_way_class_split()`` partitions
classes into train/validation/test once; every grid point is trained on
``train_indices`` and scored on ``val_indices`` only -- test classes are never
touched here (correction #3: hyperparameter selection must not use test
classes). A single plain-ARP run per seed (lambda_np=0) is also trained once
as the reference every ARP-NP grid point is compared against.

Writes ``grid.csv``/``grid.json`` (one row per seed x config) and prints the
top-N configs by mean validation mAP_full delta over plain ARP, averaged
across seeds. This script only *ranks* configs by validation performance --
it does not touch test classes and it does not decide anything for you; read
the table, pick a config (or a short list), then confirm on test classes with
``run_arp_np_controls.py --lambda-np <chosen> --neighbors-k <chosen> ...``.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from come_cbir.evaluate import evaluate_retrieval
from come_cbir.retrieval_adapter import apply_adapter, three_way_class_split
from come_cbir.train_adapter import train_adapter
from come_cbir.utils import set_global_seed, setup_logging


def _evaluate_subset(features: np.ndarray, labels: np.ndarray, paths: List[str], indices: np.ndarray, top_k: int) -> Dict:
    sub_paths = [paths[i] for i in indices]
    return evaluate_retrieval(features[indices], labels[indices], sub_paths, top_k=top_k)["metrics"]


def sweep_one_seed(
    features: np.ndarray, labels: np.ndarray, paths: List[str], seed: int, args, logger,
) -> List[Dict]:
    split = three_way_class_split(
        labels, val_class_fraction=args.val_class_fraction, test_class_fraction=args.test_class_fraction, seed=seed,
    )
    logger.info(
        "seed=%d: %d train classes / %d val classes / %d test classes",
        seed, len(split["train_classes"]), len(split["val_classes"]), len(split["test_classes"]),
    )

    base_kwargs = dict(
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        classes_per_batch=args.classes_per_batch, samples_per_class=args.samples_per_class,
        lr=args.lr, temperature=args.temperature, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        device=args.device,
    )

    # Reference point: plain ARP, once per seed, scored on validation classes.
    adapter_arp, _, _, _ = train_adapter(
        features, labels, split["train_indices"], seed=seed, logger=logger, lambda_np=0.0, **base_kwargs,
    )
    arp_val = _evaluate_subset(apply_adapter(features, adapter_arp), labels, paths, split["val_indices"], args.top_k)
    logger.info("seed=%d: plain-ARP reference val mAP_full=%.4f", seed, arp_val["mAP_full"])

    rows = []
    combos = list(itertools.product(args.lambda_np_grid, args.k_grid, args.weight_normalizations, args.graph_modes))
    logger.info("seed=%d: sweeping %d ARP-NP configurations", seed, len(combos))
    for lambda_np, k, weight_norm, graph_mode in combos:
        adapter_np, _, _, extra = train_adapter(
            features, labels, split["train_indices"], seed=seed, logger=logger,
            lambda_np=lambda_np, neighbors_k=k, symmetric_graph=(graph_mode == "symmetric"),
            np_weight_normalization=weight_norm, **base_kwargs,
        )
        val_metrics = _evaluate_subset(apply_adapter(features, adapter_np), labels, paths, split["val_indices"], args.top_k)
        rows.append({
            "seed": seed, "lambda_np": lambda_np, "neighbors_k": k,
            "weight_normalization": weight_norm, "graph_mode": graph_mode,
            "val_mAP_full": val_metrics["mAP_full"], "val_precision_at_10": val_metrics["precision_at_10"],
            "arp_reference_val_mAP_full": arp_val["mAP_full"],
            "delta_vs_arp": val_metrics["mAP_full"] - arp_val["mAP_full"],
            "final_np_loss": extra["np_loss_history"][-1],
        })
        logger.info(
            "seed=%d lambda_np=%.3g k=%d w=%s graph=%s: val_mAP_full=%.4f (delta=%+.4f)",
            seed, lambda_np, k, weight_norm, graph_mode, val_metrics["mAP_full"], rows[-1]["delta_vs_arp"],
        )
    return rows


def _write_grid(rows: List[Dict], output_dir: Path) -> None:
    fieldnames = list(rows[0].keys())
    with open(output_dir / "grid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "grid.json", "w") as f:
        json.dump(rows, f, indent=2)


def _print_ranked_summary(rows: List[Dict], logger, top_n: int = 10) -> None:
    by_config: Dict[tuple, List[float]] = {}
    for r in rows:
        key = (r["lambda_np"], r["neighbors_k"], r["weight_normalization"], r["graph_mode"])
        by_config.setdefault(key, []).append(r["delta_vs_arp"])

    ranked = sorted(by_config.items(), key=lambda kv: float(np.mean(kv[1])), reverse=True)
    logger.info("Top %d configs by mean validation mAP_full delta over plain ARP (across seeds):", top_n)
    for (lambda_np, k, w, graph_mode), deltas in ranked[:top_n]:
        logger.info(
            "  lambda_np=%.3g k=%d w=%s graph=%s: mean_delta=%+.4f (n_seeds=%d, per-seed=%s)",
            lambda_np, k, w, graph_mode, float(np.mean(deltas)), len(deltas),
            [f"{d:+.4f}" for d in deltas],
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--paths", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val-class-fraction", type=float, default=0.2)
    parser.add_argument("--test-class-fraction", type=float, default=0.2)
    parser.add_argument("--lambda-np-grid", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--k-grid", type=int, nargs="+", default=[3, 5, 10, 20])
    parser.add_argument("--weight-normalizations", type=str, nargs="+", choices=["l1", "softmax"], default=["l1", "softmax"])
    parser.add_argument("--graph-modes", type=str, nargs="+", choices=["directed", "symmetric"], default=["directed", "symmetric"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--classes-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()

    features = np.load(args.features).astype(np.float32)
    labels = np.load(args.labels)
    with open(args.paths) as f:
        paths = json.load(f)

    n_combos = len(args.lambda_np_grid) * len(args.k_grid) * len(args.weight_normalizations) * len(args.graph_modes)
    logger.info(
        "Grid: %d lambda_np x %d k x %d weight-norm x %d graph-mode = %d configs, x %d seeds "
        "(+1 plain-ARP reference per seed) = %d total adapter trainings",
        len(args.lambda_np_grid), len(args.k_grid), len(args.weight_normalizations), len(args.graph_modes),
        n_combos, len(args.seeds), (n_combos + 1) * len(args.seeds),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for seed in args.seeds:
        set_global_seed(seed)
        all_rows.extend(sweep_one_seed(features, labels, paths, seed, args, logger))

    _write_grid(all_rows, output_dir)
    logger.info("Wrote %d rows to %s", len(all_rows), output_dir / "grid.csv")
    _print_ranked_summary(all_rows, logger)


if __name__ == "__main__":
    main()
