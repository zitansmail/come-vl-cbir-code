"""
CLI: run the full ARP-NP unseen-class control matrix on already-extracted,
frozen CoME-VL descriptors, with class-disjoint train/validation/test splits
and multiple seeds.

    python -m come_cbir.run_arp_np_controls \
        --features outputs/come_fused/features.npy \
        --labels outputs/come_fused/labels.npy \
        --paths outputs/come_fused/paths.json \
        --output-dir outputs/arp_np/come_fused \
        --seeds 0 1 2 \
        --lambda-np 1.0 --neighbors-k 10

For every seed, this builds ONE class-disjoint three-way split
(``come_cbir.retrieval_adapter.three_way_class_split``) -- train classes,
validation classes, test classes, mutually exclusive -- and trains/evaluates
the five controls the ARP-NP investigation requires:

  1. frozen          -- no adapter, raw descriptor, evaluated on test classes.
  2. arp             -- plain ARP (lambda_np=0), full --epochs.
  3. arp_np          -- ARP-NP (lambda_np=--lambda-np), full --epochs.
  4. arp_few_epochs  -- plain ARP trained for --epochs-few epochs only (a
                        cheaper-compute control, NOT loss-matched).
  5. arp_loss_matched-- plain ARP re-trained from the same seed, but stopped
                        at the first epoch whose SupCon loss is <= arp_np's
                        final SupCon loss (a control matched by training loss
                        rather than by wall-clock/epoch budget).

Validation-class retrieval metrics are computed for every trained control
(so lambda_np/neighbors_k/epochs can be selected by validation performance
without ever touching test classes), and test-class retrieval metrics are
the final, one-time report. Geometric diagnostics
(``come_cbir.geometry_diagnostics.compute_all_diagnostics``) are computed
between the frozen and adapted embeddings separately on train-class and
test-class samples, for the arp and arp_np controls only.

This script never fits anything on validation or test classes -- only
``train_indices`` from ``three_way_class_split`` is ever passed to
``train_adapter`` or to ``build_knn_graph`` (indirectly, inside
``train_adapter`` when ``lambda_np > 0``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from come_cbir.evaluate import evaluate_retrieval
from come_cbir.geometry_diagnostics import compute_all_diagnostics
from come_cbir.retrieval_adapter import apply_adapter, three_way_class_split
from come_cbir.train_adapter import train_adapter
from come_cbir.utils import set_global_seed, setup_logging


def _evaluate_subset(features: np.ndarray, labels: np.ndarray, paths: List[str], indices: np.ndarray, top_k: int) -> Dict:
    sub_paths = [paths[i] for i in indices]
    return evaluate_retrieval(features[indices], labels[indices], sub_paths, top_k=top_k)["metrics"]


def _train_and_score(
    name: str, features: np.ndarray, labels: np.ndarray, paths: List[str], split: Dict,
    top_k: int, seed: int, logger, train_kwargs: Dict,
) -> Dict:
    adapter, loss_history, n_params, extra = train_adapter(
        features, labels, split["train_indices"], seed=seed, logger=logger, **train_kwargs
    )
    adapted = apply_adapter(features, adapter)
    val_metrics = _evaluate_subset(adapted, labels, paths, split["val_indices"], top_k)
    test_metrics = _evaluate_subset(adapted, labels, paths, split["test_indices"], top_k)
    return {
        "name": name, "n_params": n_params, "final_supcon_loss": extra["supcon_loss_history"][-1],
        "loss_history": loss_history, "supcon_loss_history": extra["supcon_loss_history"],
        "np_loss_history": extra["np_loss_history"],
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "adapter": adapter, "adapted_features": adapted,
    }


def run_one_seed(
    features: np.ndarray, labels: np.ndarray, paths: List[str], seed: int, args, logger,
) -> Dict:
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
        lr=args.lr, temperature=args.temperature, steps_per_epoch=args.steps_per_epoch,
        device=args.device,
    )

    results: Dict[str, Dict] = {}

    # 1. Frozen descriptor -- no adapter at all.
    results["frozen"] = {
        "name": "frozen",
        "val_metrics": _evaluate_subset(features, labels, paths, split["val_indices"], args.top_k),
        "test_metrics": _evaluate_subset(features, labels, paths, split["test_indices"], args.top_k),
    }

    # 2. Plain ARP, full epochs.
    results["arp"] = _train_and_score(
        "arp", features, labels, paths, split, args.top_k, seed, logger,
        {**base_kwargs, "epochs": args.epochs, "lambda_np": 0.0},
    )

    # 3. ARP-NP, full epochs.
    results["arp_np"] = _train_and_score(
        "arp_np", features, labels, paths, split, args.top_k, seed, logger,
        {**base_kwargs, "epochs": args.epochs, "lambda_np": args.lambda_np,
         "neighbors_k": args.neighbors_k, "symmetric_graph": args.symmetric_graph,
         "np_weight_normalization": args.np_weight_normalization},
    )

    # 4. Plain ARP, fewer epochs (cheap-compute control, not loss-matched).
    results["arp_few_epochs"] = _train_and_score(
        "arp_few_epochs", features, labels, paths, split, args.top_k, seed, logger,
        {**base_kwargs, "epochs": args.epochs_few, "lambda_np": 0.0},
    )

    # 5. Plain ARP, re-trained from scratch but stopped at the first epoch
    # whose SupCon loss is <= arp_np's final SupCon loss (loss-matched control).
    target_loss = results["arp_np"]["final_supcon_loss"]
    matched_epochs = args.epochs
    for i, v in enumerate(results["arp"]["supcon_loss_history"]):
        if v <= target_loss:
            matched_epochs = i + 1
            break
    results["arp_loss_matched"] = _train_and_score(
        "arp_loss_matched", features, labels, paths, split, args.top_k, seed, logger,
        {**base_kwargs, "epochs": matched_epochs, "lambda_np": 0.0},
    )
    results["arp_loss_matched"]["matched_epochs"] = matched_epochs
    results["arp_loss_matched"]["target_loss"] = target_loss

    # Geometric diagnostics: frozen vs. adapted, on train-class and test-class
    # samples separately, for the two controls that matter most (arp, arp_np).
    diagnostics = {}
    for control in ("arp", "arp_np"):
        adapted = results[control]["adapted_features"]
        diagnostics[control] = {
            "train_classes": compute_all_diagnostics(
                features[split["train_indices"]], adapted[split["train_indices"]], k=args.neighbors_k, seed=seed,
            ),
            "test_classes": compute_all_diagnostics(
                features[split["test_indices"]], adapted[split["test_indices"]], k=args.neighbors_k, seed=seed,
            ),
        }

    return {
        "seed": seed,
        "split": {k: v for k, v in split.items() if k != "train_indices" and k != "val_indices" and k != "test_indices"},
        "n_train_images": len(split["train_indices"]), "n_val_images": len(split["val_indices"]),
        "n_test_images": len(split["test_indices"]),
        "controls": {
            name: {k: v for k, v in r.items() if k not in ("adapter", "adapted_features")}
            for name, r in results.items()
        },
        "diagnostics": diagnostics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--paths", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val-class-fraction", type=float, default=0.2)
    parser.add_argument("--test-class-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--epochs-few", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--classes-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-np", type=float, default=1.0)
    parser.add_argument("--neighbors-k", type=int, default=10)
    parser.add_argument("--symmetric-graph", action="store_true")
    parser.add_argument("--np-weight-normalization", type=str, default="l1", choices=["l1", "softmax"])
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_seed_results = []
    for seed in args.seeds:
        set_global_seed(seed)
        result = run_one_seed(features, labels, paths, seed, args, logger)
        with open(output_dir / f"seed_{seed}.json", "w") as f:
            json.dump(result, f, indent=2)
        all_seed_results.append(result)
        logger.info("seed=%d done, wrote %s", seed, output_dir / f"seed_{seed}.json")

    with open(output_dir / "all_seeds.json", "w") as f:
        json.dump(all_seed_results, f, indent=2)
    logger.info("Wrote aggregate results to %s", output_dir / "all_seeds.json")


if __name__ == "__main__":
    main()
