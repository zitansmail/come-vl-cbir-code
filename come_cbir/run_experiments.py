"""
CLI: run a batch of CBIR descriptor/pooling/PCA configurations from a YAML
file and produce a single comparison table.

    python -m come_cbir.run_experiments --config configs/cbir_corel1000.yaml

See configs/cbir_corel1000.yaml for the expected format. Each entry under
`experiments` reuses the same loaded checkpoint (loaded once and cached by
(checkpoint, device, dtype), not once per experiment) so sweeping multiple
descriptor_mode/pooling/pca_dimension combinations does not repeatedly pay
the cost of loading the full CoME-VL model.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from come_cbir.config import ExperimentEntry, RunConfig
from come_cbir.datasets import CBIRImageDataset, cbir_collate_fn
from come_cbir.dimensionality import PCAReducer
from come_cbir.evaluate import evaluate_retrieval
from come_cbir.feature_extractor import CoMECbirFeatureExtractor
from come_cbir.retrieval_adapter import apply_adapter, save_split, stratified_holdout_split
from come_cbir.train_adapter import train_adapter
from come_cbir.utils import set_global_seed, setup_logging

_MODEL_CACHE: Dict[Tuple[str, str, str], object] = {}


def _get_or_load_model(checkpoint: str, device: str, dtype: str):
    from come_cbir.utils import parse_dtype

    key = (checkpoint, device, dtype)
    if key not in _MODEL_CACHE:
        from olmo.model import Molmo

        model = Molmo.from_checkpoint(checkpoint, device=device)
        model.eval()
        model.to(parse_dtype(dtype))
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _extract_all(
    dataset: CBIRImageDataset, extractor: CoMECbirFeatureExtractor, batch_size: int, num_workers: int
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], float]:
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=cbir_collate_fn
    )
    chunks, paths, labels, class_names = [], [], [], []
    start = time.perf_counter()
    for batch in loader:
        if batch is None:
            continue
        patches, batch_paths, batch_labels, batch_class_names = batch
        descriptors = extractor.encode_images(patches)
        chunks.append(descriptors.cpu().numpy())
        paths.extend(batch_paths)
        labels.extend(batch_labels)
        class_names.extend(batch_class_names)
    elapsed = time.perf_counter() - start
    features = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, extractor.descriptor_dim), dtype=np.float32)
    return features, paths, np.array(labels, dtype=np.int64), class_names, elapsed


def _fit_pca_split(
    features: np.ndarray, splits: List[Optional[str]], dimension: int, logger
) -> np.ndarray:
    train_mask = np.array([s in ("train", "database") for s in splits], dtype=bool)
    if train_mask.any() and not train_mask.all():
        fit_features = features[train_mask]
        logger.info(
            "Fitting PCA(%d) on %d/%d samples flagged as train/database split", dimension,
            fit_features.shape[0], features.shape[0],
        )
    else:
        fit_features = features
        logger.warning(
            "No train/database vs query/test split column found (or all rows share one split); "
            "fitting PCA(%d) on the full %d-sample set. This means PCA statistics include "
            "whatever rows are later used as evaluation queries -- see docs/CBIR.md "
            "'PCA and data leakage' before trusting absolute mAP numbers from this run.",
            dimension, features.shape[0],
        )
    reducer = PCAReducer(n_components=dimension).fit(fit_features)
    return reducer.transform(features)


def _row_from_metrics(entry: ExperimentEntry, variant: str, dim: int, metrics: Dict,
                       extraction_time: Optional[float], n_images: int, adapter_params: Optional[int] = None) -> Dict:
    return {
        "name": entry.name,
        "variant": variant,  # "original" or "adapted" -- see docs/retrieval_adapter.md
        "descriptor_mode": entry.descriptor_mode,
        "pooling": entry.pooling,
        "original_dimension": dim,
        "pca_dimension": entry.pca_dimension or "",
        "adapter_params": adapter_params or "",
        "mAP_at_10": metrics["mAP_at_10"],
        "precision_at_10": metrics["precision_at_10"],
        "recall_at_10": metrics["recall_at_10"],
        "extraction_time_seconds": extraction_time if extraction_time is not None else "",
        "extraction_seconds_per_image": (extraction_time / n_images) if extraction_time and n_images else "",
        "query_latency_seconds": metrics["query_latency_seconds_per_query_top_k"],
        "descriptor_memory_mb": metrics["descriptor_memory_mb"],
    }


def run_single_experiment(
    entry: ExperimentEntry,
    config: RunConfig,
    dataset: CBIRImageDataset,
    logger,
) -> List[Dict]:
    model = _get_or_load_model(config.model.checkpoint, config.model.device, config.model.dtype)
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode=entry.descriptor_mode,
        pooling=entry.pooling,
        device=config.model.device,
        dtype=config.model.dtype,
        gem_p=entry.gem_p,
        model=model,
    )

    features, paths, labels, class_names, extraction_time = _extract_all(
        dataset, extractor, config.batch_size, config.num_workers
    )
    original_dim = features.shape[1]

    pca_dim = entry.pca_dimension
    if pca_dim:
        splits = [s.split for s in dataset.samples]
        features = _fit_pca_split(features, splits, pca_dim, logger)

    output_dir = Path(config.output_root) / entry.name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", features)
    with open(output_dir / "paths.json", "w") as f:
        json.dump(paths, f)
    np.save(output_dir / "labels.npy", labels)

    if entry.adapter and entry.adapter.enabled:
        # Adapter comparisons must be leakage-free: train only on a stratified train
        # split, then evaluate *both* the original and the adapted descriptors on the
        # same held-out subset, so the two rows below are directly, fairly comparable
        # -- see the module docstring in come_cbir/retrieval_adapter.py.
        train_idx, holdout_idx = stratified_holdout_split(
            labels, holdout_fraction=entry.adapter.holdout_fraction, seed=config.seed
        )
        holdout_features = features[holdout_idx]
        holdout_labels = labels[holdout_idx]
        holdout_paths = [paths[i] for i in holdout_idx]

        original_eval = evaluate_retrieval(holdout_features, holdout_labels, holdout_paths, top_k=config.top_k)
        with open(output_dir / "metrics_original_holdout.json", "w") as f:
            json.dump(original_eval["metrics"], f, indent=2)

        adapter, loss_history, n_params, extra = train_adapter(
            features, labels, train_idx,
            adapter_type=entry.adapter.type, hidden_dim=entry.adapter.hidden_dim, output_dim=entry.adapter.output_dim,
            epochs=entry.adapter.epochs, steps_per_epoch=entry.adapter.steps_per_epoch,
            classes_per_batch=entry.adapter.classes_per_batch, samples_per_class=entry.adapter.samples_per_class,
            lr=entry.adapter.lr, temperature=entry.adapter.temperature,
            lambda_np=entry.adapter.lambda_np, neighbors_k=entry.adapter.neighbors_k,
            symmetric_graph=entry.adapter.symmetric_graph, np_weight_normalization=entry.adapter.np_weight_normalization,
            device=config.model.device, seed=config.seed, logger=logger,
        )
        adapter.save(str(output_dir / "adapter.pt"))
        save_split(str(output_dir / "split.json"), train_idx, holdout_idx)
        with open(output_dir / "adapter_training_log.json", "w") as f:
            json.dump({
                "loss_history": loss_history,
                "supcon_loss_history": extra["supcon_loss_history"],
                "np_loss_history": extra["np_loss_history"],
                "trainable_parameters": n_params,
            }, f, indent=2)

        adapted_holdout_features = apply_adapter(holdout_features, adapter)
        adapted_eval = evaluate_retrieval(adapted_holdout_features, holdout_labels, holdout_paths, top_k=config.top_k)
        with open(output_dir / "metrics_adapted_holdout.json", "w") as f:
            json.dump(adapted_eval["metrics"], f, indent=2)

        n_holdout = len(holdout_idx)
        return [
            _row_from_metrics(entry, "original", original_dim, original_eval["metrics"], None, n_holdout),
            _row_from_metrics(entry, "adapted", adapter.config.output_dim, adapted_eval["metrics"], None, n_holdout, n_params),
        ]

    eval_result = evaluate_retrieval(features, labels, paths, top_k=config.top_k)
    metrics = eval_result["metrics"]
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return [_row_from_metrics(entry, "original", original_dim, metrics, extraction_time, features.shape[0])]


def _write_comparison_table(rows: List[Dict], output_root: Path) -> None:
    fieldnames = list(rows[0].keys())
    with open(output_root / "comparison_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# CBIR Experiment Comparison", "", "| " + " | ".join(fieldnames) + " |",
              "|" + "---|" * len(fieldnames)]
    for row in rows:
        values = []
        for key in fieldnames:
            v = row[key]
            values.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(values) + " |")
    (output_root / "comparison_table.md").write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()
    config = RunConfig.from_yaml(args.config)
    set_global_seed(config.seed)

    if config.dataset.type == "folder":
        dataset = CBIRImageDataset(root=config.dataset.root)
    else:
        dataset = CBIRImageDataset(manifest_csv=config.dataset.manifest_csv)

    logger.info("Loaded dataset: %d images, %d classes", len(dataset), len(dataset.class_names))

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in config.experiments:
        logger.info("Running experiment '%s' (%s / %s, pca=%s, adapter=%s)",
                    entry.name, entry.descriptor_mode, entry.pooling, entry.pca_dimension,
                    entry.adapter.enabled if entry.adapter else False)
        rows.extend(run_single_experiment(entry, config, dataset, logger))

    _write_comparison_table(rows, output_root)
    logger.info("Wrote comparison table to %s", output_root / "comparison_table.csv")


if __name__ == "__main__":
    main()
