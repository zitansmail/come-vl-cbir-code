"""
CLI: extract CBIR descriptors for a dataset.

    python -m come_cbir.extract_features \\
        --dataset-root /data/corel1000 \\
        --checkpoint /models/come-vl \\
        --descriptor-mode come_fused \\
        --pooling mean \\
        --batch-size 16 \\
        --num-workers 4 \\
        --device cuda \\
        --dtype bfloat16 \\
        --output-dir outputs/corel1000/come_fused_mean

Writes to --output-dir: features.npy, paths.json, labels.npy,
class_names.json, metadata.json. Supports resuming an interrupted run: if
the output directory already contains a partial run whose recorded paths
are a prefix of the current dataset (same deterministic ordering), only the
remaining images are (re-)computed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from come_cbir.datasets import CBIRImageDataset, cbir_collate_fn
from come_cbir.feature_extractor import CoMECbirFeatureExtractor
from come_cbir.utils import get_git_commit_hash, set_global_seed, setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset-root", type=str, help="Folder-tree dataset root")
    dataset_group.add_argument("--manifest-csv", type=str, help="Manifest CSV with image_path,label,class_name,split")

    parser.add_argument("--checkpoint", type=str, required=True, help="CoME-VL/Molmo checkpoint directory")
    parser.add_argument("--descriptor-mode", type=str, default="come_fused",
                         choices=["siglip", "dino", "concat", "come_fused"])
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls", "gem", "attention"])
    parser.add_argument("--gem-p", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-corrupted", action="store_true",
                         help="Log and skip unreadable images instead of failing the whole run")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing partial run in --output-dir")
    parser.add_argument(
        "--adapter-checkpoint", type=str, default=None,
        help="Trained come_cbir.retrieval_adapter checkpoint (adapter.pt) to apply to the raw "
             "descriptors as a final step, overwriting features.npy with the adapted, compact "
             "embedding. Applied once at the end (not per-batch), so it does not interact with "
             "the resume/flush logic above -- see docs/retrieval_adapter.md. The more common "
             "workflow is to extract once without this flag, then use apply_adapter.py separately "
             "so both the original and adapted features.npy are available for comparison.",
    )
    parser.add_argument(
        "--low-memory", action="store_true",
        help="Load the checkpoint with a much lower peak-RAM footprint (meta-device construction + "
             "mmap'd, assign=True state-dict loading) instead of olmo.model.Molmo.from_checkpoint's "
             "default ~2x-model-size peak. Use this if extraction fails/hangs while loading the "
             "checkpoint on a RAM-constrained machine (e.g. free-tier Colab). See "
             "come_cbir/checkpoint_loading.py for details; only supports unsharded checkpoints "
             "(a single model.pt), not sharded/FSDP checkpoint directories.",
    )
    return parser


def _try_resume(output_dir: Path, expected_paths_order_key: List[str]) -> int:
    """Return how many leading samples of the current dataset are already extracted, or 0."""
    features_path = output_dir / "features.npy"
    paths_path = output_dir / "paths.json"
    if not (features_path.exists() and paths_path.exists()):
        return 0
    with open(paths_path) as f:
        existing_paths = json.load(f)
    n = len(existing_paths)
    if n == 0 or existing_paths != expected_paths_order_key[:n]:
        return 0  # dataset ordering changed since the interrupted run; cannot safely resume
    existing_features = np.load(features_path)
    if existing_features.shape[0] != n:
        return 0
    return n


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building dataset")
    dataset = CBIRImageDataset(
        root=args.dataset_root,
        manifest_csv=args.manifest_csv,
        skip_corrupted=args.skip_corrupted,
    )
    all_paths = [s.image_path for s in dataset.samples]
    all_labels = [s.label for s in dataset.samples]
    all_class_names = [s.class_name for s in dataset.samples]
    logger.info("Dataset has %d samples across %d classes", len(dataset), len(dataset.class_names))

    resume_from = 0
    existing_features: Optional[np.ndarray] = None
    if not args.no_resume:
        resume_from = _try_resume(output_dir, all_paths)
        if resume_from > 0:
            existing_features = np.load(output_dir / "features.npy")
            logger.info("Resuming: %d/%d samples already extracted", resume_from, len(dataset))

    if resume_from >= len(dataset):
        logger.info("Nothing to do, extraction already complete")
        remaining_indices = []
    else:
        remaining_indices = list(range(resume_from, len(dataset)))

    logger.info("Loading extractor (checkpoint=%s, mode=%s, pooling=%s)",
                args.checkpoint, args.descriptor_mode, args.pooling)
    extractor = CoMECbirFeatureExtractor(
        model_name_or_path=args.checkpoint,
        descriptor_mode=args.descriptor_mode,
        pooling=args.pooling,
        device=args.device,
        dtype=args.dtype,
        gem_p=args.gem_p,
        low_memory=args.low_memory,
    )

    subset = torch.utils.data.Subset(dataset, remaining_indices) if remaining_indices else None
    new_feature_chunks = []
    n_images_processed = 0
    extraction_start = time.time()

    if subset is not None and len(subset) > 0:
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=cbir_collate_fn,
        )
        n_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue  # every sample in this batch was a skipped/corrupted image
            patches, _paths, _labels, _class_names = batch
            descriptors = extractor.encode_images(patches)
            new_feature_chunks.append(descriptors.cpu().numpy())
            n_images_processed += descriptors.shape[0]
            if (batch_idx + 1) % max(1, n_batches // 20 or 1) == 0 or batch_idx == n_batches - 1:
                logger.info("Processed batch %d/%d (%d images so far)", batch_idx + 1, n_batches, n_images_processed)
                # Flush progress periodically so a later crash loses at most one flush interval.
                _flush_partial(output_dir, existing_features, new_feature_chunks, all_paths,
                                all_labels, all_class_names, resume_from)

    extraction_time = time.time() - extraction_start

    final_features = _flush_partial(
        output_dir, existing_features, new_feature_chunks, all_paths, all_labels, all_class_names, resume_from
    )

    if dataset.skipped:
        logger.warning(
            "%d image(s) were skipped due to load errors and are NOT present in the saved "
            "outputs: %s", len(dataset.skipped), dataset.skipped,
        )

    adapter_dim = None
    if args.adapter_checkpoint:
        from come_cbir.retrieval_adapter import AdaptiveRetrievalProjection, apply_adapter

        adapter = AdaptiveRetrievalProjection.load(args.adapter_checkpoint)
        if final_features.shape[1] != adapter.config.input_dim:
            raise ValueError(
                f"Extracted descriptor dimension {final_features.shape[1]} does not match adapter's "
                f"expected input_dim={adapter.config.input_dim} -- wrong adapter for this descriptor mode?"
            )
        raw_dim = final_features.shape[1]
        final_features = apply_adapter(final_features, adapter)
        adapter_dim = final_features.shape[1]
        np.save(output_dir / "features.npy", final_features.astype(np.float32))
        logger.info("Applied adapter %s: %d -> %d dimensions (overwrote features.npy)",
                    args.adapter_checkpoint, raw_dim, adapter_dim)

    n_total = final_features.shape[0]
    avg_time_per_image = (
        extraction_time / n_images_processed if n_images_processed > 0 else None
    )
    metadata = extractor.metadata()
    metadata.update({
        "batch_size": args.batch_size,
        "extraction_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit_hash": get_git_commit_hash(),
        "total_extraction_time_seconds": extraction_time,
        "average_time_per_image_seconds": avg_time_per_image,
        "num_images_extracted_this_run": n_images_processed,
        "num_images_total": n_total,
        "num_images_skipped": len(dataset.skipped),
        "skipped_image_paths": dataset.skipped,
        "adapter_checkpoint": args.adapter_checkpoint,
        "adapter_output_dim": adapter_dim,
    })
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(
        "Done: %d/%d images extracted (%d new this run) in %.1fs -> %s",
        n_total, len(dataset), n_images_processed, extraction_time, output_dir,
    )


def _flush_partial(output_dir, existing_features, new_feature_chunks, all_paths, all_labels,
                    all_class_names, resume_from) -> np.ndarray:
    n_new = sum(chunk.shape[0] for chunk in new_feature_chunks)
    n_done = resume_from + n_new
    if existing_features is not None and new_feature_chunks:
        combined = np.concatenate([existing_features] + new_feature_chunks, axis=0)
    elif existing_features is not None:
        combined = existing_features
    elif new_feature_chunks:
        combined = np.concatenate(new_feature_chunks, axis=0)
    else:
        combined = np.zeros((0, 0), dtype=np.float32)

    np.save(output_dir / "features.npy", combined.astype(np.float32))
    with open(output_dir / "paths.json", "w") as f:
        json.dump(all_paths[:n_done], f)
    np.save(output_dir / "labels.npy", np.array(all_labels[:n_done], dtype=np.int64))
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(sorted(set(all_class_names[:n_done])), f)
    return combined


if __name__ == "__main__":
    main()
