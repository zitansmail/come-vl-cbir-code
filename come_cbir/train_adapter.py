"""
CLI: train an Adaptive Retrieval Projection (ARP) adapter on top of
already-extracted, frozen CoME-VL descriptors.

    python -m come_cbir.train_adapter \\
        --features outputs/siglip/features.npy \\
        --labels outputs/siglip/labels.npy \\
        --output-dir outputs/siglip/adapter \\
        --holdout-fraction 0.3 \\
        --epochs 50

Trains only on a per-class stratified train split (see
come_cbir.retrieval_adapter.stratified_holdout_split); the held-out indices
are written to <output-dir>/split.json. To get a leakage-free before/after
comparison, evaluate.py must be pointed at --holdout-indices split.json for
*both* the original-descriptor run and the adapted-descriptor run -- see
docs/retrieval_adapter.md.

This operates entirely on cached feature vectors (no images, no CoME-VL
forward pass, no GPU required for the adapter itself, though the features
you point it at were presumably extracted on one) -- training a ~1M
parameter head over a few thousand 1152-3584-d vectors is seconds, not
hours, even on CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from come_cbir.retrieval_adapter import (
    build_adapter,
    build_knn_graph,
    class_level_holdout_split,
    neighborhood_preservation_loss,
    save_split,
    stratified_holdout_split,
    supervised_contrastive_loss,
)
from come_cbir.utils import set_global_seed, setup_logging


class BalancedBatchSampler:
    """Yields batches of P classes x K samples/class, drawn from a fixed index pool.

    SupCon needs at least one other same-label sample per anchor in-batch to
    have anything to pull together; uniform random sampling would frequently
    produce batches with singleton classes and nothing to learn from, so
    batches are built class-first instead.
    """

    def __init__(self, indices: np.ndarray, labels: np.ndarray, classes_per_batch: int, samples_per_class: int, seed: int = 0):
        self.samples_per_class = samples_per_class
        self.rng = np.random.RandomState(seed)
        self.by_class = {}
        for idx in indices:
            self.by_class.setdefault(int(labels[idx]), []).append(int(idx))
        # Only classes with enough training samples to fill a batch slot are usable.
        self.usable_classes = [c for c, idxs in self.by_class.items() if len(idxs) >= 2]
        if not self.usable_classes:
            raise ValueError("No class has >=2 training samples -- cannot form any contrastive batch")
        self.classes_per_batch = min(classes_per_batch, len(self.usable_classes))

    def sample_batch(self) -> np.ndarray:
        chosen_classes = self.rng.choice(self.usable_classes, size=self.classes_per_batch, replace=False)
        batch = []
        for c in chosen_classes:
            pool = self.by_class[c]
            k = min(self.samples_per_class, len(pool))
            batch.extend(self.rng.choice(pool, size=k, replace=False).tolist())
        return np.array(batch)


def train_adapter(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    adapter_type: str = "arp",
    hidden_dim: int = 256,
    output_dim: int = 256,
    epochs: int = 50,
    steps_per_epoch: int = 50,
    classes_per_batch: int = 8,
    samples_per_class: int = 4,
    lr: float = 1e-3,
    temperature: float = 0.07,
    lambda_np: float = 0.0,
    neighbors_k: int = 10,
    symmetric_graph: bool = False,
    np_weight_normalization: str = "l1",
    device: str = "cpu",
    seed: int = 0,
    logger=None,
):
    """
    lambda_np=0.0 (the default) reproduces plain ARP's SupCon-only training
    exactly -- the kNN graph is never built and the extra forward passes for
    neighbours never happen, not merely "weighted to zero after computing
    them," so there is no floating-point drift introduced by the presence of
    this code path when it is unused.

    When lambda_np > 0 (ARP-NP), the kNN graph is built ONCE, here, from
    `features[train_indices]` only -- see build_knn_graph's docstring for why
    passing anything beyond the caller's own train split would leak holdout
    geometry into training.
    """
    set_global_seed(seed)
    torch_device = torch.device(device)
    adapter = build_adapter(adapter_type, input_dim=features.shape[1], hidden_dim=hidden_dim, output_dim=output_dim)
    adapter.to(torch_device)
    n_params = adapter.count_parameters()
    if logger:
        logger.info("Adapter '%s': input_dim=%d, hidden_dim=%d, output_dim=%d, %d trainable parameters",
                    adapter_type, features.shape[1], hidden_dim, output_dim, n_params)

    sampler = BalancedBatchSampler(train_indices, labels, classes_per_batch, samples_per_class, seed=seed)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    features_t = torch.from_numpy(features).float()
    labels_t = torch.from_numpy(labels).long()

    knn_graph = None
    global_to_local = None
    if lambda_np > 0:
        train_features = features[train_indices]
        knn_graph = build_knn_graph(
            train_features, k=neighbors_k, symmetric=symmetric_graph, weight_normalization=np_weight_normalization
        )
        global_to_local = {int(g): i for i, g in enumerate(train_indices)}
        if logger:
            logger.info(
                "Built train-only kNN graph for ARP-NP: %d points, k=%d, symmetric=%s, weight_normalization=%s",
                len(train_indices), neighbors_k, symmetric_graph, np_weight_normalization,
            )

    loss_history, supcon_history, np_history = [], [], []
    adapter.train()
    for epoch in range(epochs):
        epoch_losses, epoch_supcon, epoch_np = [], [], []
        for _ in range(steps_per_epoch):
            batch_idx = sampler.sample_batch()  # global indices into `features`
            batch_labels = labels_t[batch_idx].to(torch_device)

            if lambda_np > 0:
                local_idx = np.array([global_to_local[int(g)] for g in batch_idx])
                neighbor_local = knn_graph["neighbor_indices"][local_idx]  # (B, k) local-to-train_indices
                neighbor_global = train_indices[neighbor_local]  # (B, k) global indices
                neighbor_sim = torch.from_numpy(knn_graph["neighbor_sim"][local_idx]).float().to(torch_device)
                neighbor_weight = torch.from_numpy(knn_graph["neighbor_weight"][local_idx]).float().to(torch_device)
                valid_mask = torch.from_numpy(knn_graph["valid_mask"][local_idx]).to(torch_device)

                # One forward pass over the union of anchors + their neighbours (deduplicated),
                # so both loss terms share the same computational graph and gradient path.
                unique_global, inverse = np.unique(
                    np.concatenate([batch_idx, neighbor_global.reshape(-1)]), return_inverse=True
                )
                all_features = features_t[unique_global].to(torch_device)
                all_embeddings = adapter(all_features)

                n_anchor = len(batch_idx)
                embeddings = all_embeddings[inverse[:n_anchor]]
                z_neighbors = all_embeddings[inverse[n_anchor:].reshape(neighbor_global.shape)]

                supcon = supervised_contrastive_loss(embeddings, batch_labels, temperature=temperature)
                np_loss = neighborhood_preservation_loss(embeddings, z_neighbors, neighbor_sim, neighbor_weight, valid_mask)
                loss = supcon + lambda_np * np_loss
            else:
                batch_features = features_t[batch_idx].to(torch_device)
                embeddings = adapter(batch_features)
                supcon = supervised_contrastive_loss(embeddings, batch_labels, temperature=temperature)
                np_loss = torch.tensor(0.0)
                loss = supcon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            epoch_supcon.append(float(supcon.item()))
            epoch_np.append(float(np_loss.item()))

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)
        supcon_history.append(float(np.mean(epoch_supcon)))
        np_history.append(float(np.mean(epoch_np)))
        if logger and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            if lambda_np > 0:
                logger.info("epoch %d/%d: total=%.4f (supcon=%.4f, np=%.4f)",
                            epoch + 1, epochs, mean_loss, supcon_history[-1], np_history[-1])
            else:
                logger.info("epoch %d/%d: mean SupCon loss = %.4f", epoch + 1, epochs, mean_loss)

    adapter.eval()
    extra = {"supcon_loss_history": supcon_history, "np_loss_history": np_history}
    return adapter, loss_history, n_params, extra


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--adapter-type", type=str, default="arp", choices=["arp"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--holdout-mode", type=str, default="image", choices=["image", "class"],
                         help="'image': hold out a fraction of images within every class (tests "
                              "generalization to unseen images of KNOWN classes -- the weaker check). "
                              "'class': hold out entire classes (tests generalization to genuinely "
                              "UNSEEN classes the adapter never trained on at all -- the stronger, "
                              "more meaningful check; see docs/retrieval_adapter.md).")
    parser.add_argument("--holdout-fraction", type=float, default=0.3,
                         help="--holdout-mode=image: per-class fraction of images held out. "
                              "--holdout-mode=class: fraction of classes held out entirely.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--classes-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-np", type=float, default=0.0,
                         help="Weight on the neighbourhood-preservation term (ARP-NP). 0.0 (default) "
                              "reproduces plain ARP exactly -- the kNN graph is never built when this is 0.")
    parser.add_argument("--neighbors-k", type=int, default=10,
                         help="Number of neighbours per anchor in the train-only kNN graph (ARP-NP only).")
    parser.add_argument("--symmetric-graph", action="store_true",
                         help="Symmetrize the kNN graph (union of directed edges) before training (ARP-NP only).")
    parser.add_argument("--np-weight-normalization", type=str, default="l1", choices=["l1", "softmax"],
                         help="How max(0, sim) weights are normalized within each anchor's neighbourhood.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()

    features = np.load(args.features).astype(np.float32)
    labels = np.load(args.labels)

    holdout_classes = None
    if args.holdout_mode == "class":
        train_indices, holdout_indices, holdout_classes = class_level_holdout_split(
            labels, holdout_class_fraction=args.holdout_fraction, seed=args.seed
        )
        logger.info(
            "Class-level split: %d classes held out entirely (%s), %d train images / %d holdout images",
            len(holdout_classes), holdout_classes, len(train_indices), len(holdout_indices),
        )
    else:
        train_indices, holdout_indices = stratified_holdout_split(
            labels, holdout_fraction=args.holdout_fraction, seed=args.seed
        )
        logger.info(
            "Image-level stratified split: %d train / %d holdout (of %d total, holdout_fraction=%.2f)",
            len(train_indices), len(holdout_indices), len(labels), args.holdout_fraction,
        )

    adapter, loss_history, n_params, extra = train_adapter(
        features, labels, train_indices,
        adapter_type=args.adapter_type, hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        classes_per_batch=args.classes_per_batch, samples_per_class=args.samples_per_class,
        lr=args.lr, temperature=args.temperature,
        lambda_np=args.lambda_np, neighbors_k=args.neighbors_k,
        symmetric_graph=args.symmetric_graph, np_weight_normalization=args.np_weight_normalization,
        device=args.device, seed=args.seed, logger=logger,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter.save(str(output_dir / "adapter.pt"))
    save_split(str(output_dir / "split.json"), train_indices, holdout_indices,
               mode=args.holdout_mode, holdout_classes=holdout_classes)
    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "loss_history": loss_history,
            "supcon_loss_history": extra["supcon_loss_history"],
            "np_loss_history": extra["np_loss_history"],
            "final_loss": loss_history[-1] if loss_history else None,
            "trainable_parameters": n_params,
            "args": vars(args),
        }, f, indent=2)

    logger.info(
        "Wrote adapter.pt (%d params), split.json, training_log.json to %s",
        n_params, output_dir,
    )


if __name__ == "__main__":
    main()
