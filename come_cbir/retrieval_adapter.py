"""
Adaptive Retrieval Projection (ARP): a small, frozen-backbone-compatible
projection head trained with retrieval supervision (class labels only, no
captions) to turn a CoME-VL descriptor into a more retrieval-friendly,
compact embedding.

This module operates strictly *after* feature extraction and *before*
indexing -- it never touches CoME-VL itself. Every CoME-VL parameter stays
frozen; only this small head is trained. Architecture:

    LayerNorm(d) -> Linear(d, hidden) -> GELU -> Linear(hidden, output_dim) -> L2Norm

``hidden`` and ``output_dim`` are fixed widths (both default to 256),
independent of the input dimension ``d``. This is deliberate: a naive
Linear(d, d/2) first layer would cost ~6.4M parameters for come_fused's
3584-d input alone, blowing well past a ~1M-parameter budget. A fixed,
small bottleneck (as used in SimCLR/SupCon-style projection heads,
regardless of backbone width) keeps every descriptor mode -- siglip (1152),
dino (1024), concat (2176), come_fused (3584) -- under budget; worst case
(come_fused) is ~991K parameters, see ``count_parameters()``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_ADAPTER_TYPES = ("arp",)


@dataclass
class RetrievalAdapterConfig:
    """Frozen record of an adapter's architecture -- saved next to its weights."""

    adapter_type: str = "arp"
    input_dim: int = 0
    hidden_dim: int = 256
    output_dim: int = 256

    def __post_init__(self):
        if self.adapter_type not in SUPPORTED_ADAPTER_TYPES:
            raise ValueError(f"Unsupported adapter_type='{self.adapter_type}'. Choose from: {SUPPORTED_ADAPTER_TYPES}")
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")


class AdaptiveRetrievalProjection(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear -> L2Norm, all CoME-VL parameters untouched."""

    def __init__(self, config: RetrievalAdapterConfig):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.input_dim)
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "AdaptiveRetrievalProjection":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = RetrievalAdapterConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


def build_adapter(adapter_type: str, input_dim: int, hidden_dim: int = 256, output_dim: int = 256) -> AdaptiveRetrievalProjection:
    if adapter_type != "arp":
        raise ValueError(f"Unsupported adapter_type='{adapter_type}'. Only 'arp' is implemented.")
    config = RetrievalAdapterConfig(
        adapter_type=adapter_type, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )
    return AdaptiveRetrievalProjection(config)


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al., 2020), for L2-normalized embeddings.

    Every other same-label embedding in the batch is a positive (not just one
    mined pair/triplet), which is both simpler and more stable to train than
    triplet loss's hard-negative-mining requirement, and directly optimizes
    the exact notion of similarity leave-one-out class-based retrieval
    evaluates: same-class embeddings pulled together, different-class pushed
    apart. Requires at least one other same-label sample per anchor in the
    batch (see the balanced batch sampler in train_adapter.py) -- anchors
    with no positive in the batch are skipped, not treated as zero loss.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()  # numerical stability

    labels = labels.view(-1, 1)
    same_label = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(batch_size, device=device)
    positive_mask = same_label * (1.0 - self_mask)
    logit_mask = 1.0 - self_mask  # exclude self from the denominator

    exp_sim = torch.exp(similarity) * logit_mask
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    num_positives = positive_mask.sum(dim=1)
    has_positive = num_positives > 0
    if not has_positive.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[has_positive] / num_positives[has_positive]
    return -mean_log_prob_pos.mean()


def build_knn_graph(
    features: np.ndarray,
    k: int = 10,
    symmetric: bool = False,
    weight_normalization: str = "l1",
) -> Dict[str, np.ndarray]:
    """
    Build a static k-nearest-neighbour graph over a FIXED, FROZEN feature array
    (exact cosine similarity via FAISS-flat -- reuses come_cbir.indexing, no new
    search code). This is exact brute-force search: O(n^2 d) time for n points
    of dimension d (NOT O(n log n) -- FAISS-flat does not build an approximate
    tree/graph index, it computes every pairwise distance), acceptable here
    because it runs exactly once, before training, on a small-to-moderate
    dataset (thousands of points), not per training step.

    IMPORTANT (caller's responsibility): pass ONLY training-split features.
    This function has no concept of "holdout" -- whatever array you give it is
    what the graph is built from, in full. Passing held-out or validation rows
    here would leak their identity/geometry into the training signal.

    Returns a dict of parallel (N, k) arrays (row i = point i's up to-k
    neighbours, self excluded, un-found slots masked out for tiny datasets):
      neighbor_indices: int, local index into `features` (0..N-1)
      neighbor_sim:      float, the RAW cosine sim(f_i, f_j) -- used as the
                         regression TARGET in neighborhood_preservation_loss,
                         not a weight.
      neighbor_weight:   float, nonnegative, normalized PER ANCHOR ROW --
                         used to WEIGHT the loss, never as the target itself.
      valid_mask:        bool, whether this (i, slot) is a real edge (False
                         for padding when a point has fewer than k available
                         neighbours, e.g. tiny toy datasets in tests).
    """
    if weight_normalization not in ("l1", "softmax"):
        raise ValueError(f"weight_normalization must be 'l1' or 'softmax', got '{weight_normalization}'")

    from come_cbir.indexing import build_index

    n = features.shape[0]
    k = min(k, max(n - 1, 0))
    normed = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    normed = normed.astype(np.float32)

    neighbor_indices = np.zeros((n, k), dtype=np.int64)
    neighbor_sim = np.zeros((n, k), dtype=np.float32)
    valid_mask = np.zeros((n, k), dtype=bool)

    if k > 0:
        index = build_index("faiss-flat", normed)
        search_k = min(n, k + 1)  # +1 so we can drop the self-match and still keep k real neighbours
        result = index.search(normed, search_k)
        for i in range(n):
            row_idx = result.indices[i]
            row_sim = result.scores[i]
            keep = row_idx != i
            row_idx = row_idx[keep][:k]
            row_sim = row_sim[keep][:k]
            n_found = len(row_idx)
            neighbor_indices[i, :n_found] = row_idx
            neighbor_sim[i, :n_found] = row_sim
            valid_mask[i, :n_found] = True

    if symmetric and k > 0:
        import scipy.sparse as sp

        rows = np.repeat(np.arange(n), k)[valid_mask.reshape(-1)]
        cols = neighbor_indices.reshape(-1)[valid_mask.reshape(-1)]
        vals = neighbor_sim.reshape(-1)[valid_mask.reshape(-1)]
        directed = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        symmetrized = directed.maximum(directed.T)  # union of edges; average would also be reasonable
        symmetrized = symmetrized.tolil()

        neighbor_indices = np.zeros((n, k), dtype=np.int64)
        neighbor_sim = np.zeros((n, k), dtype=np.float32)
        valid_mask = np.zeros((n, k), dtype=bool)
        for i in range(n):
            row = symmetrized.getrow(i).tocoo()
            if row.nnz == 0:
                continue
            order = np.argsort(-row.data)[:k]
            cols_i = row.col[order]
            vals_i = row.data[order]
            n_found = len(cols_i)
            neighbor_indices[i, :n_found] = cols_i
            neighbor_sim[i, :n_found] = vals_i
            valid_mask[i, :n_found] = True

    raw_weight = np.clip(neighbor_sim, a_min=0.0, a_max=None) * valid_mask  # max(0, sim(f_i, f_j))
    if weight_normalization == "l1":
        row_sum = raw_weight.sum(axis=1, keepdims=True)
        neighbor_weight = np.divide(raw_weight, row_sum, out=np.zeros_like(raw_weight), where=row_sum > 0)
    else:  # softmax over the (already nonneg-clipped) raw similarities, masked
        masked = np.where(valid_mask, raw_weight, -np.inf)
        masked = np.where(np.isneginf(masked).all(axis=1, keepdims=True), 0.0, masked)  # avoid all -inf rows
        shifted = masked - np.nanmax(np.where(valid_mask, masked, -np.inf), axis=1, keepdims=True)
        exp = np.where(valid_mask, np.exp(shifted), 0.0)
        row_sum = exp.sum(axis=1, keepdims=True)
        neighbor_weight = np.divide(exp, row_sum, out=np.zeros_like(exp), where=row_sum > 0)

    return {
        "neighbor_indices": neighbor_indices,
        "neighbor_sim": neighbor_sim,
        "neighbor_weight": neighbor_weight.astype(np.float32),
        "valid_mask": valid_mask,
    }


def neighborhood_preservation_loss(
    z_anchor: torch.Tensor,
    z_neighbors: torch.Tensor,
    neighbor_sim: torch.Tensor,
    neighbor_weight: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    L_NP = (1/|E|) * sum_{(i,j) in E} w_ij * (sim(z_i, z_j) - sim(f_i, f_j))^2

    A similarity-*preservation* loss, not an attraction-only loss: it penalizes
    the adapted-space similarity for drifting away from the frozen-space
    similarity in EITHER direction (points that were dissimilar must stay
    dissimilar, not just "similar points must stay similar"). This is the
    entire mechanism by which this loss is hypothesized (not proven) to reduce
    destructive deformation of the frozen backbone's local geometry -- see the
    module docstring's framing note below.

    z_anchor:        (B, m) adapted (L2-normalized) embeddings for the batch's anchors.
    z_neighbors:      (B, k, m) adapted (L2-normalized) embeddings for each anchor's
                      precomputed k neighbours (computed through the SAME, currently-
                      training, adapter -- these are not cached from graph-build time).
    neighbor_sim:     (B, k) raw frozen-space sim(f_i, f_j), the regression TARGET.
    neighbor_weight:  (B, k) nonnegative, per-anchor-normalized weight.
    valid_mask:       (B, k) bool, real edge vs. padding.
    """
    sim_adapted = torch.einsum("bm,bkm->bk", z_anchor, z_neighbors)
    sq_diff = (sim_adapted - neighbor_sim) ** 2
    weighted = sq_diff * neighbor_weight * valid_mask.float()
    denom = valid_mask.float().sum().clamp(min=1.0)
    return weighted.sum() / denom


def three_way_class_split(
    labels: np.ndarray, val_class_fraction: float = 0.2, test_class_fraction: float = 0.2, seed: int = 0
) -> Dict[str, object]:
    """
    Class-disjoint train/validation/test split, for the unseen-class protocol
    done properly: validation classes are used ONLY to pick lambda_np/k/epochs/
    early-stopping; test classes are touched only for the final, one-time
    report. No class ever appears in more than one of the three sets.
    """
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    if n_classes < 3:
        raise ValueError("Need at least 3 classes to form disjoint train/val/test class sets")

    shuffled = unique_classes.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(n_classes * val_class_fraction)))
    n_test = max(1, int(round(n_classes * test_class_fraction)))
    n_val = min(n_val, n_classes - 2)
    n_test = min(n_test, n_classes - n_val - 1)

    val_classes = sorted(int(c) for c in shuffled[:n_val])
    test_classes = sorted(int(c) for c in shuffled[n_val:n_val + n_test])
    train_classes = sorted(int(c) for c in shuffled[n_val + n_test:])

    val_set, test_set = set(val_classes), set(test_classes)
    labels_int = labels.astype(int)
    val_idx = np.where(np.isin(labels_int, list(val_set)))[0]
    test_idx = np.where(np.isin(labels_int, list(test_set)))[0]
    train_idx = np.where(~np.isin(labels_int, list(val_set | test_set)))[0]

    return {
        "train_indices": train_idx, "val_indices": val_idx, "test_indices": test_idx,
        "train_classes": train_classes, "val_classes": val_classes, "test_classes": test_classes,
    }


def apply_adapter(features: np.ndarray, adapter: AdaptiveRetrievalProjection, batch_size: int = 4096) -> np.ndarray:
    """Apply a trained (eval-mode) adapter to a full feature array, batched, no grad."""
    adapter.eval()
    device = next(adapter.parameters()).device
    outputs = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            chunk = torch.from_numpy(features[start:start + batch_size]).float().to(device)
            outputs.append(adapter(chunk).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, adapter.config.output_dim), dtype=np.float32)


def stratified_holdout_split(
    labels: np.ndarray, holdout_fraction: float = 0.3, seed: int = 0, min_train_per_class: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-class stratified split into (train_indices, holdout_indices), so the
    adapter never trains on an image later used to evaluate retrieval.
    Guarantees at least ``min_train_per_class`` training examples per class
    (needed for supervised-contrastive positives to exist during training)
    by shrinking that class's holdout share rather than silently dropping it.
    """
    rng = np.random.RandomState(seed)
    train_idx, holdout_idx = [], []
    for label in np.unique(labels):
        class_idx = np.where(labels == label)[0]
        rng.shuffle(class_idx)
        n_holdout = int(round(len(class_idx) * holdout_fraction))
        n_holdout = min(n_holdout, max(0, len(class_idx) - min_train_per_class))
        holdout_idx.extend(class_idx[:n_holdout].tolist())
        train_idx.extend(class_idx[n_holdout:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(holdout_idx))


def class_level_holdout_split(
    labels: np.ndarray, holdout_class_fraction: float = 0.3, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Hold out entire classes, not just images within each class -- see
    stratified_holdout_split's docstring for the weaker "seen classes, unseen
    images" test this is meant to be run alongside, not instead of. Every
    image of a held-out class goes to the holdout set; the adapter never
    sees a single example, or the label itself, of any held-out class during
    training. This is what actually tests whether the learned transformation
    generalizes to genuinely new categories, rather than just memorizing a
    closed set of classes it was directly supervised on.
    """
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        raise ValueError("Need at least 2 classes to hold any out and still have something to train on")
    shuffled = unique_classes.copy()
    rng.shuffle(shuffled)
    n_holdout_classes = max(1, int(round(len(shuffled) * holdout_class_fraction)))
    n_holdout_classes = min(n_holdout_classes, len(shuffled) - 1)  # always leave >=1 class to train on
    holdout_classes = sorted(int(c) for c in shuffled[:n_holdout_classes])
    holdout_class_set = set(holdout_classes)

    is_holdout = np.array([int(l) in holdout_class_set for l in labels])
    train_idx = np.where(~is_holdout)[0]
    holdout_idx = np.where(is_holdout)[0]
    return train_idx, holdout_idx, holdout_classes


def save_split(
    path: str, train_indices: np.ndarray, holdout_indices: np.ndarray,
    mode: str = "image", holdout_classes: Optional[List[int]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_indices": train_indices.tolist(),
        "holdout_indices": holdout_indices.tolist(),
        "holdout_mode": mode,  # "image" (seen classes, unseen images) or "class" (unseen classes entirely)
    }
    if holdout_classes is not None:
        payload["holdout_classes"] = holdout_classes
    with open(path, "w") as f:
        json.dump(payload, f)


def load_split(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    return np.array(data["train_indices"]), np.array(data["holdout_indices"])
