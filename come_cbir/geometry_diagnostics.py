"""
Geometric diagnostics for whether an adapter (ARP, ARP-NP, or otherwise)
preserves the frozen backbone's own local structure, independent of any
downstream retrieval metric. These do NOT depend on class labels -- they
compare the frozen descriptor space `f` directly against the adapted space
`z`, which is what makes them suited to checking the *mechanism* (neighbourhood
preservation) rather than only its downstream symptom (retrieval accuracy).

All functions are diagnostics only -- none of them are used inside any loss
or training loop. They exist to answer "did the geometry actually change the
way we hypothesized," not to optimize anything.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness as _sklearn_trustworthiness


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn_overlap(f: np.ndarray, z: np.ndarray, k: int = 10) -> float:
    """
    Fraction of each point's top-k frozen-space neighbours that remain among
    its top-k adapted-space neighbours, averaged over all points. 1.0 = every
    point's neighbourhood is completely unchanged; 0.0 = completely disjoint.
    """
    from come_cbir.retrieval_adapter import build_knn_graph

    graph_f = build_knn_graph(f, k=k)
    graph_z = build_knn_graph(z, k=k)
    n = f.shape[0]
    overlaps = []
    for i in range(n):
        neighbors_f = set(graph_f["neighbor_indices"][i][graph_f["valid_mask"][i]].tolist())
        neighbors_z = set(graph_z["neighbor_indices"][i][graph_z["valid_mask"][i]].tolist())
        denom = max(len(neighbors_f), 1)
        overlaps.append(len(neighbors_f & neighbors_z) / denom)
    return float(np.mean(overlaps)) if overlaps else float("nan")


def trustworthiness_score(f: np.ndarray, z: np.ndarray, k: int = 10) -> float:
    """
    Standard manifold-learning trustworthiness (Venna & Kaski): penalizes
    points that become "false neighbours" in the adapted space (close in `z`
    but were not close in `f`). 1.0 = no false neighbours introduced.
    """
    return float(_sklearn_trustworthiness(f, z, n_neighbors=k))


def continuity_score(f: np.ndarray, z: np.ndarray, k: int = 10) -> float:
    """
    The reverse of trustworthiness: penalizes points that were neighbours in
    the frozen space but are no longer neighbours after adaptation ("missing
    neighbours" / intrusions from the other direction). Computed as
    trustworthiness with the two spaces swapped, per Venna & Kaski's original
    formulation -- not a separate implementation, the same estimator.
    """
    return float(_sklearn_trustworthiness(z, f, n_neighbors=k))


def pairwise_similarity_spearman(f: np.ndarray, z: np.ndarray, n_pairs: int = 5000, seed: int = 0) -> float:
    """
    Spearman rank correlation between frozen-space and adapted-space cosine
    similarity, over a random sample of point pairs (sampling keeps this
    tractable at Corel-10K scale rather than requiring all ~5*10^7 pairs).
    1.0 = adapted similarities are a monotonic function of frozen similarities
    (a strong global preservation signal); 0 = no relationship; negative =
    similarity ordering was inverted.
    """
    rng = np.random.RandomState(seed)
    n = f.shape[0]
    i_idx = rng.randint(0, n, size=n_pairs)
    j_idx = rng.randint(0, n, size=n_pairs)
    keep = i_idx != j_idx
    i_idx, j_idx = i_idx[keep], j_idx[keep]

    f_n, z_n = _l2_normalize_rows(f), _l2_normalize_rows(z)
    sim_f = np.sum(f_n[i_idx] * f_n[j_idx], axis=1)
    sim_z = np.sum(z_n[i_idx] * z_n[j_idx], axis=1)
    rho, _ = spearmanr(sim_f, sim_z)
    return float(rho)


def local_rank_correlation(f: np.ndarray, z: np.ndarray, n_samples: int = 500, seed: int = 0) -> float:
    """
    Per-anchor rank correlation: for a random sample of anchor points, compare
    that single anchor's full similarity ranking of every other point in the
    frozen space against the adapted space, then average the per-anchor
    Spearman correlations. This is the "local" complement to
    pairwise_similarity_spearman's global, pooled correlation -- it answers
    "does each individual point's own neighbourhood ordering survive
    adaptation," not just "is there a relationship across all pairs pooled
    together." Sampled (not exhaustive over every point) to stay tractable at
    Corel-10K scale, per "if practical."
    """
    rng = np.random.RandomState(seed)
    n = f.shape[0]
    n_samples = min(n_samples, n)
    anchors = rng.choice(n, size=n_samples, replace=False)

    f_n, z_n = _l2_normalize_rows(f), _l2_normalize_rows(z)
    correlations = []
    for i in anchors:
        sim_f = f_n @ f_n[i]
        sim_z = z_n @ z_n[i]
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rho, _ = spearmanr(sim_f[mask], sim_z[mask])
        if not np.isnan(rho):
            correlations.append(rho)
    return float(np.mean(correlations)) if correlations else float("nan")


def compute_all_diagnostics(f: np.ndarray, z: np.ndarray, k: int = 10, seed: int = 0) -> Dict[str, float]:
    """Bundle of all five diagnostics, for one-line logging in an experiment script."""
    return {
        "knn_overlap": knn_overlap(f, z, k=k),
        "trustworthiness": trustworthiness_score(f, z, k=k),
        "continuity": continuity_score(f, z, k=k),
        "pairwise_similarity_spearman": pairwise_similarity_spearman(f, z, seed=seed),
        "local_rank_correlation": local_rank_correlation(f, z, seed=seed),
    }
