import numpy as np
import pytest

from come_cbir.geometry_diagnostics import (
    compute_all_diagnostics,
    continuity_score,
    knn_overlap,
    local_rank_correlation,
    pairwise_similarity_spearman,
    trustworthiness_score,
)


@pytest.fixture
def clustered_features():
    rng = np.random.RandomState(0)
    base = np.zeros((5, 16), dtype=np.float32)
    for c in range(5):
        base[c, c] = 5.0
    return np.repeat(base, 20, axis=0) + rng.normal(scale=0.05, size=(100, 16)).astype(np.float32)


def test_identical_space_gives_perfect_scores(clustered_features):
    f = clustered_features
    assert knn_overlap(f, f, k=5) == pytest.approx(1.0)
    assert trustworthiness_score(f, f, k=5) == pytest.approx(1.0, abs=1e-6)
    assert continuity_score(f, f, k=5) == pytest.approx(1.0, abs=1e-6)
    assert pairwise_similarity_spearman(f, f, n_pairs=500, seed=0) == pytest.approx(1.0, abs=1e-6)
    assert local_rank_correlation(f, f, n_samples=30, seed=0) == pytest.approx(1.0, abs=1e-6)


def test_random_unrelated_space_gives_low_knn_overlap_and_spearman(clustered_features):
    rng = np.random.RandomState(1)
    f = clustered_features
    z_random = rng.randn(*f.shape).astype(np.float32)  # unrelated to f entirely

    overlap = knn_overlap(f, z_random, k=5)
    rho = pairwise_similarity_spearman(f, z_random, n_pairs=2000, seed=0)
    # Not necessarily exactly 0, but must be far below the identical-space case.
    assert overlap < 0.5
    assert abs(rho) < 0.3


def test_compute_all_diagnostics_returns_all_five_keys(clustered_features):
    f = clustered_features
    z = clustered_features + np.random.RandomState(0).normal(scale=0.01, size=f.shape).astype(np.float32)
    result = compute_all_diagnostics(f, z, k=5, seed=0)
    assert set(result.keys()) == {
        "knn_overlap", "trustworthiness", "continuity",
        "pairwise_similarity_spearman", "local_rank_correlation",
    }
    for value in result.values():
        assert isinstance(value, float)
        assert not np.isnan(value)
