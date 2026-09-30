import numpy as np
import pytest

from come_cbir.indexing import build_index, load_index, load_paths, save_paths


def _make_clustered_features(n_per_cluster=5, n_clusters=4, dim=16, seed=0):
    """Well-separated one-hot-ish clusters so nearest-neighbor identity is unambiguous."""
    rng = np.random.RandomState(seed)
    vectors = []
    labels = []
    for c in range(n_clusters):
        base = np.zeros(dim, dtype=np.float32)
        base[c] = 5.0
        for _ in range(n_per_cluster):
            noisy = base + rng.normal(scale=0.01, size=dim).astype(np.float32)
            noisy = noisy / np.linalg.norm(noisy)
            vectors.append(noisy)
            labels.append(c)
    return np.stack(vectors).astype(np.float32), np.array(labels)


@pytest.mark.parametrize("backend", ["faiss-flat", "faiss-hnsw", "faiss-ivf", "annoy"])
def test_build_and_search_recovers_same_cluster(backend):
    features, labels = _make_clustered_features()
    index = build_index(backend, features, nlist=4) if backend == "faiss-ivf" else build_index(backend, features)

    result = index.search(features, top_k=5)
    # For each query, its own cluster-mates should dominate the top-5 (excluding rank-0 self match).
    for qi in range(features.shape[0]):
        neighbor_labels = labels[result.indices[qi][1:5]]
        same_cluster_fraction = np.mean(neighbor_labels == labels[qi])
        assert same_cluster_fraction >= 0.5, f"backend={backend} failed for query {qi}"


@pytest.mark.parametrize("backend", ["faiss-flat", "faiss-hnsw", "annoy"])
def test_save_and_load_roundtrip(tmp_path, backend):
    features, labels = _make_clustered_features()
    index = build_index(backend, features)
    index_path = str(tmp_path / f"index_{backend.replace('-', '_')}.idx")
    index.save(index_path)
    save_paths(index_path, [f"img_{i}.jpg" for i in range(features.shape[0])])

    reloaded = load_index(backend, index_path)
    paths = load_paths(index_path)
    assert len(paths) == features.shape[0]

    result = reloaded.search(features[:1], top_k=3)
    assert result.indices.shape == (1, 3)


def test_unsupported_backend_raises():
    with pytest.raises(ValueError, match="Unsupported backend"):
        build_index("not-a-backend", np.zeros((2, 4), dtype=np.float32))
