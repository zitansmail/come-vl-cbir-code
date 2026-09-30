import numpy as np
import pytest
from PIL import Image

from come_cbir.feature_extractor import CoMECbirFeatureExtractor
from come_cbir.indexing import build_index, save_paths, load_index, load_paths
from come_cbir.query import run_query


@pytest.fixture
def tmp_index_with_paths(tmp_path):
    rng = np.random.RandomState(0)
    features = rng.randn(6, 48).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    paths = [str(tmp_path / f"img_{i}.jpg") for i in range(6)]
    for p in paths:
        Image.new("RGB", (10, 10), color=(1, 2, 3)).save(p)

    index = build_index("faiss-flat", features)
    index_path = str(tmp_path / "index.faiss")
    index.save(index_path)
    save_paths(index_path, paths)
    return index_path, paths, features


def test_run_query_excludes_query_image_when_present_in_database(mock_model, tmp_index_with_paths, tmp_path):
    index_path, paths, features = tmp_index_with_paths
    index = load_index("faiss-flat", index_path)
    index_paths = load_paths(index_path)

    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="come_fused", pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )

    # Use one of the *database* images itself as the query image.
    query_path = paths[0]
    result = run_query(query_path, extractor, index, index_paths, top_k=5)

    returned_paths = [r["path"] for r in result["results"]]
    assert query_path not in returned_paths
    assert len(result["results"]) <= 5
    assert all(result["results"][i]["rank"] == i + 1 for i in range(len(result["results"])))


def test_run_query_scores_are_sorted_descending(mock_model, tmp_index_with_paths):
    index_path, paths, _ = tmp_index_with_paths
    index = load_index("faiss-flat", index_path)
    index_paths = load_paths(index_path)
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="come_fused", pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )
    result = run_query(paths[1], extractor, index, index_paths, top_k=5)
    scores = [r["score"] for r in result["results"]]
    assert scores == sorted(scores, reverse=True)
