import numpy as np
import pytest

from come_cbir.dimensionality import PCAReducer, fit_and_apply_pca


@pytest.fixture
def random_features():
    rng = np.random.RandomState(0)
    return rng.randn(200, 64).astype(np.float32)


def test_pca_fit_transform_shape_and_norm(random_features):
    reducer = PCAReducer(n_components=16).fit(random_features)
    transformed = reducer.transform(random_features)
    assert transformed.shape == (200, 16)
    norms = np.linalg.norm(transformed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_pca_transform_without_l2_normalize(random_features):
    reducer = PCAReducer(n_components=8).fit(random_features)
    transformed = reducer.transform(random_features, l2_normalize=False)
    norms = np.linalg.norm(transformed, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-4)


def test_pca_rejects_too_many_components(random_features):
    with pytest.raises(ValueError, match="n_components"):
        PCAReducer(n_components=1000).fit(random_features)


def test_pca_save_and_load_roundtrip(tmp_path, random_features):
    reducer = PCAReducer(n_components=10).fit(random_features)
    original = reducer.transform(random_features)

    model_path = tmp_path / "pca.pkl"
    reducer.save(str(model_path))

    reloaded = PCAReducer.load(str(model_path))
    reloaded_transformed = reloaded.transform(random_features)

    assert np.allclose(original, reloaded_transformed, atol=1e-5)


def test_transform_before_fit_raises():
    reducer = PCAReducer(n_components=4)
    with pytest.raises(RuntimeError, match="before fit"):
        reducer.transform(np.random.randn(5, 10).astype(np.float32))


def test_incremental_pca(random_features):
    reducer = PCAReducer(n_components=8, incremental=True, batch_size=32).fit(random_features)
    transformed = reducer.transform(random_features)
    assert transformed.shape == (200, 8)


def test_fit_and_apply_pca_convenience(tmp_path, random_features):
    query_features = random_features[:20]
    transformed = fit_and_apply_pca(
        train_features=random_features, apply_features=query_features, n_components=12,
        model_out_path=str(tmp_path / "pca.pkl"),
    )
    assert transformed.shape == (20, 12)
    assert (tmp_path / "pca.pkl").exists()
