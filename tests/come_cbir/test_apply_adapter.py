import numpy as np
import pytest

from come_cbir.apply_adapter import main
from come_cbir.retrieval_adapter import build_adapter


def test_apply_adapter_cli_writes_adapted_features(tmp_path):
    adapter = build_adapter("arp", input_dim=10, hidden_dim=8, output_dim=4)
    adapter_path = tmp_path / "adapter.pt"
    adapter.save(str(adapter_path))

    features = np.random.RandomState(0).randn(6, 10).astype(np.float32)
    features_path = tmp_path / "features.npy"
    np.save(features_path, features)

    output_path = tmp_path / "adapted.npy"
    main(["--features", str(features_path), "--adapter", str(adapter_path), "--output", str(output_path)])

    adapted = np.load(output_path)
    assert adapted.shape == (6, 4)
    norms = np.linalg.norm(adapted, axis=1)
    np.testing.assert_allclose(norms, np.ones(6), atol=1e-5)


def test_apply_adapter_cli_rejects_mismatched_dimension(tmp_path):
    adapter = build_adapter("arp", input_dim=10, output_dim=4)
    adapter_path = tmp_path / "adapter.pt"
    adapter.save(str(adapter_path))

    wrong_dim_features = np.random.RandomState(0).randn(6, 999).astype(np.float32)
    features_path = tmp_path / "features.npy"
    np.save(features_path, wrong_dim_features)

    with pytest.raises(ValueError, match="does not match adapter"):
        main([
            "--features", str(features_path),
            "--adapter", str(adapter_path),
            "--output", str(tmp_path / "adapted.npy"),
        ])
