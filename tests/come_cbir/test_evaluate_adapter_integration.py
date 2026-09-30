import json

import numpy as np

from come_cbir.evaluate import main as evaluate_main
from come_cbir.retrieval_adapter import build_adapter, save_split


def _write_toy_dataset(tmp_path, n_per_class=10, n_classes=3, dim=12):
    rng = np.random.RandomState(0)
    base = np.zeros((n_classes, dim), dtype=np.float32)
    for c in range(n_classes):
        base[c, c % dim] = 5.0
    features = np.repeat(base, n_per_class, axis=0) + rng.normal(scale=0.05, size=(n_per_class * n_classes, dim)).astype(np.float32)
    labels = np.repeat(np.arange(n_classes), n_per_class)
    paths = [f"class{l}_img{i}.jpg" for i, l in enumerate(labels)]

    features_path = tmp_path / "features.npy"
    labels_path = tmp_path / "labels.npy"
    paths_path = tmp_path / "paths.json"
    np.save(features_path, features)
    np.save(labels_path, labels)
    with open(paths_path, "w") as f:
        json.dump(paths, f)
    return features_path, labels_path, paths_path, features, labels


def test_evaluate_with_adapter_checkpoint_changes_descriptor_dim(tmp_path):
    features_path, labels_path, paths_path, features, _ = _write_toy_dataset(tmp_path)

    adapter = build_adapter("arp", input_dim=features.shape[1], hidden_dim=8, output_dim=5)
    adapter_path = tmp_path / "adapter.pt"
    adapter.save(str(adapter_path))

    output_dir = tmp_path / "eval_adapted"
    evaluate_main([
        "--features", str(features_path), "--labels", str(labels_path), "--paths", str(paths_path),
        "--top-k", "5", "--adapter-checkpoint", str(adapter_path), "--output-dir", str(output_dir),
    ])

    with open(output_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert metrics["descriptor_dim"] == 5  # adapter's output_dim, not the original 12


def test_evaluate_without_adapter_keeps_original_dim(tmp_path):
    features_path, labels_path, paths_path, features, _ = _write_toy_dataset(tmp_path)

    output_dir = tmp_path / "eval_original"
    evaluate_main([
        "--features", str(features_path), "--labels", str(labels_path), "--paths", str(paths_path),
        "--top-k", "5", "--output-dir", str(output_dir),
    ])

    with open(output_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert metrics["descriptor_dim"] == features.shape[1]


def test_evaluate_with_holdout_indices_restricts_database_size(tmp_path):
    features_path, labels_path, paths_path, features, labels = _write_toy_dataset(tmp_path, n_per_class=10, n_classes=3)

    train_idx = np.arange(len(labels))[:20]
    holdout_idx = np.arange(len(labels))[20:]
    split_path = tmp_path / "split.json"
    save_split(str(split_path), train_idx, holdout_idx)

    output_dir = tmp_path / "eval_holdout"
    evaluate_main([
        "--features", str(features_path), "--labels", str(labels_path), "--paths", str(paths_path),
        "--top-k", "5", "--holdout-indices", str(split_path), "--output-dir", str(output_dir),
    ])

    with open(output_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert metrics["num_database_images"] == len(holdout_idx)

    summary = (output_dir / "summary.md").read_text()
    assert str(split_path) in summary


def test_evaluate_with_both_holdout_and_adapter(tmp_path):
    features_path, labels_path, paths_path, features, labels = _write_toy_dataset(tmp_path, n_per_class=10, n_classes=3)

    train_idx = np.arange(len(labels))[:20]
    holdout_idx = np.arange(len(labels))[20:]
    split_path = tmp_path / "split.json"
    save_split(str(split_path), train_idx, holdout_idx)

    adapter = build_adapter("arp", input_dim=features.shape[1], hidden_dim=8, output_dim=6)
    adapter_path = tmp_path / "adapter.pt"
    adapter.save(str(adapter_path))

    output_dir = tmp_path / "eval_both"
    evaluate_main([
        "--features", str(features_path), "--labels", str(labels_path), "--paths", str(paths_path),
        "--top-k", "5", "--holdout-indices", str(split_path),
        "--adapter-checkpoint", str(adapter_path), "--output-dir", str(output_dir),
    ])

    with open(output_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert metrics["num_database_images"] == len(holdout_idx)
    assert metrics["descriptor_dim"] == 6
