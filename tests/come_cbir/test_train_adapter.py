import json

import numpy as np
import pytest
import torch

import come_cbir.train_adapter as train_adapter_module
from come_cbir.retrieval_adapter import build_adapter
from come_cbir.train_adapter import BalancedBatchSampler, main, train_adapter


@pytest.fixture
def separable_dataset():
    # 4 well-separated classes, 20 samples each, in a 16-d space.
    rng = np.random.RandomState(0)
    base = np.zeros((4, 16), dtype=np.float32)
    for c in range(4):
        base[c, c] = 5.0
    features = np.repeat(base, 20, axis=0) + rng.normal(scale=0.05, size=(80, 16)).astype(np.float32)
    labels = np.repeat(np.arange(4), 20)
    return features, labels


class TestBalancedBatchSampler:
    def test_batch_contains_only_requested_classes_per_batch(self, separable_dataset):
        _features, labels = separable_dataset
        indices = np.arange(len(labels))
        sampler = BalancedBatchSampler(indices, labels, classes_per_batch=2, samples_per_class=3, seed=0)
        batch = sampler.sample_batch()
        batch_labels = labels[batch]
        assert len(np.unique(batch_labels)) == 2
        assert len(batch) == 6  # 2 classes x 3 samples

    def test_raises_when_no_class_has_two_samples(self):
        labels = np.array([0, 1, 2, 3])  # every class is a singleton
        indices = np.arange(4)
        with pytest.raises(ValueError, match="No class has >=2"):
            BalancedBatchSampler(indices, labels, classes_per_batch=2, samples_per_class=2)

    def test_only_samples_from_given_index_pool(self, separable_dataset):
        _features, labels = separable_dataset
        # Restrict to only the first 40 samples (classes 0 and 1).
        indices = np.arange(40)
        sampler = BalancedBatchSampler(indices, labels, classes_per_batch=2, samples_per_class=4, seed=0)
        for _ in range(5):
            batch = sampler.sample_batch()
            assert all(idx < 40 for idx in batch)


class TestTrainAdapter:
    def test_reduces_loss_over_epochs(self, separable_dataset):
        features, labels = separable_dataset
        train_indices = np.arange(len(labels))  # no holdout needed for this unit test
        adapter, loss_history, n_params, extra = train_adapter(
            features, labels, train_indices,
            hidden_dim=16, output_dim=8, epochs=10, steps_per_epoch=5,
            classes_per_batch=4, samples_per_class=4, lr=1e-2, seed=0,
        )
        assert len(loss_history) == 10
        assert len(extra["supcon_loss_history"]) == 10
        assert len(extra["np_loss_history"]) == 10
        assert all(v == 0.0 for v in extra["np_loss_history"])  # lambda_np=0 by default -- never computed
        assert loss_history[-1] < loss_history[0]
        assert n_params > 0
        assert n_params < 1_000_000

    def test_adapter_only_sees_train_indices(self, separable_dataset):
        features, labels = separable_dataset
        # Only expose class 0 and 1 samples as "train" -- training must not crash
        # or silently pull from the excluded classes 2/3.
        train_indices = np.where(labels < 2)[0]
        adapter, loss_history, _, _ = train_adapter(
            features, labels, train_indices,
            hidden_dim=8, output_dim=4, epochs=3, steps_per_epoch=3,
            classes_per_batch=2, samples_per_class=4, seed=0,
        )
        assert len(loss_history) == 3


def test_train_adapter_cli_end_to_end(tmp_path):
    rng = np.random.RandomState(0)
    base = np.zeros((3, 12), dtype=np.float32)
    for c in range(3):
        base[c, c] = 4.0
    features = np.repeat(base, 15, axis=0) + rng.normal(scale=0.05, size=(45, 12)).astype(np.float32)
    labels = np.repeat(np.arange(3), 15)

    features_path = tmp_path / "features.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(features_path, features)
    np.save(labels_path, labels)

    output_dir = tmp_path / "adapter_out"
    main([
        "--features", str(features_path),
        "--labels", str(labels_path),
        "--output-dir", str(output_dir),
        "--holdout-fraction", "0.3",
        "--epochs", "3",
        "--steps-per-epoch", "3",
        "--classes-per-batch", "3",
        "--samples-per-class", "3",
        "--hidden-dim", "8",
        "--output-dim", "4",
    ])

    assert (output_dir / "adapter.pt").exists()
    assert (output_dir / "split.json").exists()
    assert (output_dir / "training_log.json").exists()

    with open(output_dir / "split.json") as f:
        split = json.load(f)
    train_idx = set(split["train_indices"])
    holdout_idx = set(split["holdout_indices"])
    assert train_idx.isdisjoint(holdout_idx)
    assert len(train_idx) + len(holdout_idx) == 45

    with open(output_dir / "training_log.json") as f:
        log = json.load(f)
    assert log["trainable_parameters"] < 1_000_000
    assert len(log["loss_history"]) == 3


def test_train_adapter_cli_class_level_holdout(tmp_path):
    # 6 classes so a 0.3 fraction holds out at least 1 class outright, with
    # enough classes remaining (>=classes_per_batch) to form contrastive batches.
    rng = np.random.RandomState(0)
    base = np.zeros((6, 12), dtype=np.float32)
    for c in range(6):
        base[c, c % 12] = 4.0
    features = np.repeat(base, 15, axis=0) + rng.normal(scale=0.05, size=(90, 12)).astype(np.float32)
    labels = np.repeat(np.arange(6), 15)

    features_path = tmp_path / "features.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(features_path, features)
    np.save(labels_path, labels)

    output_dir = tmp_path / "adapter_out_class"
    main([
        "--features", str(features_path),
        "--labels", str(labels_path),
        "--output-dir", str(output_dir),
        "--holdout-mode", "class",
        "--holdout-fraction", "0.3",
        "--epochs", "2",
        "--steps-per-epoch", "2",
        "--classes-per-batch", "4",
        "--samples-per-class", "3",
        "--hidden-dim", "8",
        "--output-dim", "4",
    ])

    with open(output_dir / "split.json") as f:
        split = json.load(f)
    assert split["holdout_mode"] == "class"
    assert len(split["holdout_classes"]) == 2  # round(6 * 0.3)

    train_idx = set(split["train_indices"])
    holdout_idx = set(split["holdout_indices"])
    holdout_classes = set(split["holdout_classes"])
    assert train_idx.isdisjoint(holdout_idx)
    # No training index may belong to a held-out class.
    assert not any(int(labels[i]) in holdout_classes for i in train_idx)
    # Every holdout index must belong to a held-out class.
    assert all(int(labels[i]) in holdout_classes for i in holdout_idx)


# ----------------------------------------------------------------------
# ARP-NP: required correctness properties, at the train_adapter() level
# ----------------------------------------------------------------------


class TestArpNpTrainAdapter:
    def test_lambda_np_zero_reproduces_plain_arp(self, separable_dataset):
        """lambda_np=0.0 must take the exact same code path as plain ARP: no
        graph is built, np_loss_history is all zeros, and total loss equals
        the SupCon loss exactly (no numerical drift from the disabled term)."""
        features, labels = separable_dataset
        train_indices = np.arange(len(labels))
        adapter, loss_history, n_params, extra = train_adapter(
            features, labels, train_indices,
            hidden_dim=16, output_dim=8, epochs=5, steps_per_epoch=5,
            classes_per_batch=4, samples_per_class=4, lr=1e-2, seed=0,
            lambda_np=0.0,
        )
        assert all(v == 0.0 for v in extra["np_loss_history"])
        np.testing.assert_allclose(loss_history, extra["supcon_loss_history"])

    def test_lambda_np_positive_builds_a_graph_and_changes_training(self, separable_dataset):
        features, labels = separable_dataset
        train_indices = np.arange(len(labels))
        adapter_arp, loss_arp, _, extra_arp = train_adapter(
            features, labels, train_indices,
            hidden_dim=16, output_dim=8, epochs=5, steps_per_epoch=5,
            classes_per_batch=4, samples_per_class=4, lr=1e-2, seed=0, lambda_np=0.0,
        )
        adapter_np, loss_np, _, extra_np = train_adapter(
            features, labels, train_indices,
            hidden_dim=16, output_dim=8, epochs=5, steps_per_epoch=5,
            classes_per_batch=4, samples_per_class=4, lr=1e-2, seed=0, lambda_np=1.0, neighbors_k=5,
        )
        assert any(v > 0.0 for v in extra_np["np_loss_history"])
        # Same seed, same data, different objective -> different learned weights.
        w_arp = adapter_arp.fc1.weight.detach().numpy()
        w_np = adapter_np.fc1.weight.detach().numpy()
        assert not np.allclose(w_arp, w_np)

    def test_gradients_reach_the_adapter_with_lambda_np(self, separable_dataset):
        features, labels = separable_dataset
        train_indices = np.arange(len(labels))
        adapter, _, _, extra = train_adapter(
            features, labels, train_indices,
            hidden_dim=16, output_dim=8, epochs=1, steps_per_epoch=1,
            classes_per_batch=4, samples_per_class=4, lr=1e-1, seed=0,
            lambda_np=1.0, neighbors_k=5,
        )
        # A single optimizer step at a high learning rate must have moved the weights
        # away from their initial (seeded, therefore known) values.
        torch.manual_seed(0)
        fresh = build_adapter("arp", input_dim=features.shape[1], hidden_dim=16, output_dim=8)
        assert not np.allclose(
            adapter.fc1.weight.detach().numpy(), fresh.fc1.weight.detach().numpy()
        )

    def test_knn_graph_is_built_only_from_train_indices(self, separable_dataset, monkeypatch):
        """Structural leakage check: patch build_knn_graph to record the shape
        of whatever array it was actually called with, and confirm it matches
        len(train_indices), not the full dataset size."""
        features, labels = separable_dataset
        # Use only classes 0/1 as "train" (40 of the 80 total samples) -- classes 2/3
        # play the role of held-out data that must never reach the graph builder.
        train_indices = np.where(labels < 2)[0]
        assert len(train_indices) < len(labels)

        seen_shapes = []
        real_build_knn_graph = train_adapter_module.build_knn_graph

        def spy(features_arg, **kwargs):
            seen_shapes.append(features_arg.shape[0])
            return real_build_knn_graph(features_arg, **kwargs)

        monkeypatch.setattr(train_adapter_module, "build_knn_graph", spy)

        train_adapter(
            features, labels, train_indices,
            hidden_dim=8, output_dim=4, epochs=1, steps_per_epoch=1,
            classes_per_batch=2, samples_per_class=4, seed=0,
            lambda_np=1.0, neighbors_k=5,
        )
        assert seen_shapes == [len(train_indices)]

    def test_saved_adapter_evaluates_correctly_via_evaluate_cli(self, separable_dataset, tmp_path):
        """An adapter trained with lambda_np>0 must be indistinguishable, from
        evaluate.py's point of view, from a plain-ARP checkpoint: same config
        shape, same save/load path, since the architecture never changed."""
        features, labels = separable_dataset
        train_indices = np.arange(len(labels))
        adapter, _, _, _ = train_adapter(
            features, labels, train_indices,
            hidden_dim=8, output_dim=4, epochs=2, steps_per_epoch=2,
            classes_per_batch=4, samples_per_class=4, seed=0, lambda_np=0.5, neighbors_k=5,
        )
        adapter_path = tmp_path / "adapter.pt"
        adapter.save(str(adapter_path))

        paths = [f"img{i}.jpg" for i in range(len(labels))]
        features_path, labels_path, paths_path = tmp_path / "f.npy", tmp_path / "l.npy", tmp_path / "p.json"
        np.save(features_path, features)
        np.save(labels_path, labels)
        with open(paths_path, "w") as f:
            json.dump(paths, f)

        from come_cbir.evaluate import main as evaluate_main

        output_dir = tmp_path / "eval"
        evaluate_main([
            "--features", str(features_path), "--labels", str(labels_path), "--paths", str(paths_path),
            "--adapter-checkpoint", str(adapter_path), "--top-k", "5", "--output-dir", str(output_dir),
        ])
        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["descriptor_dim"] == 4
