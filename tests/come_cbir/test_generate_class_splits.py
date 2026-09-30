import json

import numpy as np
import pytest

from come_cbir.generate_class_splits import generate_split, main, verify_split


@pytest.fixture
def labels():
    return np.repeat(np.arange(8), 15)  # 8 classes, 15 images each, 120 total


class TestGenerateSplit:
    def test_reuses_three_way_class_split_and_is_disjoint(self, labels):
        payload = generate_split(labels, seed=0, val_class_fraction=0.2, test_class_fraction=0.2)
        train, val, test = set(payload["train_classes"]), set(payload["val_classes"]), set(payload["test_classes"])
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)
        assert train | val | test == set(payload["all_classes"])

    def test_every_test_index_belongs_to_a_test_class(self, labels):
        payload = generate_split(labels, seed=0, val_class_fraction=0.2, test_class_fraction=0.2)
        test_classes = set(payload["test_classes"])
        for i in payload["test_indices"]:
            assert int(labels[i]) in test_classes

    def test_no_duplicate_indices(self, labels):
        payload = generate_split(labels, seed=0, val_class_fraction=0.2, test_class_fraction=0.2)
        all_idx = payload["train_indices"] + payload["val_indices"] + payload["test_indices"]
        assert len(all_idx) == len(set(all_idx))

    def test_reproducible_with_same_seed(self, labels):
        p1 = generate_split(labels, seed=0, val_class_fraction=0.2, test_class_fraction=0.2)
        p2 = generate_split(labels, seed=0, val_class_fraction=0.2, test_class_fraction=0.2)
        assert p1["test_classes"] == p2["test_classes"]
        assert p1["test_indices"] == p2["test_indices"]

    def test_different_seeds_can_give_different_splits(self, labels):
        p0 = generate_split(labels, seed=0, val_class_fraction=0.2, test_class_fraction=0.2)
        p1 = generate_split(labels, seed=1, val_class_fraction=0.2, test_class_fraction=0.2)
        assert p0["test_classes"] != p1["test_classes"] or p0["test_indices"] != p1["test_indices"]

    def test_dataset_size_and_counts_are_consistent(self, labels):
        payload = generate_split(labels, seed=0, val_class_fraction=0.2, test_class_fraction=0.2)
        assert payload["dataset_size"] == len(labels)
        assert payload["n_train_images"] + payload["n_val_images"] + payload["n_test_images"] == len(labels)
        assert payload["n_test_images"] == len(payload["test_indices"])


class TestVerifySplit:
    def test_raises_on_class_overlap(self, labels):
        split = {
            "train_classes": [0, 1], "val_classes": [1, 2], "test_classes": [3],  # 1 overlaps train/val
            "train_indices": np.array([0]), "val_indices": np.array([1]), "test_indices": np.array([2]),
        }
        with pytest.raises(ValueError, match="disjoint"):
            verify_split(split, labels)

    def test_raises_on_duplicate_indices(self, labels):
        split = {
            "train_classes": [0], "val_classes": [1], "test_classes": [2],
            "train_indices": np.array([0, 1]), "val_indices": np.array([1]), "test_indices": np.array([2]),
        }
        with pytest.raises(ValueError, match="Duplicate"):
            verify_split(split, labels)

    def test_raises_when_index_label_does_not_match_declared_class(self, labels):
        # index 0 has label 0, but declared as belonging to test_classes=[5]
        split = {
            "train_classes": [1], "val_classes": [2], "test_classes": [5],
            "train_indices": np.array([15]), "val_indices": np.array([30]), "test_indices": np.array([0]),
        }
        with pytest.raises(ValueError, match="test_indices"):
            verify_split(split, labels)


def test_cli_end_to_end(tmp_path):
    labels = np.repeat(np.arange(6), 10)
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)
    output_dir = tmp_path / "splits"

    main(["--labels", str(labels_path), "--output-dir", str(output_dir), "--seeds", "0", "1"])

    for seed in (0, 1):
        with open(output_dir / f"split_seed_{seed}.json") as f:
            payload = json.load(f)
        assert payload["seed"] == seed
        assert payload["dataset_size"] == 60
        assert set(payload["train_classes"]) | set(payload["val_classes"]) | set(payload["test_classes"]) == set(payload["all_classes"])
