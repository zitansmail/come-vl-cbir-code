import csv
import numpy as np
import pytest
from PIL import Image

from come_cbir.datasets import (
    CBIRImageDataset,
    cbir_collate_fn,
    load_manifest_dataset,
    scan_folder_dataset,
)


def _write_dummy_image(path, color=(255, 0, 0), size=(20, 20)):
    Image.new("RGB", size, color=color).save(path)


@pytest.fixture
def folder_dataset(tmp_path):
    root = tmp_path / "dataset"
    for cls, color in [("cat", (255, 0, 0)), ("dog", (0, 255, 0))]:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(2):
            _write_dummy_image(cls_dir / f"img_{i}.jpg", color=color)
    return root


def test_scan_folder_dataset_deterministic_and_labeled(folder_dataset):
    samples = scan_folder_dataset(folder_dataset)
    assert len(samples) == 4
    class_names = sorted({s.class_name for s in samples})
    assert class_names == ["cat", "dog"]
    # cat sorts before dog -> label 0; deterministic ordering
    labels_by_class = {s.class_name: s.label for s in samples}
    assert labels_by_class["cat"] == 0
    assert labels_by_class["dog"] == 1


def test_scan_folder_dataset_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_folder_dataset(tmp_path / "does_not_exist")


def test_scan_folder_dataset_no_classes_raises(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="No class subdirectories"):
        scan_folder_dataset(empty_root)


def test_load_manifest_dataset(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "class_name", "split"])
        writer.writerow(["/a.jpg", "0", "cat", "train"])
        writer.writerow(["/b.jpg", "1", "dog", "query"])
    samples = load_manifest_dataset(manifest)
    assert [s.image_path for s in samples] == ["/a.jpg", "/b.jpg"]
    assert samples[0].label == 0 and samples[0].class_name == "cat" and samples[0].split == "train"


def test_load_manifest_dataset_missing_column_raises(tmp_path):
    manifest = tmp_path / "bad_manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["not_image_path"])
        writer.writerow(["x"])
    with pytest.raises(ValueError, match="image_path"):
        load_manifest_dataset(manifest)


def test_cbir_image_dataset_getitem_shapes(folder_dataset):
    dataset = CBIRImageDataset(root=str(folder_dataset), image_size=32, patch_size=16)
    patches, path, label, class_name = dataset[0]
    n_patches = (32 // 16) ** 2
    assert patches.shape == (n_patches, 16 * 16 * 3)
    assert isinstance(path, str) and isinstance(label, int) and isinstance(class_name, str)


def test_cbir_image_dataset_requires_exactly_one_source(folder_dataset, tmp_path):
    with pytest.raises(ValueError, match="Exactly one of"):
        CBIRImageDataset(root=str(folder_dataset), manifest_csv=str(tmp_path / "m.csv"))
    with pytest.raises(ValueError, match="Exactly one of"):
        CBIRImageDataset()


def test_corrupted_image_raises_by_default(tmp_path):
    root = tmp_path / "dataset"
    cls_dir = root / "cat"
    cls_dir.mkdir(parents=True)
    (cls_dir / "broken.jpg").write_bytes(b"not a real image")
    dataset = CBIRImageDataset(root=str(root), skip_corrupted=False)
    with pytest.raises(RuntimeError, match="Failed to load image"):
        dataset[0]


def test_corrupted_image_skipped_and_recorded_when_enabled(tmp_path):
    root = tmp_path / "dataset"
    cls_dir = root / "cat"
    cls_dir.mkdir(parents=True)
    (cls_dir / "broken.jpg").write_bytes(b"not a real image")
    _write_dummy_image(cls_dir / "ok.jpg")
    dataset = CBIRImageDataset(root=str(root), skip_corrupted=True)
    # sorted() puts "broken.jpg" before "ok.jpg"
    item0 = dataset[0]
    assert item0 is None
    assert len(dataset.skipped) == 1
    assert "broken.jpg" in dataset.skipped[0]
    item1 = dataset[1]
    assert item1 is not None


def test_collate_fn_drops_none_entries(folder_dataset):
    dataset = CBIRImageDataset(root=str(folder_dataset), image_size=32, patch_size=16)
    good = dataset[0]
    batch = cbir_collate_fn([good, None, good])
    patches, paths, labels, class_names = batch
    assert patches.shape[0] == 2
    assert len(paths) == 2


def test_collate_fn_all_none_returns_none():
    assert cbir_collate_fn([None, None]) is None
