"""
Adapter integration for run_experiments.py, using the shared mock_model
fixture (see conftest.py) instead of a real checkpoint -- monkeypatches
_get_or_load_model so run_single_experiment never tries to download or load
the actual ~8B-parameter CoME-VL checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import come_cbir.run_experiments as run_experiments_module
from come_cbir.config import AdapterEntryConfig, ExperimentEntry, DatasetConfig, ModelConfigEntry, RunConfig
from come_cbir.datasets import CBIRImageDataset


def _make_folder_dataset(root: Path, n_classes: int = 3, n_per_class: int = 8):
    rng = np.random.RandomState(0)
    for c in range(n_classes):
        class_dir = root / f"class_{c}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            array = rng.randint(0, 255, size=(48, 64, 3), dtype=np.uint8)
            Image.fromarray(array, mode="RGB").save(class_dir / f"img_{i}.jpg")
    return CBIRImageDataset(root=str(root))


def test_run_single_experiment_with_adapter_returns_original_and_adapted_rows(tmp_path, mock_model, monkeypatch):
    monkeypatch.setattr(run_experiments_module, "_get_or_load_model", lambda *a, **k: mock_model)

    dataset = _make_folder_dataset(tmp_path / "data", n_classes=3, n_per_class=8)

    config = RunConfig(
        dataset=DatasetConfig(type="folder", root=str(tmp_path / "data")),
        model=ModelConfigEntry(checkpoint="unused", device="cpu", dtype="float32"),
        experiments=[],
        output_root=str(tmp_path / "outputs"),
        batch_size=4,
        num_workers=0,
        top_k=3,
        seed=0,
    )
    entry = ExperimentEntry(
        name="siglip_adapter_test",
        descriptor_mode="siglip",
        pooling="mean",
        adapter=AdapterEntryConfig(
            enabled=True, hidden_dim=8, output_dim=4, holdout_fraction=0.3,
            epochs=2, steps_per_epoch=2, classes_per_batch=3, samples_per_class=3,
        ),
    )

    import logging
    logger = logging.getLogger("test")

    rows = run_experiments_module.run_single_experiment(entry, config, dataset, logger)

    assert len(rows) == 2
    variants = {row["variant"] for row in rows}
    assert variants == {"original", "adapted"}

    adapted_row = next(r for r in rows if r["variant"] == "adapted")
    original_row = next(r for r in rows if r["variant"] == "original")
    assert adapted_row["original_dimension"] == 4  # adapter output_dim
    assert original_row["original_dimension"] == 32  # mock model's SIGLIP_DIM
    assert adapted_row["adapter_params"] != ""
    assert adapted_row["adapter_params"] < 1_000_000

    output_dir = Path(config.output_root) / entry.name
    assert (output_dir / "adapter.pt").exists()
    assert (output_dir / "split.json").exists()
    assert (output_dir / "metrics_original_holdout.json").exists()
    assert (output_dir / "metrics_adapted_holdout.json").exists()


def test_run_single_experiment_without_adapter_returns_one_original_row(tmp_path, mock_model, monkeypatch):
    monkeypatch.setattr(run_experiments_module, "_get_or_load_model", lambda *a, **k: mock_model)

    dataset = _make_folder_dataset(tmp_path / "data", n_classes=2, n_per_class=6)
    config = RunConfig(
        dataset=DatasetConfig(type="folder", root=str(tmp_path / "data")),
        model=ModelConfigEntry(checkpoint="unused", device="cpu", dtype="float32"),
        experiments=[],
        output_root=str(tmp_path / "outputs"),
        batch_size=4,
        num_workers=0,
        top_k=3,
        seed=0,
    )
    entry = ExperimentEntry(name="siglip_plain", descriptor_mode="siglip", pooling="mean")

    import logging
    logger = logging.getLogger("test")
    rows = run_experiments_module.run_single_experiment(entry, config, dataset, logger)

    assert len(rows) == 1
    assert rows[0]["variant"] == "original"
