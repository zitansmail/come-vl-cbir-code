"""Dataclasses + YAML loading for the come_cbir experiment-runner config."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DatasetConfig:
    type: str  # "folder" | "manifest"
    root: Optional[str] = None
    manifest_csv: Optional[str] = None

    def __post_init__(self):
        if self.type not in ("folder", "manifest"):
            raise ValueError(f"dataset.type must be 'folder' or 'manifest', got '{self.type}'")
        if self.type == "folder" and not self.root:
            raise ValueError("dataset.type='folder' requires dataset.root")
        if self.type == "manifest" and not self.manifest_csv:
            raise ValueError("dataset.type='manifest' requires dataset.manifest_csv")


@dataclass
class ModelConfigEntry:
    checkpoint: str
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class AdapterEntryConfig:
    """See come_cbir/retrieval_adapter.py and train_adapter.py for what each field controls."""

    enabled: bool = False
    type: str = "arp"
    hidden_dim: int = 256
    output_dim: int = 256
    holdout_fraction: float = 0.3
    epochs: int = 50
    steps_per_epoch: int = 50
    classes_per_batch: int = 8
    samples_per_class: int = 4
    lr: float = 1e-3
    temperature: float = 0.07
    lambda_np: float = 0.0  # 0.0 = plain ARP (SupCon only); >0 = ARP-NP, see docs/retrieval_adapter.md
    neighbors_k: int = 10
    symmetric_graph: bool = False
    np_weight_normalization: str = "l1"


@dataclass
class ExperimentEntry:
    name: str
    descriptor_mode: str
    pooling: str = "mean"
    pca_dimension: Optional[int] = None
    gem_p: float = 3.0
    adapter: Optional[AdapterEntryConfig] = None


@dataclass
class RunConfig:
    dataset: DatasetConfig
    model: ModelConfigEntry
    experiments: List[ExperimentEntry]
    output_root: str = "outputs/experiments"
    batch_size: int = 16
    num_workers: int = 4
    top_k: int = 10
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not raw:
            raise ValueError(f"Empty or invalid YAML config: {path}")

        dataset = DatasetConfig(**raw["dataset"])
        model = ModelConfigEntry(**raw["model"])
        experiments = []
        for entry in raw["experiments"]:
            entry = dict(entry)
            adapter_raw = entry.pop("adapter", None)
            adapter = AdapterEntryConfig(**adapter_raw) if adapter_raw else None
            experiments.append(ExperimentEntry(adapter=adapter, **entry))
        if not experiments:
            raise ValueError("Config must define at least one entry under `experiments`")
        names = [e.name for e in experiments]
        if len(names) != len(set(names)):
            raise ValueError(f"Experiment names must be unique, got: {names}")

        return cls(
            dataset=dataset,
            model=model,
            experiments=experiments,
            output_root=raw.get("output_root", "outputs/experiments"),
            batch_size=raw.get("batch_size", 16),
            num_workers=raw.get("num_workers", 4),
            top_k=raw.get("top_k", 10),
            seed=raw.get("seed", 0),
        )
