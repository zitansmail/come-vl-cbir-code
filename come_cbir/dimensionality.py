"""
PCA-based dimensionality reduction for CBIR descriptors.

Fit PCA only on a designated "database"/"train" split, never on query or
test data (that would leak query statistics into the representation used to
answer queries about that exact data -- a subtle but real form of test-set
leakage in CBIR benchmarks). Callers are responsible for passing only the
appropriate split's features into ``fit``; this module does not know about
splits, it only refuses to silently proceed against too little data and
always re-normalizes after projection.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

logger = logging.getLogger("come_cbir.dimensionality")


@dataclass
class PCAResult:
    transformed: np.ndarray
    model_path: Optional[str] = None


class PCAReducer:
    """Wraps sklearn PCA / IncrementalPCA with L2 re-normalization after projection."""

    def __init__(self, n_components: int, incremental: bool = False, batch_size: int = 4096):
        self.n_components = n_components
        self.incremental = incremental
        self.batch_size = batch_size
        self._model: Optional[PCA] = None
        self._fitted = False

    def fit(self, features: np.ndarray) -> "PCAReducer":
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features (n_samples, dim), got shape {features.shape}")
        n_samples, dim = features.shape
        if self.n_components > min(n_samples, dim):
            raise ValueError(
                f"n_components={self.n_components} must be <= min(n_samples={n_samples}, "
                f"dim={dim}). Use fewer PCA dimensions or more training samples."
            )
        if self.incremental:
            model = IncrementalPCA(n_components=self.n_components, batch_size=self.batch_size)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                if end - start < self.n_components:
                    # IncrementalPCA requires each partial batch to have >= n_components rows.
                    continue
                model.partial_fit(features[start:end])
        else:
            model = PCA(n_components=self.n_components, random_state=0)
            model.fit(features)
        self._model = model
        self._fitted = True
        logger.info(
            "Fitted %s PCA: %d -> %d dims on %d samples (explained variance ratio sum=%.4f)",
            "Incremental" if self.incremental else "standard",
            dim,
            self.n_components,
            n_samples,
            float(np.sum(model.explained_variance_ratio_)),
        )
        return self

    def transform(self, features: np.ndarray, l2_normalize: bool = True) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("PCAReducer.transform called before fit()/load()")
        projected = self._model.transform(features)
        if l2_normalize:
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            projected = projected / norms
        return projected.astype(np.float32)

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted PCAReducer")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"n_components": self.n_components, "model": self._model}, f)
        logger.info("Saved fitted PCA model to %s", path)

    @classmethod
    def load(cls, path: str) -> "PCAReducer":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        reducer = cls(n_components=payload["n_components"])
        reducer._model = payload["model"]
        reducer._fitted = True
        return reducer


def fit_and_apply_pca(
    train_features: np.ndarray,
    apply_features: np.ndarray,
    n_components: int,
    incremental: bool = False,
    model_out_path: Optional[str] = None,
) -> np.ndarray:
    """Convenience wrapper: fit on train_features, transform apply_features, optionally persist."""
    reducer = PCAReducer(n_components=n_components, incremental=incremental).fit(train_features)
    if model_out_path is not None:
        reducer.save(model_out_path)
    return reducer.transform(apply_features)
