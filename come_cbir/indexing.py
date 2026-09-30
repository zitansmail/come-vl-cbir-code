"""
Nearest-neighbor indexing backends for CBIR: exact/approximate FAISS and
Annoy. All backends assume L2-normalized input descriptors and expose
cosine similarity via inner product (for normalized vectors, inner product
and cosine similarity coincide).

The row-index <-> image-path mapping is always saved alongside the index
(as a JSON list) since a bare FAISS/Annoy index only stores an integer id.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger("come_cbir.indexing")

SUPPORTED_BACKENDS = ("faiss-flat", "faiss-hnsw", "faiss-ivf", "annoy")


def _paths_sidecar(index_path: str) -> str:
    return str(Path(index_path).with_suffix(Path(index_path).suffix + ".paths.json"))


def save_paths(index_path: str, paths: Sequence[str]) -> None:
    with open(_paths_sidecar(index_path), "w") as f:
        json.dump(list(paths), f)


def load_paths(index_path: str) -> List[str]:
    with open(_paths_sidecar(index_path)) as f:
        return json.load(f)


@dataclass
class SearchResult:
    indices: np.ndarray  # (n_queries, top_k) row indices into the original feature/path array
    scores: np.ndarray  # (n_queries, top_k) similarity scores (higher = more similar)


class BaseIndex:
    backend: str

    def build(self, features: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, queries: np.ndarray, top_k: int) -> SearchResult:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseIndex":
        raise NotImplementedError


class FaissFlatIndex(BaseIndex):
    """Exact cosine retrieval via faiss.IndexFlatIP on L2-normalized vectors."""

    backend = "faiss-flat"

    def __init__(self, dim: int):
        self.dim = dim
        self._index = None

    def build(self, features: np.ndarray) -> None:
        import faiss

        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(np.ascontiguousarray(features.astype(np.float32)))

    def search(self, queries: np.ndarray, top_k: int) -> SearchResult:
        scores, indices = self._index.search(np.ascontiguousarray(queries.astype(np.float32)), top_k)
        return SearchResult(indices=indices, scores=scores)

    def save(self, path: str) -> None:
        import faiss

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, path)

    @classmethod
    def load(cls, path: str) -> "FaissFlatIndex":
        import faiss

        index = faiss.read_index(path)
        obj = cls(dim=index.d)
        obj._index = index
        return obj


class FaissHNSWIndex(BaseIndex):
    """Approximate cosine retrieval via faiss.IndexHNSWFlat (inner product metric)."""

    backend = "faiss-hnsw"

    def __init__(self, dim: int, m: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index = None

    def build(self, features: np.ndarray) -> None:
        import faiss

        self._index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.hnsw.efSearch = self.ef_search
        self._index.add(np.ascontiguousarray(features.astype(np.float32)))

    def search(self, queries: np.ndarray, top_k: int) -> SearchResult:
        scores, indices = self._index.search(np.ascontiguousarray(queries.astype(np.float32)), top_k)
        return SearchResult(indices=indices, scores=scores)

    def save(self, path: str) -> None:
        import faiss

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, path)

    @classmethod
    def load(cls, path: str) -> "FaissHNSWIndex":
        import faiss

        index = faiss.read_index(path)
        obj = cls(dim=index.d)
        obj._index = index
        return obj


class FaissIVFIndex(BaseIndex):
    """Approximate cosine retrieval via faiss.IndexIVFFlat (inner product metric)."""

    backend = "faiss-ivf"

    def __init__(self, dim: int, nlist: int = 100, nprobe: int = 8):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self._index = None

    def build(self, features: np.ndarray) -> None:
        import faiss

        features = np.ascontiguousarray(features.astype(np.float32))
        nlist = min(self.nlist, max(1, features.shape[0] // 4))
        if nlist != self.nlist:
            logger.warning(
                "Reducing IVF nlist from %d to %d because the dataset only has %d vectors",
                self.nlist, nlist, features.shape[0],
            )
        quantizer = faiss.IndexFlatIP(self.dim)
        self._index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self._index.train(features)
        self._index.add(features)
        self._index.nprobe = self.nprobe

    def search(self, queries: np.ndarray, top_k: int) -> SearchResult:
        scores, indices = self._index.search(np.ascontiguousarray(queries.astype(np.float32)), top_k)
        return SearchResult(indices=indices, scores=scores)

    def save(self, path: str) -> None:
        import faiss

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, path)

    @classmethod
    def load(cls, path: str) -> "FaissIVFIndex":
        import faiss

        index = faiss.read_index(path)
        obj = cls(dim=index.d)
        obj._index = index
        return obj


class AnnoyIndex(BaseIndex):
    """Angular-distance retrieval via Annoy, for compatibility with a prior CBIR system."""

    backend = "annoy"

    def __init__(self, dim: int, n_trees: int = 20):
        self.dim = dim
        self.n_trees = n_trees
        self._index = None
        self._n_items = 0

    def build(self, features: np.ndarray) -> None:
        from annoy import AnnoyIndex as _AnnoyIndex

        self._index = _AnnoyIndex(self.dim, "angular")
        for i, vec in enumerate(features):
            self._index.add_item(i, vec.astype(np.float32).tolist())
        self._index.build(self.n_trees)
        self._n_items = features.shape[0]

    def search(self, queries: np.ndarray, top_k: int) -> SearchResult:
        all_indices = []
        all_scores = []
        for vec in queries:
            ids, dists = self._index.get_nns_by_vector(
                vec.astype(np.float32).tolist(), top_k, include_distances=True
            )
            # Angular distance in [0, 2] for unit vectors -> convert to a cosine-similarity-like
            # score so downstream code (ranking, thresholds) is consistent across backends.
            sims = [1.0 - (d ** 2) / 2.0 for d in dists]
            if len(ids) < top_k:
                pad = top_k - len(ids)
                ids = ids + [-1] * pad
                sims = sims + [-1.0] * pad
            all_indices.append(ids)
            all_scores.append(sims)
        return SearchResult(indices=np.array(all_indices), scores=np.array(all_scores))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._index.save(path)
        # Annoy needs dim/n_trees to reload; stash them next to the sidecar paths file.
        with open(str(Path(path).with_suffix(Path(path).suffix + ".meta.json")), "w") as f:
            json.dump({"dim": self.dim, "n_trees": self.n_trees}, f)

    @classmethod
    def load(cls, path: str) -> "AnnoyIndex":
        from annoy import AnnoyIndex as _AnnoyIndex

        with open(str(Path(path).with_suffix(Path(path).suffix + ".meta.json"))) as f:
            meta = json.load(f)
        obj = cls(dim=meta["dim"], n_trees=meta["n_trees"])
        obj._index = _AnnoyIndex(meta["dim"], "angular")
        obj._index.load(path)
        return obj


_BACKEND_CLASSES = {
    "faiss-flat": FaissFlatIndex,
    "faiss-hnsw": FaissHNSWIndex,
    "faiss-ivf": FaissIVFIndex,
    "annoy": AnnoyIndex,
}


def build_index(backend: str, features: np.ndarray, **kwargs) -> BaseIndex:
    if backend not in _BACKEND_CLASSES:
        raise ValueError(f"Unsupported backend '{backend}'. Choose from: {SUPPORTED_BACKENDS}")
    dim = features.shape[1]
    index_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    index = _BACKEND_CLASSES[backend](dim=dim, **index_kwargs)
    index.build(features)
    return index


def load_index(backend: str, path: str) -> BaseIndex:
    if backend not in _BACKEND_CLASSES:
        raise ValueError(f"Unsupported backend '{backend}'. Choose from: {SUPPORTED_BACKENDS}")
    return _BACKEND_CLASSES[backend].load(path)
