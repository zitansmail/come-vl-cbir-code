"""Small shared helpers: logging, seeding, git hash, dtype parsing."""
from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging for come_cbir CLIs and return the package logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("come_cbir")


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if importable) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_commit_hash(repo_dir: Optional[Path] = None) -> str:
    """Best-effort git commit hash of the current repo, "unknown" if unavailable."""
    repo_dir = repo_dir or Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def parse_dtype(name: str):
    """Map a CLI string ('float32'|'float16'|'bfloat16') to a torch.dtype."""
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from: {sorted(mapping)}")
    return mapping[key]


def resolve_device(name: str) -> str:
    """Validate a requested device string against actual availability."""
    import torch

    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but torch.cuda.is_available() is False")
    if name not in ("cpu", "cuda") and not name.startswith("cuda:"):
        raise ValueError(f"Unsupported device '{name}', expected 'cpu', 'cuda', or 'cuda:N'")
    return name
