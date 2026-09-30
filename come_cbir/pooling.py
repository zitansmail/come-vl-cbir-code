"""
Pooling strategies that turn a token-sequence tensor ``(batch, tokens, dim)``
into a single descriptor ``(batch, dim)``.

All poolers here operate on tokens *after* any special-token handling has
already been done by the caller (see ``feature_extractor.py``): SigLIP2 has
no CLS token at all (``num_prefix_tokens=0``, confirmed in
``docs/cbir_architecture_analysis.md`` section 1), and the DINOv3 branch has
its CLS + 4 register tokens already stripped before pooling ever sees it
(``x[:, 5:, :]`` in the original ``olmo/model.py:1896``). So for every mode
except ``"cls"``, pooling here is over patch tokens only -- no special
tokens are included or excluded by this module because none remain to
exclude for siglip/dino/come_fused token streams as produced upstream.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_POOLING_MODES = ("mean", "max", "cls", "gem", "attention")


def mean_pool(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: (B, N, D) -> (B, D). Includes all tokens passed in (see module docstring)."""
    return tokens.mean(dim=1)


def max_pool(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: (B, N, D) -> (B, D)."""
    return tokens.max(dim=1).values


def cls_pool(tokens: torch.Tensor, cls_token: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Use a genuine CLS token only. ``cls_token`` must be the encoder's own CLS
    embedding, shape (B, D); this function raises rather than silently
    falling back to patch token 0, since SigLIP2 has no CLS token
    (num_prefix_tokens=0) and using a random patch as a fake CLS would be a
    silent correctness bug, not a pooling choice.
    """
    if cls_token is None:
        raise ValueError(
            "pooling='cls' requires a real CLS token, but none is available for this "
            "descriptor mode/encoder (e.g. SigLIP2 has num_prefix_tokens=0 -- see "
            "docs/cbir_architecture_analysis.md section 1). Use 'mean', 'max', or 'gem' instead."
        )
    return cls_token


def gem_pool(tokens: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Generalized mean pooling: (mean(clamp(x, eps)^p))^(1/p) over the token dim.
    tokens: (B, N, D) -> (B, D). p=1 reduces to mean pooling, p->inf approaches max pooling.
    """
    clamped = tokens.clamp(min=eps)
    pooled = clamped.pow(p).mean(dim=1).pow(1.0 / p)
    return pooled


class GeMPooling(nn.Module):
    """Module wrapper around gem_pool with an optional learnable p (disabled by default)."""

    def __init__(self, p: float = 3.0, learnable: bool = False, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if learnable:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return gem_pool(tokens, p=float(self.p), eps=self.eps)


class AttentionPooling(nn.Module):
    """
    Lightweight single-query attention pooling head.

    Disabled by default everywhere in come_cbir because it introduces
    randomly-initialized parameters that have never been trained -- using it
    out of the box would silently produce meaningless descriptors. It is
    provided so a downstream user can fine-tune it, not for zero-shot use.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.requires_training_warning = (
            "AttentionPooling has randomly-initialized, untrained weights. "
            "Do not use its output for retrieval comparisons without first "
            "training it on a relevant dataset."
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        return pooled[:, 0, :]


def apply_pooling(
    tokens: torch.Tensor,
    pooling: str,
    cls_token: Optional[torch.Tensor] = None,
    gem_p: float = 3.0,
    attention_module: Optional[AttentionPooling] = None,
) -> torch.Tensor:
    """Dispatch to the requested pooling strategy. tokens: (B, N, D) -> (B, D)."""
    if pooling == "mean":
        return mean_pool(tokens)
    elif pooling == "max":
        return max_pool(tokens)
    elif pooling == "cls":
        return cls_pool(tokens, cls_token)
    elif pooling == "gem":
        return gem_pool(tokens, p=gem_p)
    elif pooling == "attention":
        if attention_module is None:
            raise ValueError(
                "pooling='attention' requires an AttentionPooling module to be constructed "
                "and passed in explicitly -- it is never enabled implicitly because its "
                "weights are untrained by default (see AttentionPooling docstring)."
            )
        return attention_module(tokens)
    else:
        raise ValueError(
            f"Unsupported pooling '{pooling}'. Choose from: {SUPPORTED_POOLING_MODES}"
        )


def l2_normalize(descriptor: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(descriptor, p=2, dim=dim, eps=eps)
