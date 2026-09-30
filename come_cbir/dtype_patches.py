"""
Runtime monkey-patches that fix genuine dtype bugs in the original olmo/
code when running at a non-float32 dtype (bf16/fp16) -- each confirmed
against a real CoME-VL checkpoint, not guessed from an error message alone.
These wrap existing bound *methods* at runtime; no file under olmo/ is
edited on disk, and no original logic/math is duplicated -- each patch
calls straight through to the original method and only adjusts the
dtype of what comes back out.
"""
from __future__ import annotations

import functools
import logging

logger = logging.getLogger("come_cbir.dtype_patches")

_PATCHED_MARKER = "_come_cbir_dtype_patched"


def patch_cross_rope_dtype_bug() -> None:
    """
    olmo/cross_rope.py's RotaryPositionalEncoding4D.forward computes rotation
    coefficients from `coords` (built via torch.linspace in
    make_coords_3d_4d, which defaults to float32 regardless of the model's
    running dtype) and `self.freqs` (an nn.Parameter, so it *does* follow
    model.to(dtype)). `arg = coords(float32) * freq(bf16)` promotes to
    float32, so q2/k2 = q*co + rotate_half(q)*si come out float32 whenever q
    arrives as bf16/fp16 -- while v (never touched by RoPE) stays at the
    model's real dtype. F.scaled_dot_product_attention then requires q/k/v
    to share a dtype -- confirmed on a real checkpoint: "query.dtype: float
    ... value.dtype: bfloat16", raised from
    RelativePositionalCrossAttention4D.forward (the RGCA fusion path used by
    descriptor_mode="come_fused").

    Wraps the existing forward (no duplicated math) and casts its two
    return values back to the input dtype. Idempotent -- safe to call more
    than once.
    """
    from olmo.cross_rope import RotaryPositionalEncoding4D

    if getattr(RotaryPositionalEncoding4D.forward, _PATCHED_MARKER, False):
        return  # already patched

    original_forward = RotaryPositionalEncoding4D.forward

    @functools.wraps(original_forward)
    def _dtype_safe_forward(self, q, k, coords_q, coords_k=None):
        original_dtype = q.dtype
        q2, k2 = original_forward(self, q, k, coords_q, coords_k)
        return q2.to(original_dtype), k2.to(original_dtype)

    setattr(_dtype_safe_forward, _PATCHED_MARKER, True)
    RotaryPositionalEncoding4D.forward = _dtype_safe_forward
    logger.info(
        "Patched RotaryPositionalEncoding4D.forward to preserve q/k dtype through RoPE "
        "(see come_cbir/dtype_patches.py for why)"
    )
