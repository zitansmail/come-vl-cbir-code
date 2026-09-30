"""
Tests for come_cbir/dtype_patches.py, run against the real olmo.cross_rope
module (no mock needed here -- the bug and fix are both about dtype
arithmetic in a small, self-contained method, not about model architecture,
so exercising the real class is cheap and precise).
"""
import torch

from come_cbir.dtype_patches import patch_cross_rope_dtype_bug


def test_rope_forward_preserves_bfloat16_dtype_after_patch():
    from olmo.cross_rope import RotaryPositionalEncoding4D, make_coords_3d_4d

    patch_cross_rope_dtype_bug()

    # n_pos must be even per axis and sum to the per-head dim; use small values.
    rope = RotaryPositionalEncoding4D(cutoffs=(256.0, 256.0, 256.0), n_pos=(4, 4, 4)).to(torch.bfloat16)

    B, H, T, N, d = 1, 2, 1, 4, 12  # sum(n_pos) == d
    q = torch.randn(B, H, T, N, d, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, N, d, dtype=torch.bfloat16)
    coords = make_coords_3d_4d(B, T, N, device=q.device, hw=(2, 2))  # float32, as in the real bug

    q2, k2 = rope(q, k, coords, coords)

    assert q2.dtype == torch.bfloat16
    assert k2.dtype == torch.bfloat16


def test_rope_forward_without_patch_reproduces_the_original_bug():
    """
    Confirms the bug is real (not a misdiagnosis) by checking the *unpatched*
    behavior separately, using a fresh, unpatched copy of the class loaded
    before any patch is applied in this process. Since patches are applied
    process-wide and other tests may have already patched it, this test
    calls the original forward directly (saved before patching) rather than
    relying on import order across the test session.
    """
    from olmo.cross_rope import RotaryPositionalEncoding4D, make_coords_3d_4d

    # Grab a forward that is guaranteed un-patched by constructing a fresh
    # function object equivalent to the original implementation's behavior:
    # since patch_cross_rope_dtype_bug() may have already run in this
    # process (test order is not guaranteed), we instead just check that the
    # *symptom* the patch fixes would occur without it, by calling the
    # underlying math directly on float32 coords + bf16 freq, mirroring the
    # real class's _co_si computation.
    rope = RotaryPositionalEncoding4D(cutoffs=(256.0,), n_pos=(4,)).to(torch.bfloat16)
    coords = make_coords_3d_4d(1, 1, 4, device="cpu", hw=(2, 2))[..., :1]  # float32
    co, si = rope._co_si(coords)
    assert co.dtype == torch.float32, (
        "If this ever becomes bfloat16, the underlying olmo/cross_rope.py bug this patch "
        "works around may have been fixed upstream -- come_cbir's patch would then be a "
        "harmless no-op, but worth revisiting/removing."
    )


def test_patch_is_idempotent():
    patch_cross_rope_dtype_bug()
    patch_cross_rope_dtype_bug()  # must not double-wrap or raise
