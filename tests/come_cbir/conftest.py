"""
Shared pytest fixtures for come_cbir tests, including a small mock model
that mimics the public surface of ``olmo.model.Molmo`` /
``MolmoVisionBackbone`` closely enough for ``CoMECbirFeatureExtractor`` to
run against it on CPU, with no checkpoint download and no gated
``facebook/dinov3-*`` HF access required.

Shapes below intentionally use small dims (not the real 1152/1024/3584) so
the test suite stays fast; ``CoMECbirFeatureExtractor`` reads dims from
``model.config``/``model.vision_backbone.idx1``/``idx2`` rather than
hard-coding them, so this is a faithful shape-contract test even though the
absolute sizes differ from the real checkpoint.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


SIGLIP_DIM = 32
DINO_DIM = 24
D_MODEL = 48
SIGLIP_TOKENS = 16  # e.g. a 4x4 grid, post-pooling
DINO_TOKENS = 9  # e.g. a 3x3 grid, post-pooling
FUSED_TOKENS = SIGLIP_TOKENS  # RGCA keeps the SigLIP token grid (see architecture doc section 5)


class MockVisionBackbone(nn.Module):
    """Mimics MolmoVisionBackbone.encode_image()/forward() with tiny random-but-deterministic outputs."""

    def __init__(self):
        super().__init__()
        self.idx1 = list(range(0, 6))
        self.idx2 = list(range(2, 5))
        # Deterministic (seeded) linear layers so repeated calls with the same input
        # are bit-identical -- this is what the "deterministic output" test checks.
        self.siglip_proj = nn.Linear(SIGLIP_DIM, SIGLIP_DIM)
        self.dino_proj = nn.Linear(DINO_DIM, DINO_DIM)
        self.fused_proj = nn.Linear(SIGLIP_DIM, D_MODEL)
        self.dino_cls_proj = nn.Linear(1, DINO_DIM)

    def image_vit2(self, images_flat: torch.Tensor):
        """Mimics DinoVisionTransformer(images)['hidden_states'] closely enough for cls-pooling tests."""
        pixel_summary = images_flat.mean(dim=(-2, -1), keepdim=True).squeeze(-1)  # (B, 1)
        cls_token = self.dino_cls_proj(pixel_summary)  # (B, DINO_DIM), distinct from patch tokens
        patch_tokens = torch.zeros(images_flat.shape[0], DINO_TOKENS, DINO_DIM, device=images_flat.device)
        layer = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)  # (B, 1+DINO_TOKENS, DINO_DIM)
        return {"hidden_states": tuple(layer for _ in range(3))}

    def encode_image(self, images_bt: torch.Tensor):
        # Derive tokens deterministically from the (patchified) input instead of
        # torch.randn, so identical inputs give identical outputs (see test_feature_extractor).
        # Collapse the real (variable) token/pixel dims down to a scalar per (B, T)
        # first, then broadcast out to this mock's own fixed token-grid sizes --
        # the real encode_image() also changes N (576 raw patches -> 576 OL-mixed
        # tokens for siglip, 196 for dino), so token count changing is expected.
        pixel_summary = images_bt.mean(dim=(-2, -1), keepdim=True)  # (B, T, 1, 1)
        feat_siglip = self.siglip_proj(pixel_summary.expand(-1, -1, SIGLIP_TOKENS, SIGLIP_DIM))
        feat_dino = self.dino_proj(pixel_summary.expand(-1, -1, DINO_TOKENS, DINO_DIM))
        return feat_siglip, feat_dino

    def forward(self, images_bt: torch.Tensor, image_masks: torch.Tensor):
        feat_siglip, _feat_dino = self.encode_image(images_bt)
        fused = self.fused_proj(feat_siglip)
        return fused

    def __call__(self, *args, **kwargs):  # nn.Module.__call__ already dispatches to forward()
        return super().__call__(*args, **kwargs)


class MockModel(nn.Module):
    """Mimics olmo.model.Molmo's public surface used by CoMECbirFeatureExtractor."""

    def __init__(self):
        super().__init__()
        self.vision_backbone = MockVisionBackbone()
        v1 = SimpleNamespace(image_emb_dim=SIGLIP_DIM, image_default_input_size=(384, 384), image_patch_size=16)
        v2 = SimpleNamespace(image_emb_dim=DINO_DIM, image_default_input_size=(224, 224))
        # Matches the real config.yaml, which sets this True -- see the real bug this
        # triggers in olmo/image_vit.py's SDPA attention path, documented in
        # feature_extractor.py where CoMECbirFeatureExtractor.__init__ disables it.
        self.config = SimpleNamespace(
            vision_backbone=v1, vision_backbone2=v2, d_model=D_MODEL, float32_attention=True,
            image_padding_embed="pad_and_partial_pad",
        )

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)


@pytest.fixture
def mock_model():
    torch.manual_seed(0)
    model = MockModel()
    model.eval()
    return model


@pytest.fixture
def tiny_rgb_image():
    from PIL import Image
    import numpy as np

    rng = np.random.RandomState(0)
    array = rng.randint(0, 255, size=(48, 64, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")
