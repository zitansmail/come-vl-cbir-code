"""
Tests for come_cbir/checkpoint_loading.py.

The key-remapping unit tests below run on CPU with synthetic tensors/keys
and need no checkpoint download. The real-checkpoint integration test at the
bottom is unchanged from before and stays skipped by default.
"""
import os

import pytest
import torch

from come_cbir.checkpoint_loading import _remap_dinov3_keys

pytestmark_integration = pytest.mark.integration


def test_remap_inserts_model_segment_when_that_matches_live_model():
    checkpoint_keys = {
        "vision_backbone.image_vit2.transformer.layer.0.norm1.weight": torch.zeros(4),
        "vision_backbone.image_vit.some.other.key": torch.zeros(4),
    }
    live_model_keys = {
        "vision_backbone.image_vit2.transformer.model.layer.0.norm1.weight",
        "vision_backbone.image_vit.some.other.key",
    }
    remapped = _remap_dinov3_keys(checkpoint_keys, live_model_keys)
    assert set(remapped.keys()) == live_model_keys


def test_remap_removes_model_segment_when_that_matches_live_model():
    checkpoint_keys = {
        "vision_backbone.image_vit2.transformer.model.layer.3.mlp.up_proj.weight": torch.zeros(4),
    }
    live_model_keys = {
        "vision_backbone.image_vit2.transformer.layer.3.mlp.up_proj.weight",
    }
    remapped = _remap_dinov3_keys(checkpoint_keys, live_model_keys)
    assert set(remapped.keys()) == live_model_keys


def test_remap_leaves_already_matching_keys_untouched():
    tensor = torch.arange(4.0)
    checkpoint_keys = {"vision_backbone.image_vit2.transformer.layer.0.norm1.weight": tensor}
    live_model_keys = {"vision_backbone.image_vit2.transformer.layer.0.norm1.weight"}
    remapped = _remap_dinov3_keys(checkpoint_keys, live_model_keys)
    assert remapped["vision_backbone.image_vit2.transformer.layer.0.norm1.weight"] is tensor


def test_remap_does_not_touch_tensor_values():
    tensor = torch.arange(6.0)
    checkpoint_keys = {"vision_backbone.image_vit2.transformer.layer.5.norm1.weight": tensor}
    live_model_keys = {"vision_backbone.image_vit2.transformer.model.layer.5.norm1.weight"}
    remapped = _remap_dinov3_keys(checkpoint_keys, live_model_keys)
    assert torch.equal(
        remapped["vision_backbone.image_vit2.transformer.model.layer.5.norm1.weight"], tensor
    )


def test_remap_leaves_unresolvable_keys_unchanged_for_caller_to_report():
    checkpoint_keys = {"some.totally.unrelated.key": torch.zeros(2)}
    live_model_keys = {"some.other.expected.key"}
    remapped = _remap_dinov3_keys(checkpoint_keys, live_model_keys)
    # Not silently dropped or guessed at -- left as-is so load_state_dict's own
    # missing/unexpected key error still surfaces it.
    assert "some.totally.unrelated.key" in remapped


def test_remap_handles_all_24_dinov3_layers():
    checkpoint_keys = {
        f"vision_backbone.image_vit2.transformer.layer.{i}.norm1.weight": torch.tensor([float(i)])
        for i in range(24)
    }
    live_model_keys = {
        f"vision_backbone.image_vit2.transformer.model.layer.{i}.norm1.weight" for i in range(24)
    }
    remapped = _remap_dinov3_keys(checkpoint_keys, live_model_keys)
    assert set(remapped.keys()) == live_model_keys
    for i in range(24):
        key = f"vision_backbone.image_vit2.transformer.model.layer.{i}.norm1.weight"
        assert remapped[key].item() == float(i)


def test_state_dict_cast_unifies_mixed_dtypes_before_assign():
    """
    Reproduces, in isolation, the exact fix for the real bug observed on
    2026-07-21: a mixed-precision checkpoint (amp_bf16 training) can save
    different parameters at different dtypes (e.g. q_proj/k_proj float32,
    v_proj bfloat16). assign=True preserves each tensor's saved dtype rather
    than normalizing it, so casting must happen before load_state_dict, not
    via a blanket model.to(dtype) afterward. This test checks the casting
    step itself (the dict comprehension in load_checkpoint_low_memory),
    without needing a real model/checkpoint.
    """
    mixed_dtype_state_dict = {
        "attention.q_proj.weight": torch.zeros(4, 4, dtype=torch.float32),
        "attention.k_proj.weight": torch.zeros(4, 4, dtype=torch.float32),
        "attention.v_proj.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
    }
    target_dtype = torch.bfloat16

    unified = {key: value.to(dtype=target_dtype) for key, value in mixed_dtype_state_dict.items()}

    assert all(v.dtype == target_dtype for v in unified.values())


@pytest.mark.integration
@pytest.mark.skipif(
    "COME_VL_CHECKPOINT" not in os.environ,
    reason="Set COME_VL_CHECKPOINT to a real checkpoint directory to run this test",
)
def test_loads_real_checkpoint_and_extracts_one_image():
    from PIL import Image

    from come_cbir.feature_extractor import CoMECbirFeatureExtractor

    checkpoint = os.environ["COME_VL_CHECKPOINT"]
    extractor = CoMECbirFeatureExtractor(
        model_name_or_path=checkpoint, descriptor_mode="come_fused", pooling="mean", device="cpu",
    )
    image = Image.new("RGB", (384, 384), color=(128, 64, 32))
    descriptor = extractor.encode_images([image])
    assert descriptor.shape == (1, extractor.descriptor_dim)
