import pytest
import torch

from come_cbir.feature_extractor import CoMECbirFeatureExtractor

# Must match tests/come_cbir/conftest.py's MockVisionBackbone dims exactly.
SIGLIP_DIM = 32
DINO_DIM = 24
D_MODEL = 48


@pytest.fixture
def patch_batch():
    torch.manual_seed(0)
    # (B=3, N=9, patch*patch*3=16*16*3) matches image_size=384/patch_size=16 default,
    # but the extractor's preprocessing isn't exercised here -- we feed
    # already-patchified tensors directly, as datasets.py does.
    return torch.rand(3, 9, 16 * 16 * 3)


@pytest.mark.parametrize(
    "descriptor_mode,expected_dim",
    [("siglip", SIGLIP_DIM), ("dino", DINO_DIM), ("concat", SIGLIP_DIM + DINO_DIM), ("come_fused", D_MODEL)],
)
def test_descriptor_modes_shape_and_norm(mock_model, patch_batch, descriptor_mode, expected_dim):
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode=descriptor_mode, pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )
    assert extractor.descriptor_dim == expected_dim
    descriptors = extractor.encode_images(patch_batch)
    assert descriptors.shape == (3, expected_dim)
    norms = descriptors.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_deterministic_output_for_same_input(mock_model, patch_batch):
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="come_fused", pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )
    out1 = extractor.encode_images(patch_batch)
    out2 = extractor.encode_images(patch_batch.clone())
    assert torch.allclose(out1, out2, atol=1e-6)


def test_gem_pooling_mode_runs(mock_model, patch_batch):
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="siglip", pooling="gem", gem_p=2.5, device="cpu", dtype="float32", model=mock_model,
    )
    out = extractor.encode_images(patch_batch)
    assert out.shape == (3, SIGLIP_DIM)


def test_cls_pooling_raises_for_siglip_no_cls_token(mock_model, patch_batch):
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="siglip", pooling="cls", device="cpu", dtype="float32", model=mock_model,
    )
    with pytest.raises(ValueError, match="requires a real CLS token"):
        extractor.encode_images(patch_batch)


def test_cls_pooling_works_for_dino(mock_model, patch_batch):
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="dino", pooling="cls", device="cpu", dtype="float32", model=mock_model,
    )
    out = extractor.encode_images(patch_batch)
    assert out.shape == (3, DINO_DIM)


def test_return_intermediates(mock_model, patch_batch):
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="concat", pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )
    descriptor, intermediates = extractor.encode_images(patch_batch, return_intermediates=True)
    assert "tokens_siglip" in intermediates and "tokens_dino" in intermediates
    assert descriptor.shape == (3, SIGLIP_DIM + DINO_DIM)


def test_invalid_descriptor_mode_raises(mock_model):
    with pytest.raises(ValueError, match="Unsupported descriptor_mode"):
        CoMECbirFeatureExtractor(descriptor_mode="not-a-mode", model=mock_model)


def test_invalid_pooling_raises(mock_model):
    with pytest.raises(ValueError, match="Unsupported pooling"):
        CoMECbirFeatureExtractor(descriptor_mode="siglip", pooling="not-a-pooling", model=mock_model)


def test_requires_model_or_checkpoint():
    with pytest.raises(ValueError, match="model_name_or_path or model"):
        CoMECbirFeatureExtractor(descriptor_mode="siglip")


def test_vision_backbone_forced_to_float32_even_at_bfloat16(mock_model):
    """
    Confirmed on a real checkpoint: running the vision backbone itself at bfloat16
    produced all-NaN descriptors for every image (a numerical-stability failure, not
    a crash). CoMECbirFeatureExtractor must force vision_backbone to float32
    regardless of the requested --dtype.
    """
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="come_fused", pooling="mean", device="cpu", dtype="bfloat16", model=mock_model,
    )
    assert extractor._vision_dtype == torch.float32
    for param in extractor.model.vision_backbone.parameters():
        assert param.dtype == torch.float32


def test_nan_descriptor_raises_instead_of_silently_succeeding(monkeypatch, mock_model):
    """
    A real checkpoint run once reported "1000/1000 images extracted" successfully
    while every single descriptor was actually all-NaN -- nothing in the pipeline
    checked for this. encode_images() must now fail loudly instead.
    """
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="come_fused", pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )

    def _nan_encode_come_fused(images_bt, batch_size):
        d = extractor.descriptor_dim
        return torch.full((batch_size, d), float("nan")), {}

    monkeypatch.setattr(extractor, "_encode_come_fused", _nan_encode_come_fused)

    patch_batch = torch.rand(2, 9, 16 * 16 * 3)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        extractor.encode_images(patch_batch)


def test_metadata_contains_expected_keys(mock_model, patch_batch):
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="come_fused", pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )
    metadata = extractor.metadata()
    for key in ("descriptor_mode", "pooling", "descriptor_dimension", "selected_layers", "encoder_dims"):
        assert key in metadata


def test_low_memory_flag_dispatches_to_low_memory_loader(monkeypatch, mock_model):
    """
    Verifies the low_memory=True wiring calls load_checkpoint_low_memory instead of
    Molmo.from_checkpoint -- this is a dispatch/plumbing test, not a test of the
    meta-device+mmap loading itself, which needs a real checkpoint to exercise.
    """
    calls = {}

    def fake_low_memory_loader(checkpoint_dir, device, dtype):
        calls["checkpoint_dir"] = checkpoint_dir
        calls["device"] = device
        calls["dtype"] = dtype
        return mock_model

    import come_cbir.checkpoint_loading as checkpoint_loading_module
    monkeypatch.setattr(checkpoint_loading_module, "load_checkpoint_low_memory", fake_low_memory_loader)

    extractor = CoMECbirFeatureExtractor(
        model_name_or_path="/fake/checkpoint/dir",
        descriptor_mode="siglip", pooling="mean", device="cpu", dtype="float32",
        low_memory=True,
    )
    assert calls["checkpoint_dir"] == "/fake/checkpoint/dir"
    assert calls["device"] == "cpu"
    assert extractor.model is mock_model


def test_float32_attention_is_disabled_after_construction(mock_model):
    """
    config.yaml sets float32_attention=True, which triggers a real bug in
    olmo/image_vit.py's SDPA attention path at any non-float32 runtime dtype
    (q/k get upcast to float32, v does not, and SDPA requires matching
    dtypes -- confirmed on a real checkpoint). CoMECbirFeatureExtractor must
    turn this off after loading, regardless of descriptor_mode or how the
    checkpoint was loaded.
    """
    assert mock_model.config.float32_attention is True  # sanity check on the fixture itself
    CoMECbirFeatureExtractor(
        descriptor_mode="come_fused", pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )
    assert mock_model.config.float32_attention is False


def test_image_padding_embed_is_disabled_after_construction(mock_model):
    """
    config.yaml sets image_padding_embed="pad_and_partial_pad", whose hardcoded
    float32 cast on all_pad/partial_pad silently upcasts image_features back
    to float32 under bf16, breaking the next bf16 module (image_pooling_2d) --
    confirmed on a real checkpoint. come_cbir always passes an all-ones mask,
    so this branch is a no-op for us regardless; CoMECbirFeatureExtractor must
    disable it after loading.
    """
    assert mock_model.config.image_padding_embed == "pad_and_partial_pad"  # sanity check
    CoMECbirFeatureExtractor(
        descriptor_mode="come_fused", pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )
    assert mock_model.config.image_padding_embed is None


def test_low_memory_false_does_not_call_low_memory_loader(monkeypatch, mock_model):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_checkpoint_low_memory should not be called when low_memory=False")

    import come_cbir.checkpoint_loading as checkpoint_loading_module
    monkeypatch.setattr(checkpoint_loading_module, "load_checkpoint_low_memory", fail_if_called)

    # model= bypasses checkpoint loading entirely (as it does in every other test here),
    # so this just confirms low_memory's default (False) doesn't route through the new
    # module even when it's importable.
    extractor = CoMECbirFeatureExtractor(
        descriptor_mode="siglip", pooling="mean", device="cpu", dtype="float32", model=mock_model,
    )
    assert extractor.model is mock_model
