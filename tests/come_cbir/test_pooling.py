import pytest
import torch

from come_cbir.pooling import (
    AttentionPooling,
    apply_pooling,
    cls_pool,
    gem_pool,
    l2_normalize,
    max_pool,
    mean_pool,
)


@pytest.fixture
def tokens():
    torch.manual_seed(0)
    return torch.randn(4, 10, 16)  # (B, N, D)


def test_mean_pool_shape_and_value(tokens):
    out = mean_pool(tokens)
    assert out.shape == (4, 16)
    assert torch.allclose(out, tokens.mean(dim=1))


def test_max_pool_shape_and_value(tokens):
    out = max_pool(tokens)
    assert out.shape == (4, 16)
    assert torch.allclose(out, tokens.max(dim=1).values)


def test_gem_pool_shape(tokens):
    out = gem_pool(tokens.abs(), p=3.0)  # abs() so gem's clamp doesn't degenerate on negatives
    assert out.shape == (4, 16)


def test_gem_pool_p1_equals_mean_on_positive_inputs():
    positive_tokens = torch.rand(2, 5, 8) + 0.1
    gem_out = gem_pool(positive_tokens, p=1.0)
    mean_out = mean_pool(positive_tokens)
    assert torch.allclose(gem_out, mean_out, atol=1e-5)


def test_cls_pool_raises_without_cls_token(tokens):
    with pytest.raises(ValueError, match="requires a real CLS token"):
        cls_pool(tokens, None)


def test_cls_pool_returns_given_cls_token(tokens):
    cls_token = torch.randn(4, 16)
    out = cls_pool(tokens, cls_token)
    assert torch.equal(out, cls_token)


def test_attention_pooling_shape_and_untrained_warning_attr(tokens):
    module = AttentionPooling(dim=16, num_heads=4)
    out = module(tokens)
    assert out.shape == (4, 16)
    assert "untrained" in module.requires_training_warning


def test_attention_pooling_rejects_bad_head_count():
    with pytest.raises(ValueError):
        AttentionPooling(dim=15, num_heads=4)


def test_apply_pooling_dispatch(tokens):
    assert apply_pooling(tokens, "mean").shape == (4, 16)
    assert apply_pooling(tokens, "max").shape == (4, 16)
    assert apply_pooling(tokens.abs(), "gem", gem_p=2.0).shape == (4, 16)
    with pytest.raises(ValueError):
        apply_pooling(tokens, "not-a-real-mode")
    with pytest.raises(ValueError, match="AttentionPooling module"):
        apply_pooling(tokens, "attention")


def test_l2_normalize_unit_norm(tokens):
    out = l2_normalize(mean_pool(tokens))
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
