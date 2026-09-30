"""
CoMECbirFeatureExtractor: global image descriptors for CBIR built on top of
the official CoME-VL visual pipeline (SigLIP2 + DINOv3 + entropy-selected
layer mixing + orthogonal projections + RGCA fusion), without ever running
the Qwen2 language decoder.

See docs/cbir_architecture_analysis.md for the full trace of where every
piece of this lives in ``olmo/`` and why the extraction hooks below were
chosen. In short:

- ``descriptor_mode in {"siglip", "dino", "concat"}`` call
  ``MolmoVisionBackbone.encode_image(images)`` (hook A: pre-RGCA, per-branch,
  OL-mixed tokens).
- ``descriptor_mode == "come_fused"`` calls
  ``MolmoVisionBackbone.forward(images, image_masks)`` (hook B: the official
  RGCA-fused representation, ``outputs['sig_fused']``).

Images are preprocessed as a single 384x384 "overview" crop (no multi-crop
tiling) using the real ``siglip_resize_and_pad`` / ``pixels_to_patches``
functions from ``olmo/data/model_preprocessor.py`` -- a deliberate,
documented simplification (see architecture doc, section 9).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from come_cbir.pooling import AttentionPooling, SUPPORTED_POOLING_MODES, apply_pooling, l2_normalize
from come_cbir.utils import parse_dtype, resolve_device

logger = logging.getLogger("come_cbir.feature_extractor")

SUPPORTED_DESCRIPTOR_MODES = ("siglip", "dino", "concat", "come_fused")


@dataclass
class VisionEncoderInfo:
    """Metadata about the encoder(s) backing a given descriptor mode -- used for metadata.json."""

    siglip_dim: Optional[int] = None
    dino_dim: Optional[int] = None
    d_model: Optional[int] = None
    siglip_layers: Optional[str] = None
    dino_layers: Optional[str] = None
    siglip_image_size: Optional[Tuple[int, int]] = None
    dino_image_size: Optional[Tuple[int, int]] = None
    siglip_patch_size: Optional[int] = None


def _siglip_preprocess_single_crop(
    image: Image.Image, output_size: Tuple[int, int] = (384, 384), patch_size: int = 16
) -> np.ndarray:
    """
    Reproduce the official single-crop SigLIP preprocessing path used by
    olmo/data/model_preprocessor.py, without pulling in the tokenizer-coupled
    MultiModalPreprocessor. See docs/cbir_architecture_analysis.md section 9.

    Returns patchified pixels, shape (n_patches, patch_size*patch_size*3), float32.
    """
    from olmo.data.model_preprocessor import pixels_to_patches, siglip_resize_and_pad

    rgb = image.convert("RGB")
    array = np.asarray(rgb).astype(np.float32) / 255.0  # (H, W, 3) in [0, 1]
    resized, _mask = siglip_resize_and_pad(array, output_size)
    # olmo/data/model_preprocessor.py:322-323, _normalize(..., "siglip")
    normalized = resized * 2.0 - 1.0
    patches = pixels_to_patches(normalized.astype(np.float32), patch_size)
    return patches


@dataclass
class ExtractorConfig:
    """Frozen record of how an extractor was configured -- written into metadata.json by callers."""

    model_name_or_path: Optional[str]
    descriptor_mode: str
    pooling: str
    device: str
    dtype: str
    gem_p: float = 3.0
    image_size: int = 384
    patch_size: int = 16


class CoMECbirFeatureExtractor:
    """
    Extracts L2-normalized global image descriptors for CBIR.

    Parameters
    ----------
    model_name_or_path:
        Path to a CoME-VL / Molmo checkpoint directory understood by
        ``olmo.model.Molmo.from_checkpoint`` (must contain ``model.yaml`` or
        ``config.yaml`` plus ``model.pt`` or sharded weights). Ignored if
        ``model`` is passed directly (used by tests with mock encoders).
    descriptor_mode:
        One of "siglip", "dino", "concat", "come_fused".
    pooling:
        One of "mean", "max", "cls", "gem", "attention".
    device / dtype:
        Standard torch device string and one of "float32"/"float16"/"bfloat16".
    gem_p:
        GeM pooling exponent (ignored unless pooling="gem").
    attention_pooling_module:
        Optional pre-built, already-trained ``AttentionPooling`` instance.
        Required (and only used) if pooling="attention" -- see
        ``pooling.AttentionPooling`` docstring for why this isn't built
        automatically.
    model:
        An already-constructed model object exposing the same public
        surface as ``olmo.model.Molmo`` (``.eval()``, ``.to()``, ``.config``,
        ``.vision_backbone.encode_image()``, ``.vision_backbone.forward()``).
        Passing this in lets tests inject small mock encoders instead of
        downloading the real checkpoint (see tests/come_cbir/conftest.py).
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        descriptor_mode: str = "come_fused",
        pooling: str = "mean",
        device: str = "cpu",
        dtype: str = "float32",
        gem_p: float = 3.0,
        attention_pooling_module: Optional[AttentionPooling] = None,
        image_size: int = 384,
        patch_size: int = 16,
        model: Optional[object] = None,
        low_memory: bool = False,
    ):
        if descriptor_mode not in SUPPORTED_DESCRIPTOR_MODES:
            raise ValueError(
                f"Unsupported descriptor_mode='{descriptor_mode}'. "
                f"Choose from: {SUPPORTED_DESCRIPTOR_MODES}"
            )
        if pooling not in SUPPORTED_POOLING_MODES:
            raise ValueError(
                f"Unsupported pooling='{pooling}'. Choose from: {SUPPORTED_POOLING_MODES}"
            )
        if model is None and not model_name_or_path:
            raise ValueError("Either model_name_or_path or model must be provided")

        self.config = ExtractorConfig(
            model_name_or_path=model_name_or_path,
            descriptor_mode=descriptor_mode,
            pooling=pooling,
            device=resolve_device(device),
            dtype=dtype,
            gem_p=gem_p,
            image_size=image_size,
            patch_size=patch_size,
        )
        self._torch_dtype = parse_dtype(dtype)
        # The vision backbone (SigLIP2 + DINOv3 + OL-mixing + RGCA) always runs in float32,
        # regardless of the requested --dtype. Confirmed on a real checkpoint: running it at
        # bfloat16 produced all-NaN descriptors for every single image (100% of 1000 rows) even
        # though extraction completed without any crash -- a silent numerical-stability failure,
        # not one of the three dtype-mismatch crashes patched below. config.yaml's
        # float32_attention=True exists specifically to compute attention in float32 for
        # numerical stability through many stacked layers (27 SigLIP2 blocks + 24 DINOv3 blocks +
        # RGCA); disabling it (below) fixed the crash but removed that stabilizer, and bf16
        # apparently isn't precise enough on its own for this depth of network. The vision
        # backbone is a small fraction of the full model (~700-800M params for SigLIP2+DINOv3
        # combined, vs ~7B for the Qwen2 decoder come_cbir never executes), so forcing it to
        # float32 costs only ~1.5GB extra while --dtype's memory savings still apply to the
        # (unused but still-resident) decoder.
        self._vision_dtype = torch.float32
        self.attention_pooling_module = attention_pooling_module

        if model is not None:
            self.model = model
        elif low_memory:
            from come_cbir.checkpoint_loading import load_checkpoint_low_memory

            logger.info("Loading CoME-VL checkpoint from %s (low-memory mode)", model_name_or_path)
            self.model = load_checkpoint_low_memory(
                model_name_or_path, device=self.config.device, dtype=self._torch_dtype
            )
        else:
            from olmo.model import Molmo

            logger.info("Loading CoME-VL checkpoint from %s", model_name_or_path)
            self.model = Molmo.from_checkpoint(model_name_or_path, device=self.config.device)

        self.model.eval()
        self.model.to(self._torch_dtype)
        if self._torch_dtype != self._vision_dtype:
            logger.info(
                "Forcing vision_backbone to float32 (requested dtype=%s only applies to the "
                "unused LLM decoder) -- see feature_extractor.py for why", dtype,
            )
        self.model.vision_backbone.to(self._vision_dtype)
        if self.attention_pooling_module is not None:
            self.attention_pooling_module.to(device=self.config.device, dtype=self._torch_dtype)
            self.attention_pooling_module.eval()

        # Belt-and-suspenders: with the vision backbone forced to float32 above, none of the
        # three patches below should actually be triggered anymore (their root cause was mixed
        # dtypes at bf16/fp16, which no longer occurs once everything vision-related is float32).
        # Left in place in case a future change reintroduces a reduced-precision vision path.
        # Work around a real bug in olmo/image_vit.py's ViTMultiHeadDotProductAttention.forward:
        # config.yaml sets both float32_attention=True and attention_type="sdpa". With both set,
        # at any non-float32 runtime dtype, q/k get upcast to float32 but v does not (the "direct"
        # attention_type path handles this correctly by casting attn_weights to v's dtype before
        # combining; "sdpa" does not), and F.scaled_dot_product_attention requires q/k/v to share
        # a dtype -- confirmed on a real checkpoint: "query.dtype: float ... value.dtype:
        # bfloat16", reproducing identically whether the checkpoint was loaded via
        # Molmo.from_checkpoint or load_checkpoint_low_memory, so this is unrelated to how
        # come_cbir loads weights. Disabling it here is a runtime config *value* change, not an
        # edit to any file under olmo/ -- and it's a no-op when dtype=float32 (og_dtype is already
        # float32, so the upcast in the original code was already a no-op in that case too).
        # float32_attention exists to help numerical stability during training; come_cbir only
        # ever runs inference, so there is nothing to lose by turning it off here.
        if getattr(self.model.config, "float32_attention", False):
            logger.info(
                "Disabling model.config.float32_attention (was True) to avoid a q/k-vs-v dtype "
                "mismatch inside scaled_dot_product_attention -- see feature_extractor.py for why"
            )
            self.model.config.float32_attention = False

        # Work around a second, distinct dtype bug in olmo/model.py's MolmoVisionBackbone.forward:
        # config.yaml sets image_padding_embed="pad_and_partial_pad". That branch hardcodes
        # `all_pad = (image_masks == 0).to(dtype=torch.float32)` (and the same for partial_pad)
        # regardless of the model's running dtype. pad_embed (bf16) * all_pad (fp32) promotes to
        # fp32, and adding that into image_features silently upcasts image_features back to
        # float32 for the rest of the forward pass -- confirmed on a real checkpoint: this is
        # exactly why the *next* bf16 module (image_pooling_2d) then sees a float32 query and
        # raises "mat1 and mat2 must have the same dtype". come_cbir always passes an all-ones
        # image_masks (a single, never-actually-padded crop -- see architecture doc section 9),
        # so this injection is already a semantic no-op for every image we ever encode; disabling
        # it (again, a runtime config *value* change, not an olmo/ file edit) removes the dtype
        # corruption without changing any real behavior for our use case.
        if getattr(self.model.config, "image_padding_embed", None):
            logger.info(
                "Disabling model.config.image_padding_embed (was %r) -- come_cbir never passes "
                "genuinely padded crops, and this branch corrupts dtype under bf16/fp16; see "
                "feature_extractor.py for why", self.model.config.image_padding_embed,
            )
            self.model.config.image_padding_embed = None

        if self.config.descriptor_mode == "come_fused":
            # Work around a third, distinct dtype bug, this time in olmo/cross_rope.py's
            # RotaryPositionalEncoding4D.forward (the RGCA fusion path) -- see
            # come_cbir/dtype_patches.py for the full explanation of why q/k silently become
            # float32 there while v stays at the model's real dtype, confirmed on a real
            # checkpoint. Only needed for come_fused (the only mode that runs cross_rope).
            from come_cbir.dtype_patches import patch_cross_rope_dtype_bug

            patch_cross_rope_dtype_bug()

        self._encoder_info = self._read_encoder_info()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def _read_encoder_info(self) -> VisionEncoderInfo:
        cfg = self.model.config
        v1 = getattr(cfg, "vision_backbone", None)
        v2 = getattr(cfg, "vision_backbone2", None)
        vb = getattr(self.model, "vision_backbone", None)
        idx1 = getattr(vb, "idx1", None)
        idx2 = getattr(vb, "idx2", None)
        return VisionEncoderInfo(
            siglip_dim=getattr(v1, "image_emb_dim", None),
            dino_dim=getattr(v2, "image_emb_dim", None),
            d_model=getattr(cfg, "d_model", None),
            siglip_layers=str(idx1) if idx1 is not None else None,
            dino_layers=str(idx2) if idx2 is not None else None,
            siglip_image_size=getattr(v1, "image_default_input_size", None),
            dino_image_size=getattr(v2, "image_default_input_size", None),
            siglip_patch_size=getattr(v1, "image_patch_size", None),
        )

    @property
    def descriptor_dim(self) -> int:
        info = self._encoder_info
        if self.config.descriptor_mode == "siglip":
            return int(info.siglip_dim)
        elif self.config.descriptor_mode == "dino":
            return int(info.dino_dim)
        elif self.config.descriptor_mode == "concat":
            return int(info.siglip_dim) + int(info.dino_dim)
        elif self.config.descriptor_mode == "come_fused":
            return int(info.d_model)
        raise AssertionError("unreachable")  # descriptor_mode validated in __init__

    def metadata(self) -> Dict:
        """Structured info for metadata.json (see extract_features CLI)."""
        info = self._encoder_info
        return {
            "descriptor_mode": self.config.descriptor_mode,
            "pooling": self.config.pooling,
            "descriptor_dimension": self.descriptor_dim,
            "encoder_names": {"siglip": "SigLIP2", "dino": "DINOv3"},
            "selected_layers": {"siglip": info.siglip_layers, "dino": info.dino_layers},
            "encoder_dims": {"siglip": info.siglip_dim, "dino": info.dino_dim, "d_model": info.d_model},
            "image_resolution": {"siglip": info.siglip_image_size, "dino": info.dino_image_size},
            "preprocessing": (
                "single 384x384 overview crop, SigLIP-style resize+[-1,1] normalization, "
                "16x16 patchification (see docs/cbir_architecture_analysis.md section 9)"
            ),
            "normalization": "L2",
            "dtype": self.config.dtype,
            "device": self.config.device,
            "model_checkpoint": self.config.model_name_or_path,
        }

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """PIL images -> patchified pixel tensor (B, N, patch*patch*3), single crop each."""
        patches = [
            _siglip_preprocess_single_crop(
                img, output_size=(self.config.image_size, self.config.image_size),
                patch_size=self.config.patch_size,
            )
            for img in images
        ]
        batch = np.stack(patches, axis=0)
        return torch.from_numpy(batch)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_images(
        self,
        images: Union[Sequence[Image.Image], torch.Tensor],
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Run the configured descriptor pipeline end to end.

        ``images`` may be a sequence of PIL Images (preprocessed internally)
        or an already-patchified tensor of shape (B, N, patch*patch*3) --
        the latter is what ``come_cbir.datasets`` produces so preprocessing
        only happens once per epoch in the DataLoader workers.

        Returns an L2-normalized (batch_size, descriptor_dim) tensor, plus
        (if requested) a dict of intermediate token tensors for diagnostics.
        """
        if isinstance(images, torch.Tensor):
            patch_tensor = images
        else:
            patch_tensor = self.preprocess_images(images)

        device = torch.device(self.config.device)
        # Always feed the vision backbone float32 inputs -- it always runs in float32
        # regardless of the requested --dtype (see __init__ for why).
        patch_tensor = patch_tensor.to(device=device, dtype=self._vision_dtype)
        batch_size = patch_tensor.shape[0]
        # Treat every image as a single crop (T=1); see architecture doc section 7/9.
        images_bt = patch_tensor.unsqueeze(1)  # (B, T=1, N, D_pixels)

        intermediates: Dict[str, torch.Tensor] = {}

        with torch.inference_mode():
            if self.config.descriptor_mode in ("siglip", "dino", "concat"):
                descriptor, extra = self._encode_branches(images_bt, batch_size)
            elif self.config.descriptor_mode == "come_fused":
                descriptor, extra = self._encode_come_fused(images_bt, batch_size)
            else:
                raise AssertionError("unreachable")  # validated in __init__
            intermediates.update(extra)

        descriptor = l2_normalize(descriptor.float())

        # Fail loudly instead of silently: a real checkpoint run once produced all-NaN
        # descriptors for every image while reporting "successful" extraction (no exception
        # anywhere in the pipeline checked for this). Never let that happen silently again.
        if torch.isnan(descriptor).any() or torch.isinf(descriptor).any():
            raise RuntimeError(
                f"Descriptor contains NaN/Inf values (descriptor_mode={self.config.descriptor_mode}, "
                f"pooling={self.config.pooling}, dtype={self.config.dtype}). This has happened before "
                f"due to numerical instability in the vision backbone at reduced precision -- see the "
                f"comment on self._vision_dtype in feature_extractor.py. If you are seeing this despite "
                f"that fix, please report it with the full command used."
            )

        if return_intermediates:
            return descriptor, intermediates
        return descriptor

    def _pool_tokens(self, tokens: torch.Tensor, cls_token: Optional[torch.Tensor]) -> torch.Tensor:
        return apply_pooling(
            tokens.float(),
            self.config.pooling,
            cls_token=cls_token.float() if cls_token is not None else None,
            gem_p=self.config.gem_p,
            attention_module=self.attention_pooling_module,
        )

    def _encode_branches(
        self, images_bt: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """siglip / dino / concat: hook A, olmo/model.py MolmoVisionBackbone.encode_image()."""
        vb = self.model.vision_backbone
        images_flat = images_bt.view(batch_size, images_bt.shape[2], images_bt.shape[3])
        # encode_image expects (B, T, N, D); T=1 for the single-crop CBIR pipeline.
        feat_siglip, feat_dino = vb.encode_image(images_bt)
        # (B, T=1, N, D) -> (B, N, D)
        feat_siglip = feat_siglip.squeeze(1)
        feat_dino = feat_dino.squeeze(1)

        intermediates = {"tokens_siglip": feat_siglip, "tokens_dino": feat_dino}

        cls_siglip = None  # SigLIP2 has no CLS token (num_prefix_tokens=0)
        cls_dino = None
        if self.config.pooling == "cls" and self.config.descriptor_mode in ("dino", "concat"):
            # Reproduces olmo/model.py:1890 verbatim (last-layer CLS token) without
            # duplicating any architecture -- just the same one-line indexing the
            # official encode_image() itself performs internally but doesn't return.
            hidden_states2 = vb.image_vit2(images_flat)["hidden_states"]
            cls_dino = hidden_states2[-1][:, 0]
            intermediates["cls_dino"] = cls_dino

        if self.config.descriptor_mode == "siglip":
            descriptor = self._pool_tokens(feat_siglip, cls_siglip)
        elif self.config.descriptor_mode == "dino":
            descriptor = self._pool_tokens(feat_dino, cls_dino)
        else:  # concat
            d_siglip = l2_normalize(self._pool_tokens(feat_siglip, cls_siglip))
            d_dino = l2_normalize(self._pool_tokens(feat_dino, cls_dino))
            descriptor = torch.cat([d_siglip, d_dino], dim=-1)
        return descriptor, intermediates

    def _encode_come_fused(
        self, images_bt: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """come_fused: hook B, olmo/model.py MolmoVisionBackbone.forward() -> cross_rope['sig_fused']."""
        vb = self.model.vision_backbone
        num_patch = images_bt.shape[2]
        image_masks = torch.ones(
            batch_size, 1, num_patch, device=images_bt.device, dtype=images_bt.dtype
        )
        fused = vb(images_bt, image_masks)  # (B, T=1, N_fused, d_model)
        fused = fused.squeeze(1)  # (B, N_fused, d_model)
        descriptor = self._pool_tokens(fused, cls_token=None)
        return descriptor, {"tokens_fused": fused}
