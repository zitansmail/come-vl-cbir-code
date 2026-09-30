"""
Low-peak-RAM alternative to olmo.model.Molmo.from_checkpoint for unsharded
checkpoints (a directory containing config.yaml/model.yaml + a single
model.pt -- e.g. the MBZUAI/CoME-VL release on Hugging Face).

Why this exists: Molmo.from_checkpoint's unsharded path does, in order:
  1. model = Molmo(model_config)          # constructs the FULL model on CPU,
                                           # float32, every parameter randomly
                                           # initialized -- ~1x model size in RAM
  2. state_dict = torch.load(model.pt)    # reads the WHOLE checkpoint file into
                                           # a second, separate CPU buffer --
                                           # another ~1x model size in RAM
  3. model.load_state_dict(state_dict)    # copies step 2's values into step 1's
                                           # already-allocated tensors
  4. model.to(device)                     # only now does anything move to GPU,
                                           # and only *after* this does
                                           # come_cbir's own .to(dtype) cast run

Steps 1+2 transiently co-exist in RAM, so the real peak is roughly 2x the
model's float32 size -- for CoME-VL (~8B params across the LLM + both vision
encoders) that's on the order of 60-70GB, which does not fit in RAM-limited
environments (e.g. free-tier Colab's ~12.7GB).

This loader instead:
  1. builds the model on the "meta" device -- no real storage allocated at
     all. This is not a new trick: olmo/model.py already threads
     config.init_device through consistently enough to support this, because
     it's the same mechanism config.yaml's own `low_cpu_fsdp: true` relies on
     for low-memory FSDP training initialization.
  2. reads model.pt with mmap=True, so pages are faulted in lazily from disk
     instead of the whole file being materialized in RAM up front.
  3. uses load_state_dict(..., assign=True), so each mmap'd tensor is
     assigned directly in place of its meta placeholder instead of being
     copied into a second, already-allocated destination tensor.

It still calls Molmo._make_state_dict_compatible for the exact same key
renaming/regrouping logic the official loader uses -- only the *memory
strategy* changes, the checkpoint-format understanding is untouched and
olmo/model.py itself is not modified.

Caveat, stated plainly: this has been validated against the documented
PyTorch idiom for meta-device + assign=True low-memory loading, and against
config.yaml's own low_cpu_fsdp flag implying "meta" is an already-supported
init_device value in this codebase -- but it has NOT been run against a real
CoME-VL checkpoint (no environment with the disk/network to fetch one was
available while writing this). Please report back what you see on the first
real run, including the full traceback if it fails.
"""
from __future__ import annotations

import logging
import re
from os.path import join
from pathlib import Path
from typing import Dict, Set

import torch

logger = logging.getLogger("come_cbir.checkpoint_loading")

# Observed on a real MBZUAI/CoME-VL checkpoint (2026-07-20): the DINOv3 submodule
# (vision_backbone.image_vit2.transformer, an AutoModel pulled via trust_remote_code
# from facebook/dinov3-vitl16-pretrain-lvd1689m) has a different internal attribute
# structure depending on *when* its remote code was pulled, since the download is not
# pinned to a specific revision. The checkpoint was saved against a version whose
# per-layer keys are "...transformer.layer.N...."; a later revision nests those one
# level deeper as "...transformer.model.layer.N....". Every missing/unexpected key
# pair in the observed error differed by exactly this one segment, for all 24 layers.
_DINOV3_LAYER_KEY_RE = re.compile(r"(\.transformer\.)(model\.)?(layer\.\d+\.)")


def _remap_dinov3_keys(state_dict: Dict[str, torch.Tensor], model_keys: Set[str]) -> Dict[str, torch.Tensor]:
    """
    Rename any key that doesn't match `model_keys` by inserting/removing the
    ".model." segment DINOv3's remote code may or may not use, in whichever
    direction actually lands on a real key in the live model. Same tensors,
    corrected names only -- no values are touched or dropped.
    """
    remapped: Dict[str, torch.Tensor] = {}
    renamed_count = 0
    for key, value in state_dict.items():
        if key in model_keys:
            remapped[key] = value
            continue

        match = _DINOV3_LAYER_KEY_RE.search(key)
        if match:
            has_model_segment = match.group(2) is not None
            if has_model_segment:
                candidate = key.replace(".transformer.model.layer.", ".transformer.layer.", 1)
            else:
                candidate = key.replace(".transformer.layer.", ".transformer.model.layer.", 1)
            if candidate in model_keys:
                remapped[candidate] = value
                renamed_count += 1
                continue

        remapped[key] = value  # leave as-is; load_state_dict will report it below if still wrong

    if renamed_count:
        logger.info(
            "Remapped %d DINOv3 checkpoint key(s) to match the live model's naming "
            "(transformer.<model.>layer.N -- see come_cbir/checkpoint_loading.py for why)",
            renamed_count,
        )
    return remapped


def load_checkpoint_low_memory(checkpoint_dir: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load an unsharded olmo.model.Molmo checkpoint with a much lower peak-RAM footprint."""
    from olmo.config import ModelConfig
    from olmo.model import Molmo
    from olmo.util import resource_path

    if Path(join(checkpoint_dir, "model.yaml")).exists():
        model_config = ModelConfig.load(Path(join(checkpoint_dir, "model.yaml")))
    else:
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

    state_dict_path = resource_path(checkpoint_dir, "model.pt")
    if not Path(state_dict_path).is_file():
        raise FileNotFoundError(
            f"{state_dict_path} not found. load_checkpoint_low_memory only supports "
            f"unsharded checkpoints (a single model.pt) -- for a sharded/FSDP "
            f"checkpoint directory, use olmo.model.Molmo.from_checkpoint instead."
        )

    logger.info("Building model on the meta device (no real memory allocated yet)")
    model_config.init_device = "meta"
    model = Molmo(model_config, init_params=False)

    logger.info(
        "Memory-mapping %s (pages loaded lazily, not read into RAM up front)", state_dict_path
    )
    state_dict = torch.load(state_dict_path, map_location="cpu", mmap=True)

    compatible_state_dict, _ = model._make_state_dict_compatible(state_dict)

    model_keys = set(model.state_dict().keys())
    compatible_state_dict = _remap_dinov3_keys(compatible_state_dict, model_keys)

    # Cast every tensor to the target dtype BEFORE assigning, not after. Checkpoints
    # trained under mixed precision (config.yaml: precision: amp_bf16) can save
    # different parameters at different dtypes (e.g. attention v_proj in bfloat16,
    # q_proj/k_proj in float32, for numerical-stability reasons). The *default*
    # load_state_dict path (olmo.model.Molmo.from_checkpoint's non-low-memory path)
    # never hits this: it copies values into pre-existing float32 tensors, which
    # silently normalizes dtype during the copy. assign=True instead keeps each
    # tensor's original saved dtype, so relying on a single blanket model.to(dtype)
    # call afterward is not reliable -- confirmed on a real checkpoint: it left
    # some assigned (memory-mapped) tensors at their original dtype, producing a
    # "query.dtype: float ... value.dtype: bfloat16" error inside SDPA. Casting
    # here, before assign, guarantees every parameter enters the model already
    # uniform.
    compatible_state_dict = {key: value.to(dtype=dtype) for key, value in compatible_state_dict.items()}

    logger.info("Assigning weights directly in place of meta placeholders (no double allocation)")
    try:
        model.load_state_dict(compatible_state_dict, strict=True, assign=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Loading failed even after remapping the known DINOv3 '.model.' key "
            "naming difference (see come_cbir/checkpoint_loading.py:_remap_dinov3_keys). "
            "This means there is at least one *additional* key mismatch beyond the one "
            "this loader already knows how to fix. Please report the full error below "
            "(not truncated) so the remapper can be extended -- refusing to fall back to "
            "strict=False here, since on the meta device that would silently leave some "
            "parameters uninitialized instead of loading real weights.\n\n" + str(exc)
        ) from exc

    logger.info("Moving to %s (dtype already unified to %s before assignment)", device, dtype)
    model = model.to(device=torch.device(device))
    return model.eval()
