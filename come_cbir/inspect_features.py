"""
CLI: optional intermediate-feature diagnostics for a single image. Not
required for normal extraction (each artifact here costs extra forward
passes through the raw per-layer hidden states, on top of the extraction
path used by extract_features.py).

    python -m come_cbir.inspect_features \\
        --image examples/query.jpg \\
        --checkpoint /models/come-vl \\
        --output outputs/diagnostics

Writes:
  - siglip_layer_entropy.csv, dino_layer_entropy.csv (per-layer pooled-feature
    "activation entropy" -- see note below -- plus the L2 norm of the
    per-layer mean-pooled feature)
  - selected_layers.json (idx1/idx2 actually read from the model, i.e. the
    real fixed ranges described in docs/cbir_architecture_analysis.md
    section 3, not re-derived or guessed here)
  - projected_tokens.json, fused_tokens_summary.json
  - descriptor_norms.json (pre-/post-L2-normalization norms)
  - layer_entropy.png (if matplotlib is available)

Note on "entropy": grepping the repository (see
docs/cbir_architecture_analysis.md section 3) turns up no runtime entropy
computation anywhere in olmo/ -- the "entropy-guided" layer ranges (idx1,
idx2) are fixed constants, presumably chosen offline. There is therefore no
official formula this script can reproduce. What is computed below is a
*diagnostic proxy* only: per layer, mean-pool the tokens, softmax the
pooled vector over its feature dimension, and take the Shannon entropy of
that distribution. This is a reasonable "how peaked/uniform is this layer's
activation" signal for exploratory plots -- it is explicitly NOT claimed to
be the metric that produced idx1/idx2.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from come_cbir.feature_extractor import CoMECbirFeatureExtractor, _siglip_preprocess_single_crop
from come_cbir.utils import setup_logging

logger = logging.getLogger("come_cbir.inspect_features")


def _activation_entropy(pooled_vector: torch.Tensor) -> float:
    """Diagnostic proxy entropy of a softmax'd pooled feature vector. See module docstring."""
    probs = F.softmax(pooled_vector.float(), dim=-1)
    probs = probs.clamp(min=1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return float(entropy.mean().item())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output", type=str, required=True)
    return parser


def run_diagnostics(image_path: str, extractor: CoMECbirFeatureExtractor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = extractor.model
    vb = model.vision_backbone
    device = torch.device(extractor.config.device)
    dtype = extractor._torch_dtype

    image = Image.open(image_path).convert("RGB")
    patches = _siglip_preprocess_single_crop(
        image, output_size=(extractor.config.image_size, extractor.config.image_size),
        patch_size=extractor.config.patch_size,
    )
    images_flat = torch.from_numpy(patches).unsqueeze(0).to(device=device, dtype=dtype)  # (1, N, D_pixels)
    images_bt = images_flat.unsqueeze(1)  # (1, 1, N, D_pixels)

    with torch.inference_mode():
        # --- per-layer raw hidden states (extra forward passes, diagnostics-only) ---
        siglip_hidden_states = vb.image_vit(images_flat)  # list[27] of (1, 576, 1152)
        dino_hidden_states = vb.image_vit2(images_flat)["hidden_states"]  # tuple[25] of (1, 201, 1024)

        siglip_rows = []
        for i, layer in enumerate(siglip_hidden_states):
            pooled = layer.float().mean(dim=1)  # (1, 1152)
            siglip_rows.append({
                "layer_index_0based": i,
                "entropy_proxy": _activation_entropy(pooled),
                "pooled_l2_norm": float(pooled.norm(dim=-1).mean().item()),
            })

        dino_rows = []
        for i, layer in enumerate(dino_hidden_states):
            # layer 0 is the embedding output and still has the 5 prefix tokens; strip
            # them for layers 1.. to match olmo/model.py:1896's convention, keep layer 0
            # as-is (embeddings) since it's outside the pool2 selection anyway.
            tokens = layer[:, 5:, :] if i > 0 else layer
            pooled = tokens.float().mean(dim=1)
            dino_rows.append({
                "layer_index_hf": i,
                "entropy_proxy": _activation_entropy(pooled),
                "pooled_l2_norm": float(pooled.norm(dim=-1).mean().item()),
            })

        # --- OL-mixed (entropy-selected-range + orthogonal-mixed) descriptors ---
        feat_siglip, feat_dino = vb.encode_image(images_bt)
        siglip_ol_pooled = feat_siglip.squeeze(1).float().mean(dim=1)  # (1, 1152)
        dino_ol_pooled = feat_dino.squeeze(1).float().mean(dim=1)      # (1, 1024)

        # --- projected into the shared RGCA space using the model's own proj_sig/proj_din ---
        proj_siglip = vb.cross_rope.ln_sig(vb.cross_rope.proj_sig(siglip_ol_pooled))
        proj_dino = vb.cross_rope.ln_din(vb.cross_rope.proj_din(dino_ol_pooled))
        cosine_sim_projected = float(
            F.cosine_similarity(proj_siglip.float(), proj_dino.float(), dim=-1).mean().item()
        )

        # --- RGCA-fused tokens ---
        num_patch = images_bt.shape[2]
        image_masks = torch.ones(1, 1, num_patch, device=device, dtype=dtype)
        fused = vb(images_bt, image_masks).squeeze(1)  # (1, 144, d_model)
        fused_pooled = fused.float().mean(dim=1)

        # --- final configured descriptor, pre/post L2 norm ---
        descriptor_pre_norm, intermediates = extractor.encode_images(
            torch.from_numpy(patches).unsqueeze(0), return_intermediates=True
        )
        # encode_images already L2-normalizes internally; recompute the pre-norm
        # descriptor's own norm from its pooled (unnormalized) tokens for the report.
        pre_norm_value = None
        for key in ("tokens_fused", "tokens_siglip"):
            if key in intermediates:
                pre_norm_value = float(intermediates[key].float().mean(dim=1).norm(dim=-1).mean().item())
                break
        post_norm_value = float(descriptor_pre_norm.norm(dim=-1).mean().item())

    _write_csv(output_dir / "siglip_layer_entropy.csv", siglip_rows)
    _write_csv(output_dir / "dino_layer_entropy.csv", dino_rows)

    with open(output_dir / "selected_layers.json", "w") as f:
        json.dump({"siglip_idx1": list(vb.idx1), "dino_idx2": list(vb.idx2)}, f, indent=2)

    with open(output_dir / "projected_tokens.json", "w") as f:
        json.dump({
            "projected_siglip_norm": float(proj_siglip.norm(dim=-1).mean().item()),
            "projected_dino_norm": float(proj_dino.norm(dim=-1).mean().item()),
            "cosine_similarity_projected_siglip_vs_dino": cosine_sim_projected,
        }, f, indent=2)

    with open(output_dir / "fused_tokens_summary.json", "w") as f:
        json.dump({
            "fused_token_grid_shape": list(fused.shape),
            "fused_pooled_norm": float(fused_pooled.norm(dim=-1).mean().item()),
        }, f, indent=2)

    with open(output_dir / "descriptor_norms.json", "w") as f:
        json.dump({
            "pre_normalization_norm": pre_norm_value,
            "post_normalization_norm": post_norm_value,
            "descriptor_mode": extractor.config.descriptor_mode,
            "pooling": extractor.config.pooling,
        }, f, indent=2)

    _maybe_plot(siglip_rows, dino_rows, output_dir / "layer_entropy.png")
    logger.info("Wrote diagnostics to %s", output_dir)


def _write_csv(path: Path, rows: List[dict]) -> None:
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _maybe_plot(siglip_rows: List[dict], dino_rows: List[dict], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping layer_entropy.png")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["layer_index_0based"] for r in siglip_rows], [r["entropy_proxy"] for r in siglip_rows],
            marker="o", label="SigLIP2 (per block output)")
    ax.plot([r["layer_index_hf"] for r in dino_rows], [r["entropy_proxy"] for r in dino_rows],
            marker="s", label="DINOv3 (HF hidden_states, 0=embeddings)")
    ax.set_xlabel("layer index")
    ax.set_ylabel("activation entropy proxy (diagnostic only, see docstring)")
    ax.set_title("Per-layer activation entropy proxy")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setup_logging()

    extractor = CoMECbirFeatureExtractor(
        model_name_or_path=args.checkpoint,
        descriptor_mode="come_fused",
        pooling="mean",
        device=args.device,
        dtype=args.dtype,
    )
    run_diagnostics(args.image, extractor, Path(args.output))


if __name__ == "__main__":
    main()
