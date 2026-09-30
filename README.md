<div align="center">

# come_cbir

**Open-set content-based image retrieval on top of CoME-VL — extraction pipeline, ARP/ARP-NP adapter, and graph-fusion validation study.**

Code accompanying the IEEE Access article *"Understanding the Limits of Fusion, Adaptation, and Graph Combination for Open-Set Content-Based Image Retrieval."*

[![tests](https://github.com/zitansmail/come-vl-cbir-code/actions/workflows/tests.yml/badge.svg)](https://github.com/zitansmail/come-vl-cbir-code/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

</div>

## What this is

[CoME-VL](https://github.com/mbzuai-oryx/CoME-VL) fuses a contrastive vision-language encoder (SigLIP2) with a self-supervised one (DINOv3) through a trained cross-attention module. This repository is the code behind four falsifiable investigations into whether that fusion actually helps content-based image retrieval (CBIR), and is not itself a proposed retrieval method:

1. **Fusion vs. single-encoder retrieval.** A four-way comparison on Corel-1K, Corel-10K, and Caltech-101 — the trained fusion never wins a single metric on any of the three benchmarks despite costing 3.1× the dimensionality of SigLIP2 alone.
2. **ARP (Adaptive Retrieval Projection).** A supervised-contrastive adapter that improves retrieval under a seen-class protocol but degrades under a class-disjoint unseen-class protocol — the closed-set/open-set gap the paper centers on.
3. **ARP-NP.** A neighborhood-preservation regularizer testing whether destroyed local geometry explains that degradation — a validation-tuned, multi-seed, leakage-safe sweep finds no consistent improvement.
4. **Graph-fusion pre-registration study.** Before implementing any graph-fusion method, a pre-registered test of whether the four frozen descriptor spaces carry complementary, exploitable information, using a budget-matched oracle-headroom metric, bootstrap confidence intervals, and multiple-testing correction. Complementarity is statistically detectable but falls ~5.5× below the pre-registered threshold, so graph fusion is not pursued.

This code does **not** include the CoME-VL model source or checkpoint (unmodified and publicly released — see the article's Methodology) or the datasets (Corel-1K, Corel-10K, Caltech-101 — public benchmarks, not redistributed here).

## Package layout

| Module | Role |
|---|---|
| `feature_extractor.py`, `extract_features.py` | Run CoME-VL / SigLIP2 / DINOv3 to extract pooled descriptors from an image dataset |
| `checkpoint_loading.py` | Low-memory checkpoint loading (meta-device construction, mmap, in-place weight assignment) and DINOv3 remote-code key remapping |
| `dtype_patches.py` | Runtime patch for the RGCA cross-attention RoPE dtype bug |
| `pooling.py` | Pooling strategies over patch tokens (mean, max, CLS, GeM, attention) |
| `dimensionality.py` | PCA fitting/application (`apply_pca.py`) |
| `indexing.py`, `build_index.py` | Nearest-neighbor index backends and index construction |
| `query.py` | Single-image query against a built index, with optional adapter and result-grid rendering |
| `evaluate.py` | Retrieval metrics (recall@k, mAP) over a full split or a held-out subset |
| `retrieval_adapter.py`, `train_adapter.py`, `apply_adapter.py` | The ARP / ARP-NP supervised-contrastive adapter: definition, training, and inference-time application |
| `generate_class_splits.py` | Leakage-safe seen/unseen class splits for the open-set protocol |
| `geometry_diagnostics.py` | Neighborhood-preservation diagnostics used to interpret ARP-NP |
| `validate_graph_fusion.py` | The pre-registered oracle-headroom complementarity study (bootstrap CIs, multiple-testing correction) |
| `run_experiments.py`, `run_arp_np_controls.py`, `tune_arp_np.py` | Experiment orchestration scripts |
| `config.py`, `utils.py`, `datasets.py`, `geometry_diagnostics.py`, `inspect_features.py` | Configuration, logging/IO helpers, dataset loading, feature inspection |

## Six implementation issues fixed at inference time

Running the released CoME-VL checkpoint for CBIR surfaced six previously unreported issues, none anticipated from reading the code — each is worked around at the pipeline level rather than by modifying the checkpoint itself. Full detail is in the modules linked below; the article's appendix summarizes the same table.

| # | Issue | Where | Fix |
|---|---|---|---|
| 1 | Peak memory footprint (tens of GB) at load time | [`checkpoint_loading.py`](come_cbir/checkpoint_loading.py) (`load_checkpoint_low_memory`) | Meta-device construction + mmap + `assign=True` |
| 2 | Key-naming mismatch | [`checkpoint_loading.py`](come_cbir/checkpoint_loading.py) (`_remap_dinov3_keys`) | Pattern-based key remapper (rename only) |
| 3 | Attention dtype mismatch (SigLIP2, encoder 1) | [`feature_extractor.py`](come_cbir/feature_extractor.py) | Disable the `float32_attention` config flag |
| 4 | Padding-embedding dtype corruption | [`feature_extractor.py`](come_cbir/feature_extractor.py) | Disable the `image_padding_embed` option |
| 5 | Attention dtype mismatch (RGCA cross-attention, fusion) | [`dtype_patches.py`](come_cbir/dtype_patches.py) (`patch_cross_rope_dtype_bug`) | Preserve input dtype in RoPE coordinate computation |
| 6 | Silent all-NaN descriptors at reduced precision | [`feature_extractor.py`](come_cbir/feature_extractor.py) | Force full precision for the vision backbone + explicit NaN/Inf check |

Issue 6 is the most consequential: after fixing 1–5, extraction completed without error but produced invalid (NaN) descriptors for every input, because the same numerical-stability flag disabled for issue 3 turned out to be load-bearing for stability across the encoders' depth. It is also the only one of the six that could have gone completely undetected without an explicit validity check.

## Installation

`come_cbir` calls into `olmo` (the CoME-VL / Molmo package) for checkpoint loading — install that first, then this package on top:

```bash
# 1. CoME-VL / Molmo, unmodified (provides the `olmo` package)
git clone https://github.com/mbzuai-oryx/CoME-VL.git
pip install -e ./CoME-VL

# 2. this package
git clone https://github.com/zitansmail/come-vl-cbir-code.git
cd come-vl-cbir-code
pip install -r requirements.txt
```

or as an editable package:

```bash
pip install -e .
```

## Usage

All scripts are runnable as modules from the repository root: `python -m come_cbir.<script> --help`.

**1. Extract descriptors**

```bash
python -m come_cbir.extract_features \
  --dataset-root /path/to/corel1k \
  --checkpoint /path/to/come-vl-checkpoint \
  --descriptor-mode come_fused \
  --pooling mean \
  --device cuda \
  --output-dir runs/corel1k_fused
```

`--descriptor-mode` selects which encoder(s) back the descriptor (e.g. the fused CoME-VL representation vs. SigLIP2 alone); `--dtype` controls extraction precision — the vision backbone itself is always forced to float32 internally (see issue 6 above) regardless of this flag.

**2. Build an index and evaluate**

```bash
python -m come_cbir.build_index --features runs/corel1k_fused/features.npy \
  --backend faiss-flat --output runs/corel1k_fused/index.bin

python -m come_cbir.evaluate --features runs/corel1k_fused/features.npy \
  --labels runs/corel1k_fused/labels.npy --paths runs/corel1k_fused/paths.txt \
  --top-k 10 --output-dir runs/corel1k_fused/eval
```

**3. Query a single image**

```bash
python -m come_cbir.query --image query.jpg \
  --checkpoint /path/to/come-vl-checkpoint \
  --index runs/corel1k_fused/index.bin --index-backend faiss-flat \
  --top-k 10 --output runs/corel1k_fused/query_result.png
```

**4. Open-set class splits, then train/apply the ARP adapter**

```bash
python -m come_cbir.generate_class_splits --labels runs/corel1k_fused/labels.npy \
  --output-dir runs/corel1k_fused/splits --seeds 0 1 2

python -m come_cbir.train_adapter --features runs/corel1k_fused/features.npy \
  --labels runs/corel1k_fused/labels.npy --output-dir runs/corel1k_fused/adapter \
  --adapter-type arp --holdout-mode class --lambda-np 0.0

python -m come_cbir.apply_adapter --features runs/corel1k_fused/features.npy \
  --adapter runs/corel1k_fused/adapter/adapter.pt --output runs/corel1k_fused/features_adapted.npy
```

Set `--lambda-np > 0` to enable the ARP-NP neighborhood-preservation regularizer.

**5. Graph-fusion complementarity study**

```bash
python -m come_cbir.validate_graph_fusion \
  --descriptors runs/siglip2/features.npy runs/dinov3/features.npy \
    runs/come_fused/features.npy runs/adapter/features_adapted.npy \
  --labels runs/corel1k_fused/labels.npy --paths runs/corel1k_fused/paths.txt \
  --unseen-splits runs/corel1k_fused/splits/seed0_test.json \
  --output-dir runs/graph_fusion_study
```

## Testing

```bash
pip install -r requirements.txt pytest
pytest tests/come_cbir
```

The test suite (`tests/come_cbir/`) covers every module above against a small mock model fixture (`conftest.py`) and does not require a real checkpoint or GPU — CoME-VL/`olmo` (see Installation) must still be importable, since a couple of tests exercise the real dataset/config classes it provides. All 176 tests pass as of this release.

## Data and code availability

Corel-1K, Corel-10K, and Caltech-101 are publicly available benchmark datasets commonly used in prior CBIR and image-classification literature; the CoME-VL checkpoint used is publicly released. This repository is the openly available `come_cbir` package referenced in the article's Data and Code Availability statement, built on top of the unmodified, publicly released [CoME-VL](https://github.com/mbzuai-oryx/CoME-VL) code.

## Citation

```bibtex
@article{zitane_come_cbir_2026,
  title   = {Understanding the Limits of Fusion, Adaptation, and Graph Combination for Open-Set Content-Based Image Retrieval},
  author  = {Zitane, Smail and Zeroual, Imad and Agoujil, Said},
  journal = {IEEE Access},
  year    = {2026}
}
```

See [CITATION.cff](CITATION.cff) for the software citation.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Authors

- Smail Zitane — L-STI, T-IDMS, FST Errachidia, Moulay Ismail University, Meknes, Morocco (zitansmail@gmail.com)
- Imad Zeroual — L-STI, T-IDMS, FST Errachidia, Moulay Ismail University, Meknes, Morocco
- Said Agoujil — MMIS, MAIS, FST Errachidia, Moulay Ismail University, Meknes, Morocco
