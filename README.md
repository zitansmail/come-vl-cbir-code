# come_cbir

Code accompanying the article **"Understanding the Limits of Fusion, Adaptation, and Graph Combination for Open-Set Content-Based Image Retrieval"** (IEEE Access).

This package holds the extraction and evaluation pipeline, the ARP/ARP-NP adapter and training code, and the graph-fusion validation study built on top of the unmodified, publicly released CoME-VL code. It does **not** include the CoME-VL checkpoint or model source itself — see the article's Methodology and "Data and Code Availability" sections for those.

## Contents

- `come_cbir/` — the package: feature extraction, indexing/retrieval, the ARP/ARP-NP adapter (`retrieval_adapter.py`, `train_adapter.py`), class-split generation (`generate_class_splits.py`), the graph-fusion validation study (`validate_graph_fusion.py`), evaluation (`evaluate.py`), and supporting utilities.
- `tests/come_cbir/` — the accompanying pytest test suite.

## Datasets

Corel-1K, Corel-10K, and Caltech-101 are all publicly available benchmark datasets used in this work; they are not redistributed here.

## Installation

```bash
pip install -r requirements.txt
```

## Running the tests

```bash
pytest tests/come_cbir
```

## Citation

If you use this code, please cite the accompanying article (see the repository this code was extracted from, [CoME-VL](https://github.com/mbzuai-oryx/CoME-VL), for the base model and checkpoint).
