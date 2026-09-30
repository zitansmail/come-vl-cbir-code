"""
come_cbir: a content-based image retrieval (CBIR) extension for CoME-VL.

This package reuses the official SigLIP2 / DINOv3 visual encoders and the
CoME entropy-selected-layer / orthogonal-mixing / RGCA fusion modules that
live under ``olmo/`` (see ``docs/cbir_architecture_analysis.md``) to build
global image descriptors for retrieval. It never runs the Qwen2 language
decoder.

Nothing under ``olmo/`` is modified by this package.
"""

__version__ = "0.1.0"
