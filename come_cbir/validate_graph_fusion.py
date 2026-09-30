"""
CLI: the smallest possible validation study for the "cross-modal graph fusion"
research direction -- determines whether it is even worth designing a full
algorithm for, BEFORE any fusion/diffusion method is built.

    python -m come_cbir.validate_graph_fusion \
        --descriptors siglip:/content/outputs/corel1k_v2/siglip/features.npy \
                      dino:/content/outputs/corel1k_v2/dino/features.npy \
                      concat:/content/outputs/corel1k_v2/concat/features.npy \
                      come_fused:/content/outputs/corel1k_v2/come_fused/features.npy \
        --labels /content/outputs/corel1k_v2/siglip/labels.npy \
        --paths /content/outputs/corel1k_v2/siglip/paths.json \
        --output-dir outputs/graph_fusion_validation/corel1k \
        --k-values 5 10 20 --primary-k 10 \
        --unseen-splits outputs/arp_np/corel1k/siglip/seed_0.json \
                        outputs/arp_np/corel1k/siglip/seed_1.json \
                        outputs/arp_np/corel1k/siglip/seed_2.json

Entirely training-free and read-only over already-extracted feature files --
no adapters, no gradients, no leakage to manage in the training sense. Every
descriptor mode's k-NN graph is built once per (scope, k) via
`retrieval_adapter.build_knn_graph`, reused as-is.

PROTOCOL, MATCHED EXACTLY TO run_arp_np_controls.py's unseen-class evaluation:
when a scope is restricted to a set of indices (the "overall" scope uses all
indices; each "unseen_*" scope uses one class-holdout split's test indices),
BOTH the query set and the gallery/neighbour-search universe are restricted
to exactly that subset together -- i.e. `features[indices]`, `labels[indices]`,
`[paths[i] for i in indices]` become a new, self-contained retrieval universe,
identical to what `run_arp_np_controls._evaluate_subset` does. Restricting
only the query set while leaving the gallery as the full dataset (an earlier,
incorrect version of this script did exactly that) would silently let seen
classes leak into the neighbour search for an "unseen-class" analysis.

Two hypotheses:

  H1 (complementarity): different frozen descriptor spaces' k-NN neighbour
  sets retrieve complementary relevant neighbours -- measured via mean
  pairwise overlap (redundancy) and two SEPARATE diagnostics:
    (a) Candidate Coverage Gain -- do complementary relevant candidates exist
        at all, ignoring any retrieval budget (uncapped by k). Diagnostic
        only; NOT a claim about retrieval improvement.
    (b) Budget-Matched OracleHeadroom@k -- the SAME retrieval budget k is
        used for the union and for every single descriptor. This is the
        primary go/no-go complementarity metric.

  H2 (confidence): per-query agreement between descriptor spaces correlates
  with retrieval quality (full_average_precision). Two signals are reported:
  agreement EXCLUDING the target mode (a confidence proxy usable without ever
  touching the target's own output) and agreement INCLUDING the target (what
  a real fusion method could actually observe at inference time).

Ground-truth labels are used only to compute these diagnostics. They must
never be used inside an eventual retrieval algorithm.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from come_cbir.evaluate import evaluate_retrieval
from come_cbir.retrieval_adapter import build_knn_graph
from come_cbir.utils import setup_logging


# ----------------------------------------------------------------------
# Loading, alignment validation
# ----------------------------------------------------------------------


def _load_descriptor(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _parse_descriptor_args(items: List[str]) -> Dict[str, Dict[str, str]]:
    """Each entry is 'name:features_path' or 'name:features_path:labels_path:paths_path'.
    The 4-part form lets a per-mode labels/paths file be cross-checked against
    the shared reference for alignment -- this is what actually catches the
    case where each descriptor mode was extracted into its own directory with
    its own (supposedly identical, but never verified) labels.npy/paths.json.
    """
    result: Dict[str, Dict[str, str]] = {}
    for item in items:
        parts = item.split(":")
        if len(parts) == 2:
            name, feat_path = parts
            result[name] = {"features": feat_path}
        elif len(parts) == 4:
            name, feat_path, labels_path, paths_path = parts
            result[name] = {"features": feat_path, "labels": labels_path, "paths": paths_path}
        else:
            raise ValueError(
                f"--descriptors entries must be 'NAME:FEATURES' or 'NAME:FEATURES:LABELS:PATHS', got '{item}'"
            )
    return result


def validate_alignment(
    names: List[str], descriptors: Dict[str, np.ndarray],
    reference_labels: np.ndarray, reference_paths: List[str],
    per_mode: Dict[str, Dict[str, str]],
) -> None:
    """Raises ValueError on any row-count, label, or path mismatch across
    descriptor modes. This is a correctness gate, not a warning -- silent
    misalignment here would invalidate every downstream statistic."""
    n_ref = len(reference_labels)
    if len(reference_paths) != n_ref:
        raise ValueError(f"--labels has {n_ref} rows but --paths has {len(reference_paths)} entries")

    for name in names:
        n_feat = descriptors[name].shape[0]
        if n_feat != n_ref:
            raise ValueError(
                f"Descriptor '{name}' has {n_feat} rows but --labels/--paths reference has {n_ref} rows "
                f"-- feature row count must match exactly across all descriptor modes."
            )
        mode_info = per_mode.get(name, {})
        if "labels" in mode_info:
            mode_labels = np.load(mode_info["labels"])
            if mode_labels.shape != reference_labels.shape or not np.array_equal(mode_labels, reference_labels):
                raise ValueError(
                    f"Descriptor '{name}'s own labels file ({mode_info['labels']}) does not match the "
                    f"reference --labels array row-for-row -- descriptor modes are not aligned."
                )
        if "paths" in mode_info:
            with open(mode_info["paths"]) as f:
                mode_paths = json.load(f)
            if mode_paths != reference_paths:
                raise ValueError(
                    f"Descriptor '{name}'s own paths file ({mode_info['paths']}) does not match the "
                    f"reference --paths list entry-for-entry -- descriptor modes are not aligned."
                )


def report_dataset_integrity(names: List[str], descriptors: Dict[str, np.ndarray], labels: np.ndarray, paths: List[str], logger) -> None:
    """Step 1/2 pre-flight report: shapes, dims, NaN/Inf, duplicate rows,
    class counts. Does not raise on its own (validate_alignment already
    raises on real misalignment) -- this is the human-readable inventory."""
    logger.info("===== Dataset integrity report =====")
    logger.info("n_labels=%d n_paths=%d n_unique_paths=%d", len(labels), len(paths), len(set(paths)))
    unique_classes, counts = np.unique(labels, return_counts=True)
    logger.info("n_classes=%d, samples_per_class: min=%d max=%d mean=%.1f",
                len(unique_classes), int(counts.min()), int(counts.max()), float(counts.mean()))
    for cls, cnt in zip(unique_classes, counts):
        if cnt == 0:
            logger.warning("Class %s has 0 samples -- empty class present in the dataset", cls)

    for name in names:
        feats = descriptors[name]
        n_nan = int(np.isnan(feats).sum())
        n_inf = int(np.isinf(feats).sum())
        n_unique_rows = len({tuple(row) for row in feats[: min(len(feats), 2000)].round(6)})  # capped for cost
        logger.info(
            "descriptor '%s': shape=%s dim=%d dtype=%s NaN=%d Inf=%d (dup-check on first %d rows: %d unique)",
            name, feats.shape, feats.shape[1], feats.dtype, n_nan, n_inf,
            min(len(feats), 2000), n_unique_rows,
        )
        if n_nan > 0 or n_inf > 0:
            logger.warning("descriptor '%s' contains NaN/Inf values -- results downstream will be unreliable", name)


def report_split_integrity(split_path: str, labels: np.ndarray, logger) -> Dict:
    """Step 3 pre-flight report for a single unseen-class split file. Raises
    ValueError on hard violations (out-of-range/duplicate indices); does NOT
    silently repair anything."""
    with open(split_path) as f:
        raw = json.load(f)

    indices = load_restrict_indices(split_path, labels)
    n = len(labels)

    if len(indices) != len(set(indices.tolist())):
        raise ValueError(f"{split_path}: duplicate indices found in the restricted set")
    if indices.size and (indices.min() < 0 or indices.max() >= n):
        raise ValueError(f"{split_path}: indices out of range [0, {n}) -- min={indices.min()}, max={indices.max()}")

    selected_classes = sorted(set(int(c) for c in labels[indices]))
    for cls in selected_classes:
        if int(np.sum(labels[indices] == cls)) == 0:
            raise ValueError(f"{split_path}: class {cls} listed but has 0 selected images")

    train_test_overlap = None
    if isinstance(raw, dict) and "test_classes" in raw:
        test_classes = set(raw.get("test_classes", []))
        train_classes = set(raw.get("split", {}).get("train_classes", raw.get("train_classes", [])))
        val_classes = set(raw.get("split", {}).get("val_classes", raw.get("val_classes", [])))
        overlap = (test_classes & train_classes) | (test_classes & val_classes)
        train_test_overlap = sorted(overlap)
        if overlap:
            raise ValueError(f"{split_path}: class overlap between test and train/val classes: {overlap}")

    report = {
        "file": split_path,
        "n_selected_images": int(len(indices)),
        "n_classes": len(selected_classes),
        "classes": selected_classes,
        "train_test_class_overlap": train_test_overlap,
    }
    logger.info(
        "split '%s': %d images, %d classes %s, train/test overlap=%s",
        split_path, report["n_selected_images"], report["n_classes"], selected_classes, train_test_overlap,
    )
    return report


# ----------------------------------------------------------------------
# Restricted-scope (unseen-class) index loading -- supports both file
# formats actually produced by this project's existing tooling.
# ----------------------------------------------------------------------


def load_restrict_indices(path: str, labels: np.ndarray) -> np.ndarray:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return np.array(raw, dtype=np.int64)
    if "test_indices" in raw:
        return np.array(raw["test_indices"], dtype=np.int64)
    if "holdout_indices" in raw:
        return np.array(raw["holdout_indices"], dtype=np.int64)
    if "test_classes" in raw:
        # run_arp_np_controls.py's seed_N.json only saves class lists, not
        # indices -- reconstruct indices by matching labels to those classes.
        return np.where(np.isin(labels, raw["test_classes"]))[0].astype(np.int64)
    raise ValueError(f"Could not find test_indices/holdout_indices/test_classes in {path}")


# ----------------------------------------------------------------------
# Core per-query metrics
# ----------------------------------------------------------------------


def _pairwise_overlap(neighbors_a: np.ndarray, neighbors_b: np.ndarray, k: int) -> np.ndarray:
    """Per-query |top-k(A) intersect top-k(B)| / k."""
    n = neighbors_a.shape[0]
    overlap = np.zeros(n, dtype=np.float64)
    for i in range(n):
        overlap[i] = len(set(neighbors_a[i].tolist()) & set(neighbors_b[i].tolist())) / k
    return overlap


def _same_label_hits(neighbor_indices: np.ndarray, labels: np.ndarray, query_labels: np.ndarray) -> np.ndarray:
    n = neighbor_indices.shape[0]
    hits = np.zeros(n, dtype=np.int64)
    for i in range(n):
        hits[i] = int(np.sum(labels[neighbor_indices[i]] == query_labels[i]))
    return hits


def _union_hits(neighbor_indices_by_mode: List[np.ndarray], labels: np.ndarray, query_labels: np.ndarray) -> np.ndarray:
    n = query_labels.shape[0]
    hits = np.zeros(n, dtype=np.int64)
    for i in range(n):
        union_set = set().union(*(set(neighbor_indices_by_mode[m][i].tolist()) for m in range(len(neighbor_indices_by_mode))))
        hits[i] = sum(1 for idx in union_set if labels[idx] == query_labels[i])
    return hits


def compute_candidate_coverage_gain(
    single_hits: Dict[str, np.ndarray], union_hits: np.ndarray, num_relevant: np.ndarray,
) -> np.ndarray:
    """Diagnostic ONLY -- not budget-matched, denominator is the raw total
    relevant count R_q (uncapped by any retrieval budget k). Measures whether
    complementary relevant candidates exist across descriptor spaces AT ALL,
    independent of whether a real k-sized retrieval budget could reach them.
    Must never be described as a retrieval-quality improvement."""
    safe_r = np.clip(num_relevant, 1, None)
    coverage_single = {name: hits / safe_r for name, hits in single_hits.items()}
    best_single_coverage = np.maximum.reduce(list(coverage_single.values()))
    coverage_union = union_hits / safe_r
    return coverage_union - best_single_coverage  # >= 0 always: union is a superset of every single mode


def compute_budget_matched_headroom(
    single_hits: Dict[str, np.ndarray], union_hits: np.ndarray, num_relevant: np.ndarray, k: int,
) -> np.ndarray:
    """Primary go/no-go complementarity metric. Same denominator min(k, R_q)
    for the union and for every single descriptor, and the union's numerator
    is capped at k via min(k, union_hits) -- so pooling more descriptor modes
    (a larger raw union set) can never inflate this beyond what a k-sized
    retrieval budget could actually deliver. Ground truth is used only here,
    as a diagnostic; it must never be used inside the real retrieval algorithm."""
    denom_k = np.clip(np.minimum(num_relevant, k), 1, None)
    recall_single = {name: hits / denom_k for name, hits in single_hits.items()}  # hits already <= k
    best_single_recall = np.maximum.reduce(list(recall_single.values()))
    oracle_recall_union = np.minimum(k, union_hits) / denom_k
    return oracle_recall_union - best_single_recall  # >= 0 always: union superset property


# ----------------------------------------------------------------------
# Bootstrap CIs and multiple-testing correction
# ----------------------------------------------------------------------


def bootstrap_ci_mean(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(values)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot[b] = values[idx].mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(x)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        rho, _ = spearmanr(x[idx], y[idx])
        boot[b] = 0.0 if np.isnan(rho) else rho
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def benjamini_hochberg(pvalues: List[float]) -> List[float]:
    """Standard BH step-up procedure. Deterministic: same input always gives
    the same output, no randomness involved."""
    pvals = np.asarray(pvalues, dtype=np.float64)
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = pvals[order]
    corrected_sorted = ranked * n / np.arange(1, n + 1)
    corrected_sorted = np.minimum.accumulate(corrected_sorted[::-1])[::-1]
    corrected_sorted = np.clip(corrected_sorted, 0.0, 1.0)
    corrected = np.empty(n)
    corrected[order] = corrected_sorted
    return corrected.tolist()


# ----------------------------------------------------------------------
# Main per-(scope, k) analysis
# ----------------------------------------------------------------------


def run_validation_for_k(
    descriptors: Dict[str, np.ndarray], labels: np.ndarray, paths: List[str],
    k: int, top_k: int, n_boot: int, boot_seed: int, logger,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """`descriptors`/`labels`/`paths` here are already the FULL scope universe
    (either the whole dataset, or an already-subsetted unseen-class split) --
    this function never restricts anything further itself."""
    names = list(descriptors.keys())
    n = labels.shape[0]
    query_indices = np.arange(n)
    query_labels = labels

    graphs: Dict[str, Dict[str, np.ndarray]] = {}
    per_query_ap: Dict[str, np.ndarray] = {}
    for name in names:
        feats = descriptors[name]
        logger.info("[k=%d] building k-NN graph and evaluating retrieval for '%s' (n=%d)", k, name, feats.shape[0])
        graphs[name] = build_knn_graph(feats, k=k)
        eval_result = evaluate_retrieval(feats, labels, paths, top_k=top_k, compute_full_map=True)
        ap_by_query_index = np.zeros(n, dtype=np.float64)
        for rec in eval_result["per_query"]:
            ap_by_query_index[rec["query_index"]] = rec["full_average_precision"]
        per_query_ap[name] = ap_by_query_index

    # --- Redundancy / agreement ---
    pair_overlap: Dict[str, np.ndarray] = {}
    pairwise_overlap_mean: Dict[str, float] = {}
    pairwise_overlap_ci: Dict[str, Tuple[float, float]] = {}
    for a, b in itertools.combinations(names, 2):
        overlap = _pairwise_overlap(graphs[a]["neighbor_indices"], graphs[b]["neighbor_indices"], k)
        pair_overlap[f"{a}|{b}"] = overlap
        pairwise_overlap_mean[f"{a}|{b}"] = float(overlap.mean())
        pairwise_overlap_ci[f"{a}|{b}"] = bootstrap_ci_mean(overlap, n_boot, boot_seed)
    all_overlaps = np.concatenate(list(pair_overlap.values())) if pair_overlap else np.array([0.0])
    mean_overlap = float(np.mean(list(pairwise_overlap_mean.values())))
    mean_overlap_ci = bootstrap_ci_mean(all_overlaps, n_boot, boot_seed)

    # --- Diagnostic A: Candidate Coverage Gain (uncapped by k, denominator = R_q) ---
    num_relevant = np.array([int(np.sum(labels == lbl)) - 1 for lbl in query_labels])  # R_q, excludes query itself
    single_hits = {name: _same_label_hits(graphs[name]["neighbor_indices"], labels, query_labels) for name in names}
    union_hits = _union_hits([graphs[name]["neighbor_indices"] for name in names], labels, query_labels)

    per_query_coverage_gain = compute_candidate_coverage_gain(single_hits, union_hits, num_relevant)
    coverage_gain = float(np.mean(per_query_coverage_gain))
    coverage_gain_ci = bootstrap_ci_mean(per_query_coverage_gain, n_boot, boot_seed)

    # --- Diagnostic B (primary): Budget-Matched OracleHeadroom@k ---
    per_query_headroom = compute_budget_matched_headroom(single_hits, union_hits, num_relevant, k)
    headroom = float(np.mean(per_query_headroom))
    headroom_ci = bootstrap_ci_mean(per_query_headroom, n_boot, boot_seed)

    # --- H2: agreement (excluding / including target) vs. target's own AP ---
    agreement_excl: Dict[str, np.ndarray] = {}
    agreement_incl: Dict[str, np.ndarray] = {}
    confidence_tests: List[Dict] = []
    for target in names:
        others = [nm for nm in names if nm != target]
        if len(others) >= 2:
            excl = np.zeros(n, dtype=np.float64)
            pair_count = 0
            for a, b in itertools.combinations(others, 2):
                excl += _pairwise_overlap(graphs[a]["neighbor_indices"], graphs[b]["neighbor_indices"], k)
                pair_count += 1
            excl /= pair_count
            agreement_excl[target] = excl

        incl = np.zeros(n, dtype=np.float64)
        for other in others:
            incl += _pairwise_overlap(graphs[target]["neighbor_indices"], graphs[other]["neighbor_indices"], k)
        incl /= max(len(others), 1)
        agreement_incl[target] = incl

        for signal_name, signal_values in (("excluding_target", agreement_excl.get(target)), ("including_target", incl)):
            if signal_values is None:
                continue
            rho, pval = spearmanr(signal_values, per_query_ap[target])
            rho = 0.0 if np.isnan(rho) else float(rho)
            ci_lo, ci_hi = bootstrap_ci_spearman(signal_values, per_query_ap[target], n_boot, boot_seed)
            confidence_tests.append({
                "target": target, "signal": signal_name, "k": k,
                "spearman_rho": rho, "p_value_raw": float(pval),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
            })

    summary = {
        "k": k,
        "n_queries": int(n),
        "pairwise_overlap_at_k": pairwise_overlap_mean,
        "pairwise_overlap_ci": {k2: list(v) for k2, v in pairwise_overlap_ci.items()},
        "mean_overlap": mean_overlap,
        "mean_overlap_ci": list(mean_overlap_ci),
        "candidate_coverage_gain": coverage_gain,
        "candidate_coverage_gain_ci": list(coverage_gain_ci),
        "budget_matched_oracle_headroom_at_k": headroom,
        "budget_matched_oracle_headroom_at_k_ci": list(headroom_ci),
        "mean_full_AP_per_mode": {name: float(np.mean(per_query_ap[name])) for name in names},
        "confidence_tests_raw": confidence_tests,  # p-values corrected globally later, across all scopes/k
    }

    per_query_rows = []
    for qi in range(n):
        row = {"query_index": int(qi), "query_label": int(query_labels[qi])}
        for name in names:
            row[f"ap_{name}"] = float(per_query_ap[name][qi])
        for pair_name, overlap in pair_overlap.items():
            a, b = pair_name.split("|")
            row[f"overlap_{a}_{b}"] = float(overlap[qi])
        for target, arr in agreement_excl.items():
            row[f"agreement_excl_{target}"] = float(arr[qi])
        for target, arr in agreement_incl.items():
            row[f"agreement_incl_{target}"] = float(arr[qi])
        row["candidate_coverage_gain"] = float(per_query_coverage_gain[qi])
        row["oracle_headroom_at_k"] = float(per_query_headroom[qi])
        per_query_rows.append(row)

    return summary, per_query_rows, confidence_tests


# ----------------------------------------------------------------------
# Decision logic
# ----------------------------------------------------------------------


def evaluate_h1_h2(
    summary: Dict, redundancy_threshold: float, headroom_threshold: float,
    rho_threshold: float, alpha: float,
) -> Tuple[bool, bool]:
    h1 = bool(
        summary["mean_overlap"] <= redundancy_threshold
        and summary["budget_matched_oracle_headroom_at_k"] >= headroom_threshold
        and summary["candidate_coverage_gain"] > 0
        and summary["budget_matched_oracle_headroom_at_k_ci"][0] > 0  # CI not dominated by zero
    )
    tests = summary["confidence_tests_raw"]
    if not tests:
        return h1, False
    passing = sum(
        1 for t in tests
        if t["spearman_rho"] >= rho_threshold and t.get("p_value_corrected", t["p_value_raw"]) < alpha
    )
    h2 = passing / len(tests) >= 0.5
    return h1, h2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--descriptors", type=str, nargs="+", required=True,
                         help="NAME:FEATURES or NAME:FEATURES:LABELS:PATHS (4-part form cross-checks alignment)")
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--paths", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--unseen-splits", type=str, nargs="*", default=[],
                         help="One or more split files (train_adapter.py split.json with test_indices/"
                              "holdout_indices, or run_arp_np_controls.py seed_N.json with test_classes). "
                              "Each is analyzed as its own scope, plus one pooled 'unseen_aggregate' scope.")
    parser.add_argument("--redundancy-threshold", type=float, default=0.9)
    parser.add_argument("--headroom-threshold", type=float, default=0.02)
    parser.add_argument("--rho-threshold", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true",
                         help="Run Steps 1-3 (alignment + split integrity reports) and exit -- "
                              "does not run the k-NN/agreement analysis. Use this before committing "
                              "to a full run on real data.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logging()

    descriptor_specs = _parse_descriptor_args(args.descriptors)
    names = list(descriptor_specs.keys())
    logger.info("Resolved descriptor paths:")
    for name, spec in descriptor_specs.items():
        logger.info("  %s: %s", name, spec)
    logger.info("Reference --labels: %s", args.labels)
    logger.info("Reference --paths: %s", args.paths)

    descriptors = {name: _load_descriptor(spec["features"]) for name, spec in descriptor_specs.items()}
    labels = np.load(args.labels)
    with open(args.paths) as f:
        paths = json.load(f)

    validate_alignment(names, descriptors, labels, paths, descriptor_specs)
    logger.info("Alignment validated OK across %d descriptor modes (n=%d)", len(names), len(labels))
    report_dataset_integrity(names, descriptors, labels, paths, logger)

    split_reports = []
    for split_path in args.unseen_splits:
        split_reports.append(report_split_integrity(split_path, labels, logger))

    if args.verify_only:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "verification_report.json", "w") as f:
            json.dump({"descriptor_specs": descriptor_specs, "splits": split_reports}, f, indent=2)
        logger.info("Verify-only mode: wrote verification_report.json, stopping before the real analysis.")
        return

    scopes: List[Tuple[str, np.ndarray]] = [("overall", np.arange(len(labels)))]
    for split_path in args.unseen_splits:
        idx = load_restrict_indices(split_path, labels)
        scope_name = f"unseen_{Path(split_path).stem}"
        scopes.append((scope_name, idx))
        logger.info("Scope '%s': %d indices from %s", scope_name, len(idx), split_path)
    if len(args.unseen_splits) > 1:
        aggregate_idx = np.unique(np.concatenate([
            load_restrict_indices(p, labels) for p in args.unseen_splits
        ]))
        scopes.append(("unseen_aggregate", aggregate_idx))
        logger.info("Scope 'unseen_aggregate': %d pooled unique indices across all provided splits", len(aggregate_idx))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_raw_tests: List[Dict] = []
    all_scope_k_summaries: Dict[str, List[Dict]] = {}

    for scope_name, indices in scopes:
        scope_dir = output_dir / scope_name
        scope_dir.mkdir(parents=True, exist_ok=True)
        scope_descriptors = {name: descriptors[name][indices] for name in names}
        scope_labels = labels[indices]
        scope_paths = [paths[i] for i in indices]

        summaries = []
        for k in args.k_values:
            summary, per_query_rows, tests = run_validation_for_k(
                scope_descriptors, scope_labels, scope_paths, k, args.top_k,
                args.n_bootstrap, args.bootstrap_seed, logger,
            )
            for t in tests:
                t["scope"] = scope_name
            all_raw_tests.extend(tests)
            summaries.append(summary)

            with open(scope_dir / f"summary_k{k}.json", "w") as f:
                json.dump(summary, f, indent=2)
            if per_query_rows:
                with open(scope_dir / f"per_query_k{k}.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(per_query_rows)
        all_scope_k_summaries[scope_name] = summaries

    # Global BH correction across every (scope, k, target, signal) test computed this run.
    raw_pvals = [t["p_value_raw"] for t in all_raw_tests]
    corrected_pvals = benjamini_hochberg(raw_pvals)
    for t, corrected in zip(all_raw_tests, corrected_pvals):
        t["p_value_corrected"] = corrected
    # Re-attach corrected p-values into each summary's confidence_tests_raw so
    # evaluate_h1_h2 (which reads p_value_corrected) sees them.
    tests_by_scope_k: Dict[Tuple[str, int], List[Dict]] = {}
    for t in all_raw_tests:
        tests_by_scope_k.setdefault((t["scope"], t["k"]), []).append(t)
    for scope_name, summaries in all_scope_k_summaries.items():
        for summary in summaries:
            summary["confidence_tests_raw"] = tests_by_scope_k.get((scope_name, summary["k"]), [])

    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(all_raw_tests, f, indent=2)
    with open(output_dir / "all_k_summary.json", "w") as f:
        json.dump(all_scope_k_summaries, f, indent=2)
    with open(output_dir / "validation_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Decision report, primary-k driven, per scope.
    lines = ["# Graph Fusion Validation -- Decision Report", ""]
    overall_h1: Dict[str, bool] = {}
    overall_h2: Dict[str, bool] = {}
    for scope_name, summaries in all_scope_k_summaries.items():
        lines.append(f"## Scope: {scope_name}")
        primary_summary = next((s for s in summaries if s["k"] == args.primary_k), summaries[0])
        for s in summaries:
            h1, h2 = evaluate_h1_h2(s, args.redundancy_threshold, args.headroom_threshold, args.rho_threshold, args.alpha)
            marker = " (PRIMARY)" if s["k"] == args.primary_k else ""
            lines.append(
                f"- k={s['k']}{marker}: mean_overlap={s['mean_overlap']:.3f}, "
                f"headroom={s['budget_matched_oracle_headroom_at_k']:+.4f} "
                f"[{s['budget_matched_oracle_headroom_at_k_ci'][0]:+.4f}, {s['budget_matched_oracle_headroom_at_k_ci'][1]:+.4f}], "
                f"coverage_gain={s['candidate_coverage_gain']:.4f} -> H1={h1}, H2={h2}"
            )
            if s["k"] == args.primary_k:
                overall_h1[scope_name] = h1
                overall_h2[scope_name] = h2
        lines.append("")

    overall_scope_h1 = overall_h1.get("overall", False)
    overall_scope_h2 = overall_h2.get("overall", False)
    unseen_scopes = [s for s in overall_h1 if s.startswith("unseen")]
    unseen_h1 = all(overall_h1[s] for s in unseen_scopes) if unseen_scopes else None
    unseen_h2 = all(overall_h2[s] for s in unseen_scopes) if unseen_scopes else None

    lines.append("## Final verdict (primary k={})".format(args.primary_k))
    lines.append(f"- Overall-data H1={overall_scope_h1}, H2={overall_scope_h2}")
    lines.append(f"- Unseen-class H1={unseen_h1}, H2={unseen_h2}")

    if unseen_scopes and not unseen_h1:
        case = "Case 3: H1 false on unseen classes -> STOP. Graph fusion is not supported by the data."
    elif unseen_scopes and (overall_scope_h1 and not unseen_h1):
        case = "Case 4: positive overall but not on unseen classes -> REJECT for the open-set objective."
    elif unseen_scopes and unseen_h1 and unseen_h2:
        case = "Case 1: H1 and H2 hold on unseen classes -> proceed toward confidence-aware graph fusion."
    elif unseen_scopes and unseen_h1 and not unseen_h2:
        case = "Case 2: H1 holds, H2 does not -> reject confidence-aware routing; static/equal-weight fusion only."
    elif not overall_scope_h1:
        case = "Case 3: H1 false overall -> STOP. Graph fusion is not supported by the data."
    elif overall_scope_h1 and overall_scope_h2:
        case = "Case 1 (overall data only, no unseen-class splits provided): proceed cautiously; re-check on unseen classes before committing."
    else:
        case = "Case 2 (overall data only, no unseen-class splits provided): static fusion only; re-check on unseen classes before committing."

    lines.append(f"- Decision: {case}")
    (output_dir / "decision_report.md").write_text("\n".join(lines) + "\n")

    _write_final_decision_document(
        output_dir, all_scope_k_summaries, overall_h1, overall_h2, unseen_scopes,
        overall_scope_h1, overall_scope_h2, unseen_h1, unseen_h2, case, args,
    )

    logger.info("===== %s =====", case)
    logger.info(
        "Wrote decision_report.md, final_graph_fusion_decision.md, statistical_tests.json, "
        "all_k_summary.json, validation_config.json to %s", output_dir,
    )


def _classify_robustness(summaries: List[Dict], args) -> str:
    """Whether the H1 verdict at the primary k is confirmed, contradicted, or
    only partially supported by the k=5/k=20 robustness checks."""
    verdicts = {
        s["k"]: evaluate_h1_h2(s, args.redundancy_threshold, args.headroom_threshold, args.rho_threshold, args.alpha)[0]
        for s in summaries
    }
    if len(set(verdicts.values())) == 1:
        return "robustly consistent"
    primary = verdicts.get(args.primary_k)
    agree = sum(1 for v in verdicts.values() if v == primary)
    if agree >= len(verdicts) - 1:
        return "partially consistent"
    if agree == 1:
        return "contradicted"
    return "unstable"


def _write_final_decision_document(
    output_dir: Path, all_scope_k_summaries: Dict[str, List[Dict]],
    overall_h1: Dict[str, bool], overall_h2: Dict[str, bool], unseen_scopes: List[str],
    overall_scope_h1: bool, overall_scope_h2: bool, unseen_h1: Optional[bool], unseen_h2: Optional[bool],
    case: str, args,
) -> None:
    lines = ["# Final Graph Fusion Decision", ""]

    lines.append("## 1. Overall-data evidence")
    for s in all_scope_k_summaries.get("overall", []):
        lines.append(
            f"- k={s['k']}: overlap={s['mean_overlap']:.3f}, headroom={s['budget_matched_oracle_headroom_at_k']:+.4f} "
            f"[{s['budget_matched_oracle_headroom_at_k_ci'][0]:+.4f}, {s['budget_matched_oracle_headroom_at_k_ci'][1]:+.4f}], "
            f"coverage_gain={s['candidate_coverage_gain']:.4f}"
        )
    lines.append("")

    lines.append("## 2. Unseen-class splits (individual)")
    for scope in unseen_scopes:
        if scope == "unseen_aggregate":
            continue
        lines.append(f"### {scope}")
        for s in all_scope_k_summaries.get(scope, []):
            lines.append(
                f"- k={s['k']}: overlap={s['mean_overlap']:.3f}, headroom={s['budget_matched_oracle_headroom_at_k']:+.4f} "
                f"[{s['budget_matched_oracle_headroom_at_k_ci'][0]:+.4f}, {s['budget_matched_oracle_headroom_at_k_ci'][1]:+.4f}], "
                f"coverage_gain={s['candidate_coverage_gain']:.4f}"
            )
    lines.append("")

    lines.append("## 3. Unseen-class aggregate (pooled across splits)")
    if "unseen_aggregate" in all_scope_k_summaries:
        for s in all_scope_k_summaries["unseen_aggregate"]:
            lines.append(
                f"- k={s['k']}: overlap={s['mean_overlap']:.3f}, headroom={s['budget_matched_oracle_headroom_at_k']:+.4f} "
                f"[{s['budget_matched_oracle_headroom_at_k_ci'][0]:+.4f}, {s['budget_matched_oracle_headroom_at_k_ci'][1]:+.4f}], "
                f"coverage_gain={s['candidate_coverage_gain']:.4f}"
            )
    else:
        lines.append("- Not available (fewer than 2 unseen-class splits were provided)")
    lines.append("")

    lines.append(f"## 4. k={args.primary_k} primary decision")
    lines.append(f"- Overall: H1={overall_scope_h1}, H2={overall_scope_h2}")
    lines.append(f"- Unseen (all splits must agree): H1={unseen_h1}, H2={unseen_h2}")
    lines.append("")

    non_primary_ks = sorted({s["k"] for summaries in all_scope_k_summaries.values() for s in summaries} - {args.primary_k})
    lines.append(f"## 5. Robustness at k={non_primary_ks} vs. primary k={args.primary_k}")
    for scope_name, summaries in all_scope_k_summaries.items():
        lines.append(f"- {scope_name}: {_classify_robustness(summaries, args)}")
    lines.append("")

    lines.append("## 6. H1 verdict")
    lines.append(f"- {unseen_h1 if unseen_scopes else overall_scope_h1}")
    lines.append("")
    lines.append("## 7. H2 verdict")
    lines.append(f"- {unseen_h2 if unseen_scopes else overall_scope_h2}")
    lines.append("")

    lines.append("## 8. Case classification")
    lines.append(f"- {case}")
    lines.append("")

    lines.append("## 9. Recommendation")
    if "abandon" in case.lower() or "stop" in case.lower() or "reject" in case.lower():
        recommendation = "Abandon graph fusion."
    elif "static" in case.lower():
        recommendation = "Proceed only with a minimal static/equal-weight graph-fusion baseline."
    else:
        recommendation = "Proceed with confidence-aware graph fusion."
    lines.append(f"- {recommendation}")
    lines.append("")

    lines.append("## 10. Suggested wording for the paper")
    primary_scope = "unseen_aggregate" if "unseen_aggregate" in all_scope_k_summaries else "overall"
    primary_summary = next(
        (s for s in all_scope_k_summaries.get(primary_scope, []) if s["k"] == args.primary_k), None,
    )
    primary_h1 = overall_h1.get(primary_scope, unseen_h1) if primary_scope != "overall" else overall_scope_h1
    if primary_summary is not None and not primary_h1:
        headroom = primary_summary["budget_matched_oracle_headroom_at_k"]
        ci_lo, ci_hi = primary_summary["budget_matched_oracle_headroom_at_k_ci"]
        if ci_lo > 0:
            ci_clause = f"headroom={headroom:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] excludes zero"
            effect_clause = "statistically detectable but practically negligible"
        else:
            ci_clause = f"headroom={headroom:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] includes zero"
            effect_clause = "not statistically distinguishable from zero"
        lines.append(
            f'- "Complementarity across the tested frozen descriptor spaces is {effect_clause} '
            f'({ci_clause}), relative to the pre-registered {args.headroom_threshold:.2f} threshold. '
            f'Neither training-time adaptation (ARP, ARP-NP) nor inference-time graph fusion of the tested '
            f'descriptor spaces provides a reliable solution to open-set retrieval transfer on Corel-1K, '
            f'under the tested split protocol and retrieval depths."'
        )
    else:
        lines.append("- (Fill in once a positive primary-scope result is confirmed -- do not template success wording in advance.)")

    (output_dir / "final_graph_fusion_decision.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
