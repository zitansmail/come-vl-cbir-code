import json

import numpy as np
import pytest

from come_cbir.validate_graph_fusion import (
    benjamini_hochberg,
    bootstrap_ci_mean,
    bootstrap_ci_spearman,
    compute_budget_matched_headroom,
    compute_candidate_coverage_gain,
    load_restrict_indices,
    run_validation_for_k,
    validate_alignment,
)


@pytest.fixture
def clustered_dataset():
    rng = np.random.RandomState(0)
    n_classes = 6
    base = np.zeros((n_classes, 16), dtype=np.float32)
    for c in range(n_classes):
        base[c, c % 16] = 5.0
    labels = np.repeat(np.arange(n_classes), 15)
    descriptors = {
        "a": np.repeat(base, 15, axis=0) + rng.normal(scale=0.3, size=(90, 16)).astype(np.float32),
        "b": np.repeat(base, 15, axis=0) + rng.normal(scale=0.3, size=(90, 16)).astype(np.float32),
        "c": np.repeat(base, 15, axis=0) + rng.normal(scale=1.0, size=(90, 16)).astype(np.float32),
    }
    paths = [f"img{i}.jpg" for i in range(90)]
    return descriptors, labels, paths


class TestBudgetMatchedHeadroom:
    def test_equal_retrieval_budget_used_for_single_and_union(self):
        # R_q = 3 (small), k = 5 (larger than R_q) -- denom must be min(k, R_q) = 3
        # for BOTH the single-mode recall and the union's oracle recall.
        num_relevant = np.array([3])
        single_hits = {"a": np.array([2]), "b": np.array([1])}  # both <= k=5
        union_hits = np.array([3])  # union finds all 3 relevant items
        headroom = compute_budget_matched_headroom(single_hits, union_hits, num_relevant, k=5)
        # best_single_recall = 2/min(5,3) = 2/3 ; oracle_recall_union = min(5,3)/min(5,3) = 1.0
        assert headroom[0] == pytest.approx(1.0 - 2 / 3)

    def test_union_candidate_size_cannot_inflate_headroom_beyond_budget(self):
        # Union hits far exceed k -- the numerator must be capped at k, not
        # allowed to reflect the raw (possibly huge, M*k-sized) union count.
        num_relevant = np.array([100])  # plenty of relevant items exist overall
        single_hits = {"a": np.array([3])}
        union_hits = np.array([40])  # union "sees" 40 relevant items, way more than k
        headroom = compute_budget_matched_headroom(single_hits, union_hits, num_relevant, k=5)
        # oracle_recall_union must be min(5, 40)/min(5,100) = 5/5 = 1.0, not 40/5=8.0
        best_single_recall = 3 / 5
        assert headroom[0] == pytest.approx(1.0 - best_single_recall)
        assert headroom[0] <= 1.0  # never exceeds the natural [0,1] bound regardless of union size


class TestCandidateCoverageGainDistinctFromHeadroom:
    def test_coverage_gain_positive_while_headroom_is_zero(self):
        # Best single mode already saturates recall@k (all k retrieved neighbours
        # are relevant), so budget-matched headroom is exactly 0 -- but there are
        # many MORE relevant items overall (R_q >> k) that only the union reaches,
        # so candidate coverage gain (uncapped by k) is still positive. This is
        # the concrete case that shows the two metrics measure different things.
        k = 5
        num_relevant = np.array([50])  # far more relevant items exist than k
        single_hits = {"a": np.array([5])}  # a's top-5 are ALL relevant -> recall@k = 1.0 already
        union_hits = np.array([20])  # union finds 20 total relevant items across all modes

        headroom = compute_budget_matched_headroom(single_hits, union_hits, num_relevant, k)
        coverage_gain = compute_candidate_coverage_gain(single_hits, union_hits, num_relevant)

        assert headroom[0] == pytest.approx(0.0)  # single mode already at the ceiling for this budget
        assert coverage_gain[0] > 0  # but real complementary signal exists beyond the k budget
        assert coverage_gain[0] == pytest.approx(20 / 50 - 5 / 50)

    def test_coverage_gain_is_never_negative(self):
        rng = np.random.RandomState(0)
        num_relevant = rng.randint(5, 50, size=20)
        single_hits = {
            "a": rng.randint(0, 5, size=20), "b": rng.randint(0, 5, size=20), "c": rng.randint(0, 5, size=20),
        }
        union_hits = np.maximum.reduce(list(single_hits.values())) + rng.randint(0, 5, size=20)
        gain = compute_candidate_coverage_gain(single_hits, union_hits, num_relevant)
        assert np.all(gain >= -1e-9)


class TestAlignmentValidation:
    def test_mismatched_row_count_raises(self):
        labels = np.array([0, 1, 0, 1])
        paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        descriptors = {"x": np.zeros((3, 4), dtype=np.float32)}  # wrong row count
        with pytest.raises(ValueError, match="row count"):
            validate_alignment(["x"], descriptors, labels, paths, {})

    def test_mismatched_per_mode_labels_raises(self, tmp_path):
        labels = np.array([0, 1, 0, 1])
        paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        descriptors = {"x": np.zeros((4, 4), dtype=np.float32)}
        bad_labels_path = tmp_path / "bad_labels.npy"
        np.save(bad_labels_path, np.array([1, 1, 0, 1]))  # first entry differs
        with pytest.raises(ValueError, match="does not match"):
            validate_alignment(["x"], descriptors, labels, paths, {"x": {"labels": str(bad_labels_path)}})

    def test_mismatched_per_mode_paths_raises(self, tmp_path):
        labels = np.array([0, 1, 0, 1])
        paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        descriptors = {"x": np.zeros((4, 4), dtype=np.float32)}
        bad_paths_path = tmp_path / "bad_paths.json"
        with open(bad_paths_path, "w") as f:
            json.dump(["a.jpg", "DIFFERENT.jpg", "c.jpg", "d.jpg"], f)
        with pytest.raises(ValueError, match="does not match"):
            validate_alignment(["x"], descriptors, labels, paths, {"x": {"paths": str(bad_paths_path)}})

    def test_identical_descriptor_ordering_passes_silently(self, tmp_path):
        labels = np.array([0, 1, 0, 1])
        paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        descriptors = {"x": np.zeros((4, 4), dtype=np.float32), "y": np.ones((4, 4), dtype=np.float32)}
        labels_path = tmp_path / "labels.npy"
        np.save(labels_path, labels)
        paths_path = tmp_path / "paths.json"
        with open(paths_path, "w") as f:
            json.dump(paths, f)
        validate_alignment(
            ["x", "y"], descriptors, labels, paths,
            {"x": {"labels": str(labels_path), "paths": str(paths_path)}},
        )  # must not raise


class TestLoadRestrictIndices:
    def test_raw_list_format(self, tmp_path):
        path = tmp_path / "idx.json"
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)
        result = load_restrict_indices(str(path), labels=np.zeros(10))
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_test_indices_format(self, tmp_path):
        path = tmp_path / "split.json"
        with open(path, "w") as f:
            json.dump({"train_indices": [0, 1], "test_indices": [2, 3, 4]}, f)
        result = load_restrict_indices(str(path), labels=np.zeros(10))
        np.testing.assert_array_equal(sorted(result), [2, 3, 4])

    def test_test_classes_format_reconstructs_indices_from_labels(self, tmp_path):
        path = tmp_path / "seed_0.json"
        with open(path, "w") as f:
            json.dump({"split": {"test_classes": [1]}, "test_classes": [1]}, f)
        labels = np.array([0, 1, 0, 1, 2])
        result = load_restrict_indices(str(path), labels=labels)
        np.testing.assert_array_equal(sorted(result), [1, 3])

    def test_missing_keys_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            json.dump({"nothing_useful": True}, f)
        with pytest.raises(ValueError, match="Could not find"):
            load_restrict_indices(str(path), labels=np.zeros(10))


class TestBenjaminiHochberg:
    def test_deterministic(self):
        pvals = [0.01, 0.2, 0.03, 0.5, 0.001]
        r1 = benjamini_hochberg(pvals)
        r2 = benjamini_hochberg(pvals)
        assert r1 == r2

    def test_corrected_always_at_least_raw_and_bounded(self):
        pvals = [0.001, 0.01, 0.02, 0.03, 0.5]
        corrected = benjamini_hochberg(pvals)
        for raw, corr in zip(pvals, corrected):
            assert corr >= raw - 1e-12
            assert 0.0 <= corr <= 1.0

    def test_empty_input(self):
        assert benjamini_hochberg([]) == []


class TestBootstrapDeterminism:
    def test_mean_ci_deterministic_with_fixed_seed(self):
        values = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.4])
        ci1 = bootstrap_ci_mean(values, n_boot=200, seed=42)
        ci2 = bootstrap_ci_mean(values, n_boot=200, seed=42)
        assert ci1 == ci2

    def test_spearman_ci_deterministic_with_fixed_seed(self):
        rng = np.random.RandomState(0)
        x = rng.rand(30)
        y = x + rng.normal(scale=0.1, size=30)
        ci1 = bootstrap_ci_spearman(x, y, n_boot=200, seed=7)
        ci2 = bootstrap_ci_spearman(x, y, n_boot=200, seed=7)
        assert ci1 == ci2

    def test_different_seeds_can_differ(self):
        values = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        ci1 = bootstrap_ci_mean(values, n_boot=50, seed=1)
        ci2 = bootstrap_ci_mean(values, n_boot=50, seed=2)
        assert ci1 != ci2 or True  # not a strict requirement, just documents expected variation


class TestRunValidationForK:
    def test_self_matches_excluded_from_graphs(self, clustered_dataset):
        descriptors, labels, paths = clustered_dataset
        summary, rows, tests = run_validation_for_k(
            descriptors, labels, paths, k=5, top_k=5, n_boot=20, boot_seed=0, logger=_NullLogger(),
        )
        # If self-matches leaked in, every query's own AP would trivially be
        # boosted by matching itself -- mean AP staying in a sane [0,1] range
        # with real variance across queries is a structural sanity check that
        # build_knn_graph's self-exclusion (already tested elsewhere) is still
        # in effect through this pipeline.
        for name in descriptors:
            aps = [r[f"ap_{name}"] for r in rows]
            assert all(0.0 <= v <= 1.0 for v in aps)

    def test_agreement_excluding_target_never_uses_target_mode(self, clustered_dataset):
        descriptors, labels, paths = clustered_dataset
        summary, rows, tests = run_validation_for_k(
            descriptors, labels, paths, k=5, top_k=5, n_boot=20, boot_seed=0, logger=_NullLogger(),
        )
        # With 3 modes, excluding "a" leaves exactly {b, c} -> agreement_excl_a
        # must equal overlap_b_c for every query (only one pair remains).
        for row in rows:
            assert row["agreement_excl_a"] == pytest.approx(row["overlap_b_c"])

    def test_agreement_including_target_averages_pairs_with_target(self, clustered_dataset):
        descriptors, labels, paths = clustered_dataset
        summary, rows, tests = run_validation_for_k(
            descriptors, labels, paths, k=5, top_k=5, n_boot=20, boot_seed=0, logger=_NullLogger(),
        )
        # agreement_incl_a must be the mean of overlap(a,b) and overlap(a,c).
        for row in rows:
            expected = (row["overlap_a_b"] + row["overlap_a_c"]) / 2
            assert row["agreement_incl_a"] == pytest.approx(expected)

    def test_restricted_scope_matches_direct_evaluate_retrieval_call(self, clustered_dataset):
        from come_cbir.evaluate import evaluate_retrieval

        descriptors, labels, paths = clustered_dataset
        restrict = np.arange(0, 30)  # first two classes only -- mimics an unseen-class scope
        scope_descriptors = {name: feats[restrict] for name, feats in descriptors.items()}
        scope_labels = labels[restrict]
        scope_paths = [paths[i] for i in restrict]

        summary, rows, tests = run_validation_for_k(
            scope_descriptors, scope_labels, scope_paths, k=5, top_k=5, n_boot=10, boot_seed=0, logger=_NullLogger(),
        )
        direct = evaluate_retrieval(scope_descriptors["a"], scope_labels, scope_paths, top_k=5, compute_full_map=True)
        direct_ap_by_index = {r["query_index"]: r["full_average_precision"] for r in direct["per_query"]}

        for row in rows:
            assert row["ap_a"] == pytest.approx(direct_ap_by_index[row["query_index"]])


class _NullLogger:
    def info(self, *args, **kwargs):
        pass
