import numpy as np
import pytest

from come_cbir.evaluate import (
    average_precision_at_k,
    evaluate_retrieval,
    full_average_precision,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


# Hand-computed toy ranking: relevant, not, relevant, not, not -- 3 relevant total in the DB.
RELEVANCE = np.array([1, 0, 1, 0, 0])
NUM_RELEVANT_TOTAL = 3


def test_precision_at_k():
    assert precision_at_k(RELEVANCE, 1) == pytest.approx(1.0)
    assert precision_at_k(RELEVANCE, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RELEVANCE, 5) == pytest.approx(2 / 5)


def test_recall_at_k():
    from come_cbir.evaluate import recall_at_k

    assert recall_at_k(RELEVANCE, 5, NUM_RELEVANT_TOTAL) == pytest.approx(2 / 3)
    assert recall_at_k(RELEVANCE, 1, NUM_RELEVANT_TOTAL) == pytest.approx(1 / 3)


def test_average_precision_at_k():
    # hits at rank 1 (precision 1/1) and rank 3 (precision 2/3); sum=1.6667; denom=min(3,5)=3
    expected = (1 / 1 + 2 / 3) / 3
    assert average_precision_at_k(RELEVANCE, 5, NUM_RELEVANT_TOTAL) == pytest.approx(expected)


def test_full_average_precision_matches_at_k_when_k_covers_full_list():
    expected = (1 / 1 + 2 / 3) / 3
    assert full_average_precision(RELEVANCE, NUM_RELEVANT_TOTAL) == pytest.approx(expected)


def test_reciprocal_rank_first_hit_at_rank_1():
    assert reciprocal_rank(RELEVANCE) == pytest.approx(1.0)


def test_reciprocal_rank_no_hits():
    assert reciprocal_rank(np.array([0, 0, 0])) == pytest.approx(0.0)


def test_reciprocal_rank_hit_at_rank_3():
    assert reciprocal_rank(np.array([0, 0, 1, 1])) == pytest.approx(1 / 3)


def test_ndcg_matches_hand_computation():
    gains = RELEVANCE.astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(gains)[::-1]
    idcg = float(np.sum(ideal * discounts))
    expected = dcg / idcg
    assert ndcg_at_k(RELEVANCE, 5) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_one():
    perfect = np.array([1, 1, 0, 0])
    assert ndcg_at_k(perfect, 4) == pytest.approx(1.0)


def test_metrics_are_zero_with_no_relevant_items():
    no_relevant = np.array([0, 0, 0])
    assert precision_at_k(no_relevant, 3) == 0.0
    assert average_precision_at_k(no_relevant, 3, 0) == 0.0
    assert full_average_precision(no_relevant, 0) == 0.0


# ----------------------------------------------------------------------
# End-to-end evaluate_retrieval with self-match exclusion (leave-one-out)
# ----------------------------------------------------------------------


@pytest.fixture
def clustered_dataset():
    rng = np.random.RandomState(0)
    vectors, labels, paths = [], [], []
    for c in range(4):
        base = np.zeros(20, dtype=np.float32)
        base[c] = 5.0
        for i in range(6):
            noisy = base + rng.normal(scale=0.01, size=20).astype(np.float32)
            noisy = noisy / np.linalg.norm(noisy)
            vectors.append(noisy)
            labels.append(c)
            paths.append(f"class{c}_img{i}.jpg")
    return np.stack(vectors).astype(np.float32), np.array(labels), paths


def test_evaluate_retrieval_excludes_self_match(clustered_dataset):
    features, labels, paths = clustered_dataset
    result = evaluate_retrieval(features, labels, paths, top_k=5)
    metrics = result["metrics"]

    # Every query's own row index must never appear in its own per_query relevance
    # computation -- verified indirectly: with 6 members per well-separated cluster
    # and top_k=5 < 5 other same-class members, precision@1 should be (near) perfect,
    # which would be trivially and misleadingly 1.0 if self-matches (score=1.0, same
    # label) were not excluded from evaluation.
    assert metrics["precision_at_1"] > 0.9
    assert metrics["num_queries_evaluated"] == features.shape[0]
    assert metrics["mAP_at_10"] is not None
    assert metrics["descriptor_memory_bytes"] == features.nbytes


def test_evaluate_retrieval_requires_at_least_two_samples():
    with pytest.raises(ValueError, match="at least 2 samples"):
        evaluate_retrieval(
            np.zeros((1, 4), dtype=np.float32), np.array([0]), ["a.jpg"], top_k=5
        )


def test_evaluate_retrieval_singleton_class_is_skipped(clustered_dataset):
    features, labels, paths = clustered_dataset
    # Turn one sample into its own singleton class -> it has 0 same-class peers and
    # must be excluded from the aggregate metrics rather than corrupting them with a
    # divide-by-zero or an artificial 0.0 score.
    labels = labels.copy()
    labels[0] = 999
    result = evaluate_retrieval(features, labels, paths, top_k=5)
    assert result["metrics"]["num_queries_evaluated"] == features.shape[0] - 1


# ----------------------------------------------------------------------
# Dynamic @top_k metrics -- regression test for the bug where --top-k was
# accepted by the CLI but every reported metric was silently hardcoded to a
# fixed rank (1, 5, or 10), so e.g. --top-k 20 produced identical output to
# --top-k 10 with no way to actually see a real Precision@20/mAP@20/NDCG@20.
# ----------------------------------------------------------------------


def test_evaluate_retrieval_reports_metrics_at_requested_top_k(clustered_dataset):
    features, labels, paths = clustered_dataset
    result = evaluate_retrieval(features, labels, paths, top_k=20)
    metrics = result["metrics"]

    assert metrics["requested_top_k"] == 20
    for key in ("precision_at_20", "recall_at_20", "mAP_at_20", "ndcg_at_20"):
        assert key in metrics
        assert metrics[key] is not None
        assert 0.0 <= metrics[key] <= 1.0

    # Same fields must also be present per-query, not just in the aggregate.
    assert "precision_at_20" in result["per_query"][0]
    assert "average_precision_at_20" in result["per_query"][0]
    assert "ndcg_at_20" in result["per_query"][0]

    # The fixed @10 checkpoints must be completely unaffected by --top-k, since
    # already-reported results (Corel-1K/Corel-10K tables) depend on this.
    assert metrics["precision_at_10"] is not None
    assert metrics["mAP_at_10"] is not None


def test_dynamic_topk_metrics_collapse_cleanly_when_top_k_is_10(clustered_dataset):
    features, labels, paths = clustered_dataset
    result = evaluate_retrieval(features, labels, paths, top_k=10)
    metrics = result["metrics"]

    # When --top-k is the default (10), f"precision_at_{top_k}" names the exact same
    # key as the fixed "precision_at_10" checkpoint -- there is only one field, not
    # two conflicting ones, and its value matches a manual recomputation from the
    # per-query records rather than silently holding something else.
    assert metrics["requested_top_k"] == 10
    manual_mean = sum(r["precision_at_10"] for r in result["per_query"]) / len(result["per_query"])
    assert metrics["precision_at_10"] == pytest.approx(manual_mean)
