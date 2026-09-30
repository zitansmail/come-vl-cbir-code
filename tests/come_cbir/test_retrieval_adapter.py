import numpy as np
import pytest
import torch
import torch.nn.functional as F

from come_cbir.retrieval_adapter import (
    AdaptiveRetrievalProjection,
    apply_adapter,
    build_adapter,
    build_knn_graph,
    class_level_holdout_split,
    load_split,
    neighborhood_preservation_loss,
    save_split,
    stratified_holdout_split,
    supervised_contrastive_loss,
    three_way_class_split,
)


# The four real descriptor dimensions this adapter must fit under budget for.
REAL_DESCRIPTOR_DIMS = {"siglip": 1152, "dino": 1024, "concat": 2176, "come_fused": 3584}


@pytest.mark.parametrize("mode,dim", REAL_DESCRIPTOR_DIMS.items())
def test_parameter_budget_under_one_million_for_every_descriptor_mode(mode, dim):
    adapter = build_adapter("arp", input_dim=dim)
    n_params = adapter.count_parameters()
    assert n_params < 1_000_000, f"{mode} (dim={dim}) adapter has {n_params} params, over budget"


def test_forward_pass_shape_and_l2_norm():
    adapter = build_adapter("arp", input_dim=128, hidden_dim=64, output_dim=32)
    x = torch.randn(10, 128)
    out = adapter(x)
    assert out.shape == (10, 32)
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(10), atol=1e-5)


def test_unsupported_adapter_type_rejected():
    with pytest.raises(ValueError, match="Unsupported adapter_type"):
        build_adapter("not_a_real_type", input_dim=128)


def test_save_load_round_trip(tmp_path):
    adapter = build_adapter("arp", input_dim=64, hidden_dim=32, output_dim=16)
    x = torch.randn(4, 64)
    adapter.eval()
    expected = adapter(x)

    path = tmp_path / "adapter.pt"
    adapter.save(str(path))
    loaded = AdaptiveRetrievalProjection.load(str(path))

    assert loaded.config.input_dim == 64
    assert loaded.config.hidden_dim == 32
    assert loaded.config.output_dim == 16
    actual = loaded(x)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_apply_adapter_matches_direct_forward_and_is_batch_size_invariant():
    adapter = build_adapter("arp", input_dim=32, hidden_dim=16, output_dim=8)
    adapter.eval()
    features = np.random.RandomState(0).randn(37, 32).astype(np.float32)

    with torch.no_grad():
        direct = adapter(torch.from_numpy(features)).numpy()

    via_small_batches = apply_adapter(features, adapter, batch_size=5)
    via_one_batch = apply_adapter(features, adapter, batch_size=1000)

    assert via_small_batches.shape == (37, 8)
    np.testing.assert_allclose(via_small_batches, direct, atol=1e-6)
    np.testing.assert_allclose(via_one_batch, direct, atol=1e-6)


def test_apply_adapter_empty_input():
    adapter = build_adapter("arp", input_dim=16, output_dim=8)
    out = apply_adapter(np.zeros((0, 16), dtype=np.float32), adapter)
    assert out.shape == (0, 8)


class TestSupervisedContrastiveLoss:
    def test_batch_size_less_than_two_returns_zero(self):
        embeddings = torch.randn(1, 8)
        labels = torch.tensor([0])
        loss = supervised_contrastive_loss(embeddings, labels)
        assert float(loss.detach()) == pytest.approx(0.0)

    def test_no_positive_pairs_returns_zero(self):
        # every sample its own unique class -> no positives anywhere
        embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        labels = torch.tensor([0, 1, 2, 3])
        loss = supervised_contrastive_loss(embeddings, labels)
        assert float(loss.detach()) == pytest.approx(0.0)

    def test_separated_clusters_cost_less_than_randomly_mixed_embeddings(self):
        # SupCon's softmax denominator sums over all same-label positives, so even a
        # perfectly tight, well-separated cluster has a nonzero floor of ln(num_positives)
        # (here ln(3) for 4-per-class batches) -- that's a real, known property of the
        # loss, not something a "near-zero" absolute-value assertion should expect. The
        # meaningful, robust check is relative: well-separated clusters must cost
        # noticeably less than embeddings with no class structure at all.
        rng = np.random.RandomState(0)
        cluster_a = torch.nn.functional.normalize(
            torch.from_numpy(rng.normal(loc=5.0, scale=0.001, size=(4, 8)).astype(np.float32)), dim=-1
        )
        cluster_b = torch.nn.functional.normalize(
            torch.from_numpy(rng.normal(loc=-5.0, scale=0.001, size=(4, 8)).astype(np.float32)), dim=-1
        )
        separated = torch.cat([cluster_a, cluster_b], dim=0)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        separated_loss = supervised_contrastive_loss(separated, labels, temperature=0.07)

        random_mix = torch.nn.functional.normalize(torch.from_numpy(rng.randn(8, 8).astype(np.float32)), dim=-1)
        random_loss = supervised_contrastive_loss(random_mix, labels, temperature=0.07)

        # Floor is ln(3) (3 other same-class samples per anchor); confirm we're at that floor.
        assert float(separated_loss) == pytest.approx(np.log(3), abs=1e-3)
        assert float(separated_loss) < float(random_loss)

    def test_training_step_reduces_loss(self):
        # A tiny end-to-end sanity check: a few optimizer steps on a trivially
        # separable synthetic problem should meaningfully reduce SupCon loss.
        torch.manual_seed(0)
        adapter = build_adapter("arp", input_dim=16, hidden_dim=16, output_dim=8)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-2)

        rng = np.random.RandomState(0)
        base = np.zeros((4, 16), dtype=np.float32)
        base[0, 0] = 5.0
        base[1, 1] = 5.0
        base[2, 2] = 5.0
        base[3, 3] = 5.0
        features = np.repeat(base, 8, axis=0) + rng.normal(scale=0.05, size=(32, 16)).astype(np.float32)
        labels = torch.tensor(np.repeat([0, 1, 2, 3], 8))
        features_t = torch.from_numpy(features)

        losses = []
        for _ in range(30):
            embeddings = adapter(features_t)
            loss = supervised_contrastive_loss(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        assert losses[-1] < losses[0]


class TestStratifiedHoldoutSplit:
    def test_no_overlap_and_full_coverage(self):
        labels = np.repeat(np.arange(5), 20)
        train_idx, holdout_idx = stratified_holdout_split(labels, holdout_fraction=0.3, seed=0)
        assert set(train_idx).isdisjoint(set(holdout_idx))
        assert len(train_idx) + len(holdout_idx) == len(labels)
        assert set(train_idx) | set(holdout_idx) == set(range(len(labels)))

    def test_approximately_respects_holdout_fraction_per_class(self):
        labels = np.repeat(np.arange(4), 100)
        train_idx, holdout_idx = stratified_holdout_split(labels, holdout_fraction=0.3, seed=0)
        for c in range(4):
            class_holdout = np.isin(holdout_idx, np.where(labels == c)[0]).sum()
            assert class_holdout == pytest.approx(30, abs=1)

    def test_respects_min_train_per_class_for_tiny_classes(self):
        # A class with only 2 members must keep at least min_train_per_class in train,
        # even though 0.3 * 2 would naively round to 1 holdout, leaving only 1 for train.
        labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        train_idx, holdout_idx = stratified_holdout_split(labels, holdout_fraction=0.9, seed=0, min_train_per_class=2)
        class0_train = np.isin(train_idx, np.where(labels == 0)[0]).sum()
        assert class0_train >= 2

    def test_reproducible_with_same_seed(self):
        labels = np.repeat(np.arange(5), 20)
        t1, h1 = stratified_holdout_split(labels, seed=42)
        t2, h2 = stratified_holdout_split(labels, seed=42)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(h1, h2)


class TestClassLevelHoldoutSplit:
    def test_holdout_classes_entirely_absent_from_train(self):
        labels = np.repeat(np.arange(10), 20)
        train_idx, holdout_idx, holdout_classes = class_level_holdout_split(
            labels, holdout_class_fraction=0.3, seed=0
        )
        train_classes = set(labels[train_idx].tolist())
        holdout_classes_seen = set(labels[holdout_idx].tolist())
        assert train_classes.isdisjoint(set(holdout_classes))
        assert holdout_classes_seen == set(holdout_classes)
        assert len(holdout_classes) == 3  # round(10 * 0.3)

    def test_no_overlap_and_full_coverage(self):
        labels = np.repeat(np.arange(10), 15)
        train_idx, holdout_idx, _ = class_level_holdout_split(labels, holdout_class_fraction=0.3, seed=0)
        assert set(train_idx).isdisjoint(set(holdout_idx))
        assert len(train_idx) + len(holdout_idx) == len(labels)

    def test_always_leaves_at_least_one_train_class(self):
        labels = np.repeat(np.arange(3), 10)
        # Asking to hold out "all" classes should still leave at least one to train on.
        train_idx, holdout_idx, holdout_classes = class_level_holdout_split(
            labels, holdout_class_fraction=0.99, seed=0
        )
        assert len(holdout_classes) == 2  # 3 - 1, never all of them
        assert len(train_idx) > 0

    def test_reproducible_with_same_seed(self):
        labels = np.repeat(np.arange(10), 10)
        t1, h1, c1 = class_level_holdout_split(labels, seed=7)
        t2, h2, c2 = class_level_holdout_split(labels, seed=7)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(h1, h2)
        assert c1 == c2

    def test_rejects_fewer_than_two_classes(self):
        labels = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="at least 2 classes"):
            class_level_holdout_split(labels)


def test_save_split_records_holdout_mode_and_classes(tmp_path):
    path = tmp_path / "split.json"
    save_split(str(path), np.array([0, 1]), np.array([2, 3]), mode="class", holdout_classes=[5, 7])
    with open(path) as f:
        import json
        data = json.load(f)
    assert data["holdout_mode"] == "class"
    assert data["holdout_classes"] == [5, 7]


def test_save_load_split_round_trip(tmp_path):
    train_idx = np.array([0, 2, 4])
    holdout_idx = np.array([1, 3])
    path = tmp_path / "split.json"
    save_split(str(path), train_idx, holdout_idx)
    loaded_train, loaded_holdout = load_split(str(path))
    np.testing.assert_array_equal(loaded_train, train_idx)
    np.testing.assert_array_equal(loaded_holdout, holdout_idx)


# ----------------------------------------------------------------------
# ARP-NP: build_knn_graph -- required correctness properties
# ----------------------------------------------------------------------


class TestBuildKnnGraph:
    def test_self_matches_are_excluded(self):
        rng = np.random.RandomState(0)
        features = rng.randn(30, 8).astype(np.float32)
        graph = build_knn_graph(features, k=5)
        for i in range(30):
            valid_neighbors = graph["neighbor_indices"][i][graph["valid_mask"][i]]
            assert i not in valid_neighbors

    def test_negative_teacher_similarities_never_produce_negative_weights(self):
        # Two tight, diametrically opposite clusters -> guaranteed negative
        # cross-cluster cosine similarity; a same-cluster point's true nearest
        # neighbours are same-cluster (positive sim), but we still verify no
        # negative weight can appear anywhere in the output regardless.
        rng = np.random.RandomState(0)
        cluster_a = rng.normal(loc=5.0, scale=0.01, size=(10, 8)).astype(np.float32)
        cluster_b = rng.normal(loc=-5.0, scale=0.01, size=(10, 8)).astype(np.float32)
        features = np.concatenate([cluster_a, cluster_b], axis=0)
        graph = build_knn_graph(features, k=15)  # k=15 forces some cross-cluster (negative-sim) neighbours
        assert (graph["neighbor_weight"] >= 0).all()
        # Confirm this test actually exercised a negative-similarity case, not a vacuous check.
        assert (graph["neighbor_sim"][graph["valid_mask"]] < 0).any()

    def test_l1_weights_sum_to_one_per_anchor_when_any_positive_similarity_exists(self):
        rng = np.random.RandomState(0)
        features = rng.randn(20, 8).astype(np.float32)
        graph = build_knn_graph(features, k=5, weight_normalization="l1")
        row_sums = graph["neighbor_weight"].sum(axis=1)
        # Real, non-degenerate random features always have some positive-cosine neighbour.
        np.testing.assert_allclose(row_sums, np.ones(20), atol=1e-5)

    def test_graph_only_reflects_the_features_actually_passed_in(self):
        # The function's only leakage guarantee is structural: it has no way to
        # reference data it was never given. Simulate "train-only" vs "held-out"
        # by simply never passing the held-out rows to the function at all, and
        # confirm every returned neighbour index is in-range for the train-only
        # array -- i.e. it is impossible for a held-out row to appear.
        rng = np.random.RandomState(0)
        train_only_features = rng.randn(12, 8).astype(np.float32)
        graph = build_knn_graph(train_only_features, k=4)
        assert graph["neighbor_indices"].max() < 12
        assert graph["neighbor_indices"].min() >= 0

    def test_softmax_weight_normalization_sums_to_one(self):
        rng = np.random.RandomState(0)
        features = rng.randn(20, 8).astype(np.float32)
        graph = build_knn_graph(features, k=5, weight_normalization="softmax")
        row_sums = graph["neighbor_weight"].sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(20), atol=1e-5)

    def test_invalid_weight_normalization_rejected(self):
        with pytest.raises(ValueError, match="weight_normalization"):
            build_knn_graph(np.random.randn(10, 4).astype(np.float32), k=3, weight_normalization="bogus")

    def test_symmetric_option_produces_a_valid_graph(self):
        rng = np.random.RandomState(0)
        features = rng.randn(25, 8).astype(np.float32)
        graph_sym = build_knn_graph(features, k=5, symmetric=True)
        graph_asym = build_knn_graph(features, k=5, symmetric=False)
        assert graph_sym["neighbor_indices"].shape == graph_asym["neighbor_indices"].shape
        assert (graph_sym["neighbor_weight"] >= 0).all()


# ----------------------------------------------------------------------
# ARP-NP: neighborhood_preservation_loss -- required correctness properties
# ----------------------------------------------------------------------


class TestNeighborhoodPreservationLoss:
    def test_near_zero_when_adapted_similarities_match_frozen_targets(self):
        torch.manual_seed(0)
        B, k, m = 4, 3, 8
        z_anchor = F.normalize(torch.randn(B, m), dim=-1)
        z_neighbors = F.normalize(torch.randn(B, k, m), dim=-1)
        # Set the "frozen" target to exactly the adapted-space similarity, so a
        # perfect-preservation scenario is constructed directly, not hoped for.
        neighbor_sim = torch.einsum("bm,bkm->bk", z_anchor, z_neighbors).detach()
        neighbor_weight = torch.ones(B, k) / k
        valid_mask = torch.ones(B, k, dtype=torch.bool)

        loss = neighborhood_preservation_loss(z_anchor, z_neighbors, neighbor_sim, neighbor_weight, valid_mask)
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_similarities_genuinely_differ(self):
        torch.manual_seed(0)
        B, k, m = 4, 3, 8
        z_anchor = F.normalize(torch.randn(B, m), dim=-1)
        z_neighbors = F.normalize(torch.randn(B, k, m), dim=-1)
        neighbor_sim = torch.full((B, k), -1.0)  # deliberately wrong target
        neighbor_weight = torch.ones(B, k) / k
        valid_mask = torch.ones(B, k, dtype=torch.bool)

        loss = neighborhood_preservation_loss(z_anchor, z_neighbors, neighbor_sim, neighbor_weight, valid_mask)
        assert float(loss.detach()) > 0.0

    def test_invalid_slots_are_excluded_regardless_of_their_values(self):
        torch.manual_seed(0)
        B, k, m = 2, 2, 8
        z_anchor = F.normalize(torch.randn(B, m), dim=-1)
        z_neighbors = F.normalize(torch.randn(B, k, m), dim=-1)
        neighbor_sim = torch.einsum("bm,bkm->bk", z_anchor, z_neighbors).detach()
        # Corrupt slot (0, 1)'s target so it would blow up the loss if counted --
        # but mark it invalid, so it must contribute nothing.
        neighbor_sim[0, 1] = -1.0
        neighbor_weight = torch.ones(B, k) / k
        valid_mask = torch.ones(B, k, dtype=torch.bool)
        valid_mask[0, 1] = False

        loss = neighborhood_preservation_loss(z_anchor, z_neighbors, neighbor_sim, neighbor_weight, valid_mask)
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)

    def test_gradients_flow_back_to_the_anchor_embedding(self):
        torch.manual_seed(0)
        B, k, m = 4, 3, 8
        z_anchor = F.normalize(torch.randn(B, m, requires_grad=True), dim=-1)
        z_anchor.retain_grad()
        z_neighbors = F.normalize(torch.randn(B, k, m), dim=-1)
        neighbor_sim = torch.full((B, k), -1.0)
        neighbor_weight = torch.ones(B, k) / k
        valid_mask = torch.ones(B, k, dtype=torch.bool)

        loss = neighborhood_preservation_loss(z_anchor, z_neighbors, neighbor_sim, neighbor_weight, valid_mask)
        loss.backward()
        assert z_anchor.grad is not None
        assert (z_anchor.grad.abs().sum() > 0).item()


class TestThreeWayClassSplit:
    def test_disjoint_and_full_coverage(self):
        labels = np.repeat(np.arange(10), 20)
        result = three_way_class_split(labels, val_class_fraction=0.2, test_class_fraction=0.2, seed=0)
        train_c, val_c, test_c = set(result["train_classes"]), set(result["val_classes"]), set(result["test_classes"])
        assert train_c.isdisjoint(val_c)
        assert train_c.isdisjoint(test_c)
        assert val_c.isdisjoint(test_c)
        assert train_c | val_c | test_c == set(range(10))

        train_i, val_i, test_i = set(result["train_indices"]), set(result["val_indices"]), set(result["test_indices"])
        assert train_i.isdisjoint(val_i) and train_i.isdisjoint(test_i) and val_i.isdisjoint(test_i)
        assert len(train_i) + len(val_i) + len(test_i) == len(labels)

    def test_rejects_fewer_than_three_classes(self):
        labels = np.repeat(np.arange(2), 10)
        with pytest.raises(ValueError, match="at least 3 classes"):
            three_way_class_split(labels)

    def test_reproducible_with_same_seed(self):
        labels = np.repeat(np.arange(10), 15)
        r1 = three_way_class_split(labels, seed=3)
        r2 = three_way_class_split(labels, seed=3)
        assert r1["val_classes"] == r2["val_classes"]
        assert r1["test_classes"] == r2["test_classes"]
        np.testing.assert_array_equal(r1["train_indices"], r2["train_indices"])
