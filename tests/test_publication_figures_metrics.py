import numpy as np


def _expected_balance_pair_weights(train_df, test_df):
    all_genus = np.concatenate([train_df["genus"].astype(str).to_numpy(), test_df["genus"].astype(str).to_numpy()])
    categories = {value: idx for idx, value in enumerate(np.unique(all_genus))}
    genus_codes_x = np.array([categories[value] for value in train_df["genus"].astype(str).to_numpy()], dtype=np.int32)
    genus_codes_y = np.array([categories[value] for value in test_df["genus"].astype(str).to_numpy()], dtype=np.int32)

    pair_ids = []
    for idx1 in range(len(train_df)):
        for idx2 in range(len(test_df)):
            if idx1 == idx2:
                continue
            lo = min(genus_codes_x[idx1], genus_codes_y[idx2])
            hi = max(genus_codes_x[idx1], genus_codes_y[idx2])
            pair_ids.append(lo * len(categories) + hi)

    pair_ids = np.array(pair_ids, dtype=np.int64)
    class_counts = np.bincount(pair_ids)
    nonzero_classes = class_counts > 0
    n_classes = int(np.count_nonzero(nonzero_classes))
    n_samples = int(pair_ids.shape[0])
    return (n_samples / (n_classes * class_counts[pair_ids])).astype(np.float32)


def _patch_curve_functions(monkeypatch, metrics_module, captured):
    def fake_compute_sim_numba(X, Y, i1, i2, metric_code):
        captured["pairs"] = (np.array(i1, copy=True), np.array(i2, copy=True), metric_code)
        if len(i1) == 0:
            return np.array([], dtype=np.float32)
        return np.linspace(0.1, 0.9, len(i1), dtype=np.float32)

    def fake_roc_curve(y_true, y_score, drop_intermediate=False, sample_weight=None):
        captured["roc"] = {
            "y_true": np.array(y_true, copy=True),
            "y_score": np.array(y_score, copy=True),
            "sample_weight": None if sample_weight is None else np.array(sample_weight, copy=True),
        }
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    def fake_precision_recall_curve(y_true, y_score, sample_weight=None, drop_intermediate=False):
        captured["pr"] = {
            "y_true": np.array(y_true, copy=True),
            "y_score": np.array(y_score, copy=True),
            "sample_weight": None if sample_weight is None else np.array(sample_weight, copy=True),
        }
        return np.array([1.0, 0.5, 0.0]), np.array([0.0, 0.5, 1.0]), np.array([0.7, 0.3])

    monkeypatch.setattr(metrics_module, "compute_sim_numba", fake_compute_sim_numba)
    monkeypatch.setattr(metrics_module, "roc_curve", fake_roc_curve)
    monkeypatch.setattr(metrics_module, "precision_recall_curve", fake_precision_recall_curve)


def test_binary_similarity_curves_test_only_skips_diagonal_and_balances_pairs(
    monkeypatch,
    metrics_module,
    balanced_pair_frames,
):
    train_df, test_df = balanced_pair_frames
    captured = {}
    _patch_curve_functions(monkeypatch, metrics_module, captured)

    metrics_module._binary_similarity_curves(
        train_df,
        test_df,
        distance_metric="euclidean",
        max_pairs=None,
        between_species=False,
        balance_pairs=True,
        remove_singleton_genera=False,
        test_only=True,
    )

    i1, i2, metric_code = captured["pairs"]
    assert metric_code == 1
    assert len(i1) == 12
    assert np.all(i1 != i2)

    expected_weights = _expected_balance_pair_weights(train_df, test_df)
    np.testing.assert_allclose(captured["roc"]["sample_weight"], expected_weights)
    np.testing.assert_allclose(captured["pr"]["sample_weight"], expected_weights)


def test_binary_similarity_curves_max_pairs_samples_without_replacement(
    monkeypatch,
    metrics_module,
    balanced_pair_frames,
):
    train_df, test_df = balanced_pair_frames
    captured = {}
    _patch_curve_functions(monkeypatch, metrics_module, captured)

    metrics_module._binary_similarity_curves(
        train_df,
        test_df,
        distance_metric="cosine",
        max_pairs=5,
        between_species=False,
        balance_pairs=False,
        remove_singleton_genera=False,
        test_only=False,
    )

    i1, i2, metric_code = captured["pairs"]
    assert metric_code == 0
    assert len(i1) == 5
    assert len(np.unique(np.stack([i1, i2], axis=1), axis=0)) == 5


def test_binary_similarity_curves_between_species_filters_same_species(
    monkeypatch,
    metrics_module,
    between_species_frames,
):
    train_df, test_df = between_species_frames
    captured = {}
    _patch_curve_functions(monkeypatch, metrics_module, captured)

    metrics_module._binary_similarity_curves(
        train_df,
        test_df,
        distance_metric="euclidean",
        max_pairs=5,
        between_species=True,
        balance_pairs=False,
        remove_singleton_genera=False,
        test_only=False,
    )

    i1, i2, _ = captured["pairs"]
    assert len(i1) == 5
    assert np.all(train_df.loc[i1, "species"].to_numpy() != test_df.loc[i2, "species"].to_numpy())


def test_binary_similarity_curves_remove_singleton_genera_drops_singleton_rows(
    monkeypatch,
    metrics_module,
    singleton_genera_frames,
):
    train_df, test_df = singleton_genera_frames
    captured = {}
    _patch_curve_functions(monkeypatch, metrics_module, captured)

    metrics_module._binary_similarity_curves(
        train_df,
        test_df,
        distance_metric="euclidean",
        max_pairs=100,
        between_species=False,
        balance_pairs=False,
        remove_singleton_genera=True,
        test_only=False,
    )

    i1, i2, _ = captured["pairs"]
    assert len(i1) == 4
    assert set(train_df.loc[i1, "genus"]) == {"Genus_1"}
    assert set(test_df.loc[i2, "genus"]) == {"Genus_1"}


def test_binary_curves_plot_wraps_gathered_embeddings(
    monkeypatch,
    metrics_module,
    balanced_pair_frames,
):
    train_df, test_df = balanced_pair_frames

    def fake_gather_embeddings(dataset, target, split_type, rki_disjoint=None):
        return {
            "metadata": {"dataset": dataset, "target": target, "split_type": split_type},
            "cosine": {"train": [train_df.copy(deep=True)], "test": [test_df.copy(deep=True)]},
            "clip_transformer": {"train": [train_df.copy(deep=True)], "test": [test_df.copy(deep=True)]},
            "cross_encoder": {"train": [train_df.copy(deep=True)], "test": [test_df.copy(deep=True)]},
        }

    def fake_compute_interpolated_curves(
        df1,
        df2,
        method,
        data_idx,
        interp_points,
        metric,
        max_pairs,
        between_species=False,
        balance_pairs=False,
        remove_singleton_genera=False,
        test_only=False,
    ):
        curve = np.linspace(0.0, 1.0, len(interp_points), dtype=np.float32)
        return (
            (
                curve,
                curve,
                curve,
            ),
            (0.5, 0.9, 0.8),
        )

    monkeypatch.setattr(metrics_module, "gather_embeddings", fake_gather_embeddings)
    monkeypatch.setattr(metrics_module, "_compute_interpolated_curves", fake_compute_interpolated_curves)

    result = metrics_module.binary_curves_plot(
        dataset="driams-c",
        target="species",
        split_type="species",
        test_only=False,
        max_pairs=5,
        cosine_ablation=False,
        n_jobs=1,
        between_species=False,
        balance_pairs=False,
        rki_disjoint=None,
        remove_singleton_genera=False,
    )

    assert "roc" in result and "precision_recall" in result and "fdr" in result
    assert "metrics_table" in result
    assert "cosine_intensity_agnostic" in result["metrics_table"]["method"].tolist()
    assert "cross_encoder" not in result["metrics_table"]["method"].tolist()
