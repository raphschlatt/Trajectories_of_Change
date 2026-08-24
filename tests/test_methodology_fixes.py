from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from trajectories_of_change.metrics_density import KDEDensity
from trajectories_of_change.metrics_kld import KLDPrecompute, VocabularyKLD
from trajectories_of_change.multimetric import _prepare_sync_welch, summarize_kld_sync


def _kld_model(corpus: pd.DataFrame, **kwargs) -> VocabularyKLD:
    defaults = dict(
        target_name="Target, T.",
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=1,
        min_docs_global_freq=1,
        min_tokens_target_slice=1,
        min_tokens_field_slice=1,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        allow_name_fallback=True,
    )
    defaults.update(kwargs)
    return VocabularyKLD(corpus, **defaults)


def _direct_term_stats(
    token_rows,
    *,
    global_vocab_set: set[str],
    allowed_terms: set[str],
    min_docs_required: int,
) -> tuple[dict[str, tuple[float, float, int]], int]:
    sums: dict[str, float] = {}
    sum_sq: dict[str, float] = {}
    n_analyzable = 0
    for tokens in token_rows:
        counts_all = Counter(token for token in tokens if token in global_vocab_set)
        doc_len = sum(counts_all.values())
        if doc_len <= 0:
            continue
        n_analyzable += 1
        inv_len = 1.0 / float(doc_len)
        for term, count in counts_all.items():
            if term not in allowed_terms:
                continue
            rel = float(count) * inv_len
            sums[term] = sums.get(term, 0.0) + rel
            sum_sq[term] = sum_sq.get(term, 0.0) + rel * rel

    if n_analyzable < min_docs_required:
        return {}, n_analyzable
    if n_analyzable == 1:
        return {term: (value, 0.0, 1) for term, value in sums.items()}, 1

    stats: dict[str, tuple[float, float, int]] = {}
    n_float = float(n_analyzable)
    for term, sum_val in sums.items():
        variance = (float(sum_sq.get(term, 0.0)) - (float(sum_val) ** 2) / n_float) / float(
            max(n_analyzable - 1, 1)
        )
        stats[term] = (float(sum_val) / n_float, float(np.sqrt(max(variance, 0.0))), n_analyzable)
    return stats, n_analyzable


def test_welch_term_stats_use_full_analyzable_document_length() -> None:
    corpus = pd.DataFrame(
        [
            {
                "Bibcode": "t1",
                "Year": 2000,
                "Author": ["Target, T."],
                "tokens": ["candidate"] + ["other"] * 99,
            },
            {
                "Bibcode": "t2",
                "Year": 2000,
                "Author": ["Target, T."],
                "tokens": ["other"] * 100,
            },
            {
                "Bibcode": "f1",
                "Year": 2000,
                "Author": ["Field, F."],
                "tokens": ["other"] * 100,
            },
        ]
    )
    model = _kld_model(corpus)

    stats, n_docs = _direct_term_stats(
        corpus.iloc[:2]["tokens"],
        global_vocab_set=model.global_vocab_set,
        allowed_terms={"candidate"},
        min_docs_required=1,
    )

    assert n_docs == 2
    assert stats["candidate"][0] == pytest.approx(0.005)


def test_welch_excludes_documents_without_global_vocab_tokens() -> None:
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": ["candidate", "other"]},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "tokens": ["other", "other"]},
        ]
    )
    model = _kld_model(corpus)
    df_slice = pd.DataFrame(
        [
            {"tokens": ["candidate", "other"]},
            {"tokens": ["not_in_global_vocab"]},
        ]
    )

    stats, n_docs = _direct_term_stats(
        df_slice["tokens"],
        global_vocab_set=model.global_vocab_set,
        allowed_terms={"candidate"},
        min_docs_required=1,
    )

    assert n_docs == 1
    assert stats["candidate"][0] == pytest.approx(0.5)


def test_top_k_none_tests_terms_observed_in_target_or_field() -> None:
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": ["target_only", "common"]},
            {"Bibcode": "t2", "Year": 2000, "Author": ["Target, T."], "tokens": ["target_only", "common"]},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "tokens": ["field_only", "common"]},
            {"Bibcode": "f2", "Year": 2000, "Author": ["Field, F."], "tokens": ["field_only", "common"]},
        ]
    )
    model = _kld_model(corpus, top_k_kld_terms=None)

    welch = model.perform_welch_tests_all_pairs(sync_only=True)

    assert {"target_only", "field_only", "common"}.issubset(set(welch["term"]))


def test_precomputed_welch_moments_match_direct_document_stats() -> None:
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": ["target", "common"]},
            {"Bibcode": "t2", "Year": 2000, "Author": ["Target, T."], "tokens": ["common", "common"]},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "tokens": ["field", "common"]},
            {"Bibcode": "f2", "Year": 2000, "Author": ["Field, F."], "tokens": ["field", "field"]},
            {"Bibcode": "empty", "Year": 2000, "Author": ["Field, F."], "tokens": ["out_of_vocab"]},
        ]
    )
    model = _kld_model(corpus, min_docs_global_freq=2)
    allowed_terms = {"target", "common", "field"}
    allowed_indices = tuple(model.vocab_index.get_loc(term) for term in allowed_terms if term in model.vocab_index)

    direct_target, n_target = _direct_term_stats(
        model.target_corpus["tokens"],
        global_vocab_set=model.global_vocab_set,
        allowed_terms=allowed_terms,
        min_docs_required=1,
    )
    direct_field, n_field = _direct_term_stats(
        model.field_corpus["tokens"],
        global_vocab_set=model.global_vocab_set,
        allowed_terms=allowed_terms,
        min_docs_required=1,
    )
    fast_target, fast_n_target = model.core._compute_term_stats(
        2000,
        allowed_indices=allowed_indices,
        side="target",
        min_docs_required=1,
    )
    fast_field, fast_n_field = model.core._compute_term_stats(
        2000,
        allowed_indices=allowed_indices,
        side="field",
        min_docs_required=1,
    )

    assert fast_n_target == n_target
    assert fast_n_field == n_field
    for term, direct_values in direct_target.items():
        assert fast_target[term] == pytest.approx(direct_values)
    for term, direct_values in direct_field.items():
        assert fast_field[term] == pytest.approx(direct_values)


def test_explicit_full_slice_moment_precompute_matches_lazy_stats() -> None:
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": ["a", "b", "b"]},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "tokens": ["b", "c", "c"]},
            {"Bibcode": "f2", "Year": 2000, "Author": ["Field, F."], "tokens": ["a", "c"]},
        ]
    )
    lazy = _kld_model(corpus, top_k_kld_terms=None)
    precompute = KLDPrecompute(
        corpus,
        year_col="Year",
        token_col="tokens",
        start_year=None,
        end_year=None,
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=1,
        min_docs_global_freq=1,
        max_vocab_size=None,
        precompute_slice_moments=True,
    )
    eager = _kld_model(corpus, top_k_kld_terms=None, shared_precompute=precompute)

    lazy_welch = lazy.perform_welch_tests_all_pairs(sync_only=True).sort_values("term").reset_index(drop=True)
    eager_welch = eager.perform_welch_tests_all_pairs(sync_only=True).sort_values("term").reset_index(drop=True)

    assert precompute.precompute_slice_moments is True
    pd.testing.assert_frame_equal(lazy_welch, eager_welch, check_exact=False, atol=1e-12, rtol=1e-12)


def test_fractional_token_weights_survive_kld_precompute_without_truncation() -> None:
    corpus = pd.DataFrame(
        [
            {
                "Bibcode": "t1",
                "Year": 2000,
                "Author": ["Target, T."],
                "tokens": {"shared": 0.5, "target_pair": 0.5},
            },
            {
                "Bibcode": "f1",
                "Year": 2000,
                "Author": ["Field, F."],
                "tokens": {"shared": 1.0},
            },
        ]
    )
    model = _kld_model(corpus, min_token_global_freq=0)

    sync, pointwise = model.calculate_kld_sync()

    assert model.precompute.global_total == pytest.approx(2.0)
    assert model.slice_token_counts[2000]["target_tokens"] == pytest.approx(1.0)
    assert model.slice_token_counts[2000]["field_tokens"] == pytest.approx(1.0)
    assert not sync.empty
    assert sync.loc[0, "kld_all"] >= 0
    assert set(pointwise["term"]) == {"shared", "target_pair"}


def test_fractional_kld_thresholds_are_not_truncated_to_integers() -> None:
    corpus = pd.DataFrame(
        [
            {
                "Bibcode": "t1",
                "Year": 2000,
                "Author": ["Target, T."],
                "tokens": {"fractional": 0.3, "shared": 0.3},
            },
            {
                "Bibcode": "f1",
                "Year": 2000,
                "Author": ["Field, F."],
                "tokens": {"fractional": 0.3, "shared": 0.3},
            },
        ]
    )

    loose = _kld_model(
        corpus,
        min_token_global_freq=0.5,
        min_tokens_target_slice=0.5,
        min_tokens_field_slice=0.5,
    )
    strict_global = _kld_model(
        corpus,
        min_token_global_freq=0.7,
        min_tokens_target_slice=0.5,
        min_tokens_field_slice=0.5,
    )
    strict_slice = _kld_model(
        corpus,
        min_token_global_freq=0.5,
        min_tokens_target_slice=0.7,
        min_tokens_field_slice=0.5,
    )

    loose_sync, _ = loose.calculate_kld_sync()
    strict_global_sync, _ = strict_global.calculate_kld_sync()
    strict_slice_sync, _ = strict_slice.calculate_kld_sync()

    assert loose.vocab_size == 2
    assert not loose_sync.empty
    assert strict_global.vocab_size == 0
    assert strict_global_sync.empty
    assert strict_slice.vocab_size == 2
    assert strict_slice_sync.empty


def test_kld_precompute_sparse_counts_match_masked_counter_sums() -> None:
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": {"a": 0.25, "b": 0.75}},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "tokens": {"b": 1.5, "c": 0.5}},
            {"Bibcode": "f2", "Year": 2001, "Author": ["Field, F."], "tokens": ["a", "c", "c"]},
        ]
    )
    precompute = KLDPrecompute(
        corpus,
        year_col="Year",
        token_col="tokens",
        start_year=None,
        end_year=None,
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=0,
        min_docs_global_freq=1,
        max_vocab_size=None,
    )

    assert precompute.doc_term_matrix.shape == (3, precompute.vocab_size)
    assert precompute.doc_term_matrix.nnz > 0

    counts, total = precompute.counts_for_mask(np.array([True, True, False]))
    by_term = dict(zip(precompute.global_vocab, counts))

    assert total == pytest.approx(3.0)
    assert by_term["a"] == pytest.approx(0.25)
    assert by_term["b"] == pytest.approx(2.25)
    assert by_term["c"] == pytest.approx(0.5)


def test_bh_adjustment_is_applied_per_slice() -> None:
    raw = pd.DataFrame(
        [
            {"target_slice": 2000, "field_slice": 2000, "term": "a", "pvalue": 0.01},
            {"target_slice": 2000, "field_slice": 2000, "term": "b", "pvalue": 0.04},
            {"target_slice": 2001, "field_slice": 2001, "term": "c", "pvalue": 0.03},
            {"target_slice": 2001, "field_slice": 2001, "term": "d", "pvalue": 0.90},
        ]
    )

    adjusted = _prepare_sync_welch(raw, method="fdr_bh", scope="slice")

    by_term = adjusted.set_index("term")["p_adj"]
    assert by_term["a"] == pytest.approx(0.02)
    assert by_term["b"] == pytest.approx(0.04)
    assert by_term["c"] == pytest.approx(0.06)
    assert by_term["d"] == pytest.approx(0.90)


def test_kld_sig_abs_preserves_magnitude_when_signed_contributions_cancel() -> None:
    sync = pd.DataFrame([{"slice": 2000, "kld_all": 0.10}])
    welch = pd.DataFrame(
        [
            {"slice": 2000, "term": "over", "p_adj": 0.01, "kld_contribution": 0.05},
            {"slice": 2000, "term": "under", "p_adj": 0.01, "kld_contribution": -0.05},
        ]
    )

    summary = summarize_kld_sync(sync, welch, alpha=0.05)

    assert summary["kld_sig_level"] == pytest.approx(0.0)
    assert summary["kld_sig_abs_level"] == pytest.approx(0.10)
    assert summary["kld_sig_abs_ratio"] == pytest.approx(1.0)


def test_async_kld_uses_independent_target_and_field_slice_thresholds() -> None:
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": ["target"]},
            {"Bibcode": "t2", "Year": 2000, "Author": ["Target, T."], "tokens": ["target"]},
            {"Bibcode": "f1", "Year": 2001, "Author": ["Field, F."], "tokens": ["field"]},
            {"Bibcode": "f2", "Year": 2001, "Author": ["Field, F."], "tokens": ["field"]},
        ]
    )
    model = _kld_model(
        corpus,
        min_docs_target_slice=2,
        min_docs_field_slice=2,
    )

    async_df = model.calculate_kld_async()

    assert set(async_df["target_slice"]) == {2000}
    assert set(async_df["field_slice"]) == {2001}


def test_invalid_window_size_raises() -> None:
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": ["a"], "x": 0.0, "y": 0.0},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "tokens": ["b"], "x": 1.0, "y": 1.0},
        ]
    )

    with pytest.raises(ValueError, match="window_size"):
        _kld_model(corpus, window_size=0)
    with pytest.raises(ValueError, match="window_size"):
        KDEDensity(
            corpus.rename(columns={"x": "embedding_2d_x", "y": "embedding_2d_y"}),
            "Target, T.",
            window_size=-1,
        )


def test_density_supports_5d_standardized_coordinates() -> None:
    rows = []
    for i in range(12):
        is_target = i % 3 == 0
        row = {
            "Bibcode": f"d{i}",
            "Year": 2000 + i % 2,
            "Author": ["Target, T."] if is_target else ["Field, F."],
        }
        for dim in range(5):
            row[f"embedding_5d_{dim}"] = float(i + dim)
        rows.append(row)
    corpus = pd.DataFrame(rows)

    density = KDEDensity(
        corpus,
        "Target, T.",
        embedding_cols=[f"embedding_5d_{dim}" for dim in range(5)],
        window_size=1,
        skip_incomplete_slices=False,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        allow_name_fallback=True,
    )
    sync_df, point_df = density.calculate_density_sync()

    assert density.standardize is True
    assert len(density.embedding_cols) == 5
    assert np.isfinite(sync_df["density_neglog_median"]).all()
    assert np.isfinite(point_df["density_neglog"]).all()


def test_density_rejects_nonfinite_or_nonnumeric_coordinates() -> None:
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "embedding_2d_x": 0.0, "embedding_2d_y": 0.0},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "embedding_2d_x": "bad", "embedding_2d_y": 1.0},
        ]
    )

    with pytest.raises(ValueError, match="finite numeric"):
        KDEDensity(corpus, "Target, T.", allow_name_fallback=True)
