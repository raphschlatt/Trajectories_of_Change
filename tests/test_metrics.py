from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from trajectories_of_change.contract import build_dataset_bundle
from trajectories_of_change.metrics_density import KDEDensity
from trajectories_of_change.metrics_kld import VocabularyKLD


def _publications() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Bibcode": "1995PASP..107..803U",
                "Year": 1995,
                "Author": ["Hawking, S. W."],
                "AuthorUID": ["uid:hawking"],
                "References": ["1962RSPSA.269...21B", "1962RSPSA.270..103S"],
                "tokens": ["particle", "creation", "black", "hole"],
                "embedding_2d_x": 0.0,
                "embedding_2d_y": 0.1,
            },
            {
                "Bibcode": "1995ApJ...000..001B",
                "Year": 1995,
                "Author": ["Bondi, H."],
                "AuthorUID": ["uid:bondi"],
                "References": ["1962RSPSA.269...21B", "1970ApJ...000..002B"],
                "tokens": ["gravity", "wave", "field"],
                "embedding_2d_x": 1.0,
                "embedding_2d_y": 1.1,
            },
            {
                "Bibcode": "1997PASP..107..804U",
                "Year": 1997,
                "Author": ["Hawking, S. W."],
                "AuthorUID": ["uid:hawking"],
                "References": ["1962RSPSA.270..103S", "1971ApJ...000..003C"],
                "tokens": ["black", "hole", "entropy"],
                "embedding_2d_x": 0.2,
                "embedding_2d_y": 0.3,
            },
            {
                "Bibcode": "1997ApJ...000..004D",
                "Year": 1997,
                "Author": ["Ellis, G. F. R."],
                "AuthorUID": ["uid:ellis"],
                "References": ["1970ApJ...000..002B", "1971ApJ...000..003C"],
                "tokens": ["cosmology", "field", "equation"],
                "embedding_2d_x": 1.2,
                "embedding_2d_y": 1.3,
            },
        ]
    )


def _references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Bibcode": "1962RSPSA.269...21B", "Author": ["Bondi, H."], "AuthorUID": ["uid:bondi"]},
            {"Bibcode": "1962RSPSA.270..103S", "Author": ["Sachs, R. K."], "AuthorUID": ["uid:sachs"]},
            {"Bibcode": "1970ApJ...000..002B", "Author": ["Ellis, G. F. R."], "AuthorUID": ["uid:ellis"]},
            {"Bibcode": "1971ApJ...000..003C", "Author": ["Carter, B."], "AuthorUID": ["uid:carter"]},
        ]
    )


def test_metric_smoke_tests_run_on_normalized_bundle() -> None:
    bundle = build_dataset_bundle(_publications(), _references())

    vocab = VocabularyKLD(
        bundle.publications,
        "uid:hawking",
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
    )
    vocab_sync, _ = vocab.calculate_kld_sync()
    assert not vocab_sync.empty

    density = KDEDensity(
        bundle.publications,
        "uid:hawking",
        window_size=1,
        skip_incomplete_slices=False,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
    )
    density_sync, _ = density.calculate_density_sync()
    assert not density_sync.empty


def test_density_result_reuses_field_kde_fits_for_async(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = build_dataset_bundle(_publications(), _references())
    density = KDEDensity(
        bundle.publications,
        "uid:hawking",
        window_size=1,
        skip_incomplete_slices=False,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
    )
    original_fit = density._fit_kde
    fit_calls = 0

    def counted_fit(coords):  # type: ignore[no-untyped-def]
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(coords)

    monkeypatch.setattr(density, "_fit_kde", counted_fit)
    result = density.result(include_async=True)

    assert result.async_df is not None
    assert fit_calls > 0
    assert fit_calls == len(density._field_kde_cache)


def test_kld_accepts_counter_tokens_equivalent_to_lists() -> None:
    list_corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": ["a", "a", "b"]},
            {"Bibcode": "t2", "Year": 2001, "Author": ["Target, T."], "tokens": ["a", "c"]},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "tokens": ["b", "b", "c"]},
            {"Bibcode": "f2", "Year": 2001, "Author": ["Field, F."], "tokens": ["c", "d"]},
        ]
    )
    counter_corpus = list_corpus.copy()
    counter_corpus["tokens"] = counter_corpus["tokens"].apply(Counter)

    kwargs = dict(
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
        top_k_kld_terms=None,
        allow_name_fallback=True,
    )
    list_model = VocabularyKLD(list_corpus, **kwargs)
    counter_model = VocabularyKLD(counter_corpus, **kwargs)

    pd.testing.assert_frame_equal(
        list_model.calculate_kld_sync()[0],
        counter_model.calculate_kld_sync()[0],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_frame_equal(
        list_model.perform_welch_tests_all_pairs(sync_only=True).sort_values(["target_slice", "term"]).reset_index(
            drop=True
        ),
        counter_model.perform_welch_tests_all_pairs(sync_only=True)
        .sort_values(["target_slice", "term"])
        .reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_target_mask_must_be_one_dimensional() -> None:
    # A non-1-D mask with a matching element count must be rejected, not silently
    # mis-broadcast (guards the per-target parallelism path against bad masks).
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": ["a", "b"]},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "tokens": ["b", "c"]},
        ]
    )
    with pytest.raises(ValueError, match="1-D"):
        VocabularyKLD(
            corpus,
            "Target, T.",
            target_mask=[[True], [False]],
            window_size=1,
            skip_incomplete_slices=False,
            allow_name_fallback=True,
        )


def test_sync_kld_contribution_cache_feeds_pointwise_and_welch() -> None:
    corpus = pd.DataFrame(
        [
            {"Bibcode": "t1", "Year": 2000, "Author": ["Target, T."], "tokens": ["a", "a", "b"]},
            {"Bibcode": "f1", "Year": 2000, "Author": ["Field, F."], "tokens": ["b", "c", "c"]},
        ]
    )
    model = VocabularyKLD(
        corpus,
        "Target, T.",
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
        top_k_kld_terms=2,
        allow_name_fallback=True,
    )

    sync_df, pointwise_df = model.calculate_kld_sync()
    cached = model._sync_contrib_cache[2000]
    welch_df = model.perform_welch_tests_all_pairs(sync_only=True)

    assert sync_df.loc[0, "kld_all"] == pytest.approx(float(cached.sum()))
    pointwise_by_term = pointwise_df.set_index("term")["kld_contribution"]
    for term, contribution in pointwise_by_term.items():
        assert contribution == pytest.approx(float(cached[model.vocab_index.get_loc(term)]))
    welch_by_term = welch_df.set_index("term")["kld_contribution"]
    for term, contribution in welch_by_term.items():
        assert contribution == pytest.approx(float(cached[model.vocab_index.get_loc(term)]))

