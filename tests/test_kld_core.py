from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trajectories_of_change.kld_core import DocumentFeatureMatrix, KLDCore
from trajectories_of_change.metrics_kld import VocabularyKLD


def _vocab_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Bibcode": "t2000",
                "Year": 2000,
                "Author": ["Target, T."],
                "author_uids": ["uid:target"],
                "tokens": ["alpha", "shared", "shared"],
            },
            {
                "Bibcode": "f2000",
                "Year": 2000,
                "Author": ["Field, F."],
                "author_uids": ["uid:field"],
                "tokens": ["beta", "shared"],
            },
            {
                "Bibcode": "t2001",
                "Year": 2001,
                "Author": ["Target, T."],
                "author_uids": ["uid:target"],
                "tokens": ["gamma", "shared"],
            },
            {
                "Bibcode": "f2001",
                "Year": 2001,
                "Author": ["Field, G."],
                "author_uids": ["uid:field2"],
                "tokens": ["beta", "delta", "shared"],
            },
        ]
    )


def test_kld_core_matches_vocabulary_sync_async_and_welch() -> None:
    corpus = _vocab_fixture()
    target_mask = corpus["author_uids"].map(lambda values: "uid:target" in values).to_numpy(dtype=bool)
    kwargs = {
        "window_size": 1,
        "skip_incomplete_slices": False,
        "min_token_global_freq": 1.0,
        "min_docs_global_freq": 1,
        "min_tokens_target_slice": 1e-12,
        "min_tokens_field_slice": 1e-12,
        "min_docs_target_slice": 1,
        "min_docs_field_slice": 1,
        "top_k_kld_terms": 3,
        "min_docs_target_test": 1,
        "min_docs_field_test": 1,
        "target_mask": target_mask,
        "allow_name_fallback": False,
    }
    reference = VocabularyKLD(corpus, "", target_author_uid="uid:target", **kwargs)
    reference_sync, reference_pointwise = reference.calculate_kld_sync()
    reference_async = reference.calculate_kld_async()
    reference_welch = reference.perform_welch_tests_all_pairs(sync_only=True)

    matrix = DocumentFeatureMatrix.from_token_frame(
        corpus,
        year_col="Year",
        token_col="tokens",
        start_year=None,
        end_year=None,
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=1.0,
        min_docs_global_freq=1,
        max_vocab_size=None,
    )
    core = KLDCore(
        matrix,
        target_mask=target_mask,
        lambda_param=0.5,
        epsilon=1e-12,
        min_tokens_target_slice=1e-12,
        min_tokens_field_slice=1e-12,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        top_k_kld_terms=3,
    )
    sync, pointwise = core.calculate_sync()
    async_df = core.calculate_async()
    welch = core.perform_welch_tests(sync_only=True)

    pd.testing.assert_frame_equal(sync, reference_sync, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(
        pointwise.sort_values(["slice", "term"]).reset_index(drop=True),
        reference_pointwise.sort_values(["slice", "term"]).reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        async_df.sort_values(["target_slice", "field_slice"]).reset_index(drop=True),
        reference_async.sort_values(["target_slice", "field_slice"]).reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        welch.sort_values(["target_slice", "field_slice", "term"]).reset_index(drop=True),
        reference_welch.sort_values(["target_slice", "field_slice", "term"]).reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_async_welch_reuses_slice_moments_across_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = pd.DataFrame(
        [
            {"Year": 2000, "tokens": ["alpha", "shared", "old"], "author_uids": ["uid:target"]},
            {"Year": 2000, "tokens": ["beta", "shared", "field"], "author_uids": ["uid:field"]},
            {"Year": 2001, "tokens": ["alpha", "gamma", "shared"], "author_uids": ["uid:target"]},
            {"Year": 2001, "tokens": ["beta", "delta", "shared"], "author_uids": ["uid:field"]},
            {"Year": 2002, "tokens": ["gamma", "future", "shared"], "author_uids": ["uid:target"]},
            {"Year": 2002, "tokens": ["delta", "future", "shared"], "author_uids": ["uid:field"]},
        ]
    )
    target_mask = corpus["author_uids"].map(lambda values: "uid:target" in values).to_numpy(dtype=bool)
    matrix = DocumentFeatureMatrix.from_token_frame(
        corpus,
        year_col="Year",
        token_col="tokens",
        start_year=None,
        end_year=None,
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=1.0,
        min_docs_global_freq=1,
        max_vocab_size=None,
        precompute_slice_moments=False,
    )
    original = DocumentFeatureMatrix.moments_for_mask
    calls: list[tuple[int, int | None]] = []

    def counted_moments(self, label, mask, allowed_indices):  # type: ignore[no-untyped-def]
        if self is matrix:
            calls.append((int(label), None if allowed_indices is None else len(tuple(allowed_indices))))
        return original(self, label, mask, allowed_indices)

    monkeypatch.setattr(DocumentFeatureMatrix, "moments_for_mask", counted_moments)
    core = KLDCore(
        matrix,
        target_mask=target_mask,
        lambda_param=0.5,
        epsilon=1e-12,
        min_tokens_target_slice=1e-12,
        min_tokens_field_slice=1e-12,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        top_k_kld_terms=2,
    )

    welch = core.perform_welch_tests(sync_only=False)

    assert not welch.empty
    assert len(calls) <= 2 * len(matrix.slices)
    assert all(size is None for _, size in calls)


def test_sparse_moments_match_scalar_reference_for_edge_cases() -> None:
    corpus = pd.DataFrame(
        [
            {"Year": 2000, "tokens": {"alpha": 2.0, "shared": 1.0}},
            {"Year": 2000, "tokens": {}},
            {"Year": 2000, "tokens": {"beta": 3.0, "shared": 2.0}},
            {"Year": 2001, "tokens": {"alpha": 1.0, "gamma": 4.0}},
        ]
    )
    matrix = DocumentFeatureMatrix.from_token_frame(
        corpus,
        year_col="Year",
        token_col="tokens",
        start_year=None,
        end_year=None,
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=1.0,
        min_docs_global_freq=1,
        max_vocab_size=None,
    )
    alpha = matrix.vocab_lookup["alpha"]
    shared = matrix.vocab_lookup["shared"]

    def scalar(mask: np.ndarray, allowed: tuple[int, ...] | None) -> tuple[dict[int, float], dict[int, float], int]:
        selected = set(allowed) if allowed is not None else None
        sums: dict[int, float] = {}
        sum_sq: dict[int, float] = {}
        n = 0
        combined = np.asarray(mask, dtype=bool) & matrix.slice_masks[2000]
        for row in np.flatnonzero(combined):
            length = float(matrix.doc_lengths[row])
            if length <= 0:
                continue
            n += 1
            for idx, count in matrix.doc_counts[row].items():
                if selected is not None and idx not in selected:
                    continue
                relative = float(count) / length
                sums[idx] = sums.get(idx, 0.0) + relative
                sum_sq[idx] = sum_sq.get(idx, 0.0) + relative * relative
        return sums, sum_sq, n

    masks = [
        np.array([True, True, True, True]),
        np.array([True, False, False, False]),
        np.zeros(len(corpus), dtype=bool),
    ]
    allowed_sets = [None, (alpha, shared), (), (shared, shared, -1, matrix.vocab_size + 1)]
    for mask in masks:
        for allowed in allowed_sets:
            actual_sums, actual_sq, actual_n = matrix.moments_for_mask(2000, mask, allowed)
            expected_sums, expected_sq, expected_n = scalar(mask, allowed)
            assert actual_n == expected_n
            assert actual_sums.keys() == expected_sums.keys()
            assert actual_sq.keys() == expected_sq.keys()
            assert all(
                np.isclose(actual_sums[idx], expected_sums[idx], rtol=1e-12, atol=1e-12) for idx in expected_sums
            )
            assert all(np.isclose(actual_sq[idx], expected_sq[idx], rtol=1e-12, atol=1e-12) for idx in expected_sq)
