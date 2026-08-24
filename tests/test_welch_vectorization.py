from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

import trajectories_of_change.kld_core as kld_core_module
from trajectories_of_change.kld_core import DocumentFeatureMatrix, KLDCore
from trajectories_of_change.stats_utils import add_pvalue_adjustments


def _edge_case_core(*, top_k_kld_terms: int | None) -> KLDCore:
    corpus = pd.DataFrame(
        [
            {"Year": 2000, "tokens": ["constant", "target"], "is_target": True},
            {"Year": 2000, "tokens": ["constant", "target"], "is_target": True},
            {"Year": 2000, "tokens": ["constant", "field"], "is_target": False},
            {"Year": 2000, "tokens": ["constant", "field"], "is_target": False},
            {"Year": 2001, "tokens": ["constant", "solo"], "is_target": True},
            {"Year": 2001, "tokens": ["constant", "other"], "is_target": False},
            {"Year": 2002, "tokens": ["constant", "target-only"], "is_target": True},
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
        min_token_global_freq=1,
        min_docs_global_freq=1,
        max_vocab_size=None,
    )
    return KLDCore(
        matrix,
        target_mask=corpus["is_target"].to_numpy(dtype=bool),
        min_tokens_target_slice=1e-12,
        min_tokens_field_slice=1e-12,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        top_k_kld_terms=top_k_kld_terms,
    )


@pytest.mark.parametrize("top_k_kld_terms", [3, None])
def test_vectorized_welch_matches_scalar_reference_and_decisions(
    monkeypatch: pytest.MonkeyPatch,
    top_k_kld_terms: int | None,
) -> None:
    core = _edge_case_core(top_k_kld_terms=top_k_kld_terms)
    scipy_ttest = kld_core_module.ttest_ind_from_stats

    def scalar_reference(mean_t, std_t, n_t, mean_f, std_f, n_f, *, equal_var):  # type: ignore[no-untyped-def]
        arrays = np.broadcast_arrays(mean_t, std_t, n_t, mean_f, std_f, n_f)
        pvalues = [
            float(scipy_ttest(*values, equal_var=equal_var).pvalue)
            for values in zip(*(array.flat for array in arrays))
        ]
        return SimpleNamespace(pvalue=np.asarray(pvalues, dtype=float))

    monkeypatch.setattr(kld_core_module, "ttest_ind_from_stats", scalar_reference)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scalar = core.perform_welch_tests(sync_only=False)

    vector_calls: list[int] = []

    def vector_spy(mean_t, std_t, n_t, mean_f, std_f, n_f, *, equal_var):  # type: ignore[no-untyped-def]
        vector_calls.append(int(np.asarray(mean_t).size))
        return scipy_ttest(mean_t, std_t, n_t, mean_f, std_f, n_f, equal_var=equal_var)

    monkeypatch.setattr(kld_core_module, "ttest_ind_from_stats", vector_spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vectorized = core.perform_welch_tests(sync_only=False)

    assert list(vectorized.columns) == list(scalar.columns)
    assert vectorized[["target_slice", "field_slice", "term"]].equals(
        scalar[["target_slice", "field_slice", "term"]]
    )
    pd.testing.assert_frame_equal(
        vectorized,
        scalar,
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
    )

    rows_per_pair = (
        vectorized.groupby(["target_slice", "field_slice"], sort=False)
        .size()
        .tolist()
    )
    assert vector_calls == rows_per_pair
    assert vectorized["pvalue"].isna().any()

    for method in ("none", "bonferroni", "holm", "fdr_bh", "fdr_by"):
        scalar_adjusted = add_pvalue_adjustments(
            scalar,
            method=method,
            group_cols=["target_slice", "field_slice"],
        )["p_adj"]
        vector_adjusted = add_pvalue_adjustments(
            vectorized,
            method=method,
            group_cols=["target_slice", "field_slice"],
        )["p_adj"]
        for alpha in (0.05, 0.1, 0.2):
            assert (scalar_adjusted < alpha).equals(vector_adjusted < alpha)


def test_vectorized_welch_preserves_empty_schema() -> None:
    corpus = pd.DataFrame(
        [
            {"Year": 2000, "tokens": [], "is_target": True},
            {"Year": 2000, "tokens": [], "is_target": False},
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
        min_token_global_freq=1,
        min_docs_global_freq=1,
        max_vocab_size=None,
    )
    core = KLDCore(
        matrix,
        target_mask=corpus["is_target"].to_numpy(dtype=bool),
        min_tokens_target_slice=1e-12,
        min_tokens_field_slice=1e-12,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        top_k_kld_terms=None,
    )

    result = core.perform_welch_tests(sync_only=False)

    assert result.empty
    assert list(result.columns) == [
        "target_slice",
        "field_slice",
        "term",
        "pvalue",
        "kld_contribution",
        "mean_target",
        "mean_field",
    ]
