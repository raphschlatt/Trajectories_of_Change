from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trajectories_of_change import (
    MetricResult,
    load_dataset_bundle,
    run_metric,
    run_metrics,
    run_top_authors_metrics_from_parquets,
)
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA = ROOT / "examples" / "data"
PUBLICATIONS = EXAMPLE_DATA / "publications.parquet"
REFERENCES = EXAMPLE_DATA / "references.parquet"
TARGET_UID = "uid:stable_vocab_distinct"


def test_run_metric_supports_all_simple_metric_keys() -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES, auto_discover_sidecars=True)

    results = {
        metric: run_metric(
            bundle,
            metric=metric,
            target_author_uid=TARGET_UID,
            window_size=1,
            top_k_kld_terms=2,
            run_welch=True,
            skip_incomplete_slices=False,
        )
        for metric in ("own_vocab", "density", "ref_vocab", "citation_identity")
    }

    assert all(isinstance(result, MetricResult) for result in results.values())
    assert results["own_vocab"].metric == "own_vocab"
    assert results["density"].metric == "density"
    assert results["ref_vocab"].metric == "ref_vocab"
    assert results["citation_identity"].metric == "citation_identity"
    assert results["own_vocab"].async_df is not None
    assert results["ref_vocab"].async_df is not None
    assert results["citation_identity"].async_df is not None
    assert results["density"].async_df is not None
def test_run_metric_accepts_parquet_paths() -> None:
    result = run_metric(
        PUBLICATIONS,
        REFERENCES,
        metric="own_vocab",
        target_author_uid=TARGET_UID,
        window_size=1,
        top_k_kld_terms=2,
        include_async=False,
        run_welch=False,
        skip_incomplete_slices=False,
    )

    assert result.metric == "own_vocab"
    assert not result.sync.empty
    assert result.async_df is None
    assert result.welch is None


def test_run_metric_rejects_unknown_metric() -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES)

    with pytest.raises(ValueError, match="metric must be one of"):
        run_metric(bundle, metric="unknown", target_author_uid=TARGET_UID)

    with pytest.raises(ValueError, match="metric must be one of"):
        run_metric(bundle, metric="own-vocab", target_author_uid=TARGET_UID)


def test_run_metric_rejects_typoed_metric_option_with_suggestion() -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES)

    with pytest.raises(TypeError, match="bandwith.*density_bandwidth"):
        run_metric(
            bundle,
            metric="density",
            target_author_uid=TARGET_UID,
            density_bandwith=0.5,
        )


def test_run_metrics_matches_existing_full_run() -> None:
    kwargs = dict(
        targets=[TARGET_UID],
        select_by="uid",
        window_size=1,
        skip_incomplete_slices=False,
        top_k_kld_terms=2,
        run_welch=False,
        include_async=False,
        show_progress=False,
        n_jobs=1,
    )

    expected = run_top_authors_metrics_from_parquets(PUBLICATIONS, REFERENCES, **kwargs)
    actual = run_metrics(PUBLICATIONS, REFERENCES, **kwargs)

    pd.testing.assert_frame_equal(actual, expected)


def test_run_metrics_bundle_path_has_exact_path_parity_without_temp_parquet(monkeypatch) -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES, auto_discover_sidecars=True)
    kwargs = dict(
        targets=[TARGET_UID],
        select_by="uid",
        include=("own_vocab",),
        window_size=1,
        skip_incomplete_slices=False,
        top_k_kld_terms=2,
        run_welch=False,
        include_async=False,
        show_progress=False,
        n_jobs=1,
    )
    expected = run_metrics(
        PUBLICATIONS,
        REFERENCES,
        auto_discover_sidecars=True,
        **kwargs,
    )

    def unexpected_write(*args, **kwargs):
        raise AssertionError("the in-memory bundle path must not write temporary Parquets")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", unexpected_write)
    actual = run_metrics(bundle, **kwargs)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_run_metrics_include_argument_maps_to_existing_toggles() -> None:
    result = run_metrics(
        PUBLICATIONS,
        REFERENCES,
        targets=[TARGET_UID],
        include=("own_vocab", "density"),
        window_size=1,
        skip_incomplete_slices=False,
        top_k_kld_terms=2,
        run_welch=False,
        show_progress=False,
    )

    assert "vocab_kld_all_level" in result.columns
    assert "density_neglog_level" in result.columns
    assert "cocit_kld_all_level" not in result.columns
    assert "ref_vocab_kld_all_level" not in result.columns


def test_run_metrics_rejects_non_core_include() -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        run_metrics(PUBLICATIONS, REFERENCES, top_n=1, include=("citation_image",))


def test_run_metrics_rejects_typoed_advanced_option() -> None:
    with pytest.raises(TypeError, match="min_docs_target_slic.*min_docs_target_slice"):
        run_metrics(
            PUBLICATIONS,
            REFERENCES,
            targets=[TARGET_UID],
            include=("own_vocab",),
            show_progress=False,
            min_docs_target_slic=1,
        )
