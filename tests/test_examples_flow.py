from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from trajectories_of_change import (
    KDEDensity,
    MetricResult,
    VocabularyKLD,
    load_dataset_bundle,
    run_metric,
    run_metrics,
)
from trajectories_of_change.defaults import (
    DEFAULT_ALPHA,
    DEFAULT_MULTIPLE_TESTING,
    DEFAULT_MULTIPLE_TESTING_SCOPE,
    DEFAULT_TOP_K_KLD_TERMS,
    DEFAULT_WINDOW_SIZE,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA = ROOT / "examples" / "data"
PUBLICATIONS = EXAMPLE_DATA / "publications.parquet"
REFERENCES = EXAMPLE_DATA / "references.parquet"
TARGET_UID = "uid:stable_vocab_distinct"
TARGET_LABEL = "Stable Vocabulary, V."
TARGETS = [
    "uid:field_like",
    "uid:stable_vocab_distinct",
    "uid:spiky_vocab_distinct",
    "uid:citation_distinct",
    "uid:density_shift",
    "uid:correlated_distinct",
    "uid:converging_distinct",
    "uid:geometry_trap",
]


def test_example_bundle_supports_core_metrics() -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES, auto_discover_sidecars=True)

    assert len(bundle.publications) == 340
    assert len(bundle.references) == 32
    assert bundle.manifest is not None
    assert bundle.manifest["run_id"] == "synthetic-oracle-v1-test"
    assert bundle.manifest["oracle"]["expected"]["density_geometric_trap"] == "uid:geometry_trap"
    assert bundle.validation is not None
    assert bundle.validation.errors == []
    assert bundle.validation.metric_availability["vocabulary_kld"] is True
    assert bundle.validation.metric_availability["density"] is True
    assert bundle.validation.metric_availability["cocitation_authors"] is True
    assert bundle.validation.metric_availability["author_identity"] is True
    assert bundle.validation.metric_availability["referenced_vocabulary"] is True
    assert "Citation Count" in bundle.publications.columns


@pytest.mark.parametrize(
    "embedding_cols",
    [
        ("embedding_2d_x", "embedding_2d_y"),
        tuple(f"embedding_5d_{idx}" for idx in range(5)),
        tuple(f"embedding_10d_{idx}" for idx in range(10)),
    ],
)
def test_example_density_runs_for_2d_5d_and_10d(embedding_cols) -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES)
    density = KDEDensity(
        bundle.publications,
        TARGET_LABEL,
        target_author_uid=TARGET_UID,
        embedding_cols=embedding_cols,
        window_size=DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices=False,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        allow_name_fallback=False,
    )

    sync_df, point_df = density.calculate_density_sync()

    assert not sync_df.empty
    assert not point_df.empty
    assert np.isfinite(sync_df["density_neglog_median"].to_numpy(dtype=float)).all()
    assert np.isfinite(point_df["density_neglog"].to_numpy(dtype=float)).all()


def test_example_multimetric_run_matches_quickstart_contract() -> None:
    df = run_metrics(
        PUBLICATIONS,
        REFERENCES,
        auto_discover_sidecars=True,
        targets=[TARGET_UID],
        select_by="uid",
        window_size=DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices=False,
        top_k_kld_terms=DEFAULT_TOP_K_KLD_TERMS,
        citation_identity_counting="multiplicity",
        target_exclusion="none",
        remove_self_loops=False,
        show_progress=False,
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["author_uid"] == TARGET_UID
    assert row["top_k_kld_terms"] == DEFAULT_TOP_K_KLD_TERMS
    assert row["multiple_testing"] == DEFAULT_MULTIPLE_TESTING
    assert row["multiple_testing_scope"] == DEFAULT_MULTIPLE_TESTING_SCOPE
    assert np.isfinite(float(row["density_neglog_level"]))
    assert float(row["vocab_kld_all_level"]) > 10.0
    assert np.isfinite(float(row["cocit_kld_all_level"]))
    assert "ref_vocab_kld_all_level" in df.columns
    assert np.isfinite(float(row["ref_vocab_kld_all_level"]))


def test_example_simple_metric_facade_matches_quickstart_contract() -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES, auto_discover_sidecars=True)

    result = run_metric(
        bundle,
        metric="citation_identity",
        target_author_uid=TARGET_UID,
        window_size=DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices=False,
        top_k_kld_terms=2,
    )

    assert isinstance(result, MetricResult)
    assert result.metric == "citation_identity"
    assert result.async_df is not None
    assert result.welch is not None


def test_example_dashboard_helpers_return_figures_without_export(tmp_path) -> None:
    pytest.importorskip("plotly")
    from trajectories_of_change.plotting.density import plot_density_dashboards
    from trajectories_of_change.plotting.kld import (
        plot_kld_dashboards,
        prepare_kld_dashboard_inputs,
    )
    from trajectories_of_change.plotting.multimetric import (
        plot_correlation_heatmaps,
        plot_level_agreement,
        plot_slope_agreement,
    )

    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES)
    metrics_df = run_metrics(
        PUBLICATIONS,
        REFERENCES,
        targets=TARGETS[:4],
        select_by="uid",
        window_size=DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices=False,
        top_k_kld_terms=DEFAULT_TOP_K_KLD_TERMS,
        show_progress=False,
    )
    assert plot_level_agreement(metrics_df) is not None
    assert plot_slope_agreement(metrics_df) is not None
    assert plot_correlation_heatmaps(metrics_df) is not None

    vkld = VocabularyKLD(
        bundle.publications,
        TARGET_LABEL,
        target_author_uid=TARGET_UID,
        window_size=DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices=False,
        top_k_kld_terms=DEFAULT_TOP_K_KLD_TERMS,
        min_token_global_freq=1,
        min_docs_global_freq=1,
        min_tokens_target_slice=1,
        min_tokens_field_slice=1,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        allow_name_fallback=False,
    )
    kld_inputs = prepare_kld_dashboard_inputs(
        vkld.calculate_kld_sync()[0],
        vkld.calculate_kld_async(),
        vkld.perform_welch_tests_all_pairs(),
        alpha=DEFAULT_ALPHA,
        multiple_testing=DEFAULT_MULTIPLE_TESTING,
        multiple_testing_scope="pair",
    )
    kld_figs = plot_kld_dashboards(
        *kld_inputs,
        target_name=TARGET_LABEL,
        alpha=DEFAULT_ALPHA,
        pointwise_alpha=DEFAULT_ALPHA,
        window_size=DEFAULT_WINDOW_SIZE,
        sync_plot_width=800,
        sync_plot_height=360,
        export=False,
        show=False,
    )

    density = KDEDensity(
        bundle.publications,
        TARGET_LABEL,
        target_author_uid=TARGET_UID,
        window_size=DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices=False,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        allow_name_fallback=False,
    )
    density_sync, density_points = density.calculate_density_sync()
    density_figs = plot_density_dashboards(
        density_sync,
        density_points,
        density.calculate_density_async(),
        target_name=TARGET_LABEL,
        window_size=DEFAULT_WINDOW_SIZE,
        sync_plot_width=800,
        sync_plot_height=360,
        export=False,
        show=False,
        export_dir=tmp_path,
    )

    assert set(kld_figs) == {"sync", "pointwise", "async"}
    assert set(density_figs) == {"sync", "pointwise", "async"}
    assert kld_figs["sync"] is not None
    assert kld_figs["pointwise"] is not None
    assert kld_figs["async"] is not None
    assert density_figs["sync"] is not None
    assert density_figs["pointwise"] is not None
    assert density_figs["async"] is not None
    assert list(tmp_path.iterdir()) == []


def test_quickstart_colab_uses_simple_facade() -> None:
    notebook = json.loads((ROOT / "examples" / "quickstart_colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "run_metric" in source
    assert "run_metrics" in source
    assert "plot_metric" in source
    assert "plot_multimetric" in source
    assert "run_top_authors_metrics_from_parquets" not in source
    assert "VocabularyKLD" not in source
    assert "KDEDensity" not in source
