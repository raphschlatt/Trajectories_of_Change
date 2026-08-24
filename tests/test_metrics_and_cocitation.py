from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from trajectories_of_change import load_dataset_bundle
from trajectories_of_change.citation_identity_event import CitationIdentitySyncKLDResult
from trajectories_of_change.contract import DatasetValidationError
from trajectories_of_change.metrics_density import KDEDensity
from trajectories_of_change.metrics_kld import VocabularyKLD
import trajectories_of_change.multimetric as multimetric_module
from trajectories_of_change.multimetric import pick_top_authors, run_top_authors_metrics_from_parquets


def test_author_uid_selection_beats_name_variants(canonical_bundle_paths) -> None:
    bundle = load_dataset_bundle(*canonical_bundle_paths)

    uid_model = VocabularyKLD(
        bundle.publications,
        "",
        target_author_uid="uid:smith",
        min_token_global_freq=1,
        min_docs_global_freq=1,
        min_tokens_target_slice=1,
        min_tokens_field_slice=1,
        top_k_kld_terms=2,
        allow_name_fallback=True,
    )
    name_model = VocabularyKLD(
        bundle.publications,
        "Smith, A.",
        min_token_global_freq=1,
        min_docs_global_freq=1,
        min_tokens_target_slice=1,
        min_tokens_field_slice=1,
        top_k_kld_terms=2,
        allow_name_fallback=True,
    )

    assert sorted(uid_model.target_corpus["Bibcode"].tolist()) == [
        "2000A&A...000..001S",
        "2001A&A...000..002S",
    ]
    assert name_model.target_corpus["Bibcode"].tolist() == ["2000A&A...000..001S"]


def test_metric_models_disable_name_fallback_for_explicit_uids(canonical_bundle_paths) -> None:
    bundle = load_dataset_bundle(*canonical_bundle_paths)

    with pytest.raises(DatasetValidationError, match="name fallback disabled"):
        VocabularyKLD(
            bundle.publications,
            "Smith, A.",
            target_author_uid="uid:missing",
            min_token_global_freq=1,
            min_docs_global_freq=1,
            allow_name_fallback=False,
        )

    with pytest.raises(DatasetValidationError, match="name fallback disabled"):
        KDEDensity(
            bundle.publications,
            "Smith, A.",
            target_author_uid="uid:missing",
            allow_name_fallback=False,
        )

    with pytest.raises(DatasetValidationError, match="name fallback disabled"):
        run_top_authors_metrics_from_parquets(
            *canonical_bundle_paths,
            targets=["uid:missing"],
            select_by="uid",
            top_n=1,
            window_size=1,
            vocab_min_token_global_freq=1,
            vocab_min_docs_global_freq=1,
            show_progress=False,
        )


def test_pick_top_authors_skips_placeholder_uids() -> None:
    df = pd.DataFrame(
        [
            {"Author": ["No author"], "author_uids": ["run::n.author::1"]},
            {"Author": ["No author"], "author_uids": ["run::n.author::1"]},
            {"Author": ["Unknown"], "author_uids": ["run::unknown::0"]},
            {"Author": ["Real, A."], "author_uids": ["uid:real"]},
        ]
    )

    assert pick_top_authors(df, "Author", top_n=1, prefer_id_col="author_uids") == ["uid:real"]


def test_list_like_numpy_arrays_work_for_author_helpers() -> None:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "p1",
                "Year": 2000,
                "Author": np.array(["Author, A."], dtype=object),
                "author_uids": np.array(["uid:a"], dtype=object),
                "author_display_names": np.array(["Author, A."], dtype=object),
                "References": np.array(["r1", "r2"], dtype=object),
                "tokens": np.array(["alpha"], dtype=object),
            },
            {
                "Bibcode": "p2",
                "Year": 2001,
                "Author": np.array(["Author, B."], dtype=object),
                "author_uids": np.array(["uid:b"], dtype=object),
                "author_display_names": np.array(["Author, B."], dtype=object),
                "References": np.array(["r1", "r2"], dtype=object),
                "tokens": np.array(["beta"], dtype=object),
            },
        ]
    )

    assert pick_top_authors(publications, "Author", top_n=1, prefer_id_col="author_uids") == ["uid:a"]


def test_density_accepts_canonical_coordinates(canonical_bundle_paths) -> None:
    bundle = load_dataset_bundle(*canonical_bundle_paths)

    density = KDEDensity(
        bundle.publications,
        "",
        target_author_uid="uid:smith",
        bandwidth=1.0,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        allow_name_fallback=True,
    )
    sync_df, _ = density.calculate_density_sync()

    assert not sync_df.empty
    assert set(sync_df.columns) >= {"slice", "density_neglog_median"}


def test_density_alias_layer_accepts_producer_umap_aliases(alias_publications_df) -> None:
    raw_df = alias_publications_df.drop(columns=["AuthorUID", "AuthorDisplayName"])

    density = KDEDensity(
        raw_df,
        "Smith, A.",
        embedding_cols=("UMAP-1", "UMAP-2"),
        bandwidth=1.0,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        allow_name_fallback=True,
    )
    sync_df, _ = density.calculate_density_sync()

    assert not sync_df.empty


def test_run_top_authors_metrics_from_parquets_smoke(canonical_bundle_paths) -> None:
    metrics_df = run_top_authors_metrics_from_parquets(
        *canonical_bundle_paths,
        top_n=1,
        select_by="uid",
        window_size=1,
        skip_incomplete_slices=False,
        vocab_min_token_global_freq=1,
        vocab_min_docs_global_freq=1,
        vocab_min_tokens_target_slice=1,
        vocab_min_tokens_field_slice=1,
        vocab_min_docs_target_slice=1,
        vocab_min_docs_field_slice=1,
        vocab_min_docs_target_test=1,
        vocab_min_docs_field_test=1,
        cocit_min_token_global_freq=1,
        cocit_min_docs_global_freq=1,
        cocit_min_tokens_target_slice=1,
        cocit_min_tokens_field_slice=1,
        cocit_min_docs_target_slice=1,
        cocit_min_docs_field_slice=1,
        cocit_min_docs_target_test=1,
        cocit_min_docs_field_test=1,
        density_min_docs_target_slice=1,
        density_min_docs_field_slice=1,
        density_bandwidth=1.0,
        top_k_kld_terms=2,
    )

    assert not metrics_df.empty
    assert metrics_df.loc[0, "author_uid"] == "uid:smith"
    assert metrics_df.loc[0, "selection_mode"] == "author_uids"
    assert metrics_df.loc[0, "multiple_testing"] == "fdr_bh"
    assert metrics_df.loc[0, "multiple_testing_scope"] == "slice"
    assert metrics_df.loc[0, "top_k_kld_terms"] == 2
    assert metrics_df.loc[0, "density_embedding_cols"] == ("embedding_2d_x", "embedding_2d_y")
    assert bool(metrics_df.loc[0, "density_standardize"]) is True
    assert metrics_df.loc[0, "window_size"] == 1
    assert metrics_df.loc[0, "cocit_mode"] == "authors"
    assert metrics_df.loc[0, "citation_identity_counting"] == "document_fractional"
    assert metrics_df.loc[0, "citation_author_scope"] == "first_author"
    assert metrics_df.loc[0, "target_exclusion"] == "all_docs"
    assert bool(metrics_df.loc[0, "welch_enabled"]) is True
    assert "cocit_support_size" in metrics_df.columns
    assert "cocit_dropped_pair_mass" in metrics_df.columns
    assert "vocab_kld_sig_abs_level" in metrics_df.columns
    assert "cocit_kld_sig_abs_level" in metrics_df.columns
    assert metrics_df.loc[0, "vocab_slices_kld"] >= 1
    assert metrics_df.loc[0, "density_slices_total"] >= metrics_df.loc[0, "density_slices_sync"]


def test_run_top_authors_metrics_writes_detail_tables(canonical_bundle_paths, tmp_path) -> None:
    details_dir = tmp_path / "details"

    metrics_df = run_top_authors_metrics_from_parquets(
        *canonical_bundle_paths,
        top_n=1,
        select_by="uid",
        window_size=1,
        skip_incomplete_slices=False,
        vocab_min_token_global_freq=1,
        vocab_min_docs_global_freq=1,
        vocab_min_tokens_target_slice=1,
        vocab_min_tokens_field_slice=1,
        vocab_min_docs_target_slice=1,
        vocab_min_docs_field_slice=1,
        vocab_min_docs_target_test=1,
        vocab_min_docs_field_test=1,
        cocit_min_token_global_freq=1,
        cocit_min_docs_global_freq=1,
        cocit_min_tokens_target_slice=1,
        cocit_min_tokens_field_slice=1,
        cocit_min_docs_target_slice=1,
        cocit_min_docs_field_slice=1,
        cocit_min_docs_target_test=1,
        cocit_min_docs_field_test=1,
        density_min_docs_target_slice=1,
        density_min_docs_field_slice=1,
        density_bandwidth=1.0,
        top_k_kld_terms=2,
        details_out_dir=details_dir,
        show_progress=False,
    )

    target_dir = details_dir / "uid_smith"
    assert metrics_df.loc[0, "author_uid"] == "uid:smith"
    expected_files = {
        "vocab_kld_sync.parquet",
        "vocab_kld_pointwise.parquet",
        "vocab_welch_sync.parquet",
        "vocab_dashboard_sync.parquet",
        "cocit_kld_sync.parquet",
        "cocit_kld_pointwise.parquet",
        "cocit_welch_sync.parquet",
        "cocit_dashboard_sync.parquet",
        "cocit_diagnostics.parquet",
        "cocit_diagnostics_by_slice.parquet",
        "density_sync.parquet",
        "density_pointwise.parquet",
    }
    assert expected_files.issubset({path.name for path in target_dir.iterdir()})
    dashboard = pd.read_parquet(target_dir / "vocab_dashboard_sync.parquet")
    welch = pd.read_parquet(target_dir / "vocab_welch_sync.parquet")
    pointwise = pd.read_parquet(target_dir / "vocab_kld_pointwise.parquet")
    assert {"slice", "kld_all", "kld_sig", "kld_sig_abs"}.issubset(dashboard.columns)
    assert {"slice", "term", "pvalue", "kld_contribution", "p_adj"}.issubset(welch.columns)
    assert {"slice", "term", "kld_contribution"}.issubset(pointwise.columns)
    diagnostics = pd.read_parquet(target_dir / "cocit_diagnostics_by_slice.parquet")
    assert {"slice", "candidate_pair_mass", "target_excluded_pair_mass"}.issubset(diagnostics.columns)


def test_run_top_authors_metrics_can_skip_welch(canonical_bundle_paths) -> None:
    metrics_df = run_top_authors_metrics_from_parquets(
        *canonical_bundle_paths,
        targets=["uid:smith"],
        select_by="uid",
        window_size=1,
        skip_incomplete_slices=False,
        vocab_min_token_global_freq=1,
        vocab_min_docs_global_freq=1,
        vocab_min_tokens_target_slice=1,
        vocab_min_tokens_field_slice=1,
        vocab_min_docs_target_slice=1,
        vocab_min_docs_field_slice=1,
        cocit_min_token_global_freq=1,
        cocit_min_docs_global_freq=1,
        cocit_min_tokens_target_slice=1,
        cocit_min_tokens_field_slice=1,
        cocit_min_docs_target_slice=1,
        cocit_min_docs_field_slice=1,
        density_min_docs_target_slice=1,
        density_min_docs_field_slice=1,
        density_bandwidth=1.0,
        top_k_kld_terms=2,
        run_welch=False,
        show_progress=False,
    )

    assert bool(metrics_df.loc[0, "welch_enabled"]) is False
    assert pd.isna(metrics_df.loc[0, "vocab_kld_sig_level"])
    assert pd.isna(metrics_df.loc[0, "cocit_sig_terms_total"])


def test_citation_identity_uses_single_event_core_path_multimetric(tmp_path) -> None:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "2000T",
                "Year": 2000,
                "Author": ["Target, T."],
                "author_uids": ["uid:target"],
                "References": ["ref:a", "ref:b"],
                "tokens": ["alpha", "shared"],
                "embedding_2d_x": 0.0,
                "embedding_2d_y": 0.0,
            },
            {
                "Bibcode": "2000F",
                "Year": 2000,
                "Author": ["Field, F."],
                "author_uids": ["uid:field"],
                "References": ["ref:a", "ref:c", "ref:t"],
                "tokens": ["beta", "shared"],
                "embedding_2d_x": 1.0,
                "embedding_2d_y": 1.0,
            },
            {
                "Bibcode": "2001T",
                "Year": 2001,
                "Author": ["Target, T."],
                "author_uids": ["uid:target"],
                "References": ["ref:b", "ref:c"],
                "tokens": ["gamma", "shared"],
                "embedding_2d_x": 0.1,
                "embedding_2d_y": 0.1,
            },
            {
                "Bibcode": "2001F",
                "Year": 2001,
                "Author": ["Field, G."],
                "author_uids": ["uid:field2"],
                "References": ["ref:a", "ref:b"],
                "tokens": ["delta", "shared"],
                "embedding_2d_x": 1.1,
                "embedding_2d_y": 1.1,
            },
        ]
    )
    references = pd.DataFrame(
        [
            {"Bibcode": "ref:a", "Author": ["A, A."], "author_uids": ["uid:a"]},
            {"Bibcode": "ref:b", "Author": ["B, B."], "author_uids": ["uid:b"]},
            {"Bibcode": "ref:c", "Author": ["C, C."], "author_uids": ["uid:c"]},
            {"Bibcode": "ref:t", "Author": ["Target, T."], "author_uids": ["uid:target"]},
        ]
    )
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    publications.to_parquet(publications_path, index=False)
    references.to_parquet(references_path, index=False)

    kwargs = {
        "targets": ["uid:target"],
        "run_welch": False,
        "window_size": 1,
        "skip_incomplete_slices": False,
        "assume_valid": False,
        "show_progress": False,
        "min_token_global_freq": 1.0,
        "min_tokens_target_slice": 1e-12,
        "min_tokens_field_slice": 1e-12,
    }
    metrics = run_top_authors_metrics_from_parquets(
        publications_path,
        references_path,
        **kwargs,
    )

    assert "citation_identity_backend" not in metrics.columns
    assert pd.notna(metrics.loc[0, "cocit_kld_all_level"])
    assert pd.notna(metrics.loc[0, "cocit_kld_all_slope"])
    assert metrics.loc[0, "cocit_support_size"] > 0
    assert pd.notna(metrics.loc[0, "cocit_field_entropy_level"])
    assert metrics.loc[0, "cocit_dropped_pair_mass"] >= 0


def test_removed_citation_identity_backend_argument_is_rejected(canonical_bundle_paths) -> None:
    with pytest.raises(ValueError, match="citation_identity_backend has been removed"):
        run_top_authors_metrics_from_parquets(
            *canonical_bundle_paths,
            targets=["uid:smith"],
            citation_identity_backend="reference",
            select_by="uid",
            window_size=1,
            skip_incomplete_slices=False,
            vocab_min_token_global_freq=1,
            vocab_min_docs_global_freq=1,
            vocab_min_tokens_target_slice=1,
            vocab_min_tokens_field_slice=1,
            vocab_min_docs_target_slice=1,
            vocab_min_docs_field_slice=1,
            cocit_min_token_global_freq=1,
            cocit_min_docs_global_freq=1,
            cocit_min_tokens_target_slice=1,
            cocit_min_tokens_field_slice=1,
            cocit_min_docs_target_slice=1,
            cocit_min_docs_field_slice=1,
            density_min_docs_target_slice=1,
            density_min_docs_field_slice=1,
            density_bandwidth=1.0,
            top_k_kld_terms=2,
            run_welch=False,
            show_progress=False,
        )


def test_single_event_core_path_supports_citation_identity_async_and_welch(tmp_path) -> None:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "2000T",
                "Year": 2000,
                "Author": ["Target, T."],
                "author_uids": ["uid:target"],
                "References": ["ref:a", "ref:b"],
                "tokens": ["alpha", "shared"],
                "embedding_2d_x": 0.0,
                "embedding_2d_y": 0.0,
            },
            {
                "Bibcode": "2000F",
                "Year": 2000,
                "Author": ["Field, F."],
                "author_uids": ["uid:field"],
                "References": ["ref:a", "ref:c", "ref:t"],
                "tokens": ["beta", "shared"],
                "embedding_2d_x": 1.0,
                "embedding_2d_y": 1.0,
            },
            {
                "Bibcode": "2001T",
                "Year": 2001,
                "Author": ["Target, T."],
                "author_uids": ["uid:target"],
                "References": ["ref:b", "ref:c"],
                "tokens": ["gamma", "shared"],
                "embedding_2d_x": 0.1,
                "embedding_2d_y": 0.1,
            },
            {
                "Bibcode": "2001F",
                "Year": 2001,
                "Author": ["Field, G."],
                "author_uids": ["uid:field2"],
                "References": ["ref:a", "ref:b"],
                "tokens": ["delta", "shared"],
                "embedding_2d_x": 1.1,
                "embedding_2d_y": 1.1,
            },
        ]
    )
    references = pd.DataFrame(
        [
            {"Bibcode": "ref:a", "Author": ["A, A."], "author_uids": ["uid:a"]},
            {"Bibcode": "ref:b", "Author": ["B, B."], "author_uids": ["uid:b"]},
            {"Bibcode": "ref:c", "Author": ["C, C."], "author_uids": ["uid:c"]},
            {"Bibcode": "ref:t", "Author": ["Target, T."], "author_uids": ["uid:target"]},
        ]
    )
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    publications.to_parquet(publications_path, index=False)
    references.to_parquet(references_path, index=False)

    metrics = run_top_authors_metrics_from_parquets(
        publications_path,
        references_path,
        targets=["uid:target"],
        run_welch=True,
        include_async=True,
        window_size=1,
        skip_incomplete_slices=False,
        assume_valid=False,
        show_progress=False,
        min_token_global_freq=1.0,
        min_tokens_target_slice=1e-12,
        min_tokens_field_slice=1e-12,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        top_k_kld_terms=3,
    )

    assert "citation_identity_backend" not in metrics.columns
    assert bool(metrics.loc[0, "welch_enabled"]) is True
    assert pd.notna(metrics.loc[0, "cocit_kld_async_min"])
    assert pd.notna(metrics.loc[0, "cocit_kld_sig_level"])


def test_aggregate_cocit_coverage_counts_zero_significance_slices() -> None:
    result = CitationIdentitySyncKLDResult(
        sync=pd.DataFrame({"slice": [2000, 2001, 2002], "kld_all": [0.1, 0.2, 0.3]}),
        pointwise=pd.DataFrame(),
        metadata={
            "slices_total": 3,
            "slice_token_counts": {
                2000: {"target_docs": 2, "field_docs": 10, "target_tokens": 1.0, "field_tokens": 5.0},
                2001: {"target_docs": 2, "field_docs": 10, "target_tokens": 1.0, "field_tokens": 5.0},
                2002: {"target_docs": 2, "field_docs": 10, "target_tokens": 1.0, "field_tokens": 5.0},
            },
            "welch_target_doc_counts": {2000: 2, 2001: 2, 2002: 2},
            "welch_field_doc_counts": {2000: 10, 2001: 10, 2002: 10},
        },
    )
    df_welch_sync = pd.DataFrame(
        [
            {"slice": 2000, "p_adj": 0.01},
            {"slice": 2000, "p_adj": 0.02},
            {"slice": 2001, "p_adj": 0.5},
            {"slice": 2002, "p_adj": 0.03},
        ]
    )

    summary = multimetric_module._summarize_aggregate_cocit_coverage(
        result,
        result.sync,
        df_welch_sync,
        alpha=0.05,
        welch_enabled=True,
    )

    assert summary["sig_terms_total"] == 3
    assert summary["sig_terms_median_per_slice"] == 1.0


def test_document_fractional_citation_identity_defaults_remain_analyzable(tmp_path) -> None:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "target-2000",
                "Year": 2000,
                "Author": ["Target, T."],
                "AuthorUID": ["uid:target"],
                "References": ["target-ref", "a-ref", "b-ref"],
                "tokens": ["target", "alpha"],
                "embedding_2d_x": 0.0,
                "embedding_2d_y": 0.0,
            },
            {
                "Bibcode": "field-2000",
                "Year": 2000,
                "Author": ["Field, F."],
                "AuthorUID": ["uid:field"],
                "References": ["a-ref", "b-ref"],
                "tokens": ["field", "beta"],
                "embedding_2d_x": 1.0,
                "embedding_2d_y": 1.0,
            },
        ]
    )
    references = pd.DataFrame(
        [
            {"Bibcode": "target-ref", "Author": ["Target, T."], "AuthorUID": ["uid:target"]},
            {"Bibcode": "a-ref", "Author": ["Alpha, A."], "AuthorUID": ["uid:alpha"]},
            {"Bibcode": "b-ref", "Author": ["Beta, B."], "AuthorUID": ["uid:beta"]},
        ]
    )
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    publications.to_parquet(publications_path, index=False)
    references.to_parquet(references_path, index=False)

    metrics_df = run_top_authors_metrics_from_parquets(
        publications_path,
        references_path,
        targets=["uid:target"],
        select_by="uid",
        window_size=1,
        skip_incomplete_slices=False,
        vocab_min_token_global_freq=1,
        vocab_min_docs_global_freq=1,
        vocab_min_tokens_target_slice=1,
        vocab_min_tokens_field_slice=1,
        vocab_min_docs_target_slice=1,
        vocab_min_docs_field_slice=1,
        cocit_min_token_global_freq=1,
        cocit_min_docs_global_freq=1,
        density_min_docs_target_slice=1,
        density_min_docs_field_slice=1,
        density_bandwidth=1.0,
        top_k_kld_terms=2,
        run_welch=False,
        show_progress=False,
    )

    assert metrics_df.loc[0, "citation_identity_counting"] == "document_fractional"
    assert metrics_df.loc[0, "target_exclusion"] == "all_docs"
    assert pd.notna(metrics_df.loc[0, "cocit_kld_all_level"])
    assert metrics_df.loc[0, "cocit_support_size"] > 0


def test_document_fractional_defaults_allow_subunit_retained_pair_mass(tmp_path) -> None:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "target-2000",
                "Year": 2000,
                "Author": ["Target, T."],
                "AuthorUID": ["uid:target"],
                "References": ["a-ref", "b-ref", "c-ref"],
                "tokens": ["target", "alpha"],
                "embedding_2d_x": 0.0,
                "embedding_2d_y": 0.0,
            },
            {
                "Bibcode": "field-2000-a",
                "Year": 2000,
                "Author": ["Field, F."],
                "AuthorUID": ["uid:field-a"],
                "References": ["a-ref", "b-ref"],
                "tokens": ["field", "beta"],
                "embedding_2d_x": 1.0,
                "embedding_2d_y": 1.0,
            },
            {
                "Bibcode": "field-2000-b",
                "Year": 2000,
                "Author": ["Field, G."],
                "AuthorUID": ["uid:field-b"],
                "References": ["a-ref", "b-ref"],
                "tokens": ["field", "gamma"],
                "embedding_2d_x": 2.0,
                "embedding_2d_y": 2.0,
            },
        ]
    )
    references = pd.DataFrame(
        [
            {"Bibcode": "a-ref", "Author": ["Alpha, A."], "AuthorUID": ["uid:alpha"]},
            {"Bibcode": "b-ref", "Author": ["Beta, B."], "AuthorUID": ["uid:beta"]},
            {"Bibcode": "c-ref", "Author": ["Gamma, C."], "AuthorUID": ["uid:gamma"]},
        ]
    )
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    publications.to_parquet(publications_path, index=False)
    references.to_parquet(references_path, index=False)

    metrics_df = run_top_authors_metrics_from_parquets(
        publications_path,
        references_path,
        targets=["uid:target"],
        select_by="uid",
        window_size=1,
        skip_incomplete_slices=False,
        vocab_min_token_global_freq=1,
        vocab_min_docs_global_freq=1,
        vocab_min_tokens_target_slice=1,
        vocab_min_tokens_field_slice=1,
        vocab_min_docs_target_slice=1,
        vocab_min_docs_field_slice=1,
        density_min_docs_target_slice=1,
        density_min_docs_field_slice=1,
        density_bandwidth=1.0,
        top_k_kld_terms=2,
        run_welch=False,
        show_progress=False,
    )

    assert pd.notna(metrics_df.loc[0, "cocit_kld_all_level"])
    assert metrics_df.loc[0, "cocit_target_tokens_median_kld"] == pytest.approx(1 / 3)
    assert metrics_df.loc[0, "cocit_support_size"] == 1


def test_multimetric_uses_event_core_path_for_target_exclusion(canonical_bundle_paths) -> None:
    metrics_df = run_top_authors_metrics_from_parquets(
        *canonical_bundle_paths,
        targets=["uid:smith", "uid:jones"],
        select_by="uid",
        window_size=1,
        skip_incomplete_slices=False,
        vocab_min_token_global_freq=1,
        vocab_min_docs_global_freq=1,
        vocab_min_tokens_target_slice=1,
        vocab_min_tokens_field_slice=1,
        vocab_min_docs_target_slice=1,
        vocab_min_docs_field_slice=1,
        cocit_min_token_global_freq=1,
        cocit_min_docs_global_freq=1,
        cocit_min_tokens_target_slice=1,
        cocit_min_tokens_field_slice=1,
        cocit_min_docs_target_slice=1,
        cocit_min_docs_field_slice=1,
        density_min_docs_target_slice=1,
        density_min_docs_field_slice=1,
        density_bandwidth=1.0,
        top_k_kld_terms=2,
        run_welch=False,
        show_progress=False,
    )

    assert metrics_df["author_uid"].tolist() == ["uid:smith", "uid:jones"]
    assert set(metrics_df["target_exclusion"]) == {"all_docs"}


def test_assume_valid_column_projection_matches_full_load(canonical_bundle_paths) -> None:
    kwargs = dict(
        top_n=1,
        select_by="uid",
        window_size=1,
        skip_incomplete_slices=False,
        vocab_min_token_global_freq=1,
        vocab_min_docs_global_freq=1,
        vocab_min_tokens_target_slice=1,
        vocab_min_tokens_field_slice=1,
        vocab_min_docs_target_slice=1,
        vocab_min_docs_field_slice=1,
        vocab_min_docs_target_test=1,
        vocab_min_docs_field_test=1,
        cocit_min_token_global_freq=1,
        cocit_min_docs_global_freq=1,
        cocit_min_tokens_target_slice=1,
        cocit_min_tokens_field_slice=1,
        cocit_min_docs_target_slice=1,
        cocit_min_docs_field_slice=1,
        cocit_min_docs_target_test=1,
        cocit_min_docs_field_test=1,
        density_min_docs_target_slice=1,
        density_min_docs_field_slice=1,
        density_bandwidth=1.0,
        top_k_kld_terms=2,
        show_progress=False,
    )

    full = run_top_authors_metrics_from_parquets(*canonical_bundle_paths, assume_valid=False, **kwargs)
    projected = run_top_authors_metrics_from_parquets(*canonical_bundle_paths, assume_valid=True, **kwargs)

    pd.testing.assert_frame_equal(full, projected, check_exact=False, atol=1e-12, rtol=1e-12)


def test_run_top_authors_metrics_from_parquets_routes_prefixed_metric_kwargs(monkeypatch) -> None:
    from trajectories_of_change.contract import DatasetBundle
    import trajectories_of_change.multimetric as multimetric

    publications = pd.DataFrame(
        [
            {
                "Bibcode": "2000A&A...000..001S",
                "Year": 2000,
                "Author": ["Smith, A."],
                "author_uids": ["uid:smith"],
                "author_display_names": ["Smith, Alice"],
                "References": ["1990A&A...000..001R", "1991A&A...000..002R"],
                "tokens": ["alpha", "beta"],
                "embedding_2d_x": 0.1,
                "embedding_2d_y": 0.2,
            },
            {
                "Bibcode": "2000A&A...000..002R",
                "Year": 2000,
                "Author": ["Roe, B."],
                "author_uids": ["uid:roe"],
                "author_display_names": ["Roe, Bob"],
                "References": ["1990A&A...000..001R", "1991A&A...000..002R"],
                "tokens": ["gamma", "delta"],
                "embedding_2d_x": 1.1,
                "embedding_2d_y": 1.2,
            },
        ]
    )
    references = pd.DataFrame(
        [
            {"Bibcode": "1990A&A...000..001R", "Author": ["Ref, A."]},
            {"Bibcode": "1991A&A...000..002R", "Author": ["Ref, B."]},
        ]
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        multimetric,
        "_load_bundle_arg",
        lambda *args, **kwargs: DatasetBundle(publications=publications, references=references, manifest=None),
    )
    monkeypatch.setattr(multimetric, "pick_top_authors", lambda *args, **kwargs: ["uid:smith"])

    def _fake_run_one_target(**kwargs):
        config = kwargs["config"]
        target = kwargs["target"]
        captured["vocab_kwargs"] = config.vocab_kwargs
        captured["cocit_kwargs"] = config.cocit_kwargs
        return {"author": target.label, "author_uid": target.uid, "selection_mode": "author_uids"}

    monkeypatch.setattr(multimetric, "_run_one_target", _fake_run_one_target)

    metrics_df = run_top_authors_metrics_from_parquets(
        "publications.parquet",
        "references.parquet",
        top_n=1,
        vocab_min_token_global_freq=3,
        vocab_min_docs_target_slice=4,
        cocit_min_token_global_freq=5,
        cocit_min_docs_field_slice=6,
        density_bandwidth=1.0,
        top_k_kld_terms=2,
    )

    assert metrics_df.loc[0, "author_uid"] == "uid:smith"
    assert captured["vocab_kwargs"] == {
        "min_token_global_freq": 3,
        "min_docs_target_slice": 4,
    }
    assert captured["cocit_kwargs"] == {
        "min_token_global_freq": 5,
        "min_docs_field_slice": 6,
        "min_tokens_target_slice": 1e-12,
        "min_tokens_field_slice": 1e-12,
    }


def test_assume_valid_metrics_uses_column_projection(monkeypatch) -> None:
    from trajectories_of_change.contract import DatasetBundle
    import trajectories_of_change.multimetric as multimetric

    publications = pd.DataFrame(
        [
            {
                "Bibcode": "2000A&A...000..001S",
                "Year": 2000,
                "Author": ["Smith, A."],
                "author_uids": ["uid:smith"],
                "author_display_names": ["Smith, Alice"],
                "References": ["1990A&A...000..001R", "1991A&A...000..002R"],
                "tokens": ["alpha", "beta"],
                "embedding_2d_x": 0.1,
                "embedding_2d_y": 0.2,
            },
            {
                "Bibcode": "2000A&A...000..002R",
                "Year": 2000,
                "Author": ["Roe, B."],
                "author_uids": ["uid:roe"],
                "author_display_names": ["Roe, Bob"],
                "References": ["1990A&A...000..001R", "1991A&A...000..002R"],
                "tokens": ["gamma", "delta"],
                "embedding_2d_x": 1.1,
                "embedding_2d_y": 1.2,
            },
        ]
    )
    references = pd.DataFrame(
        [
            {"Bibcode": "1990A&A...000..001R", "Author": ["Ref, A."]},
            {"Bibcode": "1991A&A...000..002R", "Author": ["Ref, B."]},
        ]
    )
    captured: dict[str, object] = {}

    def _fake_load(*args, **kwargs):
        captured.update(kwargs)
        return DatasetBundle(publications=publications, references=references, manifest=None)

    monkeypatch.setattr(multimetric, "_load_bundle_arg", _fake_load)
    monkeypatch.setattr(multimetric, "pick_top_authors", lambda *args, **kwargs: ["uid:smith"])
    monkeypatch.setattr(
        multimetric,
        "_run_one_target",
        lambda **kwargs: {
            "author": kwargs["target"].label,
            "author_uid": kwargs["target"].uid,
        },
    )

    run_top_authors_metrics_from_parquets(
        "publications.parquet",
        "references.parquet",
        assume_valid=True,
        top_n=1,
        density_embedding_cols=("embedding_2d_x", "embedding_2d_y"),
        show_progress=False,
    )

    assert captured["assume_valid"] is True
    assert set(captured["publication_columns"]) >= {
        "Bibcode",
        "Year",
        "Author",
        "author_uids",
        "author_display_names",
        "References",
        "tokens",
        "embedding_2d_x",
        "embedding_2d_y",
    }
    assert set(captured["reference_columns"]) >= {"Bibcode", "Author", "author_uids", "author_display_names"}
