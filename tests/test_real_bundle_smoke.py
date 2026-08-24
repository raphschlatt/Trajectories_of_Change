from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trajectories_of_change import load_dataset_bundle, prepare_dataset_bundle
from trajectories_of_change.contract import DatasetValidationError, build_dataset_bundle, build_target_mask
from trajectories_of_change.metrics_density import KDEDensity
from trajectories_of_change.multimetric import (
    pick_top_authors,
    run_top_authors_metrics_from_parquets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATIONS_PATH = REPO_ROOT / "data" / "publications.parquet"
REFERENCES_PATH = REPO_ROOT / "data" / "references.parquet"
MANIFEST_PATH = REPO_ROOT / "data" / "dataset_manifest.json"
RUN_SUMMARY_PATH = REPO_ROOT / "data" / "run_summary.yaml"
CONFIG_PATH = REPO_ROOT / "data" / "config_used.yaml"
COORDS_2D = ("embedding_2d_x", "embedding_2d_y")
COORDS_5D = tuple(f"embedding_5d_{idx}" for idx in range(5))

pytestmark = pytest.mark.skipif(
    not PUBLICATIONS_PATH.exists() or not REFERENCES_PATH.exists(),
    reason="real canonical data bundle is not present",
)


@pytest.fixture(scope="module")
def prepared_real_bundle_paths(tmp_path_factory):
    bundle = prepare_dataset_bundle(
        PUBLICATIONS_PATH,
        REFERENCES_PATH,
        auto_discover_sidecars=True,
    )
    out_dir = tmp_path_factory.mktemp("prepared_real_bundle")
    publications_path = out_dir / "publications.parquet"
    references_path = out_dir / "references.parquet"
    bundle.publications.to_parquet(publications_path, index=False)
    bundle.references.to_parquet(references_path, index=False)
    return publications_path, references_path, bundle


@pytest.fixture(scope="module")
def real_bundle(prepared_real_bundle_paths):
    return prepared_real_bundle_paths[2]


@pytest.fixture(scope="module")
def real_multimetric_sample_paths(tmp_path_factory, real_bundle):
    target = _coauthor_targets(real_bundle, n=1)[0]
    target_mask = build_target_mask(
        real_bundle.publications,
        target_author_uid=target,
        target_name="",
        allow_name_fallback=False,
    )
    target_docs = real_bundle.publications[target_mask]
    field_docs = (
        real_bundle.publications[~target_mask]
        .sample(n=min(500, int((~target_mask).sum())), random_state=42)
        .sort_values("Year")
    )
    publications = pd.concat([target_docs, field_docs], ignore_index=True)
    referenced = {ref for refs in publications["References"] for ref in refs}
    references = real_bundle.references[real_bundle.references["Bibcode"].isin(referenced)].copy()
    out_dir = tmp_path_factory.mktemp("real_multimetric_sample")
    publications_path = out_dir / "publications.parquet"
    references_path = out_dir / "references.parquet"
    publications.to_parquet(publications_path, index=False)
    references.to_parquet(references_path, index=False)
    return publications_path, references_path, target


def test_raw_real_bundle_reports_prepare_diagnostics(prepared_real_bundle_paths) -> None:
    with pytest.raises(DatasetValidationError):
        load_dataset_bundle(
            PUBLICATIONS_PATH,
            REFERENCES_PATH,
            auto_discover_sidecars=True,
            validate=True,
        )

    _, _, bundle = prepared_real_bundle_paths
    report = bundle.cleaning_report
    assert report is not None
    assert report["bibcode"]["publications_duplicate_rows_removed"] >= 1
    assert report["bibcode"]["references_duplicate_rows_removed"] >= 1
    assert report["references"]["missing_ids_removed"] >= 1


def _coauthor_targets(bundle, n: int = 1) -> list[str]:
    candidates = pick_top_authors(
        bundle.publications,
        "Author",
        top_n=20,
        prefer_id_col="author_uids",
    )
    selected: list[str] = []
    for candidate in candidates:
        mask = build_target_mask(
            bundle.publications,
            target_author_uid=candidate,
            target_name="",
            allow_name_fallback=False,
        )
        if bool(mask.any()) and bool((~mask).any()):
            selected.append(candidate)
        if len(selected) == n:
            break
    assert len(selected) == n
    return selected


def test_real_bundle_contract_and_coordinates(real_bundle) -> None:
    report = real_bundle.validation
    assert report is not None
    assert report.errors == []
    assert all(report.metric_availability.values())
    if MANIFEST_PATH.exists():
        assert real_bundle.manifest is not None
        assert real_bundle.manifest["counts"]["publications"] == len(real_bundle.publications)
    if RUN_SUMMARY_PATH.exists() and CONFIG_PATH.exists():
        assert real_bundle.provenance is not None
        assert real_bundle.provenance["config"]["topic_model"]["reduction_method"] == "pacmap"

    referenced = {
        ref
        for refs in real_bundle.publications["References"]
        for ref in refs
    }
    known = set(real_bundle.references["Bibcode"])
    assert referenced <= known

    for column in (*COORDS_2D, *COORDS_5D):
        values = pd.to_numeric(real_bundle.publications[column], errors="coerce")
        assert values.notna().all()
        assert np.isfinite(values.to_numpy(dtype=float)).all()


def test_duplicate_author_uids_are_reported_as_warning() -> None:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "pub-1",
                "Year": 2000,
                "Author": ["Hawking, S. W.", "Hawking, S. W."],
                "AuthorUID": ["uid:hawking", "uid:hawking"],
                "References": ["ref-1"],
            },
        ]
    )
    references = pd.DataFrame(
        [
            {
                "Bibcode": "ref-1",
                "Author": ["Bondi, H."],
                "AuthorUID": ["uid:bondi"],
            },
        ]
    )

    bundle = build_dataset_bundle(publications, references)

    assert bundle.validation is not None
    assert any(
        "publications.author_uids contains duplicate values" in warning
        for warning in bundle.validation.warnings
    )


@pytest.mark.parametrize("embedding_cols", [COORDS_2D, COORDS_5D])
def test_real_bundle_density_runs_for_2d_and_5d(real_bundle, embedding_cols) -> None:
    target = _coauthor_targets(real_bundle, n=1)[0]

    density = KDEDensity(
        real_bundle.publications,
        "",
        target_author_uid=target,
        embedding_cols=embedding_cols,
        window_size=5,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        allow_name_fallback=False,
        standardize=True,
    )
    sync_df, points_df = density.calculate_density_sync()

    assert density.standardize is True
    assert tuple(density.embedding_cols) == embedding_cols
    assert not sync_df.empty
    assert not points_df.empty
    assert np.isfinite(sync_df["density_neglog_median"].to_numpy(dtype=float)).all()
    assert np.isfinite(points_df["density_neglog"].to_numpy(dtype=float)).all()


@pytest.mark.parametrize("embedding_cols", [COORDS_2D, COORDS_5D])
def test_real_bundle_multimetric_from_parquets_runs_for_2d_and_5d(
    real_multimetric_sample_paths,
    embedding_cols,
) -> None:
    publications_path, references_path, target = real_multimetric_sample_paths

    metrics_df = run_top_authors_metrics_from_parquets(
        publications_path,
        references_path,
        targets=[target],
        select_by="uid",
        window_size=5,
        include_async=False,
        alpha=0.2,
        multiple_testing="fdr_bh",
        multiple_testing_scope="slice",
        top_k_kld_terms=20,
        density_embedding_cols=embedding_cols,
        density_standardize=True,
    )

    assert len(metrics_df) == 1
    row = metrics_df.iloc[0]
    assert row["author_uid"] == target
    assert row["selection_mode"] == "author_uids"
    assert row["multiple_testing"] == "fdr_bh"
    assert row["multiple_testing_scope"] == "slice"
    assert row["top_k_kld_terms"] == 20
    assert row["window_size"] == 5
    assert row["cocit_mode"] == "authors"
    assert tuple(row["density_embedding_cols"]) == embedding_cols
    assert bool(row["density_standardize"]) is True
    assert row["density_slices_sync"] > 0
    assert row["density_slices_total"] >= row["density_slices_sync"]
    assert "vocab_kld_sig_abs_level" in metrics_df.columns
    assert "vocab_slices_kld" in metrics_df.columns
    assert np.isfinite(float(row["density_neglog_level"]))
    assert np.isfinite(float(row["vocab_kld_all_level"]))
