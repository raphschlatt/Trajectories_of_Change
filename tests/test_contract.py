from __future__ import annotations

import json

import pandas as pd
import pytest

import trajectories_of_change.contract as contract_module
from trajectories_of_change.contract import (
    DatasetValidationError,
    build_dataset_bundle,
    build_target_mask,
    load_dataset_bundle,
    prepare_dataset_bundle,
)


def _sample_publications() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Bibcode": "1995PASP..107..803U",
                "Year": 1995,
                "Author": ["Hawking, S. W."],
                "AuthorUID": ["uid:hawking"],
                "AuthorDisplayName": ["Hawking, Stephen W."],
                "References": ["1962RSPSA.269...21B", "1962RSPSA.270..103S"],
                "tokens": ["particle", "creation", "black", "hole"],
                "UMAP-1": 0.0,
                "UMAP-2": 0.2,
                "Title": "Particle Creation by Black Holes",
                "Abstract": "Testing the alias layer.",
            },
            {
                "Bibcode": "1996ApJ...000..001A",
                "Year": 1996,
                "Author": ["Bondi, H."],
                "AuthorUID": ["uid:bondi"],
                "AuthorDisplayName": ["Bondi, Hermann"],
                "References": ["1962RSPSA.269...21B", "1970ApJ...000..002B"],
                "tokens": ["gravity", "wave", "field"],
                "UMAP-1": 1.0,
                "UMAP-2": 1.1,
            },
        ]
    )


def _sample_references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Bibcode": "1962RSPSA.269...21B",
                "Author": ["Bondi, H."],
                "AuthorUID": ["uid:bondi"],
                "AuthorDisplayName": ["Bondi, Hermann"],
            },
            {
                "Bibcode": "1962RSPSA.270..103S",
                "Author": ["Sachs, R. K."],
                "AuthorUID": ["uid:sachs"],
                "AuthorDisplayName": ["Sachs, Rainer K."],
            },
            {
                "Bibcode": "1970ApJ...000..002B",
                "Author": ["Ellis, G. F. R."],
                "AuthorUID": ["uid:ellis"],
                "AuthorDisplayName": ["Ellis, George F. R."],
            },
        ]
    )


def test_build_dataset_bundle_normalizes_aliases() -> None:
    bundle = build_dataset_bundle(_sample_publications(), _sample_references())

    assert "author_uids" in bundle.publications.columns
    assert "author_display_names" in bundle.publications.columns
    assert "embedding_2d_x" in bundle.publications.columns
    assert "embedding_2d_y" in bundle.publications.columns
    assert "Title_en" in bundle.publications.columns
    assert "Abstract_en" in bundle.publications.columns
    assert bundle.validation is not None
    assert bundle.validation.metric_availability["vocabulary_kld"] is True
    assert bundle.validation.metric_availability["density"] is True
    assert bundle.provenance is None


def test_missing_reference_is_error_in_strict_mode() -> None:
    publications = _sample_publications()
    publications.at[0, "References"] = ["missing-bibcode"]

    with pytest.raises(DatasetValidationError):
        build_dataset_bundle(publications, _sample_references(), strict=True)


def test_missing_reference_can_be_dropped_in_lenient_mode() -> None:
    publications = _sample_publications()
    publications.at[0, "References"] = ["missing-bibcode", "1962RSPSA.269...21B"]

    bundle = build_dataset_bundle(
        publications,
        _sample_references(),
        strict=False,
        drop_missing_references=True,
    )

    assert bundle.publications.at[0, "References"] == ["1962RSPSA.269...21B"]
    assert bundle.validation is not None
    assert bundle.validation.errors == []


def test_prepare_dataset_bundle_cleans_bibcodes_references_and_author_identities(
    tmp_path,
    monkeypatch,
) -> None:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": " pub-1 ",
                "Year": 2000,
                "Author": ["No author"],
                "AuthorUID": ["run::n.author::1"],
                "AuthorDisplayName": ["No author"],
                "References": [" ref-1 ", "", "ref-1", "missing-ref"],
                "tokens": ["a"],
            },
            {
                "Bibcode": "dup",
                "Year": 2001,
                "Author": ["A"],
                "AuthorUID": ["uid:a"],
                "AuthorDisplayName": ["A"],
                "References": ["ref-1"],
                "tokens": ["short"],
            },
            {
                "Bibcode": " dup ",
                "Year": 2001,
                "Author": ["A", "A", "Unknown"],
                "AuthorUID": ["uid:a", "uid:a", "run::unknown::0"],
                "AuthorDisplayName": ["A", "A again", "Unknown"],
                "References": ["ref-1", "ref-2"],
                "tokens": ["long"],
            },
            {
                "Bibcode": " ",
                "Year": 2002,
                "Author": ["Dropped"],
                "References": ["ref-1"],
            },
        ]
    )
    references = pd.DataFrame(
        [
            {
                "Bibcode": " ref-1 ",
                "Author": ["R", "R"],
                "AuthorUID": ["uid:r", "uid:r"],
                "AuthorDisplayName": ["R", "R duplicate"],
            },
            {
                "Bibcode": "ref-2",
                "Author": ["No author"],
                "AuthorUID": ["run::n.author::1"],
                "AuthorDisplayName": ["No author"],
            },
            {
                "Bibcode": " ref-2 ",
                "Author": ["No author"],
                "AuthorUID": ["run::n.author::1"],
                "AuthorDisplayName": ["No author"],
            },
            {
                "Bibcode": "",
                "Author": ["Dropped"],
            },
        ]
    )
    pub_path = tmp_path / "publications.parquet"
    ref_path = tmp_path / "references.parquet"
    publications.to_parquet(pub_path)
    references.to_parquet(ref_path)
    calls = {"publications": 0, "references": 0}
    real_publications = contract_module.normalize_publications_frame
    real_references = contract_module.normalize_references_frame

    def normalize_publications_once(frame):
        calls["publications"] += 1
        return real_publications(frame)

    def normalize_references_once(frame):
        calls["references"] += 1
        return real_references(frame)

    monkeypatch.setattr(contract_module, "normalize_publications_frame", normalize_publications_once)
    monkeypatch.setattr(contract_module, "normalize_references_frame", normalize_references_once)

    bundle = prepare_dataset_bundle(pub_path, ref_path)

    assert calls == {"publications": 1, "references": 1}
    assert bundle.validation is not None
    assert bundle.validation.errors == []
    assert bundle.publications["Bibcode"].tolist() == ["pub-1", "dup"]
    assert bundle.references["Bibcode"].tolist() == ["ref-1", "ref-2"]
    assert bundle.publications.loc[bundle.publications["Bibcode"] == "pub-1", "References"].iloc[0] == ["ref-1"]
    kept_dup = bundle.publications.loc[bundle.publications["Bibcode"] == "dup"].iloc[0]
    assert kept_dup["References"] == ["ref-1", "ref-2"]
    assert kept_dup["tokens"] == ["long"]
    assert kept_dup["Author"] == ["A", "A", "Unknown"]
    assert kept_dup["author_uids"] == ["uid:a"]
    assert kept_dup["author_display_names"] == ["A"]
    assert bundle.publications.loc[bundle.publications["Bibcode"] == "pub-1", "Author"].iloc[0] == ["No author"]
    assert bundle.publications.loc[bundle.publications["Bibcode"] == "pub-1", "author_uids"].iloc[0] == []
    assert bundle.references.loc[bundle.references["Bibcode"] == "ref-1", "author_uids"].iloc[0] == ["uid:r"]
    assert bundle.references.loc[bundle.references["Bibcode"] == "ref-2", "author_uids"].iloc[0] == []

    report = bundle.cleaning_report
    assert report is not None
    assert report["bibcode"]["publications_empty_dropped"] == 1
    assert report["bibcode"]["references_empty_dropped"] == 1
    assert report["bibcode"]["publications_duplicate_rows_removed"] == 1
    assert report["bibcode"]["references_duplicate_rows_removed"] == 1
    assert report["references"]["empty_ids_removed"] == 1
    assert report["references"]["duplicate_ids_removed"] == 1
    assert report["references"]["missing_ids_removed"] == 1
    assert report["references"]["missing_unique_ids_removed"] == 1
    assert report["author_identities"]["publications_duplicate_uids_removed"] == 1
    assert report["author_identities"]["publications_placeholder_uids_removed"] == 2
    assert report["author_identities"]["references_duplicate_uids_removed"] == 1
    assert report["author_identities"]["references_placeholder_uids_removed"] == 1
    assert bundle.manifest is not None
    assert bundle.manifest["counts"] == {"publications": 2, "references": 2}
    assert bundle.manifest["cleaning"] == report


def test_build_target_mask_uses_exact_matching() -> None:
    bundle = build_dataset_bundle(_sample_publications(), _sample_references())

    mask = build_target_mask(bundle.publications, target_author_uid="uid:hawking")
    assert mask.tolist() == [True, False]

    name_mask = build_target_mask(bundle.publications, target_name="Hawking, S. W.")
    assert name_mask.tolist() == [True, False]

    no_partial = build_target_mask(bundle.publications, target_name="hawking")
    assert no_partial.tolist() == [False, False]


def test_build_target_mask_requires_explicit_name_fallback() -> None:
    bundle = build_dataset_bundle(_sample_publications(), _sample_references())

    uid_mask = build_target_mask(
        bundle.publications,
        target_author_uid="uid:hawking",
        target_name="Hawking, S. W.",
        allow_name_fallback=False,
    )
    assert uid_mask.tolist() == [True, False]

    with pytest.raises(DatasetValidationError, match="name fallback disabled"):
        build_target_mask(
            bundle.publications,
            target_author_uid="uid:missing",
            target_name="Hawking, S. W.",
            allow_name_fallback=False,
        )

    with pytest.raises(DatasetValidationError, match="name fallback disabled"):
        build_target_mask(
            bundle.publications,
            target_name="Hawking, S. W.",
            allow_name_fallback=False,
        )

    name_mask = build_target_mask(
        bundle.publications,
        target_name="Hawking, S. W.",
        allow_name_fallback=True,
    )
    assert name_mask.tolist() == [True, False]


def test_load_dataset_bundle_accepts_optional_provenance_sidecars(tmp_path) -> None:
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    manifest_path = tmp_path / "dataset_manifest.json"
    run_summary_path = tmp_path / "run_summary.yaml"
    config_path = tmp_path / "config_used.yaml"

    _sample_publications().to_parquet(publications_path)
    _sample_references().to_parquet(references_path)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-test",
                "producer": "ads_bib",
                "producer_version": "0.1.0",
                "and_enabled": True,
                "counts": {"publications": 2, "references": 3},
            }
        ),
        encoding="utf-8",
    )
    run_summary_path.write_text(
        "\n".join(
            [
                "schema_version: 2",
                "run:",
                "  run_id: run-test",
                "reproducibility:",
                "  config_file: config_used.yaml",
                "  git_commit: abc123",
            ]
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 42",
                "search:",
                "  query: 'author:\"Hawking, S*\"'",
                "topic_model:",
                "  embedding_provider: openrouter",
                "  embedding_model: qwen/qwen3-embedding-8b",
                "  reduction_method: pacmap",
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_dataset_bundle(
        publications_path,
        references_path,
        manifest_path=manifest_path,
        run_summary_path=run_summary_path,
        validate=True,
    )

    assert bundle.manifest is not None
    assert bundle.manifest["run_id"] == "run-test"
    assert bundle.provenance is not None
    assert bundle.provenance["run_summary"]["reproducibility"]["git_commit"] == "abc123"
    assert bundle.provenance["config"]["search"]["query"] == 'author:"Hawking, S*"'
    assert bundle.provenance["config"]["topic_model"]["reduction_method"] == "pacmap"


def test_load_dataset_bundle_can_auto_discover_sidecars(tmp_path) -> None:
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    manifest_path = tmp_path / "dataset_manifest.json"

    _sample_publications().to_parquet(publications_path)
    _sample_references().to_parquet(references_path)
    manifest_path.write_text(json.dumps({"run_id": "auto-run"}), encoding="utf-8")

    bundle = load_dataset_bundle(
        publications_path,
        references_path,
        auto_discover_sidecars=True,
        validate=True,
    )

    assert bundle.manifest is not None
    assert bundle.manifest["run_id"] == "auto-run"


def test_manifest_count_mismatch_is_warning_not_error(tmp_path) -> None:
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    manifest_path = tmp_path / "dataset_manifest.json"

    _sample_publications().to_parquet(publications_path)
    _sample_references().to_parquet(references_path)
    manifest_path.write_text(
        json.dumps({"counts": {"publications": 999, "references": 3}}),
        encoding="utf-8",
    )

    bundle = load_dataset_bundle(
        publications_path,
        references_path,
        manifest_path=manifest_path,
        validate=True,
    )

    assert bundle.validation is not None
    assert any("manifest counts mismatch for publications" in w for w in bundle.validation.warnings)


def test_manifest_artifact_mismatch_is_warning(tmp_path) -> None:
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    manifest_path = tmp_path / "dataset_manifest.json"

    _sample_publications().to_parquet(publications_path)
    _sample_references().to_parquet(references_path)
    manifest_path.write_text(
        json.dumps(
            {
                "counts": {"publications": 2, "references": 3},
                "artifacts": {
                    "publications": {"path": "publications.parquet", "bytes": 1},
                    "references": {"path": "references.parquet", "bytes": references_path.stat().st_size},
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = load_dataset_bundle(
        publications_path,
        references_path,
        manifest_path=manifest_path,
        validate=True,
    )

    assert bundle.validation is not None
    assert any("manifest artifact bytes mismatch for publications" in w for w in bundle.validation.warnings)


def test_provenance_run_id_mismatch_warns_and_can_be_strict(tmp_path) -> None:
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    manifest_path = tmp_path / "dataset_manifest.json"
    run_summary_path = tmp_path / "run_summary.yaml"

    _sample_publications().to_parquet(publications_path)
    _sample_references().to_parquet(references_path)
    manifest_path.write_text(json.dumps({"run_id": "manifest-run"}), encoding="utf-8")
    run_summary_path.write_text(
        "\n".join(["run:", "  run_id: summary-run"]),
        encoding="utf-8",
    )

    bundle = load_dataset_bundle(
        publications_path,
        references_path,
        manifest_path=manifest_path,
        run_summary_path=run_summary_path,
        validate=True,
    )

    assert bundle.validation is not None
    assert any("provenance run_id mismatch" in w for w in bundle.validation.warnings)

    with pytest.raises(DatasetValidationError, match="provenance run_id mismatch"):
        load_dataset_bundle(
            publications_path,
            references_path,
            manifest_path=manifest_path,
            run_summary_path=run_summary_path,
            validate=True,
            strict_provenance=True,
        )


def test_missing_config_referenced_by_run_summary_warns(tmp_path) -> None:
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    run_summary_path = tmp_path / "run_summary.yaml"

    _sample_publications().to_parquet(publications_path)
    _sample_references().to_parquet(references_path)
    run_summary_path.write_text(
        "\n".join(
            [
                "run:",
                "  run_id: run-test",
                "reproducibility:",
                "  config_file: missing_config.yaml",
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_dataset_bundle(
        publications_path,
        references_path,
        run_summary_path=run_summary_path,
        validate=True,
    )

    assert bundle.validation is not None
    assert any("provenance config missing" in w for w in bundle.validation.warnings)


def test_load_dataset_bundle_raises_for_explicit_missing_sidecar(tmp_path) -> None:
    pubs = pd.DataFrame(
        [
            {
                "Bibcode": "pub-1",
                "Year": 2000,
                "Author": ["A"],
                "References": ["ref-1"],
            }
        ]
    )
    refs = pd.DataFrame([{"Bibcode": "ref-1", "Author": ["B"]}])
    pub_path = tmp_path / "publications.parquet"
    ref_path = tmp_path / "references.parquet"
    pubs.to_parquet(pub_path)
    refs.to_parquet(ref_path)

    with pytest.raises(DatasetValidationError, match="Required sidecar file does not exist"):
        load_dataset_bundle(pub_path, ref_path, manifest_path=tmp_path / "missing_manifest.json")
