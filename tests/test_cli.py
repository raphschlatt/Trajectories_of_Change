from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import trajectories_of_change.api as api
import trajectories_of_change.cli as cli_module
from trajectories_of_change.cli import main
from trajectories_of_change.cli import build_parser
from trajectories_of_change.defaults import (
    DEFAULT_ALPHA,
    DEFAULT_COCIT_MODE,
    DEFAULT_EPSILON,
    DEFAULT_LAMBDA_PARAM,
    DEFAULT_MULTIPLE_TESTING,
    DEFAULT_MULTIPLE_TESTING_SCOPE,
    DEFAULT_TOP_K_KLD_TERMS,
    DEFAULT_WINDOW_SIZE,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA = ROOT / "examples" / "data"
EXAMPLE_PUBLICATIONS = EXAMPLE_DATA / "publications.parquet"
EXAMPLE_REFERENCES = EXAMPLE_DATA / "references.parquet"
EXAMPLE_TARGET_UID = "uid:stable_vocab_distinct"


def test_cli_metrics_defaults_match_public_defaults() -> None:
    args = build_parser().parse_args(
        ["metrics", "publications.parquet", "references.parquet", "--out", "metrics.parquet"]
    )

    assert args.window_size == DEFAULT_WINDOW_SIZE
    assert args.top_k_kld_terms == DEFAULT_TOP_K_KLD_TERMS
    assert args.alpha == DEFAULT_ALPHA
    assert args.multiple_testing == DEFAULT_MULTIPLE_TESTING
    assert args.multiple_testing_scope == DEFAULT_MULTIPLE_TESTING_SCOPE
    assert args.cocit_mode == DEFAULT_COCIT_MODE
    assert args.lambda_param == DEFAULT_LAMBDA_PARAM
    assert args.epsilon == DEFAULT_EPSILON
    assert args.citation_identity_counting == "document_fractional"
    assert args.citation_author_scope == "first_author"
    assert args.target_exclusion == "all_docs"
    assert args.select_by == "uid"
    assert not hasattr(args, "citation_identity_backend")
    assert args.run_welch is True


@pytest.mark.parametrize(
    "metric_key",
    ["own_vocab", "ref_vocab", "density", "citation_identity"],
)
def test_cli_metric_parser_accepts_simple_metric_keys(metric_key: str) -> None:
    args = build_parser().parse_args(
        [
            "metric",
            metric_key,
            "publications.parquet",
            "references.parquet",
            "--target-author-uid",
            "uid:author",
            "--out-dir",
            "outputs/metric",
        ]
    )

    assert args.metric == metric_key
    assert args.target_author_uid == "uid:author"
    assert args.out_dir == "outputs/metric"
    assert args.include_async is True
    assert args.run_welch is True


@pytest.mark.parametrize("removed_flag", [["--target-name", "Author"], ["--allow-name-fallback"]])
def test_cli_metric_rejects_name_fallback_flags(removed_flag) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "metric",
                "own_vocab",
                "publications.parquet",
                "references.parquet",
                "--target-author-uid",
                "uid:author",
                "--out-dir",
                "outputs/metric",
                *removed_flag,
            ]
        )


def test_cli_metrics_new_citation_identity_flags_parse() -> None:
    args = build_parser().parse_args(
        [
            "metrics",
            "publications.parquet",
            "references.parquet",
            "--out",
            "metrics.parquet",
            "--citation-identity-counting",
            "binary",
            "--citation-author-scope",
            "all-authors",
            "--target-exclusion",
            "target-docs-only",
            "--no-welch",
        ]
    )

    assert args.citation_identity_counting == "binary"
    assert args.citation_author_scope == "all_authors"
    assert args.target_exclusion == "target_docs_only"
    assert args.run_welch is False


def test_cli_metrics_rejects_removed_citation_identity_backend_flag() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "metrics",
                "publications.parquet",
                "references.parquet",
                "--out",
                "metrics.parquet",
                "--citation-identity-backend",
                "reference",
            ]
        )


@pytest.mark.parametrize(
    "removed_flag",
    [["--name-matching"], ["--self-citation-mode", "none"], ["--use-all-reference-authors"]],
)
def test_cli_metrics_rejects_removed_legacy_flags(removed_flag) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["metrics", "publications.parquet", "references.parquet", "--out", "metrics.parquet", *removed_flag]
        )


def test_cli_metrics_select_by_and_tuning_flags_parse() -> None:
    args = build_parser().parse_args(
        [
            "metrics",
            "publications.parquet",
            "references.parquet",
            "--out",
            "metrics.parquet",
            "--select-by",
            "name",
            "--density-bandwidth",
            "0.5",
            "--density-min-docs-target",
            "2",
            "--density-min-docs-field",
            "3",
            "--lambda-param",
            "0.4",
            "--epsilon",
            "1e-9",
        ]
    )

    assert args.select_by == "name"
    assert args.density_bandwidth == 0.5
    assert args.density_min_docs_target == 2
    assert args.density_min_docs_field == 3
    assert args.lambda_param == 0.4
    assert args.epsilon == 1e-9


def test_cli_validate_reports_bundle(canonical_bundle_paths, capsys) -> None:
    pubs, refs = canonical_bundle_paths

    code = main(["validate", str(pubs), str(refs), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["publications_rows"] == 4
    assert payload["references_rows"] == 4
    assert payload["metric_availability"]["vocabulary_kld"] is True


def test_cli_prepare_writes_clean_bundle(tmp_path, capsys) -> None:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": " pub ",
                "Year": 2000,
                "Author": ["No author"],
                "AuthorUID": ["run::n.author::1"],
                "AuthorDisplayName": ["No author"],
                "References": [" ref ", "missing"],
            },
        ]
    )
    references = pd.DataFrame([{"Bibcode": " ref ", "Author": ["A"], "AuthorUID": ["uid:a"]}])
    pub_path = tmp_path / "raw_publications.parquet"
    ref_path = tmp_path / "raw_references.parquet"
    out_dir = tmp_path / "prepared"
    publications.to_parquet(pub_path)
    references.to_parquet(ref_path)

    code = main(["prepare", str(pub_path), str(ref_path), "--out-dir", str(out_dir)])

    assert code == 0
    assert "Prepared dataset bundle" in capsys.readouterr().out
    assert (out_dir / "publications.parquet").exists()
    assert (out_dir / "references.parquet").exists()
    assert (out_dir / "dataset_manifest.json").exists()
    assert (out_dir / "cleaning_report.json").exists()
    prepared_pubs = pd.read_parquet(out_dir / "publications.parquet")
    assert prepared_pubs.loc[0, "Bibcode"] == "pub"
    assert list(prepared_pubs.loc[0, "References"]) == ["ref"]
    assert list(prepared_pubs.loc[0, "author_uids"]) == []
    report = json.loads((out_dir / "cleaning_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert report["references"]["missing_ids_removed"] == 1
    assert manifest["cleaning"] == report
    assert manifest["counts"] == {"publications": 1, "references": 1}
    assert manifest["artifacts"]["publications"]["path"] == "publications.parquet"
    assert manifest["artifacts"]["references"]["path"] == "references.parquet"
    assert "sha256" in manifest["artifacts"]["publications"]
    assert manifest["source_artifacts"]["publications"]["bytes"] == pub_path.stat().st_size
    assert manifest["source_artifacts"]["references"]["bytes"] == ref_path.stat().st_size


def test_cli_metrics_writes_table(canonical_bundle_paths, tmp_path) -> None:
    pubs, refs = canonical_bundle_paths
    out = tmp_path / "metrics.parquet"
    details_dir = tmp_path / "details"

    code = main(
        [
            "metrics",
            str(pubs),
            str(refs),
            "--target",
            "uid:smith",
            "--window-size",
            "1",
            "--top-k-kld-terms",
            "2",
            "--no-progress",
            "--details-out-dir",
            str(details_dir),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df.loc[0, "author_uid"] == "uid:smith"
    assert df.loc[0, "alpha"] == pytest.approx(0.2)
    assert df.loc[0, "multiple_testing"] == "fdr_bh"
    assert df.loc[0, "citation_identity_counting"] == "document_fractional"
    assert df.loc[0, "citation_author_scope"] == "first_author"
    assert df.loc[0, "target_exclusion"] == "all_docs"
    assert bool(df.loc[0, "welch_enabled"]) is True
    assert (details_dir / "uid_smith" / "vocab_welch_sync.parquet").exists()
    assert (details_dir / "uid_smith" / "cocit_diagnostics_by_slice.parquet").exists()


def test_cli_metrics_routes_through_simple_facade(monkeypatch, canonical_bundle_paths, tmp_path) -> None:
    pubs, refs = canonical_bundle_paths
    out = tmp_path / "metrics.parquet"
    called = {}
    real_load = cli_module.load_dataset_bundle

    def tracked_load(*args, **kwargs):
        called["load_count"] = called.get("load_count", 0) + 1
        return real_load(*args, **kwargs)

    def fake_run_metrics(publications, references=None, **kwargs):
        called["publications"] = publications
        called["references"] = references
        called["kwargs"] = kwargs
        return pd.DataFrame([{"author_uid": "uid:smith", "author": "Smith"}])

    monkeypatch.setattr(api, "run_metrics", fake_run_metrics)
    monkeypatch.setattr(cli_module, "load_dataset_bundle", tracked_load)

    code = main(
        [
            "metrics",
            str(pubs),
            str(refs),
            "--target",
            "uid:smith",
            "--out",
            str(out),
            "--no-progress",
        ]
    )

    assert code == 0
    assert len(called["publications"].publications) > 0
    assert called["references"] is None
    assert called["load_count"] == 1
    assert called["kwargs"]["targets"] == ["uid:smith"]
    assert pd.read_parquet(out).loc[0, "author_uid"] == "uid:smith"


def test_cli_metric_writes_metric_result_folder(tmp_path) -> None:
    out_dir = tmp_path / "own_vocab"

    code = main(
        [
            "metric",
            "own_vocab",
            str(EXAMPLE_PUBLICATIONS),
            str(EXAMPLE_REFERENCES),
            "--target-author-uid",
            EXAMPLE_TARGET_UID,
            "--auto-discover-sidecars",
            "--window-size",
            "1",
            "--top-k-kld-terms",
            "2",
            "--no-async",
            "--no-welch",
            "--alpha",
            "0.05",
            "--multiple-testing",
            "holm",
            "--multiple-testing-scope",
            "global",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    manifest = json.loads((out_dir / "metric_result.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["metric"] == "own_vocab"
    assert manifest["kind"] == "kld"
    assert manifest["target_author_uid"] == EXAMPLE_TARGET_UID
    assert manifest["config"]["alpha"] == 0.05
    assert manifest["config"]["multiple_testing"] == "holm"
    assert manifest["config"]["multiple_testing_scope"] == "global"
    assert manifest["provenance"]["dataset_manifest"]["run_id"] == "synthetic-oracle-v1-test"
    assert manifest["tables"]["sync"] == "sync.parquet"
    assert (out_dir / "sync.parquet").exists()
    assert (out_dir / "pointwise.parquet").exists()


def test_cli_plot_metric_writes_html(tmp_path) -> None:
    result_dir = tmp_path / "own_vocab"
    figures_dir = result_dir / "figures"

    code = main(
        [
            "metric",
            "own_vocab",
            str(EXAMPLE_PUBLICATIONS),
            str(EXAMPLE_REFERENCES),
            "--target-author-uid",
            EXAMPLE_TARGET_UID,
            "--auto-discover-sidecars",
            "--window-size",
            "1",
            "--top-k-kld-terms",
            "2",
            "--out-dir",
            str(result_dir),
        ]
    )
    assert code == 0

    code = main(["plot", "metric", str(result_dir), "--out-dir", str(figures_dir), "--format", "html"])

    assert code == 0
    assert list(figures_dir.glob("*.html"))


def test_cli_metrics_no_welch_writes_nan_significant_summaries(canonical_bundle_paths, tmp_path) -> None:
    pubs, refs = canonical_bundle_paths
    out = tmp_path / "metrics.parquet"
    details_dir = tmp_path / "details"

    code = main(
        [
            "metrics",
            str(pubs),
            str(refs),
            "--target",
            "uid:smith",
            "--window-size",
            "1",
            "--top-k-kld-terms",
            "2",
            "--no-welch",
            "--no-progress",
            "--details-out-dir",
            str(details_dir),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    df = pd.read_parquet(out)
    assert bool(df.loc[0, "welch_enabled"]) is False
    assert pd.isna(df.loc[0, "vocab_kld_sig_level"])
    assert pd.isna(df.loc[0, "cocit_kld_sig_abs_level"])
    assert (details_dir / "uid_smith" / "vocab_welch_sync.parquet").exists()
    assert pd.read_parquet(details_dir / "uid_smith" / "vocab_welch_sync.parquet").empty


def test_cli_metrics_run_dir_writes_reproducible_run(canonical_bundle_paths, tmp_path) -> None:
    pubs, refs = canonical_bundle_paths
    run_dir = tmp_path / "runs" / "smoke"

    code = main(
        [
            "metrics",
            str(pubs),
            str(refs),
            "--target",
            "uid:smith",
            "--window-size",
            "1",
            "--top-k-kld-terms",
            "2",
            "--no-progress",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert code == 0
    assert (run_dir / "config_used.yaml").exists()
    assert (run_dir / "run_summary.yaml").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "logs" / "metrics.log").exists()
    result_path = run_dir / "results" / "multimetric.parquet"
    assert result_path.exists()
    df = pd.read_parquet(result_path)
    assert len(df) == 1
    summary_text = (run_dir / "run_summary.yaml").read_text(encoding="utf-8")
    assert "duration_seconds" in summary_text
    assert "multimetric" in summary_text
    assert "Outputs" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_cli_metrics_rejects_out_with_run_dir() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "metrics",
                "publications.parquet",
                "references.parquet",
                "--out",
                "metrics.parquet",
                "--run-dir",
                "runs/smoke",
            ]
        )


def test_cli_plot_multimetric_writes_html(tmp_path, monkeypatch) -> None:
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda self: pytest.fail("CLI opened a browser"))
    metrics_path = tmp_path / "metrics.parquet"
    out_dir = tmp_path / "plots"
    pd.DataFrame(
        [
            {
                "author": "A",
                "density_neglog_level": 1.0,
                "density_neglog_slope": 0.1,
                "vocab_kld_all_level": 0.2,
                "vocab_kld_all_slope": 0.02,
                "cocit_kld_all_level": 0.3,
                "cocit_kld_all_slope": 0.03,
                "density_slices_sync": 2,
                "vocab_slices_kld": 2,
                "cocit_slices_kld": 2,
            },
            {
                "author": "B",
                "density_neglog_level": 2.0,
                "density_neglog_slope": -0.1,
                "vocab_kld_all_level": 0.4,
                "vocab_kld_all_slope": -0.02,
                "cocit_kld_all_level": 0.1,
                "cocit_kld_all_slope": -0.01,
                "density_slices_sync": 2,
                "vocab_slices_kld": 2,
                "cocit_slices_kld": 2,
            },
        ]
    ).to_parquet(metrics_path, index=False)

    code = main(["plot", "multimetric", str(metrics_path), "--out-dir", str(out_dir)])

    assert code == 0
    assert (out_dir / "multimetric_slope_agreement.html").exists()
    assert (out_dir / "multimetric_level_agreement.html").exists()
    assert (out_dir / "multimetric_correlations.html").exists()


def test_cli_plot_multimetric_run_dir_uses_default_paths(tmp_path) -> None:
    pytest.importorskip("plotly")
    run_dir = tmp_path / "runs" / "plot-smoke"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "author": "A",
                "density_neglog_level": 1.0,
                "density_neglog_slope": 0.1,
                "vocab_kld_all_level": 0.2,
                "vocab_kld_all_slope": 0.02,
                "cocit_kld_all_level": 0.3,
                "cocit_kld_all_slope": 0.03,
            },
            {
                "author": "B",
                "density_neglog_level": 2.0,
                "density_neglog_slope": -0.1,
                "vocab_kld_all_level": 0.4,
                "vocab_kld_all_slope": -0.02,
                "cocit_kld_all_level": 0.1,
                "cocit_kld_all_slope": -0.01,
            },
        ]
    ).to_parquet(results_dir / "multimetric.parquet", index=False)

    code = main(["plot", "multimetric", "--run-dir", str(run_dir)])

    assert code == 0
    assert (run_dir / "figures" / "multimetric" / "multimetric_slope_agreement.html").exists()
    assert "figures:" in (run_dir / "run_summary.yaml").read_text(encoding="utf-8")


def test_cli_validate_raw_error_prints_prepare_hint(tmp_path, capsys) -> None:
    pubs = pd.DataFrame(
        [
            {"Bibcode": "dup", "Year": 2000, "Author": ["A"], "References": ["missing"]},
            {"Bibcode": "dup", "Year": 2001, "Author": ["B"], "References": []},
        ]
    )
    refs = pd.DataFrame([{"Bibcode": "ref", "Author": ["R"]}])
    pub_path = tmp_path / "publications.parquet"
    ref_path = tmp_path / "references.parquet"
    pubs.to_parquet(pub_path, index=False)
    refs.to_parquet(ref_path, index=False)

    code = main(["validate", str(pub_path), str(ref_path)])

    assert code == 2
    err = capsys.readouterr().err
    assert "hint:" in err
    assert "toc prepare" in err


def test_cli_missing_input_is_clean_user_error(tmp_path, capsys) -> None:
    code = main(
        [
            "validate",
            str(tmp_path / "missing-publications.parquet"),
            str(tmp_path / "missing-references.parquet"),
        ]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "publications parquet not found" in err
    assert "Traceback" not in err
