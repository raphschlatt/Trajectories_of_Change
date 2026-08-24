"""Command line interface for Trajectories of Change."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from . import __version__
from .contract import DatasetValidationError, file_artifact, load_dataset_bundle, prepare_dataset_bundle
from .defaults import (
    DEFAULT_ALPHA,
    DEFAULT_CITATION_AUTHOR_SCOPE,
    DEFAULT_CITATION_IDENTITY_COUNTING,
    DEFAULT_COCIT_MODE,
    DEFAULT_DENSITY_EMBEDDING_COLS,
    DEFAULT_EPSILON,
    DEFAULT_LAMBDA_PARAM,
    DEFAULT_MULTIPLE_TESTING,
    DEFAULT_MULTIPLE_TESTING_SCOPE,
    DEFAULT_REFERENCE_POLICY,
    DEFAULT_TARGET_EXCLUSION,
    DEFAULT_TOP_K_KLD_TERMS,
    DEFAULT_WINDOW_SIZE,
    METRIC_KEYS,
)
from .metric_result import MetricResult


class CLIError(Exception):
    """Expected user-facing CLI failure."""


def _parse_top_k(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "all", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("top-k must be a positive integer or 'none'")
    return parsed


def _policy_value(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _parse_jobs(value: str) -> int | str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        return int(normalized)
    except ValueError:
        raise argparse.ArgumentTypeError("--jobs must be 'auto' or an integer")


def _add_sidecar_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("provenance sidecars")
    group.add_argument("--manifest", dest="manifest_path", help="dataset_manifest.json path")
    group.add_argument("--run-summary", dest="run_summary_path", help="producer run_summary.yaml path")
    group.add_argument("--config", dest="config_path", help="producer config_used.yaml path")
    group.add_argument(
        "--auto-discover-sidecars",
        action="store_true",
        help="discover standard sidecar filenames beside publications.parquet",
    )
    group.add_argument(
        "--strict-provenance",
        action="store_true",
        help="treat provenance sidecar mismatches as errors instead of warnings",
    )


def _sidecar_source(args: argparse.Namespace, filename: str, explicit_attr: str) -> Path | None:
    explicit = getattr(args, explicit_attr, None)
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    if args.auto_discover_sidecars:
        candidate = Path(args.publications).parent / filename
        return candidate if candidate.exists() else None
    return None


def _dataframe_from_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, orient="records", lines=True)
    raise ValueError(f"Unsupported table format for input path: {path}")


def _write_dataframe(df: pd.DataFrame, path: Path, fmt: str | None = None) -> None:
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt in {"jsonl", "ndjson"}:
        path.write_text(df.to_json(orient="records", lines=True), encoding="utf-8")
    else:
        raise ValueError("output format must be one of: parquet, csv, jsonl")


def _write_metric_result(result: MetricResult, out_dir: Path) -> None:
    result.save(out_dir)


def _read_metric_result(result_dir: Path) -> MetricResult:
    return MetricResult.load(result_dir)


def _write_mapping(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return json.loads(path.read_text(encoding="utf-8"))
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _artifact(path: Path | str, *, base_dir: Path | None = None, include_sha256: bool = True) -> dict[str, Any]:
    path = Path(path)
    artifact_path = path.name
    if base_dir is not None:
        try:
            artifact_path = path.resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            artifact_path = str(path)
    return file_artifact(path, artifact_path=artifact_path, include_sha256=include_sha256)


def _prepared_manifest(
    manifest: dict[str, Any],
    *,
    raw_publications: Path,
    raw_references: Path,
    prepared_publications: Path,
    prepared_references: Path,
    cleaning_report: dict[str, Any] | None,
) -> dict[str, Any]:
    prepared = dict(manifest)
    source_artifacts = dict(prepared.get("source_artifacts") or {})
    if isinstance(prepared.get("artifacts"), dict):
        source_artifacts.update(prepared["artifacts"])
    source_artifacts["publications"] = _artifact(raw_publications)
    source_artifacts["references"] = _artifact(raw_references)

    prepared["source_artifacts"] = source_artifacts
    prepared["artifacts"] = {
        "publications": _artifact(prepared_publications, base_dir=prepared_publications.parent),
        "references": _artifact(prepared_references, base_dir=prepared_references.parent),
    }
    prepared["publications_path"] = prepared_publications.name
    prepared["references_path"] = prepared_references.name
    prepared["counts"] = {
        "publications": int(pd.read_parquet(prepared_publications, columns=["Bibcode"]).shape[0]),
        "references": int(pd.read_parquet(prepared_references, columns=["Bibcode"]).shape[0]),
    }
    if cleaning_report is not None:
        prepared["cleaning"] = cleaning_report
    return prepared


def _copy_input_sidecars(args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for filename, attr in (
        ("dataset_manifest.json", "manifest_path"),
        ("run_summary.yaml", "run_summary_path"),
        ("config_used.yaml", "config_path"),
    ):
        source = _sidecar_source(args, filename, attr)
        if source is None:
            continue
        destination = inputs_dir / filename
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied[filename] = destination.relative_to(run_dir).as_posix()

    cleaning_report = Path(args.publications).parent / "cleaning_report.json"
    if cleaning_report.exists():
        destination = inputs_dir / "cleaning_report.json"
        if cleaning_report.resolve() != destination.resolve():
            shutil.copy2(cleaning_report, destination)
        copied["cleaning_report.json"] = destination.relative_to(run_dir).as_posix()
    return copied


def _metrics_config(
    args: argparse.Namespace,
    *,
    output_path: Path,
    run_options: dict[str, Any],
) -> dict[str, Any]:
    config = {
        "command": "metrics",
        "publications": str(args.publications),
        "references": str(args.references),
        "output": str(output_path),
        "package_version": __version__,
    }
    config.update(
        {
            key: (
                str(value)
                if isinstance(value, Path)
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in run_options.items()
        }
    )
    return config


def _spearman_section(df: pd.DataFrame) -> list[str]:
    from scipy.stats import spearmanr

    pairs = [
        ("vocab_kld_all_level", "cocit_kld_all_level"),
        ("vocab_kld_all_level", "density_neglog_level"),
        ("cocit_kld_all_level", "density_neglog_level"),
        ("vocab_kld_all_slope", "cocit_kld_all_slope"),
        ("vocab_kld_all_slope", "density_neglog_slope"),
        ("cocit_kld_all_slope", "density_neglog_slope"),
    ]
    rows: list[str] = []
    for left, right in pairs:
        if left not in df.columns or right not in df.columns:
            continue
        sub = df[[left, right]].dropna()
        if len(sub) < 2:
            continue
        result = spearmanr(sub[left], sub[right])
        rows.append(f"- `{left}` vs `{right}`: n={len(sub)}, rho={result.statistic:.3f}, p={result.pvalue:.4g}")
    return rows


def _write_metrics_report(
    path: Path,
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    df: pd.DataFrame,
    warnings: Sequence[str],
) -> None:
    lines = [
        "# Trajectories of Change Analysis Run",
        "",
        "## Configuration",
        f"- window_size: `{config['window_size']}`",
        f"- top_n: `{config['top_n']}`",
        f"- top_k_kld_terms: `{config['top_k_kld_terms']}`",
        f"- alpha: `{config['alpha']}`",
        f"- multiple_testing: `{config['multiple_testing']}`",
        f"- multiple_testing_scope: `{config['multiple_testing_scope']}`",
        f"- citation_identity_counting: `{config.get('citation_identity_counting')}`",
        f"- citation_author_scope: `{config.get('citation_author_scope')}`",
        f"- target_exclusion: `{config.get('target_exclusion')}`",
        f"- run_welch: `{config.get('run_welch')}`",
        f"- select_by: `{config.get('select_by')}`",
        f"- density_cols: `{config['density_embedding_cols'] or 'default 2D'}`",
        f"- reference_policy: `{config.get('reference_policy')}`",
        f"- metrics: `{config.get('include')}`",
        "",
        "## Runtime",
        f"- status: `{summary['run']['status']}`",
        f"- duration_seconds: `{summary['run']['duration_seconds']}`",
        f"- result_rows: `{len(df)}`",
        "",
        "## Outputs",
    ]
    for name, artifact in summary.get("outputs", {}).items():
        if isinstance(artifact, dict) and "path" in artifact:
            lines.append(f"- {name}: `{artifact['path']}`")
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    spearman_lines = _spearman_section(df)
    if spearman_lines:
        lines.extend(["", "## Spearman Correlations"])
        lines.extend(spearman_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_metrics_run_summary(run_dir: Path) -> dict[str, Any]:
    return _read_mapping(run_dir / "run_summary.yaml")


def _write_metrics_run_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    _write_mapping(run_dir / "run_summary.yaml", summary)


def _prepare_hint(message: str) -> str | None:
    raw_markers = (
        "Bibcode must be unique",
        "Bibcode must not be empty",
        "referenced Bibcodes are missing",
    )
    if any(marker in message for marker in raw_markers):
        return (
            "This looks like a raw handoff. Prepare it first with: "
            "toc prepare publications.parquet references.parquet --out-dir data/prepared"
        )
    return None


def _cmd_prepare(args: argparse.Namespace) -> int:
    bundle = prepare_dataset_bundle(
        args.publications,
        args.references,
        manifest_path=args.manifest_path,
        run_summary_path=args.run_summary_path,
        config_path=args.config_path,
        auto_discover_sidecars=args.auto_discover_sidecars,
        strict_provenance=args.strict_provenance,
        strict=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    publications_out = out_dir / "publications.parquet"
    references_out = out_dir / "references.parquet"
    bundle.publications.to_parquet(publications_out, index=False)
    bundle.references.to_parquet(references_out, index=False)

    manifest = _prepared_manifest(
        bundle.manifest or {},
        raw_publications=Path(args.publications),
        raw_references=Path(args.references),
        prepared_publications=publications_out,
        prepared_references=references_out,
        cleaning_report=bundle.cleaning_report,
    )
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if bundle.cleaning_report is not None:
        (out_dir / "cleaning_report.json").write_text(
            json.dumps(bundle.cleaning_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for filename, attr in (("run_summary.yaml", "run_summary_path"), ("config_used.yaml", "config_path")):
        source = _sidecar_source(args, filename, attr)
        if source is not None:
            destination = out_dir / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

    report = bundle.cleaning_report or {}
    print(
        "Prepared dataset bundle: "
        f"{len(bundle.publications)} publications, {len(bundle.references)} references -> {out_dir}"
    )
    if report:
        bib = report.get("bibcode", {})
        refs = report.get("references", {})
        ids = report.get("author_identities", {})
        print(
            "Cleaning summary: "
            f"publication duplicate rows removed={bib.get('publications_duplicate_rows_removed', 0)}, "
            f"reference duplicate rows removed={bib.get('references_duplicate_rows_removed', 0)}, "
            f"missing reference mentions removed={refs.get('missing_ids_removed', 0)}, "
            f"placeholder author UIDs removed="
            f"{ids.get('publications_placeholder_uids_removed', 0) + ids.get('references_placeholder_uids_removed', 0)}"
        )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    bundle = load_dataset_bundle(
        args.publications,
        args.references,
        manifest_path=args.manifest_path,
        run_summary_path=args.run_summary_path,
        config_path=args.config_path,
        auto_discover_sidecars=args.auto_discover_sidecars,
        strict_provenance=args.strict_provenance,
        validate=True,
    )
    report = bundle.validation
    payload = {
        "publications_rows": int(len(bundle.publications)),
        "references_rows": int(len(bundle.references)),
        "manifest_loaded": bundle.manifest is not None,
        "provenance_loaded": bundle.provenance is not None,
        "warnings": list(report.warnings if report else []),
        "metric_availability": dict(report.metric_availability if report else {}),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            "Valid dataset bundle: "
            f"{payload['publications_rows']} publications, {payload['references_rows']} references"
        )
        for metric, available in payload["metric_availability"].items():
            print(f"- {metric}: {'yes' if available else 'no'}")
        for warning in payload["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


def _common_metric_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Translate shared CLI flags to the canonical Python API names once."""

    return {
        "manifest_path": args.manifest_path,
        "run_summary_path": args.run_summary_path,
        "config_path": args.config_path,
        "auto_discover_sidecars": args.auto_discover_sidecars,
        "strict_provenance": args.strict_provenance,
        "assume_valid": args.assume_valid,
        "include_async": args.include_async,
        "run_welch": args.run_welch,
        "window_size": args.window_size,
        "skip_incomplete_slices": not args.keep_incomplete_slices,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "top_k_kld_terms": args.top_k_kld_terms,
        "lambda_param": args.lambda_param,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "multiple_testing": args.multiple_testing,
        "multiple_testing_scope": args.multiple_testing_scope,
        "show_progress": not args.no_progress,
        "verbose": args.verbose,
    }


def _multimetric_run_options(
    args: argparse.Namespace,
    *,
    details_out_dir: Path | None,
) -> dict[str, Any]:
    """Translate the CLI namespace to the canonical multimetric option set once."""

    return {
        **_common_metric_kwargs(args),
        "top_n": args.top_n,
        "targets": args.target,
        "select_by": args.select_by,
        "cocit_mode": args.cocit_mode,
        "remove_self_loops": args.remove_self_loops,
        "citation_identity_counting": args.citation_identity_counting,
        "citation_author_scope": args.citation_author_scope,
        "target_exclusion": args.target_exclusion,
        "density_embedding_cols": tuple(args.density_cols) if args.density_cols else None,
        "density_standardize": not args.no_density_standardize,
        "density_bandwidth": args.density_bandwidth,
        "density_min_docs_target_slice": args.density_min_docs_target,
        "density_min_docs_field_slice": args.density_min_docs_field,
        "reference_policy": args.reference_policy,
        "include": tuple(args.metrics),
        "n_jobs": args.jobs,
        "details_out_dir": details_out_dir,
    }


def _cmd_metric(args: argparse.Namespace) -> int:
    from .api import run_metric

    metric_kwargs: dict[str, Any] = {
        **_common_metric_kwargs(args),
        "metric": args.metric,
        "target_author_uid": args.target_author_uid,
    }
    if args.metric == "density":
        metric_kwargs.update(
            {
                "density_embedding_cols": tuple(args.density_cols) if args.density_cols else None,
                "density_standardize": not args.no_density_standardize,
                "density_bandwidth": args.density_bandwidth,
                "density_min_docs_target_slice": args.density_min_docs_target,
                "density_min_docs_field_slice": args.density_min_docs_field,
            }
        )
    elif args.metric == "ref_vocab":
        metric_kwargs["reference_policy"] = args.reference_policy
    elif args.metric == "citation_identity":
        metric_kwargs.update(
            {
                "cocit_mode": args.cocit_mode,
                "remove_self_loops": args.remove_self_loops,
                "citation_identity_counting": args.citation_identity_counting,
                "citation_author_scope": args.citation_author_scope,
                "target_exclusion": args.target_exclusion,
            }
        )
    result = run_metric(args.publications, args.references, **metric_kwargs)
    out_dir = Path(args.out_dir)
    _write_metric_result(result, out_dir)
    print(f"Wrote {result.metric} metric result to {out_dir}")
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    from .api import run_metrics
    from .multimetric import _metrics_publication_columns, _metrics_reference_columns

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "results" / f"multimetric.{args.format or 'parquet'}"
        details_out_dir = None
        if args.details_out_dir:
            details_out_dir = Path(args.details_out_dir)
            if not details_out_dir.is_absolute():
                details_out_dir = run_dir / "results" / details_out_dir
        copied_inputs = _copy_input_sidecars(args, run_dir)
        started_at = dt.datetime.now(dt.timezone.utc)
        start = time.perf_counter()
        log_path = run_dir / "logs" / "metrics.log"
        log_path.write_text(f"started_at_utc: {started_at.isoformat()}\n", encoding="utf-8")
    else:
        output_path = Path(args.out)
        details_out_dir = Path(args.details_out_dir) if args.details_out_dir else None
        copied_inputs = {}
        started_at = dt.datetime.now(dt.timezone.utc)
        start = time.perf_counter()

    run_options = _multimetric_run_options(args, details_out_dir=details_out_dir)
    config = _metrics_config(args, output_path=output_path, run_options=run_options) if run_dir else {}
    if run_dir is not None:
        _write_mapping(run_dir / "config_used.yaml", config)

    status = "completed"
    error = None
    warnings: list[str] = []
    df = pd.DataFrame()
    try:
        loader_keys = {
            "manifest_path",
            "run_summary_path",
            "config_path",
            "auto_discover_sidecars",
            "strict_provenance",
            "assume_valid",
        }
        bundle = load_dataset_bundle(
            args.publications,
            args.references,
            manifest_path=run_options["manifest_path"],
            run_summary_path=run_options["run_summary_path"],
            config_path=run_options["config_path"],
            auto_discover_sidecars=run_options["auto_discover_sidecars"],
            strict_provenance=run_options["strict_provenance"],
            validate=not run_options["assume_valid"],
            publication_columns=_metrics_publication_columns(
                author_col="Author",
                author_id_col="author_uids",
                year_col="Year",
                tokens_col="tokens",
                density_embedding_cols=run_options["density_embedding_cols"]
                or DEFAULT_DENSITY_EMBEDDING_COLS,
            )
            if run_options["assume_valid"]
            else None,
            reference_columns=_metrics_reference_columns(
                author_col="Author",
                author_id_col="author_uids",
                include_tokens="ref_vocab" in run_options["include"],
            )
            if run_options["assume_valid"]
            else None,
        )
        warnings = list(bundle.validation.warnings if bundle.validation else [])
        compute_options = {
            key: value for key, value in run_options.items() if key not in loader_keys
        }
        df = run_metrics(bundle, **compute_options)
        _write_dataframe(df, output_path, args.format)
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        if run_dir is not None:
            ended_at = dt.datetime.now(dt.timezone.utc)
            duration = time.perf_counter() - start
            summary = {
                "schema_version": 1,
                "run": {
                    "run_id": run_dir.name,
                    "started_at_utc": started_at.isoformat(),
                    "ended_at_utc": ended_at.isoformat(),
                    "duration_seconds": round(duration, 2),
                    "status": status,
                    "error": error,
                },
                "package": {"name": "trajectories-of-change", "version": __version__},
                "inputs": {
                    "publications": _artifact(args.publications),
                    "references": _artifact(args.references),
                    "sidecars": copied_inputs,
                },
                "config_file": "config_used.yaml",
                "warnings": list(warnings),
                "outputs": {},
            }
            if output_path.exists():
                summary["outputs"]["multimetric"] = _artifact(output_path, base_dir=run_dir)
            if details_out_dir is not None and details_out_dir.exists():
                summary["outputs"]["details_dir"] = {"path": details_out_dir.relative_to(run_dir).as_posix()}
            _write_metrics_run_summary(run_dir, summary)
            if status == "completed":
                _write_metrics_report(run_dir / "report.md", config=config, summary=summary, df=df, warnings=warnings)
            with (run_dir / "logs" / "metrics.log").open("a", encoding="utf-8") as handle:
                handle.write(f"ended_at_utc: {ended_at.isoformat()}\n")
                handle.write(f"duration_seconds: {duration:.2f}\n")
                handle.write(f"status: {status}\n")
                if error:
                    handle.write(f"error: {error}\n")
                handle.write(f"rows: {len(df)}\n")
    print(f"Wrote {len(df)} metric row(s) to {output_path}")
    return 0


def _cmd_plot_multimetric(args: argparse.Namespace) -> int:
    try:
        from .plotting import plot_multimetric
    except ImportError as exc:
        raise CLIError(
            "Plotting requires optional dependencies. Install trajectories-of-change[plotting]."
        ) from exc

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        metrics_path = Path(args.metrics) if args.metrics else run_dir / "results" / "multimetric.parquet"
        out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures" / "multimetric"
    else:
        if not args.metrics or not args.out_dir:
            raise ValueError("plot multimetric requires METRICS and --out-dir unless --run-dir is used")
        metrics_path = Path(args.metrics)
        out_dir = Path(args.out_dir)
    df = _dataframe_from_path(metrics_path)
    suffix = "html" if args.format == "html" else "png"
    try:
        figures = plot_multimetric(df, export_dir=out_dir, format=args.format, show=False)
    except ValueError as exc:
        print(f"warning: skipped multimetric plots: {exc}", file=sys.stderr)
        figures = {}
    written = len(figures)

    if written == 0:
        print("No plots were written; the metrics table lacks the required columns.", file=sys.stderr)
        return 1
    if run_dir is not None:
        summary = _load_metrics_run_summary(run_dir)
        outputs = summary.setdefault("outputs", {})
        plot_outputs = {}
        for path in sorted(out_dir.glob(f"*.{suffix}")):
            plot_outputs[path.stem] = _artifact(path, base_dir=run_dir, include_sha256=False)
        outputs["figures"] = plot_outputs
        _write_metrics_run_summary(run_dir, summary)
    print(f"Wrote {written} plot file(s) to {out_dir}")
    return 0


def _cmd_plot_metric(args: argparse.Namespace) -> int:
    try:
        from .plotting import plot_metric
    except ImportError as exc:
        raise CLIError(
            "Plotting requires optional dependencies. Install trajectories-of-change[plotting]."
        ) from exc
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    result = _read_metric_result(result_dir)
    figures = plot_metric(result, export_dir=out_dir, format=args.format, show=False)
    written = sum(1 for path in out_dir.glob(f"*.{args.format}") if path.is_file())
    print(f"Wrote {written or len(figures)} plot file(s) to {out_dir}")
    return 0


def _add_metric_run_args(parser: argparse.ArgumentParser, *, single_target: bool) -> None:
    """Add the shared metric controls with consistent names, defaults, and help."""

    time_group = parser.add_argument_group("time slicing")
    time_group.add_argument("--start-year", type=int, help="first publication year to include")
    time_group.add_argument("--end-year", type=int, help="last publication year to include")
    time_group.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"non-overlapping slice width in years (default: {DEFAULT_WINDOW_SIZE})",
    )
    time_group.add_argument(
        "--keep-incomplete-slices",
        action="store_true",
        help="retain incomplete edge slices",
    )

    statistics = parser.add_argument_group("KLD statistics")
    statistics.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"significance threshold (default: {DEFAULT_ALPHA})",
    )
    statistics.add_argument(
        "--multiple-testing",
        choices=["none", "bonferroni", "holm", "fdr_bh", "fdr_by"],
        default=DEFAULT_MULTIPLE_TESTING,
        help=f"p-value adjustment method (default: {DEFAULT_MULTIPLE_TESTING})",
    )
    statistics.add_argument(
        "--multiple-testing-scope",
        choices=["slice", "pair", "global"],
        default=DEFAULT_MULTIPLE_TESTING_SCOPE,
        help=f"adjustment family (default: {DEFAULT_MULTIPLE_TESTING_SCOPE})",
    )
    statistics.add_argument(
        "--top-k-kld-terms",
        type=_parse_top_k,
        default=DEFAULT_TOP_K_KLD_TERMS,
        help=f"terms tested per slice pair, or 'none' for all (default: {DEFAULT_TOP_K_KLD_TERMS})",
    )
    statistics.add_argument(
        "--no-welch",
        dest="run_welch",
        action="store_false",
        help="skip per-term Welch tests",
    )
    parser.set_defaults(run_welch=True)

    computation = parser.add_argument_group("computation")
    if single_target:
        computation.add_argument(
            "--no-async",
            dest="include_async",
            action="store_false",
            help="skip asynchronous slice-pair matrices",
        )
        parser.set_defaults(include_async=True)
    else:
        computation.add_argument(
            "--include-async",
            action="store_true",
            help="compute asynchronous slice-pair summaries (off by default for multi-target runs)",
        )
    computation.add_argument(
        "--assume-valid",
        action="store_true",
        help="skip strict bundle validation; use only after toc prepare/toc validate",
    )
    computation.add_argument("--no-progress", action="store_true", help="hide progress bars")

    advanced = parser.add_argument_group("advanced metric options")
    advanced.add_argument(
        "--lambda-param",
        type=float,
        default=DEFAULT_LAMBDA_PARAM,
        help=f"KLD smoothing lambda (default: {DEFAULT_LAMBDA_PARAM})",
    )
    advanced.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"KLD numerical floor (default: {DEFAULT_EPSILON:g})",
    )
    advanced.add_argument("--density-cols", nargs="+", help="explicit numeric density coordinate columns")
    advanced.add_argument(
        "--no-density-standardize",
        action="store_true",
        help="do not standardize density coordinates globally",
    )
    advanced.add_argument("--density-bandwidth", type=float, default=None, help="KDE bandwidth (default: Scott)")
    advanced.add_argument("--density-min-docs-target", type=int, default=1)
    advanced.add_argument("--density-min-docs-field", type=int, default=1)
    advanced.add_argument(
        "--reference-policy",
        type=_policy_value,
        choices=["inclusive", "external_only"],
        default=DEFAULT_REFERENCE_POLICY,
        help="referenced-vocabulary target-reference policy",
    )
    advanced.add_argument("--cocit-mode", choices=["authors", "works"], default=DEFAULT_COCIT_MODE)
    advanced.add_argument("--remove-self-loops", action="store_true", default=True)
    advanced.add_argument("--keep-self-loops", dest="remove_self_loops", action="store_false")
    advanced.add_argument(
        "--citation-identity-counting",
        type=_policy_value,
        choices=["document_fractional", "binary", "multiplicity"],
        default=DEFAULT_CITATION_IDENTITY_COUNTING,
    )
    advanced.add_argument(
        "--citation-author-scope",
        type=_policy_value,
        choices=["first_author", "all_authors"],
        default=DEFAULT_CITATION_AUTHOR_SCOPE,
    )
    advanced.add_argument(
        "--target-exclusion",
        type=_policy_value,
        choices=["none", "target_docs_only", "all_docs"],
        default=DEFAULT_TARGET_EXCLUSION,
    )
    _add_sidecar_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toc", description="Trajectories of Change CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare a raw two-parquet bundle for analysis")
    prepare.add_argument("publications")
    prepare.add_argument("references")
    prepare.add_argument("--out-dir", required=True)
    _add_sidecar_args(prepare)
    prepare.set_defaults(func=_cmd_prepare)

    validate = subparsers.add_parser("validate", help="validate a canonical two-parquet bundle")
    validate.add_argument("publications")
    validate.add_argument("references")
    validate.add_argument("--json", action="store_true")
    _add_sidecar_args(validate)
    validate.set_defaults(func=_cmd_validate)

    metric = subparsers.add_parser(
        "metric",
        help="run one metric for one target",
        description="Run one package metric for one author UID and write a MetricResult folder.",
    )
    metric.add_argument("metric", choices=METRIC_KEYS, help="metric to compute")
    metric.add_argument("publications", help="canonical publications.parquet")
    metric.add_argument("references", help="canonical references.parquet")
    metric.add_argument("--target-author-uid", required=True, help="exact author UID")
    metric.add_argument("--out-dir", required=True, help="MetricResult output directory")
    _add_metric_run_args(metric, single_target=True)
    metric.set_defaults(func=_cmd_metric)

    metrics = subparsers.add_parser("metrics", help="run top-author metrics")
    metrics.add_argument("publications")
    metrics.add_argument("references")
    metrics_output = metrics.add_mutually_exclusive_group(required=True)
    metrics_output.add_argument("--out")
    metrics_output.add_argument("--run-dir")
    metrics.add_argument("--format", choices=["parquet", "csv", "jsonl"])
    metrics.add_argument("--top-n", type=int, default=5)
    metrics.add_argument("--target", action="append", help="target author UID or name; can be repeated")
    metrics.add_argument(
        "--select-by",
        choices=["uid", "name"],
        default="uid",
        help="select --target values by author UID (default) or display name",
    )
    _add_metric_run_args(metrics, single_target=False)
    metrics.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_KEYS,
        default=list(METRIC_KEYS),
        help="metrics to compute; defaults to all four package metrics",
    )
    metrics.add_argument(
        "--jobs",
        type=_parse_jobs,
        default="auto",
        help="parallel workers for the per-target loop: 'auto' (RAM-aware, default), 1 (serial), or N",
    )
    metrics.add_argument("--details-out-dir", help="write per-target detail tables for dashboard/deep-dive analysis")
    metrics.set_defaults(func=_cmd_metrics)

    plot = subparsers.add_parser("plot", help="plot exported result tables")
    plot_subparsers = plot.add_subparsers(dest="plot_command", required=True)
    metric_plot = plot_subparsers.add_parser("metric", help="plot a single metric result folder")
    metric_plot.add_argument("result_dir")
    metric_plot.add_argument("--out-dir", required=True)
    metric_plot.add_argument("--format", choices=["html", "png"], default="html")
    metric_plot.set_defaults(func=_cmd_plot_metric)

    multimetric = plot_subparsers.add_parser("multimetric", help="plot a multimetric result table")
    multimetric.add_argument("metrics", nargs="?")
    multimetric.add_argument("--out-dir")
    multimetric.add_argument("--run-dir")
    multimetric.add_argument("--format", choices=["html", "png"], default="html")
    multimetric.set_defaults(func=_cmd_plot_multimetric)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        return int(args.func(args))
    except (CLIError, DatasetValidationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        hint = _prepare_hint(str(exc)) if getattr(args, "command", None) == "validate" else None
        if hint:
            print(f"hint: {hint}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
