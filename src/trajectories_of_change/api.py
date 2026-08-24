"""Small user-facing metric facade.

The classes remain the advanced API. These functions provide the simple path:
load a bundle, run one named metric, or run the consolidated multimetric path.
"""

from __future__ import annotations

from dataclasses import replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Sequence, Unpack

from .citation_identity import CitationIdentityConfig
from .citation_identity_event import CitationIdentityEventIndex, CitationIdentityKLD
from .contract import DatasetBundle, _load_bundle_arg, build_target_mask, resolve_target_label
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
from .metrics_density import KDEDensity
from .metrics_kld import VocabularyKLD
from .multimetric import _RunMetricsOptions, run_top_authors_metrics_from_parquets
from .referenced_vocabulary import ReferencedVocabularyKLD
from .stats_utils import MULTIPLE_TESTING_METHODS


_CORE_RUN_METRICS = set(METRIC_KEYS)
_KLD_THRESHOLD_KEYS = {
    "min_token_global_freq",
    "min_docs_global_freq",
    "max_vocab_size",
    "min_tokens_target_slice",
    "min_tokens_field_slice",
    "min_docs_target_slice",
    "min_docs_field_slice",
    "min_docs_target_test",
    "min_docs_field_test",
}
_METRIC_OPTIONS = {
    "own_vocab": _KLD_THRESHOLD_KEYS | {"author_col", "author_id_col", "year_col", "token_col", "docid_col"},
    "ref_vocab": _KLD_THRESHOLD_KEYS | {"reference_policy"},
    "density": {
        "density_embedding_cols",
        "density_bandwidth",
        "density_standardize",
        "density_min_docs_target_slice",
        "density_min_docs_field_slice",
        "year_col",
        "docid_col",
    },
    "citation_identity": _KLD_THRESHOLD_KEYS
    | {
        "cocit_mode",
        "citation_identity_counting",
        "citation_author_scope",
        "target_exclusion",
        "remove_self_loops",
    },
}


def _metric_error() -> ValueError:
    return ValueError(f"metric must be one of {METRIC_KEYS}")


def _coerce_metric(metric: str) -> str:
    normalized = str(metric).strip().lower()
    if normalized not in METRIC_KEYS:
        raise _metric_error()
    return normalized


def _coerce_include(include: Sequence[str] | None) -> set[str] | None:
    if include is None:
        return None
    return {_coerce_metric(metric) for metric in include}


def _target_mask(bundle: DatasetBundle, *, target_author_uid: str):
    return build_target_mask(
        bundle.publications,
        target_name="",
        target_author_uid=target_author_uid,
        allow_name_fallback=False,
    ).to_numpy(dtype=bool)

def _validate_metric_options(metric: str, options: dict[str, Any]) -> None:
    allowed = _METRIC_OPTIONS[metric]
    unknown = sorted(set(options) - allowed)
    if not unknown:
        return
    key = unknown[0]
    suggestion = get_close_matches(key, sorted(allowed), n=1)
    hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
    raise TypeError(f"run_metric(metric={metric!r}) got an unexpected option {key!r}{hint}")


def _result_provenance(bundle: DatasetBundle) -> dict[str, Any]:
    warnings = list(bundle.validation.warnings) if bundle.validation is not None else []
    return {
        "dataset_manifest": dict(bundle.manifest or {}),
        "producer_provenance": dict(bundle.provenance or {}),
        "validation_warnings": warnings,
    }


def run_metric(
    bundle_or_publications_path: DatasetBundle | str | Path,
    references_path: str | Path | None = None,
    *,
    metric: str,
    target_author_uid: str,
    manifest_path: str | Path | None = None,
    run_summary_path: str | Path | None = None,
    config_path: str | Path | None = None,
    auto_discover_sidecars: bool = False,
    strict_provenance: bool = False,
    assume_valid: bool = False,
    include_async: bool = True,
    run_welch: bool = True,
    window_size: int = DEFAULT_WINDOW_SIZE,
    skip_incomplete_slices: bool = True,
    start_year: int | None = None,
    end_year: int | None = None,
    top_k_kld_terms: int | None = DEFAULT_TOP_K_KLD_TERMS,
    lambda_param: float = DEFAULT_LAMBDA_PARAM,
    epsilon: float = DEFAULT_EPSILON,
    alpha: float = DEFAULT_ALPHA,
    multiple_testing: str = DEFAULT_MULTIPLE_TESTING,
    multiple_testing_scope: str = DEFAULT_MULTIPLE_TESTING_SCOPE,
    **kwargs: Any,
) -> MetricResult:
    """Run one named metric for one target author UID."""
    metric_key = _coerce_metric(metric)
    if multiple_testing not in MULTIPLE_TESTING_METHODS:
        raise ValueError(f"multiple_testing must be one of {sorted(MULTIPLE_TESTING_METHODS)}")
    if multiple_testing_scope not in {"slice", "pair", "global"}:
        raise ValueError("multiple_testing_scope must be one of: slice, pair, global")
    if not str(target_author_uid).strip():
        raise ValueError("target_author_uid is required")
    bundle = _load_bundle_arg(
        bundle_or_publications_path,
        references_path,
        manifest_path=manifest_path,
        run_summary_path=run_summary_path,
        config_path=config_path,
        auto_discover_sidecars=auto_discover_sidecars,
        strict_provenance=strict_provenance,
        assume_valid=assume_valid,
    )
    target_uid = str(target_author_uid)
    target_label = resolve_target_label(bundle.publications, target_uid)

    target_mask = _target_mask(
        bundle,
        target_author_uid=target_uid,
    )
    show_progress = bool(kwargs.pop("show_progress", False))
    verbose = bool(kwargs.pop("verbose", False))
    _validate_metric_options(metric_key, kwargs)
    common = dict(
        start_year=start_year,
        end_year=end_year,
        window_size=window_size,
        skip_incomplete_slices=skip_incomplete_slices,
        lambda_param=lambda_param,
        epsilon=epsilon,
        top_k_kld_terms=top_k_kld_terms,
        target_mask=target_mask,
        show_progress=show_progress,
        verbose=verbose,
    )

    if metric_key == "own_vocab":
        model = VocabularyKLD(
            bundle.publications,
            target_label,
            target_author_uid=target_uid,
            allow_name_fallback=False,
            **common,
            **kwargs,
        )
        result = model.result(include_async=include_async, run_welch=run_welch)

    elif metric_key == "density":
        density_kwargs = {
            "embedding_cols": kwargs.pop("density_embedding_cols", None) or DEFAULT_DENSITY_EMBEDDING_COLS,
            "bandwidth": kwargs.pop("density_bandwidth", None),
            "standardize": kwargs.pop("density_standardize", True),
            "min_docs_target_slice": kwargs.pop("density_min_docs_target_slice", 1),
            "min_docs_field_slice": kwargs.pop("density_min_docs_field_slice", 1),
        }
        model = KDEDensity(
            bundle.publications,
            target_label,
            target_author_uid=target_uid,
            year_col=kwargs.pop("year_col", "Year"),
            docid_col=kwargs.pop("docid_col", "Bibcode"),
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
            target_mask=target_mask,
            allow_name_fallback=False,
            **density_kwargs,
        )
        result = model.result(include_async=include_async)

    elif metric_key == "ref_vocab":
        if "tokens" not in bundle.references.columns:
            raise ValueError("Referenced Vocabulary requires references.tokens")
        model = ReferencedVocabularyKLD(
            bundle.publications,
            bundle.references,
            target_author_uid=target_uid,
            policy=kwargs.pop("reference_policy", DEFAULT_REFERENCE_POLICY),
            precompute_slice_moments="auto" if run_welch else False,
            **common,
            **kwargs,
        )
        result = model.result(include_async=include_async, run_welch=run_welch)

    elif metric_key == "citation_identity":
        config = CitationIdentityConfig(
            mode=kwargs.pop("cocit_mode", DEFAULT_COCIT_MODE),  # type: ignore[arg-type]
            counting=kwargs.pop("citation_identity_counting", DEFAULT_CITATION_IDENTITY_COUNTING),  # type: ignore[arg-type]
            author_scope=kwargs.pop("citation_author_scope", DEFAULT_CITATION_AUTHOR_SCOPE),  # type: ignore[arg-type]
            target_exclusion=kwargs.pop("target_exclusion", DEFAULT_TARGET_EXCLUSION),  # type: ignore[arg-type]
            remove_self_loops=bool(kwargs.pop("remove_self_loops", True)),
        )
        index = CitationIdentityEventIndex._from_normalized_frames(
            bundle.publications,
            bundle.references,
            config=CitationIdentityConfig(
                mode=config.mode,
                counting=config.counting,
                author_scope=config.author_scope,
                target_exclusion="none",
                remove_self_loops=config.remove_self_loops,
            ),
        )
        model = CitationIdentityKLD(
            index,
            config=config,
            target_name=target_label,
            target_author_uid=target_uid,
            target_mask=target_mask,
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
            lambda_param=lambda_param,
            epsilon=epsilon,
            top_k_kld_terms=top_k_kld_terms,
            min_token_global_freq=float(kwargs.pop("min_token_global_freq", 1.0)),
            min_docs_global_freq=int(kwargs.pop("min_docs_global_freq", 1)),
            min_tokens_target_slice=float(kwargs.pop("min_tokens_target_slice", 1e-12)),
            min_tokens_field_slice=float(kwargs.pop("min_tokens_field_slice", 1e-12)),
            min_docs_target_slice=int(kwargs.pop("min_docs_target_slice", 1)),
            min_docs_field_slice=int(kwargs.pop("min_docs_field_slice", 1)),
            min_docs_target_test=int(kwargs.pop("min_docs_target_test", 2)),
            min_docs_field_test=int(kwargs.pop("min_docs_field_test", 2)),
            max_vocab_size=kwargs.pop("max_vocab_size", None),
            precompute_slice_moments="auto" if run_welch else False,
            show_progress=show_progress,
            verbose=verbose,
        )
        result = replace(model.result(include_async=include_async, run_welch=run_welch), metric="citation_identity")
    else:
        raise _metric_error()

    resolved = {
        **result.config,
        "metric": metric_key,
        "target_author_uid": target_uid,
        "include_async": bool(include_async),
        "run_welch": bool(run_welch),
        "alpha": float(alpha),
        "multiple_testing": str(multiple_testing),
        "multiple_testing_scope": str(multiple_testing_scope),
    }
    return replace(
        result,
        target_author_uid=target_uid,
        target_name=target_label,
        config=resolved,
        provenance=_result_provenance(bundle),
    )


def run_metrics(
    bundle_or_publications_path: DatasetBundle | str | Path,
    references_path: str | Path | None = None,
    **kwargs: Unpack[_RunMetricsOptions],
) -> pd.DataFrame:
    """Run the consolidated top-author metrics through the friendly API name."""
    include = kwargs.pop("include", None)
    include_metrics = _coerce_include(include)
    if include_metrics is not None:
        unknown_for_full_run = include_metrics - _CORE_RUN_METRICS
        if unknown_for_full_run:
            raise ValueError(f"include contains unsupported metric(s): {sorted(unknown_for_full_run)}")
        kwargs.setdefault("include", tuple(metric for metric in METRIC_KEYS if metric in include_metrics))

    if isinstance(bundle_or_publications_path, DatasetBundle):
        if references_path is not None:
            raise ValueError("references_path must be omitted when passing a DatasetBundle")
        return run_top_authors_metrics_from_parquets(bundle_or_publications_path, **kwargs)
    if references_path is None:
        raise ValueError("references_path is required when the first argument is not a DatasetBundle")
    return run_top_authors_metrics_from_parquets(bundle_or_publications_path, references_path, **kwargs)


__all__ = ["METRIC_KEYS", "run_metric", "run_metrics"]
