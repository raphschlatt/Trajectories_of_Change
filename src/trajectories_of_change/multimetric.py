"""Core helpers for multimetric comparison runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, TypedDict, Unpack

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .citation_identity import CitationIdentityConfig
from .citation_identity_event import (
    CitationIdentityEventIndex,
    CitationIdentitySyncKLDResult,
    calculate_citation_identity_sync_kld_from_event_index,
)
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
from .metrics_density import (
    DensityPrecompute,
    KDEDensity,
    summarize_density_async,
    summarize_density_sync,
)
from .metrics_kld import KLDPrecompute, VocabularyKLD
from .referenced_vocabulary import (
    ReferenceTokenCache,
    ReferencedVocabularyKLD,
    build_reference_token_cache,
)
from .stats_utils import (
    MULTIPLE_TESTING_METHODS,
    _async_min_leadlag,
    _level_slope,
    add_pvalue_adjustments,
)
from ._parallel import limit_blas_threads, resolve_n_jobs
from ._filenames import safe_filename_component
from .contract import (
    DatasetBundle,
    _build_uid_display_name_map,
    _coerce_list,
    _load_bundle_arg,
    build_target_mask,
    canonicalize_column_name,
    is_placeholder_author_uid,
)


def pick_top_authors(
    df: pd.DataFrame,
    author_col: str,
    top_n: int,
    *,
    prefer_id_col: Optional[str] = "author_uids",
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Pick the most frequent author identities from list-based author columns."""
    if exclude is None:
        exclude = ["", "nan", "none", "unknown", "no author"]
    exclude_norm = {str(x).strip().lower() for x in exclude}
    count_col = prefer_id_col if prefer_id_col and prefer_id_col in df.columns else author_col
    exploded = df[count_col].apply(lambda value: _coerce_list(value, split_semicolon=True))
    counts = exploded.explode().astype(str).str.strip().value_counts()
    if not counts.empty:
        idx = counts.index.to_series().astype(str).str.strip().str.lower()
        placeholder_mask = counts.index.to_series().map(is_placeholder_author_uid).to_numpy(dtype=bool)
        counts = counts[(~idx.isin(exclude_norm)) & (~placeholder_mask)]
    return counts.head(top_n).index.tolist()


def _significant_kld_by_slice(
    df_sync: pd.DataFrame,
    df_welch_sync: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Attach signed and absolute significant KLD mass to each sync slice."""
    p_col = "p_adj" if "p_adj" in df_welch_sync.columns else "pvalue"
    sig = df_welch_sync[df_welch_sync[p_col] < alpha] if not df_welch_sync.empty else df_welch_sync
    if sig.empty:
        aggregates = pd.DataFrame(
            {
                "slice": df_sync["slice"],
                "kld_sig": np.zeros(len(df_sync), dtype=float),
                "kld_sig_abs": np.zeros(len(df_sync), dtype=float),
            }
        )
    else:
        aggregates = (
            sig.assign(kld_sig_abs=sig["kld_contribution"].abs())
            .groupby("slice", as_index=False)
            .agg(
                kld_sig=("kld_contribution", "sum"),
                kld_sig_abs=("kld_sig_abs", "sum"),
            )
        )
    return df_sync.merge(aggregates, on="slice", how="left").fillna(
        {"kld_sig": 0.0, "kld_sig_abs": 0.0}
    )


def summarize_kld_sync(
    df_sync: pd.DataFrame,
    df_welch_sync: pd.DataFrame,
    alpha: float,
    *,
    welch_enabled: bool = True,
) -> pd.Series:
    if df_sync.empty:
        return pd.Series(
            {
                "kld_all_level": np.nan,
                "kld_all_slope": np.nan,
                "kld_sig_level": np.nan,
                "kld_sig_slope": np.nan,
                "kld_sig_ratio": np.nan,
                "kld_sig_abs_level": np.nan,
                "kld_sig_abs_slope": np.nan,
                "kld_sig_abs_ratio": np.nan,
            }
        )
    x = df_sync["slice"].to_numpy()
    y = df_sync["kld_all"].to_numpy()
    all_level, all_slope = _level_slope(x, y)
    if not welch_enabled:
        return pd.Series(
            {
                "kld_all_level": all_level,
                "kld_all_slope": all_slope,
                "kld_sig_level": np.nan,
                "kld_sig_slope": np.nan,
                "kld_sig_ratio": np.nan,
                "kld_sig_abs_level": np.nan,
                "kld_sig_abs_slope": np.nan,
                "kld_sig_abs_ratio": np.nan,
            }
        )

    merged = _significant_kld_by_slice(df_sync, df_welch_sync, alpha=alpha)
    sig_slope = float(np.polyfit(merged["slice"], merged["kld_sig"], 1)[0]) if len(merged) >= 2 else np.nan
    sig_abs_slope = (
        float(np.polyfit(merged["slice"], merged["kld_sig_abs"], 1)[0]) if len(merged) >= 2 else np.nan
    )
    nonzero = merged["kld_all"].replace(0, np.nan)
    sig_ratio = (merged["kld_sig"] / nonzero).replace([np.inf, -np.inf], np.nan)
    sig_abs_ratio = (merged["kld_sig_abs"] / nonzero).replace([np.inf, -np.inf], np.nan)
    return pd.Series(
        {
            "kld_all_level": all_level,
            "kld_all_slope": all_slope,
            "kld_sig_level": float(merged["kld_sig"].median()),
            "kld_sig_slope": sig_slope,
            "kld_sig_ratio": float(sig_ratio.median()) if sig_ratio.notna().any() else np.nan,
            "kld_sig_abs_level": float(merged["kld_sig_abs"].median()),
            "kld_sig_abs_slope": sig_abs_slope,
            "kld_sig_abs_ratio": float(sig_abs_ratio.median()) if sig_abs_ratio.notna().any() else np.nan,
        }
    )


def summarize_kld_async(df_async: pd.DataFrame) -> pd.Series:
    if df_async.empty:
        return pd.Series({"kld_async_min": np.nan, "kld_async_leadlag": np.nan})
    min_mean, leadlag = _async_min_leadlag(df_async, "kld")
    return pd.Series({"kld_async_min": min_mean, "kld_async_leadlag": leadlag})


def _prepare_sync_welch(df: pd.DataFrame, *, method: str, scope: str) -> pd.DataFrame:
    sync = df[df["target_slice"] == df["field_slice"]].copy()
    sync.rename(columns={"target_slice": "slice"}, inplace=True)
    if method != "none":
        sync = add_pvalue_adjustments(
            sync,
            p_col="pvalue",
            method=method,
            group_cols=["slice"] if scope in {"slice", "pair"} else None,
            out_col="p_adj",
        )
    return sync


_KLD_OVERRIDE_KEYS = {
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


def _split_metric_kwargs(metric_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Route prefixed metric kwargs to the correct KLD implementation."""
    vocab_kwargs: dict[str, Any] = {}
    ref_vocab_kwargs: dict[str, Any] = {}
    cocit_kwargs: dict[str, Any] = {}

    for key, value in metric_kwargs.items():
        prefix = ""
        option = key
        for candidate in ("vocab_", "ref_vocab_", "cocit_"):
            if key.startswith(candidate):
                prefix = candidate
                option = key.removeprefix(candidate)
                break
        if option not in _KLD_OVERRIDE_KEYS:
            candidates = [
                f"{candidate}{allowed}"
                for candidate in ("vocab_", "ref_vocab_", "cocit_")
                for allowed in sorted(_KLD_OVERRIDE_KEYS)
            ] + sorted(_KLD_OVERRIDE_KEYS)
            suggestion = get_close_matches(key, candidates, n=1)
            hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
            raise TypeError(f"unexpected multimetric option {key!r}{hint}")
        if prefix == "vocab_":
            vocab_kwargs[option] = value
        elif prefix == "ref_vocab_":
            ref_vocab_kwargs[option] = value
        elif prefix == "cocit_":
            cocit_kwargs[option] = value
        else:
            vocab_kwargs[option] = value
            ref_vocab_kwargs[option] = value
            cocit_kwargs[option] = value

    return vocab_kwargs, ref_vocab_kwargs, cocit_kwargs


def _build_kld_precompute(
    corpus: pd.DataFrame,
    *,
    year_col: str,
    token_col: str,
    start_year: Optional[int],
    end_year: Optional[int],
    window_size: int,
    skip_incomplete_slices: bool,
    metric_kwargs: dict[str, Any],
    run_welch: bool,
) -> KLDPrecompute:
    return KLDPrecompute(
        corpus,
        year_col=year_col,
        token_col=token_col,
        start_year=start_year,
        end_year=end_year,
        window_size=window_size,
        skip_incomplete_slices=skip_incomplete_slices,
        min_token_global_freq=float(metric_kwargs.get("min_token_global_freq", 2)),
        min_docs_global_freq=int(metric_kwargs.get("min_docs_global_freq", 1)),
        max_vocab_size=metric_kwargs.get("max_vocab_size"),
        precompute_slice_moments="auto" if run_welch else False,
    )


def _build_ref_vocab_precompute(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    probe_target_uid: str,
    reference_cache,
    window_size: int,
    start_year: Optional[int],
    end_year: Optional[int],
    skip_incomplete_slices: bool,
    lambda_param: float,
    epsilon: float,
    top_k_kld_terms: Optional[int],
    run_welch: bool,
    ref_vocab_kwargs: dict[str, Any],
) -> Any:
    """Build the inclusive-policy referenced-vocabulary matrix once, for reuse.

    For policy="inclusive" the event matrix is target-independent (the external_only
    ``continue`` filter never fires), so it is built once via the standard
    ``ReferencedVocabularyKLD`` construction with any valid target and the ``.matrix``
    is reused across all targets. Using the real constructor (rather than duplicated
    build logic) guarantees the matrix is byte-identical to a per-target build.
    Diagnostics are target-dependent, so they are NOT built here (build_diagnostics=False)
    and never shared across targets; the consolidated path does not consume them.
    """
    probe = ReferencedVocabularyKLD(
        publications,
        references,
        target_author_uid=probe_target_uid,
        policy="inclusive",
        reference_cache=reference_cache,
        window_size=window_size,
        start_year=start_year,
        end_year=end_year,
        skip_incomplete_slices=skip_incomplete_slices,
        lambda_param=lambda_param,
        epsilon=epsilon,
        top_k_kld_terms=top_k_kld_terms,
        precompute_slice_moments="auto" if run_welch else False,
        build_diagnostics=False,
        **ref_vocab_kwargs,
    )
    return probe.matrix


def _build_uid_position_index(df: pd.DataFrame, author_ids_col: str) -> dict[str, list[int]]:
    """Map each casefolded author id to the row positions that contain it.

    Mirrors the matching in ``contract._row_contains_target`` so a per-target
    mask is an O(1) lookup instead of an O(corpus) scan. Built once per run and
    reused for every target (the author_uids selection path).
    """
    index: dict[str, list[int]] = {}
    if author_ids_col not in df.columns:
        return index
    for position, values in enumerate(df[author_ids_col].to_numpy()):
        for value in _coerce_list(values, split_semicolon=True):
            index.setdefault(value.casefold(), []).append(position)
    return index


def _metrics_publication_columns(
    *,
    author_col: str,
    author_id_col: str,
    year_col: str,
    tokens_col: str,
    density_embedding_cols: Sequence[str],
) -> list[str]:
    columns = [
        "Bibcode",
        year_col,
        author_col,
        author_id_col,
        "author_display_names",
        "References",
        tokens_col,
        *density_embedding_cols,
    ]
    return list(dict.fromkeys(columns))


def _metrics_reference_columns(
    *,
    author_col: str,
    author_id_col: str,
    include_tokens: bool = False,
) -> list[str]:
    columns = ["Bibcode", author_col, author_id_col, "author_display_names"]
    if include_tokens:
        columns.extend(["tokens", "Title", "Title_en", "Abstract", "Abstract_en", "Title_lang", "Abstract_lang"])
    return list(dict.fromkeys(columns))


def _write_detail_tables(
    details_out_dir: Path | str | None,
    *,
    target_label: str,
    tables: dict[str, pd.DataFrame],
) -> None:
    if details_out_dir is None:
        return
    target_dir = Path(details_out_dir) / safe_filename_component(target_label)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(target_dir / f"{name}.parquet", index=False)


def _kld_dashboard_sync(
    df_sync: pd.DataFrame,
    df_welch_sync: pd.DataFrame,
    *,
    alpha: float,
    welch_enabled: bool = True,
) -> pd.DataFrame:
    if df_sync.empty:
        return pd.DataFrame(columns=["slice", "kld_all", "kld_sig", "kld_sig_abs"])
    if not welch_enabled:
        return df_sync.assign(kld_sig=np.nan, kld_sig_abs=np.nan)
    return _significant_kld_by_slice(df_sync, df_welch_sync, alpha=alpha)


def _median_or_nan(values: Sequence[float]) -> float:
    cleaned = [float(value) for value in values if pd.notna(value)]
    return float(np.median(cleaned)) if cleaned else np.nan


def _coverage_summary(
    *,
    slices_total: int,
    slice_token_counts: dict,
    target_doc_counts: dict,
    field_doc_counts: dict,
    df_sync: pd.DataFrame,
    df_welch_sync: pd.DataFrame,
    alpha: float,
    welch_enabled: bool,
) -> pd.Series:
    kld_labels = [int(value) for value in df_sync["slice"].tolist()] if not df_sync.empty else []
    welch_labels = (
        sorted(int(value) for value in df_welch_sync["slice"].dropna().unique())
        if welch_enabled and "slice" in df_welch_sync.columns and not df_welch_sync.empty
        else []
    )
    kld_counts = [slice_token_counts.get(label, {}) for label in kld_labels]
    if welch_enabled:
        p_col = "p_adj" if "p_adj" in df_welch_sync.columns else "pvalue"
        sig = df_welch_sync[df_welch_sync[p_col] < alpha] if not df_welch_sync.empty else df_welch_sync
        sig_counts = (
            sig.groupby("slice").size().reindex(kld_labels, fill_value=0).tolist()
            if kld_labels and not sig.empty
            else ([0] * len(kld_labels))
        )
        sig_terms_total = int(len(sig))
        sig_terms_median = _median_or_nan(sig_counts)
    else:
        sig_terms_total = np.nan
        sig_terms_median = np.nan

    return pd.Series(
        {
            "slices_total": int(slices_total),
            "slices_kld": int(len(kld_labels)),
            "slices_welch": int(len(welch_labels)),
            "welch_rows": int(len(df_welch_sync)) if welch_enabled else 0,
            "target_docs_median_kld": _median_or_nan([counts.get("target_docs", np.nan) for counts in kld_counts]),
            "field_docs_median_kld": _median_or_nan([counts.get("field_docs", np.nan) for counts in kld_counts]),
            "target_tokens_median_kld": _median_or_nan(
                [counts.get("target_tokens", np.nan) for counts in kld_counts]
            ),
            "field_tokens_median_kld": _median_or_nan([counts.get("field_tokens", np.nan) for counts in kld_counts]),
            "target_docs_median_welch": _median_or_nan(
                [target_doc_counts.get(label, np.nan) for label in welch_labels]
            ),
            "field_docs_median_welch": _median_or_nan(
                [field_doc_counts.get(label, np.nan) for label in welch_labels]
            ),
            "sig_terms_total": sig_terms_total,
            "sig_terms_median_per_slice": sig_terms_median,
        }
    )


def _summarize_kld_coverage(
    model: VocabularyKLD,
    df_sync: pd.DataFrame,
    df_welch_sync: pd.DataFrame,
    *,
    alpha: float,
    welch_enabled: bool = True,
) -> pd.Series:
    return _coverage_summary(
        slices_total=len(model.slices),
        slice_token_counts=model.slice_token_counts,
        target_doc_counts=model.doc_counts_target,
        field_doc_counts=model.doc_counts_field,
        df_sync=df_sync,
        df_welch_sync=df_welch_sync,
        alpha=alpha,
        welch_enabled=welch_enabled,
    )


def _summarize_aggregate_cocit_coverage(
    result: CitationIdentitySyncKLDResult,
    df_sync: pd.DataFrame,
    df_welch_sync: pd.DataFrame,
    *,
    alpha: float,
    welch_enabled: bool,
) -> pd.Series:
    metadata = result.metadata
    return _coverage_summary(
        slices_total=int(metadata.get("slices_total", 0)),
        slice_token_counts=metadata.get("slice_token_counts", {}),
        target_doc_counts=metadata.get("welch_target_doc_counts", {}),
        field_doc_counts=metadata.get("welch_field_doc_counts", {}),
        df_sync=df_sync,
        df_welch_sync=df_welch_sync,
        alpha=alpha,
        welch_enabled=welch_enabled,
    )


def _aggregate_field_entropy_level(result: CitationIdentitySyncKLDResult, labels: Sequence[int]) -> float:
    entropies = result.metadata.get("field_entropies", {})
    return _median_or_nan([entropies.get(int(label), np.nan) for label in labels])


def _summarize_density_coverage(density: KDEDensity, df_sync: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "density_slices_total": int(len(density.slices)),
            "density_target_docs_median_sync": _median_or_nan(
                df_sync["target_docs"].tolist() if "target_docs" in df_sync.columns else []
            ),
            "density_field_docs_median_sync": _median_or_nan(
                df_sync["field_docs"].tolist() if "field_docs" in df_sync.columns else []
            ),
        }
    )


def _empty_welch_sync() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["slice", "field_slice", "term", "pvalue", "kld_contribution", "mean_target", "mean_field"]
    )


def _extract_provenance_metadata(bundle) -> dict[str, Any]:
    manifest = bundle.manifest or {}
    provenance = bundle.provenance or {}
    run_summary = provenance.get("run_summary", {})
    config = provenance.get("config", {})
    topic_model = config.get("topic_model", {})
    run_config = config.get("run", {})
    search_config = config.get("search", {})
    return {
        "dataset_run_id": manifest.get("run_id") or run_summary.get("run", {}).get("run_id"),
        "producer": manifest.get("producer"),
        "producer_version": manifest.get("producer_version"),
        "source_query": search_config.get("query"),
        "embedding_provider": topic_model.get("embedding_provider"),
        "embedding_model": topic_model.get("embedding_model"),
        "reduction_method": topic_model.get("reduction_method"),
        "random_seed": run_config.get("random_seed"),
        "and_enabled": manifest.get("and_enabled"),
        "upstream_git_commit": run_summary.get("reproducibility", {}).get("git_commit"),
    }


class _RunMetricsOptions(TypedDict, total=False):
    """Optional public arguments shared by the eager and streaming runners."""

    manifest_path: Path | str | None
    run_summary_path: Path | str | None
    config_path: Path | str | None
    auto_discover_sidecars: bool
    strict_provenance: bool
    assume_valid: bool
    top_n: int
    targets: Optional[List[str]]
    select_by: str
    author_col: str
    author_id_col: str
    year_col: str
    tokens_col: str
    start_year: Optional[int]
    end_year: Optional[int]
    window_size: int
    skip_incomplete_slices: bool
    lambda_param: float
    epsilon: float
    alpha: float
    multiple_testing: str
    multiple_testing_scope: str
    top_k_kld_terms: Optional[int]
    cocit_mode: str
    remove_self_loops: bool
    citation_identity_counting: str
    citation_author_scope: str
    target_exclusion: str
    run_welch: bool
    include_async: bool
    density_bandwidth: Optional[float]
    density_embedding_cols: Optional[Sequence[str]]
    density_standardize: bool
    density_min_docs_target_slice: int
    density_min_docs_field_slice: int
    reference_policy: str
    include: Optional[Sequence[str]]
    n_jobs: int | str
    details_out_dir: Path | str | None
    show_progress: bool
    verbose: bool


@dataclass(frozen=True)
class _ResolvedRunConfig:
    sidecar_paths: tuple[Path | str | None, Path | str | None, Path | str | None]
    input_flags: tuple[bool, bool, bool]
    top_n: int
    targets: Optional[List[str]]
    select_by: str
    columns: tuple[str, str, str, str]
    years: tuple[Optional[int], Optional[int]]
    window: tuple[int, bool]
    smoothing: tuple[float, float]
    testing: tuple[float, str, str, Optional[int], bool]
    citation: CitationIdentityConfig
    include_async: bool
    density: tuple[Optional[float], tuple[str, ...], bool, int, int]
    reference_policy: str
    selected_metrics: frozenset[str]
    execution: tuple[int | str, Path | str | None, bool, bool]
    vocab_kwargs: dict[str, Any]
    ref_vocab_kwargs: dict[str, Any]
    cocit_kwargs: dict[str, Any]

    def includes(self, metric: str) -> bool:
        return metric in self.selected_metrics


@dataclass(frozen=True)
class _RunState:
    bundle: DatasetBundle
    vocab_precompute: Optional[KLDPrecompute]
    density_precompute: Optional[DensityPrecompute]
    reference_token_cache: Optional[ReferenceTokenCache]
    ref_vocab_matrix: Any | None
    citation_identity_event_index: Optional[CitationIdentityEventIndex]
    uid_position_index: dict[str, list[int]]
    uid_display_names: dict[str, Optional[str]]
    record_metadata: dict[str, Any]


@dataclass(frozen=True)
class _TargetSpec:
    label: str
    name: str
    uid: Optional[str]
    mask: np.ndarray


def _resolve_run_config(options: dict[str, Any]) -> _ResolvedRunConfig:
    """Validate raw runner options and resolve every default in one place."""

    raw = dict(options)
    metric_kwargs = {
        key: value
        for key, value in raw.items()
        if key not in _RunMetricsOptions.__annotations__
    }
    if "citation_identity_backend" in metric_kwargs:
        raise ValueError(
            "citation_identity_backend has been removed; Citation Identity uses the event/core path."
        )
    multiple_testing = raw.get("multiple_testing", DEFAULT_MULTIPLE_TESTING)
    multiple_testing_scope = raw.get("multiple_testing_scope", DEFAULT_MULTIPLE_TESTING_SCOPE)
    select_by = raw.get("select_by", "uid")
    include = raw.get("include")
    selected_metrics = frozenset(
        METRIC_KEYS if include is None else (str(item) for item in include)
    )
    unknown_metrics = selected_metrics - set(METRIC_KEYS)
    if unknown_metrics:
        raise ValueError(f"include contains unsupported metric(s): {sorted(unknown_metrics)}")

    if multiple_testing not in MULTIPLE_TESTING_METHODS:
        raise ValueError(f"multiple_testing must be one of {sorted(MULTIPLE_TESTING_METHODS)}")
    if multiple_testing_scope not in {"slice", "pair", "global"}:
        raise ValueError("multiple_testing_scope must be one of: slice, pair, global")
    if select_by not in {"uid", "name"}:
        raise ValueError(f"select_by must be 'uid' or 'name', got {select_by!r}")

    vocab_kwargs, ref_vocab_kwargs, cocit_kwargs = _split_metric_kwargs(metric_kwargs)
    citation_identity_counting = raw.get(
        "citation_identity_counting",
        DEFAULT_CITATION_IDENTITY_COUNTING,
    )
    if citation_identity_counting == "document_fractional":
        cocit_kwargs.setdefault("min_token_global_freq", 1.0)
        cocit_kwargs.setdefault("min_tokens_target_slice", 1e-12)
        cocit_kwargs.setdefault("min_tokens_field_slice", 1e-12)

    return _ResolvedRunConfig(
        sidecar_paths=(
            raw.get("manifest_path"),
            raw.get("run_summary_path"),
            raw.get("config_path"),
        ),
        input_flags=(
            raw.get("auto_discover_sidecars", False),
            raw.get("strict_provenance", False),
            raw.get("assume_valid", False),
        ),
        top_n=raw.get("top_n", 5),
        targets=raw.get("targets"),
        select_by=select_by,
        columns=(
            raw.get("author_col", "Author"),
            raw.get("author_id_col", "author_uids"),
            raw.get("year_col", "Year"),
            raw.get("tokens_col", "tokens"),
        ),
        years=(raw.get("start_year"), raw.get("end_year")),
        window=(
            raw.get("window_size", DEFAULT_WINDOW_SIZE),
            raw.get("skip_incomplete_slices", True),
        ),
        smoothing=(
            raw.get("lambda_param", DEFAULT_LAMBDA_PARAM),
            raw.get("epsilon", DEFAULT_EPSILON),
        ),
        testing=(
            raw.get("alpha", DEFAULT_ALPHA),
            multiple_testing,
            multiple_testing_scope,
            raw.get("top_k_kld_terms", DEFAULT_TOP_K_KLD_TERMS),
            raw.get("run_welch", True),
        ),
        citation=CitationIdentityConfig(
            mode=raw.get("cocit_mode", DEFAULT_COCIT_MODE),  # type: ignore[arg-type]
            counting=citation_identity_counting,  # type: ignore[arg-type]
            author_scope=raw.get("citation_author_scope", DEFAULT_CITATION_AUTHOR_SCOPE),  # type: ignore[arg-type]
            target_exclusion=raw.get("target_exclusion", DEFAULT_TARGET_EXCLUSION),  # type: ignore[arg-type]
            remove_self_loops=raw.get("remove_self_loops", True),
        ),
        include_async=raw.get("include_async", False),
        density=(
            raw.get("density_bandwidth"),
            tuple(raw.get("density_embedding_cols") or DEFAULT_DENSITY_EMBEDDING_COLS),
            raw.get("density_standardize", True),
            raw.get("density_min_docs_target_slice", 1),
            raw.get("density_min_docs_field_slice", 1),
        ),
        reference_policy=raw.get("reference_policy", DEFAULT_REFERENCE_POLICY),
        selected_metrics=selected_metrics,
        execution=(
            raw.get("n_jobs", 1),
            raw.get("details_out_dir"),
            raw.get("show_progress", True),
            raw.get("verbose", False),
        ),
        vocab_kwargs=vocab_kwargs,
        ref_vocab_kwargs=ref_vocab_kwargs,
        cocit_kwargs=cocit_kwargs,
    )


def _run_one_target(
    *,
    config: _ResolvedRunConfig,
    state: _RunState,
    target: _TargetSpec,
) -> dict[str, Any]:
    vocab_df = state.bundle.publications
    references_df = state.bundle.references
    target_mask = target.mask
    target_label = target.label
    target_name = target.name
    target_uid = target.uid
    author_col, author_id_col, year_col, tokens_col = config.columns
    start_year, end_year = config.years
    window_size, skip_incomplete_slices = config.window
    lambda_param, epsilon = config.smoothing
    alpha, multiple_testing, multiple_testing_scope, top_k_kld_terms, run_welch = config.testing
    include_async = config.include_async
    include_own_vocab = config.includes("own_vocab")
    include_density = config.includes("density")
    include_cocit = config.includes("citation_identity")
    (
        density_bandwidth,
        density_embedding_cols,
        density_standardize,
        density_min_docs_target_slice,
        density_min_docs_field_slice,
    ) = config.density
    vocab_kwargs = config.vocab_kwargs
    ref_vocab_kwargs = config.ref_vocab_kwargs
    vocab_precompute = state.vocab_precompute
    density_precompute = state.density_precompute
    reference_policy = config.reference_policy
    reference_token_cache = state.reference_token_cache
    ref_vocab_matrix = state.ref_vocab_matrix
    _, details_out_dir, show_progress, verbose = config.execution
    allow_name_fallback = target_uid is None
    common_kld = {
        "window_size": window_size,
        "start_year": start_year,
        "end_year": end_year,
        "skip_incomplete_slices": skip_incomplete_slices,
        "lambda_param": lambda_param,
        "epsilon": epsilon,
        "top_k_kld_terms": top_k_kld_terms,
        "show_progress": show_progress,
        "verbose": verbose,
        "target_mask": target_mask,
    }
    v_sync = pd.DataFrame(columns=["slice", "kld_all"])
    v_pointwise = pd.DataFrame(columns=["slice", "term", "kld_contribution"])
    v_welch_sync = _empty_welch_sync()
    v_async_df = pd.DataFrame(columns=["target_slice", "field_slice", "time_diff", "kld"])
    vocab_summary: Optional[pd.Series] = None
    if include_own_vocab:
        vkld = VocabularyKLD(
            vocab_df,
            target_name,
            target_author_uid=target_uid,
            author_col=author_col,
            author_id_col=author_id_col,
            year_col=year_col,
            token_col=tokens_col,
            allow_name_fallback=allow_name_fallback,
            shared_precompute=vocab_precompute,
            **common_kld,
            **vocab_kwargs,
        )
        v_sync, v_pointwise = vkld.calculate_kld_sync()
        if run_welch:
            v_welch_sync = _prepare_sync_welch(
                vkld.perform_welch_tests_all_pairs(sync_only=True),
                method=multiple_testing,
                scope=multiple_testing_scope,
            )
        vocab_summary = summarize_kld_sync(v_sync, v_welch_sync, alpha, welch_enabled=run_welch)
        vocab_coverage = _summarize_kld_coverage(vkld, v_sync, v_welch_sync, alpha=alpha, welch_enabled=run_welch)
        vocab_summary.index = ["vocab_" + key for key in vocab_summary.index]
        vocab_coverage.index = ["vocab_" + key for key in vocab_coverage.index]
        vocab_summary = pd.concat([vocab_summary, vocab_coverage])
        if include_async:
            v_async_df = vkld.calculate_kld_async()
            v_async = summarize_kld_async(v_async_df)
            v_async.index = ["vocab_" + key for key in v_async.index]
            vocab_summary = pd.concat([vocab_summary, v_async])

    c_sync = pd.DataFrame(columns=["slice", "kld_all"])
    c_pointwise = pd.DataFrame()
    c_welch_sync = _empty_welch_sync()
    c_async_df = pd.DataFrame(columns=["target_slice", "field_slice", "time_diff", "kld"])
    cocit_diagnostics_documents = pd.DataFrame()
    cocit_diagnostics_by_slice = pd.DataFrame()
    cocit_summary: Optional[pd.Series] = None
    cocit_welch_enabled = False
    if include_cocit:
        if state.citation_identity_event_index is None:
            raise RuntimeError("Citation Identity event index was not initialized")
        event_cocit = calculate_citation_identity_sync_kld_from_event_index(
            state.citation_identity_event_index,
            config=config.citation,
            target_name=target_name,
            target_author_uid=target_uid,
            target_mask=target_mask,
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
            lambda_param=lambda_param,
            epsilon=epsilon,
            min_token_global_freq=float(config.cocit_kwargs.get("min_token_global_freq", 1.0)),
            min_docs_global_freq=int(config.cocit_kwargs.get("min_docs_global_freq", 1)),
            min_tokens_target_slice=float(config.cocit_kwargs.get("min_tokens_target_slice", 1e-12)),
            min_tokens_field_slice=float(config.cocit_kwargs.get("min_tokens_field_slice", 1e-12)),
            min_docs_target_slice=int(config.cocit_kwargs.get("min_docs_target_slice", 1)),
            min_docs_field_slice=int(config.cocit_kwargs.get("min_docs_field_slice", 1)),
            min_docs_target_test=int(config.cocit_kwargs.get("min_docs_target_test", 2)),
            min_docs_field_test=int(config.cocit_kwargs.get("min_docs_field_test", 2)),
            max_vocab_size=config.cocit_kwargs.get("max_vocab_size"),
            top_k_kld_terms=top_k_kld_terms,
            include_async=include_async,
            run_welch=run_welch,
            welch_sync_only=True,
        )
        c_sync = event_cocit.sync
        c_pointwise = event_cocit.pointwise
        if run_welch and event_cocit.welch is not None:
            c_welch_sync = _prepare_sync_welch(
                event_cocit.welch,
                method=multiple_testing,
                scope=multiple_testing_scope,
            )
            cocit_welch_enabled = True
        cocit_summary = summarize_kld_sync(
            c_sync,
            c_welch_sync,
            alpha,
            welch_enabled=cocit_welch_enabled,
        )
        cocit_coverage = _summarize_aggregate_cocit_coverage(
            event_cocit,
            c_sync,
            c_welch_sync,
            alpha=alpha,
            welch_enabled=cocit_welch_enabled,
        )
        cocit_summary.index = ["cocit_" + key for key in cocit_summary.index]
        cocit_coverage.index = ["cocit_" + key for key in cocit_coverage.index]
        cocit_summary = pd.concat([cocit_summary, cocit_coverage])
        if include_async and event_cocit.async_df is not None:
            c_async_df = event_cocit.async_df
            c_async = summarize_kld_async(c_async_df)
            c_async.index = ["cocit_" + key for key in c_async.index]
            cocit_summary = pd.concat([cocit_summary, c_async])
        cocit_labels = [int(value) for value in c_sync["slice"].tolist()] if not c_sync.empty else []
        cocit_summary = pd.concat(
            [
                cocit_summary,
                pd.Series(
                    {
                        "cocit_support_size": int(event_cocit.metadata.get("support_size", 0)),
                        "cocit_field_entropy_level": _aggregate_field_entropy_level(
                            event_cocit,
                            cocit_labels,
                        ),
                    }
                ),
            ]
        )
        diagnostics_summary = event_cocit.metadata.get("diagnostics_summary", {})
        diag_summary = {
            f"cocit_{key}": value
            for key, value in diagnostics_summary.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
        }
        diag_summary["cocit_dropped_pair_mass"] = float(
            diagnostics_summary.get("self_loop_pair_mass", 0.0)
        ) + float(diagnostics_summary.get("target_excluded_pair_mass", 0.0))
        cocit_summary = pd.concat([cocit_summary, pd.Series(diag_summary)])
        cocit_diagnostics_documents = event_cocit.metadata.get("diagnostics_documents", pd.DataFrame())
        cocit_diagnostics_by_slice = event_cocit.metadata.get("diagnostics_by_slice", pd.DataFrame())

    rv_sync = pd.DataFrame(columns=["slice", "kld_all"])
    rv_pointwise = pd.DataFrame(columns=["slice", "term", "kld_contribution"])
    rv_welch_sync = _empty_welch_sync()
    rv_async_df = pd.DataFrame(columns=["target_slice", "field_slice", "time_diff", "kld"])
    rv_welch_enabled = False
    ref_vocab_summary: Optional[pd.Series] = None
    if reference_token_cache is not None and references_df is not None and target_uid is not None:
        rvkld = ReferencedVocabularyKLD(
            vocab_df,
            references_df,
            target_author_uid=target_uid,
            policy=reference_policy,
            reference_cache=reference_token_cache,
            prebuilt_matrix=ref_vocab_matrix,
            build_diagnostics=False,
            precompute_slice_moments="auto" if run_welch else False,
            **common_kld,
            **ref_vocab_kwargs,
        )
        rv_sync, rv_pointwise = rvkld.calculate_kld_sync()
        if run_welch:
            rv_welch_sync = _prepare_sync_welch(
                rvkld.perform_welch_tests_all_pairs(sync_only=True),
                method=multiple_testing,
                scope=multiple_testing_scope,
            )
            rv_welch_enabled = True
        ref_vocab_summary = summarize_kld_sync(rv_sync, rv_welch_sync, alpha, welch_enabled=rv_welch_enabled)
        ref_vocab_coverage = _summarize_kld_coverage(
            rvkld,
            rv_sync,
            rv_welch_sync,
            alpha=alpha,
            welch_enabled=rv_welch_enabled,
        )
        ref_vocab_summary.index = ["ref_vocab_" + key for key in ref_vocab_summary.index]
        ref_vocab_coverage.index = ["ref_vocab_" + key for key in ref_vocab_coverage.index]
        ref_vocab_summary = pd.concat([ref_vocab_summary, ref_vocab_coverage])
        if include_async:
            rv_async_df = rvkld.calculate_kld_async()
            rv_async = summarize_kld_async(rv_async_df)
            rv_async.index = ["ref_vocab_" + key for key in rv_async.index]
            ref_vocab_summary = pd.concat([ref_vocab_summary, rv_async])

    d_sync = pd.DataFrame()
    d_pointwise = pd.DataFrame()
    d_async_df = pd.DataFrame(
        columns=["target_slice", "field_slice", "time_diff", "density_neglog_median", "target_docs", "field_docs"]
    )
    density_record: dict[str, Any] = {}
    if include_density:
        density = KDEDensity(
            vocab_df,
            target_name,
            target_author_uid=target_uid,
            author_col=author_col,
            author_id_col=author_id_col,
            year_col=year_col,
            docid_col="Bibcode",
            embedding_cols=density_embedding_cols,
            window_size=window_size,
            start_year=start_year,
            end_year=end_year,
            skip_incomplete_slices=skip_incomplete_slices,
            bandwidth=density_bandwidth,
            min_docs_target_slice=density_min_docs_target_slice,
            min_docs_field_slice=density_min_docs_field_slice,
            allow_name_fallback=allow_name_fallback,
            standardize=density_standardize,
            shared_precompute=density_precompute,
            target_mask=target_mask,
        )
        d_sync, d_pointwise = density.calculate_density_sync()
        density_summary = summarize_density_sync(d_sync)
        density_summary = pd.concat([density_summary, _summarize_density_coverage(density, d_sync)])
        if include_async:
            d_async_df = density.calculate_density_async()
            density_summary = pd.concat([density_summary, summarize_density_async(d_async_df)])
        density_record = {
            "density_bandwidth": density.bandwidth,
            "density_embedding_cols": tuple(density.embedding_cols),
            "density_standardize": density.standardize,
            "density_slices_sync": int(len(d_sync)),
            **density_summary.to_dict(),
        }

    detail_tables: dict[str, pd.DataFrame] = {}
    if include_own_vocab:
        detail_tables.update({
            "vocab_kld_sync": v_sync,
            "vocab_kld_pointwise": v_pointwise,
            "vocab_welch_sync": v_welch_sync,
            "vocab_dashboard_sync": _kld_dashboard_sync(v_sync, v_welch_sync, alpha=alpha, welch_enabled=run_welch),
            "vocab_kld_async": v_async_df,
        })
    if include_cocit:
        detail_tables.update({
            "cocit_kld_sync": c_sync,
            "cocit_kld_pointwise": c_pointwise,
            "cocit_welch_sync": c_welch_sync,
            "cocit_dashboard_sync": _kld_dashboard_sync(
                c_sync,
                c_welch_sync,
                alpha=alpha,
                welch_enabled=cocit_welch_enabled,
            ),
            "cocit_kld_async": c_async_df,
            "cocit_diagnostics": cocit_diagnostics_documents,
            "cocit_diagnostics_by_slice": cocit_diagnostics_by_slice,
        })
    if include_density:
        detail_tables.update({
            "density_sync": d_sync,
            "density_pointwise": d_pointwise,
            "density_async": d_async_df,
        })
    if ref_vocab_summary is not None:
        detail_tables.update({
            "ref_vocab_kld_sync": rv_sync,
            "ref_vocab_kld_pointwise": rv_pointwise,
            "ref_vocab_welch_sync": rv_welch_sync,
            "ref_vocab_dashboard_sync": _kld_dashboard_sync(
                rv_sync, rv_welch_sync, alpha=alpha, welch_enabled=rv_welch_enabled
            ),
            "ref_vocab_kld_async": rv_async_df,
        })
    _write_detail_tables(
        details_out_dir,
        target_label=target_label,
        tables=detail_tables,
    )

    record = {
        "author": target_label,
        "selection_mode": author_id_col if target_uid else canonicalize_column_name(author_col),
        "alpha": alpha,
        "multiple_testing": multiple_testing,
        "multiple_testing_scope": multiple_testing_scope,
        "top_k_kld_terms": top_k_kld_terms,
        **density_record,
        **(vocab_summary.to_dict() if vocab_summary is not None else {}),
        **(cocit_summary.to_dict() if cocit_summary is not None else {}),
        **(ref_vocab_summary.to_dict() if ref_vocab_summary is not None else {}),
    }
    record.update(state.record_metadata)
    if target_uid is not None:
        record["author_uid"] = target_uid
        record["author_display_name"] = state.uid_display_names.get(str(target_uid))
    return record


def iter_top_authors_metrics_from_parquets(
    publications_path,
    references_path=None,
    **options: Unpack[_RunMetricsOptions],
) -> Iterator[dict[str, Any]]:
    """Yield per-target metric records from the canonical two-parquet input contract."""
    config = _resolve_run_config(dict(options))
    manifest_path, run_summary_path, config_path = config.sidecar_paths
    auto_discover_sidecars, strict_provenance, assume_valid = config.input_flags
    author_col, author_id_col, year_col, tokens_col = config.columns
    _, density_embedding_cols, density_standardize, _, _ = config.density
    bundle = _load_bundle_arg(
        publications_path,
        references_path,
        manifest_path=manifest_path,
        run_summary_path=run_summary_path,
        config_path=config_path,
        auto_discover_sidecars=auto_discover_sidecars,
        strict_provenance=strict_provenance,
        assume_valid=assume_valid,
        publication_columns=_metrics_publication_columns(
            author_col=author_col,
            author_id_col=author_id_col,
            year_col=year_col,
            tokens_col=tokens_col,
            density_embedding_cols=density_embedding_cols,
        ),
        reference_columns=_metrics_reference_columns(
            author_col=author_col,
            author_id_col=author_id_col,
            include_tokens=config.includes("ref_vocab"),
        ),
        combined_bundle_error=True,
    )
    vocab_df = bundle.publications.copy(deep=False)
    include_own_vocab = config.includes("own_vocab")
    include_referenced_vocab = config.includes("ref_vocab")
    include_density = config.includes("density")
    include_cocit = config.includes("citation_identity")
    start_year, end_year = config.years
    window_size, skip_incomplete_slices = config.window
    _, _, _, _, run_welch = config.testing
    density_bandwidth, _, _, _, _ = config.density
    vocab_kwargs = config.vocab_kwargs
    ref_vocab_kwargs = config.ref_vocab_kwargs
    targets = config.targets
    vocab_precompute = (
        _build_kld_precompute(
            vocab_df,
            year_col=year_col,
            token_col=tokens_col,
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
            metric_kwargs=vocab_kwargs,
            run_welch=run_welch,
        )
        if include_own_vocab
        else None
    )
    density_precompute = (
        DensityPrecompute(
            vocab_df,
            year_col=year_col,
            embedding_cols=density_embedding_cols,
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
            bandwidth=density_bandwidth,
            standardize=density_standardize,
        )
        if include_density
        else None
    )
    reference_token_cache = (
        build_reference_token_cache(bundle.references)
        if include_referenced_vocab and "tokens" in bundle.references.columns
        else None
    )
    record_metadata = {
        "window_size": int(window_size),
        "start_year": start_year,
        "end_year": end_year,
        "lambda_param": float(config.smoothing[0]),
        "epsilon": float(config.smoothing[1]),
        "welch_enabled": bool(run_welch),
        **(
            {
                "cocit_mode": config.citation.mode,
                "remove_self_loops": bool(config.citation.remove_self_loops),
                "citation_identity_counting": config.citation.counting,
                "citation_author_scope": config.citation.author_scope,
                "target_exclusion": config.citation.target_exclusion,
            }
            if include_cocit
            else {}
        ),
        **_extract_provenance_metadata(bundle),
    }
    use_target_uids = bool(config.select_by == "uid" and author_id_col in vocab_df.columns)
    if targets is None:
        targets = pick_top_authors(
            vocab_df,
            author_col,
            config.top_n,
            prefer_id_col=author_id_col if use_target_uids else None,
        )

    uid_position_index = _build_uid_position_index(vocab_df, author_id_col) if use_target_uids else {}
    uid_display_names = (
        _build_uid_display_name_map(vocab_df, author_ids_col=author_id_col)
        if use_target_uids
        else {}
    )
    n_docs_total = len(vocab_df)

    ref_vocab_matrix = None
    if (
        include_referenced_vocab
        and reference_token_cache is not None
        and config.reference_policy == "inclusive"
        and use_target_uids
        and len(targets) > 0
    ):
        ref_vocab_matrix = _build_ref_vocab_precompute(
            vocab_df,
            bundle.references,
            probe_target_uid=str(targets[0]),
            reference_cache=reference_token_cache,
            window_size=window_size,
            start_year=start_year,
            end_year=end_year,
            skip_incomplete_slices=skip_incomplete_slices,
            lambda_param=config.smoothing[0],
            epsilon=config.smoothing[1],
            top_k_kld_terms=config.testing[3],
            run_welch=run_welch,
            ref_vocab_kwargs=ref_vocab_kwargs,
        )

    citation_identity_event_index: CitationIdentityEventIndex | None = None
    if include_cocit:
        base_citation_identity_config = CitationIdentityConfig(
            mode=config.citation.mode,
            counting=config.citation.counting,
            author_scope=config.citation.author_scope,
            target_exclusion="none",
            remove_self_loops=config.citation.remove_self_loops,
        )
        citation_identity_event_index = CitationIdentityEventIndex._from_normalized_frames(
            bundle.publications,
            bundle.references,
            config=base_citation_identity_config,
        )

    state = _RunState(
        bundle=bundle,
        vocab_precompute=vocab_precompute,
        density_precompute=density_precompute,
        reference_token_cache=reference_token_cache,
        ref_vocab_matrix=ref_vocab_matrix,
        citation_identity_event_index=citation_identity_event_index,
        uid_position_index=uid_position_index,
        uid_display_names=uid_display_names,
        record_metadata=record_metadata,
    )

    def _compute_target(target_label: str):
        target_uid = target_label if use_target_uids else None
        target_name = "" if use_target_uids else target_label
        if use_target_uids:
            positions = state.uid_position_index.get(str(target_label).strip().casefold())
            if positions is not None:
                target_mask = np.zeros(n_docs_total, dtype=bool)
                target_mask[positions] = True
            else:
                target_mask = build_target_mask(
                    vocab_df,
                    target_name=target_name,
                    target_author_uid=target_uid,
                    author_col=author_col,
                    author_ids_col=author_id_col,
                    allow_name_fallback=False,
                ).to_numpy(dtype=bool)
        else:
            target_mask = build_target_mask(
                vocab_df,
                target_name=target_name,
                target_author_uid=target_uid,
                author_col=author_col,
                author_ids_col=author_id_col,
                allow_name_fallback=True,
            ).to_numpy(dtype=bool)
        return _run_one_target(
            config=config,
            state=state,
            target=_TargetSpec(
                label=target_label,
                name=target_name,
                uid=target_uid,
                mask=target_mask,
            ),
        )

    resolved_n_jobs = resolve_n_jobs(config.execution[0])
    progress_total = len(targets) if hasattr(targets, "__len__") else None
    if resolved_n_jobs <= 1:
        for target in tqdm(
            targets,
            desc="Overall progress (authors)",
            unit="author",
            disable=not config.execution[2],
        ):
            yield _compute_target(target)
    else:
        with limit_blas_threads(), ThreadPoolExecutor(max_workers=resolved_n_jobs) as executor:
            for record in tqdm(
                executor.map(_compute_target, targets),
                total=progress_total,
                desc=f"Overall progress (authors, {resolved_n_jobs} jobs)",
                unit="author",
                disable=not config.execution[2],
            ):
                yield record


def run_top_authors_metrics_from_parquets(
    publications_path,
    references_path=None,
    **options: Unpack[_RunMetricsOptions],
) -> pd.DataFrame:
    """Eager wrapper around the canonical streaming runner."""

    return pd.DataFrame(
        list(
            iter_top_authors_metrics_from_parquets(
                publications_path,
                references_path,
                **options,
            )
        )
    )
