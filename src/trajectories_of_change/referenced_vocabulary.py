"""Referenced Vocabulary as a core document-feature KLD metric."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import inspect
from itertools import repeat
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .contract import _coerce_list, _coerce_list_column, build_target_mask
from .defaults import (
    DEFAULT_EPSILON,
    DEFAULT_LAMBDA_PARAM,
    DEFAULT_TOP_K_KLD_TERMS,
    DEFAULT_WINDOW_SIZE,
)
from .kld_core import DocumentFeatureMatrix, token_counts
from .metrics_kld import FeatureKLDBase
from .stats_utils import _level_slope

REFERENCE_POLICIES = ("inclusive", "external_only")
_REFERENCED_VOCAB_OPTION_KEYS = (
    "min_token_global_freq",
    "min_docs_global_freq",
    "min_tokens_target_slice",
    "min_tokens_field_slice",
    "min_docs_target_slice",
    "min_docs_field_slice",
    "min_docs_target_test",
    "min_docs_field_test",
    "max_vocab_size",
)


@dataclass(frozen=True)
class ReferencedVocabularyEvents:
    doc_indices: np.ndarray
    feature_ids: np.ndarray
    weights: np.ndarray
    diagnostics: pd.DataFrame
    label_for_feature_id: Callable[[int], str]
    token_labels: list[str]


@dataclass(frozen=True)
class ReferenceTokenCache:
    ref_lookup: dict[str, int]
    ref_tokens: list[Counter[str]]
    ref_language_flags: list[dict[str, bool]]
    ref_author_uids: list[set[str]]


@dataclass(frozen=True)
class _ReferencedVocabularyEventCache:
    publications: pd.DataFrame
    reference_cache: ReferenceTokenCache
    reference_lists: list[list[str]]
    bibcodes: list[Any]
    years: np.ndarray
    reference_token_weights: list[tuple[tuple[str, float], ...]]


def _language_flags(values: list[Any]) -> dict[str, bool]:
    title, title_en, abstract, abstract_en, title_lang, abstract_lang = (
        str(value or "").strip() for value in values
    )
    title_lang = title_lang.lower()
    abstract_lang = abstract_lang.lower()
    title_has_text = bool(title_en or title)
    abstract_has_text = bool(abstract_en or abstract)
    title_nonenglish = title_has_text and title_lang not in {"", "en", "unknown", "none", "nan"}
    abstract_nonenglish = abstract_has_text and abstract_lang not in {"", "en", "unknown", "none", "nan"}
    return {
        "title_nonenglish": bool(title_nonenglish),
        "abstract_nonenglish": bool(abstract_nonenglish),
        "title_untranslated": bool(title_nonenglish and title_en and title and title_en == title),
        "abstract_untranslated": bool(abstract_nonenglish and abstract_en and abstract and abstract_en == abstract),
    }


def build_reference_token_cache(references: pd.DataFrame) -> ReferenceTokenCache:
    if "tokens" not in references.columns:
        raise ValueError(
            "Referenced Vocabulary requires tokenized references: references.tokens is missing. "
            "Build a *_ref_tokens prepared bundle before running this metric."
        )
    ref_lookup: dict[str, int] = {}
    ref_tokens: list[Counter[str]] = []
    ref_language_flags: list[dict[str, bool]] = []
    ref_author_uids: list[set[str]] = []

    language_columns = ("Title", "Title_en", "Abstract", "Abstract_en", "Title_lang", "Abstract_lang")
    rows = zip(
        references["Bibcode"].to_numpy(copy=False),
        references["tokens"].to_numpy(copy=False),
        references["author_uids"].to_numpy(copy=False)
        if "author_uids" in references
        else repeat([]),
        *(
            references[column].to_numpy(copy=False) if column in references else repeat("")
            for column in language_columns
        ),
    )
    for ref_int, row in enumerate(rows):
        ref_id, tokens, author_uids, *language_values = row
        ref_id = str(ref_id)
        ref_lookup[ref_id] = int(ref_int)
        ref_tokens.append(token_counts(tokens))
        ref_language_flags.append(_language_flags(language_values))
        ref_author_uids.append(set(_coerce_list(author_uids, split_semicolon=True)))

    return ReferenceTokenCache(
        ref_lookup=ref_lookup,
        ref_tokens=ref_tokens,
        ref_language_flags=ref_language_flags,
        ref_author_uids=ref_author_uids,
    )


def _target_ref_mask_from_cache(cache: ReferenceTokenCache, *, target_author_uid: str) -> np.ndarray:
    target = str(target_author_uid)
    return np.asarray([target in author_uids for author_uids in cache.ref_author_uids], dtype=bool)


def _normalized_reference_token_weights(counter: Counter[str]) -> tuple[tuple[str, float], ...]:
    token_total = float(sum(counter.values()))
    if token_total <= 0:
        return tuple()
    return tuple((str(token), float(count) / token_total) for token, count in counter.items())


def _build_referenced_vocab_event_cache(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    reference_cache: ReferenceTokenCache | None = None,
) -> _ReferencedVocabularyEventCache:
    reset_publications = publications.reset_index(drop=True)
    if "References" in reset_publications.columns:
        reference_lists = [
            [str(reference_id) for reference_id in reference_ids]
            for reference_ids in _coerce_list_column(reset_publications["References"], split_semicolon=True)
        ]
    else:
        reference_lists = [[] for _ in range(len(reset_publications))]
    bibcodes = reset_publications["Bibcode"].tolist() if "Bibcode" in reset_publications.columns else [""] * len(reset_publications)
    years = (
        reset_publications["Year"].astype(int).to_numpy(copy=False)
        if "Year" in reset_publications.columns
        else np.zeros(len(reset_publications), dtype=int)
    )
    cache = reference_cache or build_reference_token_cache(references)
    return _ReferencedVocabularyEventCache(
        publications=reset_publications,
        reference_cache=cache,
        reference_lists=reference_lists,
        bibcodes=bibcodes,
        years=np.asarray(years, dtype=int),
        reference_token_weights=[_normalized_reference_token_weights(counter) for counter in cache.ref_tokens],
    )


def build_referenced_vocab_events(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    target_author_uid: str,
    policy: str,
    reference_cache: ReferenceTokenCache | None = None,
    collect_events: bool = True,
    _event_cache: _ReferencedVocabularyEventCache | None = None,
) -> ReferencedVocabularyEvents:
    if policy not in REFERENCE_POLICIES:
        raise ValueError(f"policy must be one of {REFERENCE_POLICIES}, got {policy!r}")

    event_cache = _event_cache or _build_referenced_vocab_event_cache(
        publications,
        references,
        reference_cache=reference_cache,
    )
    cache = event_cache.reference_cache
    target_ref_mask = _target_ref_mask_from_cache(cache, target_author_uid=target_author_uid)
    target_doc_mask = build_target_mask(
        event_cache.publications,
        target_name="",
        target_author_uid=target_author_uid,
        author_col="Author",
        author_ids_col="author_uids",
        allow_name_fallback=False,
    ).to_numpy(dtype=bool)
    token_lookup: dict[str, int] = {}
    token_labels: list[str] = []
    doc_indices: list[int] = []
    feature_ids: list[int] = []
    weights: list[float] = []
    diagnostics_rows: list[dict[str, Any]] = []

    for doc_idx, reference_ids in enumerate(event_cache.reference_lists):
        doc_ref_counters: list[tuple[tuple[tuple[str, float], ...], bool]] = []
        missing_reference_mentions = 0
        missing_text_reference_mentions = 0
        target_authored_reference_mentions = 0
        removed_target_authored_reference_mentions = 0
        title_nonenglish_mentions = 0
        abstract_nonenglish_mentions = 0
        title_untranslated_mentions = 0
        abstract_untranslated_mentions = 0

        for reference_id in reference_ids:
            ref_int = cache.ref_lookup.get(str(reference_id))
            if ref_int is None:
                missing_reference_mentions += 1
                continue
            is_target_ref = bool(target_ref_mask[ref_int])
            if is_target_ref:
                target_authored_reference_mentions += 1
            flags = cache.ref_language_flags[ref_int]
            title_nonenglish_mentions += int(flags["title_nonenglish"])
            abstract_nonenglish_mentions += int(flags["abstract_nonenglish"])
            title_untranslated_mentions += int(flags["title_untranslated"])
            abstract_untranslated_mentions += int(flags["abstract_untranslated"])
            if policy == "external_only" and is_target_ref:
                removed_target_authored_reference_mentions += 1
                continue
            token_weights = event_cache.reference_token_weights[ref_int]
            if not token_weights:
                missing_text_reference_mentions += 1
                continue
            doc_ref_counters.append((token_weights, is_target_ref))

        doc_counter: dict[str, float] = {}
        target_authored_kept_reference_mass = 0.0
        document_reference_vocab_mass = 0.0
        if doc_ref_counters:
            ref_weight = 1.0 / float(len(doc_ref_counters))
            for token_weights, is_target_ref in doc_ref_counters:
                if is_target_ref:
                    target_authored_kept_reference_mass += ref_weight
                document_reference_vocab_mass += ref_weight
                if collect_events:
                    for token, token_weight in token_weights:
                        doc_counter[token] = doc_counter.get(token, 0.0) + ref_weight * token_weight

        if collect_events:
            for token, weight in doc_counter.items():
                token_id = token_lookup.get(token)
                if token_id is None:
                    token_id = len(token_labels)
                    token_lookup[token] = token_id
                    token_labels.append(token)
                doc_indices.append(int(doc_idx))
                feature_ids.append(int(token_id))
                weights.append(float(weight))

        diagnostics_rows.append(
            {
                "Bibcode": event_cache.bibcodes[int(doc_idx)],
                "slice": int(event_cache.years[int(doc_idx)]),
                "is_target_document": bool(target_doc_mask[int(doc_idx)]) if int(doc_idx) < len(target_doc_mask) else False,
                "reference_policy": policy,
                "reference_mentions": int(len(reference_ids)),
                "known_reference_mentions": int(len(reference_ids) - missing_reference_mentions),
                "kept_reference_mentions": int(len(doc_ref_counters)),
                "missing_reference_mentions": int(missing_reference_mentions),
                "missing_text_reference_mentions": int(missing_text_reference_mentions),
                "target_authored_reference_mentions": int(target_authored_reference_mentions),
                "removed_target_authored_reference_mentions": int(removed_target_authored_reference_mentions),
                "target_authored_kept_reference_mass": float(target_authored_kept_reference_mass),
                "title_nonenglish_reference_mentions": int(title_nonenglish_mentions),
                "abstract_nonenglish_reference_mentions": int(abstract_nonenglish_mentions),
                "title_untranslated_reference_mentions": int(title_untranslated_mentions),
                "abstract_untranslated_reference_mentions": int(abstract_untranslated_mentions),
                "document_reference_vocab_mass": float(document_reference_vocab_mass),
                "documents_without_referenced_vocab_after_filters": int(document_reference_vocab_mass <= 0),
            }
        )

    def label_for_feature_id(feature_id: int) -> str:
        return token_labels[int(feature_id)]

    return ReferencedVocabularyEvents(
        doc_indices=np.asarray(doc_indices, dtype=np.int32),
        feature_ids=np.asarray(feature_ids, dtype=np.int64),
        weights=np.asarray(weights, dtype=float),
        diagnostics=pd.DataFrame(diagnostics_rows),
        label_for_feature_id=label_for_feature_id,
        token_labels=token_labels,
    )


def build_referenced_vocab_diagnostics(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    target_author_uid: str,
    policy: str,
    reference_cache: ReferenceTokenCache | None = None,
    _event_cache: _ReferencedVocabularyEventCache | None = None,
) -> pd.DataFrame:
    return build_referenced_vocab_events(
        publications,
        references,
        target_author_uid=target_author_uid,
        policy=policy,
        reference_cache=reference_cache,
        collect_events=False,
        _event_cache=_event_cache,
    ).diagnostics


def referenced_vocab_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve caller overrides from the metric's canonical defaults."""
    parameters = inspect.signature(ReferencedVocabularyKLD).parameters
    out = {key: parameters[key].default for key in _REFERENCED_VOCAB_OPTION_KEYS}
    if kwargs:
        out.update(kwargs)
    return out


def summarize_referenced_vocab_sync(sync: pd.DataFrame) -> dict[str, float | int]:
    if sync.empty:
        return {
            "ref_vocab_kld_all_level": np.nan,
            "ref_vocab_kld_all_slope": np.nan,
            "ref_vocab_sync_slices": 0,
        }
    level, slope = _level_slope(sync["slice"].astype(float).to_numpy(), sync["kld_all"].astype(float).to_numpy())
    return {
        "ref_vocab_kld_all_level": level,
        "ref_vocab_kld_all_slope": slope,
        "ref_vocab_sync_slices": int(len(sync)),
    }


def summarize_referenced_vocab_async_matrix(matrix: pd.DataFrame) -> dict[str, float | int]:
    if matrix.empty:
        return {
            "ref_vocab_kld_async_min": np.nan,
            "ref_vocab_kld_async_leadlag": np.nan,
            "ref_vocab_async_target_slices": 0,
            "ref_vocab_async_pairs": 0,
            "ref_vocab_async_edge_minima": 0,
            "ref_vocab_async_edge_minima_share": np.nan,
        }
    expected = matrix["field_slice"].astype(float) - matrix["target_slice"].astype(float)
    if not np.allclose(matrix["time_diff"].astype(float), expected):
        raise ValueError("Referenced Vocabulary async matrix has invalid time_diff")
    minima = matrix.loc[matrix.groupby("target_slice")["kld"].idxmin()].copy()
    field_min = float(matrix["field_slice"].min())
    field_max = float(matrix["field_slice"].max())
    edge_mask = minima["field_slice"].astype(float).isin([field_min, field_max])
    return {
        "ref_vocab_kld_async_min": float(minima["kld"].mean()) if not minima.empty else np.nan,
        "ref_vocab_kld_async_leadlag": float(minima["time_diff"].mean()) if not minima.empty else np.nan,
        "ref_vocab_async_target_slices": int(minima["target_slice"].nunique()),
        "ref_vocab_async_pairs": int(len(matrix)),
        "ref_vocab_async_edge_minima": int(edge_mask.sum()),
        "ref_vocab_async_edge_minima_share": float(edge_mask.mean()) if len(minima) else np.nan,
    }


def summarize_referenced_vocab_diagnostics(diagnostics: pd.DataFrame) -> dict[str, float | int]:
    if diagnostics.empty:
        return {}
    sum_columns = [
        "reference_mentions",
        "known_reference_mentions",
        "kept_reference_mentions",
        "missing_reference_mentions",
        "missing_text_reference_mentions",
        "target_authored_reference_mentions",
        "removed_target_authored_reference_mentions",
        "target_authored_kept_reference_mass",
        "title_nonenglish_reference_mentions",
        "abstract_nonenglish_reference_mentions",
        "title_untranslated_reference_mentions",
        "abstract_untranslated_reference_mentions",
        "document_reference_vocab_mass",
        "documents_without_referenced_vocab_after_filters",
    ]

    def summarize_frame(frame: pd.DataFrame, prefix: str = "") -> dict[str, float | int]:
        sums = {
            f"{prefix}{column}": float(frame[column].sum())
            for column in sum_columns
            if column in frame.columns
        }
        reference_mentions = sums.get(f"{prefix}reference_mentions", 0.0)
        known_mentions = sums.get(f"{prefix}known_reference_mentions", 0.0)
        kept_mentions = sums.get(f"{prefix}kept_reference_mentions", 0.0)
        sums[f"{prefix}reference_coverage_ratio"] = known_mentions / reference_mentions if reference_mentions else np.nan
        sums[f"{prefix}token_text_coverage_ratio"] = kept_mentions / known_mentions if known_mentions else np.nan
        sums[f"{prefix}target_authored_reference_share"] = (
            sums.get(f"{prefix}target_authored_reference_mentions", 0.0) / reference_mentions
            if reference_mentions
            else np.nan
        )
        sums[f"{prefix}nonenglish_reference_share"] = (
            max(
                sums.get(f"{prefix}title_nonenglish_reference_mentions", 0.0),
                sums.get(f"{prefix}abstract_nonenglish_reference_mentions", 0.0),
            )
            / reference_mentions
            if reference_mentions
            else np.nan
        )
        sums[f"{prefix}untranslated_reference_share"] = (
            max(
                sums.get(f"{prefix}title_untranslated_reference_mentions", 0.0),
                sums.get(f"{prefix}abstract_untranslated_reference_mentions", 0.0),
            )
            / reference_mentions
            if reference_mentions
            else np.nan
        )
        return sums

    out = summarize_frame(diagnostics)
    if "is_target_document" in diagnostics.columns:
        target_side = diagnostics[diagnostics["is_target_document"].astype(bool)]
        field_side = diagnostics[~diagnostics["is_target_document"].astype(bool)]
        out.update(summarize_frame(target_side, prefix="target_side_"))
        out.update(summarize_frame(field_side, prefix="field_side_"))
    return out


class ReferencedVocabularyKLD(FeatureKLDBase):
    """KLD analysis for the vocabulary of literature cited by publications."""

    METRIC = "ref_vocab"

    def __init__(
        self,
        publications: pd.DataFrame,
        references: pd.DataFrame,
        *,
        target_author_uid: str,
        policy: str = "inclusive",
        reference_cache: ReferenceTokenCache | None = None,
        prebuilt_matrix: DocumentFeatureMatrix | None = None,
        build_diagnostics: bool = True,
        target_mask: Optional[np.ndarray] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices: bool = True,
        lambda_param: float = DEFAULT_LAMBDA_PARAM,
        epsilon: float = DEFAULT_EPSILON,
        min_token_global_freq: float = 0.0,
        min_docs_global_freq: int = 2,
        max_vocab_size: Optional[int] = 50_000,
        min_tokens_target_slice: float = 1e-12,
        min_tokens_field_slice: float = 1e-12,
        min_docs_target_slice: int = 1,
        min_docs_field_slice: int = 1,
        min_docs_target_test: int = 2,
        min_docs_field_test: int = 2,
        top_k_kld_terms: Optional[int] = DEFAULT_TOP_K_KLD_TERMS,
        precompute_slice_moments: bool | str = False,
        show_progress: bool = False,
        verbose: bool = False,
        _event_cache: _ReferencedVocabularyEventCache | None = None,
    ) -> None:
        if policy not in REFERENCE_POLICIES:
            raise ValueError(f"policy must be one of {REFERENCE_POLICIES}, got {policy!r}")
        self.publications = publications.copy(deep=False)
        self.references = references.copy(deep=False)
        self.target_author_uid = str(target_author_uid)
        self.policy = policy
        self.start_year = start_year
        self.end_year = end_year
        self.window_size = int(window_size)
        self.skip_incomplete_slices = bool(skip_incomplete_slices)
        self.min_token_global_freq = float(min_token_global_freq)
        self.min_docs_global_freq = int(min_docs_global_freq)
        self.max_vocab_size = max_vocab_size
        self.reference_cache = reference_cache or build_reference_token_cache(self.references)

        if prebuilt_matrix is None:
            self.events = build_referenced_vocab_events(
                self.publications,
                self.references,
                target_author_uid=self.target_author_uid,
                policy=self.policy,
                reference_cache=self.reference_cache,
                _event_cache=_event_cache,
            )
            matrix = DocumentFeatureMatrix.from_weighted_events(
                years=self.publications["Year"].astype(int).to_numpy(copy=False),
                doc_indices=self.events.doc_indices,
                feature_ids=self.events.feature_ids,
                weights=self.events.weights,
                label_for_feature_id=self.events.label_for_feature_id,
                start_year=start_year,
                end_year=end_year,
                window_size=window_size,
                skip_incomplete_slices=skip_incomplete_slices,
                min_token_global_freq=float(min_token_global_freq),
                min_docs_global_freq=int(min_docs_global_freq),
                max_vocab_size=max_vocab_size,
                precompute_slice_moments=precompute_slice_moments,
            )
        else:
            matrix = prebuilt_matrix
            self.events = None
        # Diagnostics are target-dependent, so they are built per target and never shared
        # from a probe target. The consolidated multimetric path does not consume them and
        # passes build_diagnostics=False to skip the work entirely.
        if not build_diagnostics:
            self.diagnostics = None
        elif self.events is not None:
            self.diagnostics = self.events.diagnostics
        else:
            self.diagnostics = build_referenced_vocab_diagnostics(
                self.publications,
                self.references,
                target_author_uid=self.target_author_uid,
                policy=self.policy,
                reference_cache=self.reference_cache,
                _event_cache=_event_cache,
            )
        if target_mask is None:
            target_mask = build_target_mask(
                self.publications,
                target_name="",
                target_author_uid=self.target_author_uid,
                author_col="Author",
                author_ids_col="author_uids",
                allow_name_fallback=False,
            ).to_numpy(dtype=bool)
        else:
            target_mask = np.asarray(target_mask, dtype=bool)
        super().__init__(
            matrix=matrix,
            target_mask=target_mask,
            mode="ref_vocab",
            target_label=self.target_author_uid,
            lambda_param=lambda_param,
            epsilon=epsilon,
            min_tokens_target_slice=min_tokens_target_slice,
            min_tokens_field_slice=min_tokens_field_slice,
            min_docs_target_slice=min_docs_target_slice,
            min_docs_field_slice=min_docs_field_slice,
            min_docs_target_test=min_docs_target_test,
            min_docs_field_test=min_docs_field_test,
            top_k_kld_terms=top_k_kld_terms,
            show_progress=show_progress,
            verbose=verbose,
        )
        self.metadata = {
            "support_size": int(self.matrix.vocab_size),
            "slices_total": int(len(self.matrix.slices)),
            "slice_token_counts": {int(label): dict(values) for label, values in self.core.slice_token_counts.items()},
        }

    def perform_welch_tests_all_pairs(self, sync_only: bool = False) -> pd.DataFrame:
        welch = super().perform_welch_tests_all_pairs(sync_only=sync_only)
        self.metadata["welch_target_doc_counts"] = dict(self.core.doc_counts_target)
        self.metadata["welch_field_doc_counts"] = dict(self.core.doc_counts_field)
        return welch
