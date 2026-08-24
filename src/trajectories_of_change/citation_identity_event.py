"""Compact Citation Identity event index for the production KLD path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .citation_identity import CitationIdentityConfig, _diagnostics_by_slice, _reference_entities
from .contract import (
    DatasetValidationError,
    _coerce_list_column,
    _ensure_citation_identity_columns,
    build_target_mask,
    normalize_publications_frame,
    normalize_references_frame,
)
from .kld_core import DocumentFeatureMatrix, _sorted_membership_positions
from .metric_result import MetricResult
from .metrics_kld import FeatureKLDBase
from .defaults import DEFAULT_EPSILON, DEFAULT_LAMBDA_PARAM

# Legacy name kept for back-compat; the canonical result type is MetricResult.
CitationIdentitySyncKLDResult = MetricResult


@dataclass(frozen=True)
class _EventRecord:
    pair_keys: np.ndarray
    left_refs: np.ndarray
    right_refs: np.ndarray
    reference_mentions: int
    analyzable_reference_mentions: int
    empty_reference_entities: int
    candidate_pair_mass: float
    self_loop_pair_mass: float


def _pair_arrays(
    cited_groups: list[tuple[int, tuple[int, ...]]],
    *,
    single_entity: bool,
    remove_self_loops: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build co-reference pairs in the canonical document order."""
    group_count = len(cited_groups)
    if group_count < 2:
        return (
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            0.0,
            0.0,
        )

    if not single_entity:
        pair_keys: list[int] = []
        left_refs: list[int] = []
        right_refs: list[int] = []
        candidate_pair_mass = 0.0
        self_loop_pair_mass = 0.0
        for left_pos in range(group_count - 1):
            left_ref, left_entities = cited_groups[left_pos]
            for right_pos in range(left_pos + 1, group_count):
                right_ref, right_entities = cited_groups[right_pos]
                for left_entity in left_entities:
                    for right_entity in right_entities:
                        candidate_pair_mass += 1.0
                        if left_entity == right_entity:
                            self_loop_pair_mass += 1.0
                            if remove_self_loops:
                                continue
                        low, high = sorted((int(left_entity), int(right_entity)))
                        pair_keys.append((int(low) << 32) | int(high))
                        left_refs.append(int(left_ref))
                        right_refs.append(int(right_ref))
        return (
            np.asarray(pair_keys, dtype=np.uint64),
            np.asarray(left_refs, dtype=np.int32),
            np.asarray(right_refs, dtype=np.int32),
            candidate_pair_mass,
            self_loop_pair_mass,
        )

    reference_ids = np.fromiter(
        (reference_id for reference_id, _ in cited_groups),
        dtype=np.int32,
        count=group_count,
    )
    entity_ids = np.fromiter(
        (entities[0] for _, entities in cited_groups),
        dtype=np.uint64,
        count=group_count,
    )
    left_positions, right_positions = np.triu_indices(group_count, k=1)
    left_entities = entity_ids[left_positions]
    right_entities = entity_ids[right_positions]
    self_loops = left_entities == right_entities
    candidate_pair_mass = float(left_positions.size)
    self_loop_pair_mass = float(np.count_nonzero(self_loops))

    if remove_self_loops and self_loop_pair_mass:
        keep = ~self_loops
        left_positions = left_positions[keep]
        right_positions = right_positions[keep]
        left_entities = left_entities[keep]
        right_entities = right_entities[keep]

    low = np.minimum(left_entities, right_entities)
    high = np.maximum(left_entities, right_entities)
    pair_keys = (low << np.uint64(32)) | high
    return (
        pair_keys.astype(np.uint64, copy=False),
        reference_ids[left_positions],
        reference_ids[right_positions],
        candidate_pair_mass,
        self_loop_pair_mass,
    )


class CitationIdentityEventIndex:
    """Compact integer-pair representation of Citation Identity events."""

    def __init__(
        self,
        publications: pd.DataFrame,
        references: pd.DataFrame,
        records: list[_EventRecord],
        *,
        ref_lookup: dict[str, int],
        entity_labels: list[str],
        config: CitationIdentityConfig,
    ) -> None:
        self.publications = publications
        self.references = references
        self.records = records
        # Target-independent per-document record scalars: materialized once here and
        # reused across all targets (each CitationIdentityKLD reads these instead of
        # rebuilding them per target).
        n_docs = len(records)
        self.rec_pair_size = np.fromiter((r.pair_keys.size for r in records), dtype=np.int64, count=n_docs)
        self.rec_ref_mentions = np.fromiter((r.reference_mentions for r in records), dtype=np.int64, count=n_docs)
        self.rec_analyzable = np.fromiter((r.analyzable_reference_mentions for r in records), dtype=np.int64, count=n_docs)
        self.rec_empty_entities = np.fromiter((r.empty_reference_entities for r in records), dtype=np.int64, count=n_docs)
        self.rec_candidate_mass = np.fromiter((r.candidate_pair_mass for r in records), dtype=np.float64, count=n_docs)
        self.rec_self_loop_mass = np.fromiter((r.self_loop_pair_mass for r in records), dtype=np.float64, count=n_docs)
        self.ref_lookup = ref_lookup
        self.entity_labels = entity_labels
        self.config = config

    @classmethod
    def from_frames(
        cls,
        publications: pd.DataFrame,
        references: pd.DataFrame,
        *,
        config: CitationIdentityConfig,
    ) -> "CitationIdentityEventIndex":
        return cls._from_normalized_frames(
            normalize_publications_frame(publications),
            normalize_references_frame(references),
            config=config,
        )

    @classmethod
    def _from_normalized_frames(
        cls,
        publications: pd.DataFrame,
        references: pd.DataFrame,
        *,
        config: CitationIdentityConfig,
    ) -> "CitationIdentityEventIndex":
        """Build from frames that already passed the package contract."""
        _ensure_citation_identity_columns(publications, references=references, mode=config.mode)

        ref_lookup: dict[str, int] = {}
        ref_entities: dict[str, tuple[int, ...]] = {}
        entity_lookup: dict[str, int] = {}
        entity_labels: list[str] = []

        for ref_idx, ref_row in enumerate(references.to_dict(orient="records")):
            ref_id = str(ref_row["Bibcode"])
            ref_lookup[ref_id] = int(ref_idx)
            entities = _reference_entities(ref_row, config=config)
            if not entities:
                ref_entities[ref_id] = tuple()
                continue
            entity_ids: list[int] = []
            for entity in entities:
                entity_label = str(entity)
                entity_id = entity_lookup.get(entity_label)
                if entity_id is None:
                    entity_id = len(entity_labels)
                    entity_lookup[entity_label] = entity_id
                    entity_labels.append(entity_label)
                entity_ids.append(int(entity_id))
            ref_entities[ref_id] = tuple(entity_ids)

        records: list[_EventRecord] = []
        reference_lists = (
            _coerce_list_column(publications["References"], split_semicolon=True)
            if "References" in publications.columns
            else [[] for _ in range(len(publications))]
        )
        for reference_ids in reference_lists:
            if not isinstance(reference_ids, list):
                raise DatasetValidationError("publications.References must be list-like after normalization")

            cited_groups: list[tuple[int, tuple[int, ...]]] = []
            missing_refs: list[str] = []
            empty_entities = 0
            for reference_id in reference_ids:
                reference_id = str(reference_id)
                ref_int = ref_lookup.get(reference_id)
                if ref_int is None:
                    missing_refs.append(reference_id)
                    continue
                entity_ids = ref_entities.get(reference_id, tuple())
                if not entity_ids:
                    empty_entities += 1
                    continue
                cited_groups.append((int(ref_int), entity_ids))

            if missing_refs:
                preview = missing_refs[:5]
                raise DatasetValidationError(
                    f"co-citation build encountered missing reference rows, e.g. {preview}"
                )

            (
                pair_keys_array,
                left_refs_array,
                right_refs_array,
                candidate_pair_mass,
                self_loop_pair_mass,
            ) = _pair_arrays(
                cited_groups,
                single_entity=config.mode == "works" or config.author_scope == "first_author",
                remove_self_loops=config.remove_self_loops,
            )

            records.append(
                _EventRecord(
                    pair_keys=pair_keys_array,
                    left_refs=left_refs_array,
                    right_refs=right_refs_array,
                    reference_mentions=int(len(reference_ids)),
                    analyzable_reference_mentions=int(len(cited_groups)),
                    empty_reference_entities=int(empty_entities),
                    candidate_pair_mass=float(candidate_pair_mass),
                    self_loop_pair_mass=float(self_loop_pair_mass),
                )
            )

        return cls(
            publications,
            references,
            records,
            ref_lookup=ref_lookup,
            entity_labels=entity_labels,
            config=config,
        )


def _term_from_pair_key(pair_key: np.uint64, entity_labels: list[str]) -> str:
    key = int(pair_key)
    left = key >> 32
    right = key & 0xFFFFFFFF
    return f"{entity_labels[left]} | {entity_labels[right]}"


def _target_reference_mask(
    index: CitationIdentityEventIndex,
    *,
    target_name: str,
    target_author_uid: Optional[str],
) -> np.ndarray:
    try:
        reference_hits_target = build_target_mask(
            index.references,
            target_name=target_name,
            target_author_uid=target_author_uid,
            allow_name_fallback=target_author_uid is None,
        )
    except ValueError:
        reference_hits_target = pd.Series(False, index=index.references.index)
    out = np.zeros(len(index.references), dtype=bool)
    for ref_id in index.references.loc[reference_hits_target, "Bibcode"].astype(str):
        ref_int = index.ref_lookup.get(str(ref_id))
        if ref_int is not None:
            out[int(ref_int)] = True
    return out


@dataclass(frozen=True)
class _CitationIdentityTargetEventCache:
    mode: str
    author_scope: str
    target_exclusion: str
    remove_self_loops: bool
    target_name: str
    target_author_uid: Optional[str]
    target_mask: np.ndarray
    allowed_pair_keys: Optional[np.ndarray]
    u_doc: np.ndarray
    u_key: np.ndarray
    u_counts: np.ndarray
    per_doc_unique: np.ndarray
    target_excluded_per_doc: np.ndarray
    kept_count_per_doc: np.ndarray


def _normalize_allowed_pair_keys(allowed_pair_keys: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if allowed_pair_keys is None:
        return None
    return np.unique(np.asarray(allowed_pair_keys, dtype=np.uint64))


def _validate_index_config(index: CitationIdentityEventIndex, config: CitationIdentityConfig) -> None:
    if config.mode != index.config.mode or config.author_scope != index.config.author_scope:
        raise ValueError("event index config differs from requested config")
    if config.remove_self_loops != index.config.remove_self_loops:
        raise ValueError("event index self-loop policy differs from requested config")


def _build_citation_identity_target_event_cache(
    index: CitationIdentityEventIndex,
    *,
    config: CitationIdentityConfig,
    target_name: str = "",
    target_author_uid: Optional[str] = None,
    target_mask: np.ndarray,
    allowed_pair_keys: Optional[np.ndarray] = None,
) -> _CitationIdentityTargetEventCache:
    """Build counting-independent target event arrays for Citation Identity."""

    _validate_index_config(index, config)
    if config.target_exclusion != "none" and not (target_author_uid or str(target_name).strip()):
        raise ValueError("target_name or target_author_uid is required")

    target_mask_array = np.asarray(target_mask, dtype=bool)
    if target_mask_array.ndim != 1 or target_mask_array.size != len(index.records):
        raise ValueError("target_mask must be 1-D with length matching Citation Identity index publications")

    allowed_keys = _normalize_allowed_pair_keys(allowed_pair_keys)
    target_ref_mask = (
        _target_reference_mask(
            index,
            target_name=target_name,
            target_author_uid=target_author_uid,
        )
        if config.target_exclusion != "none"
        else np.zeros(len(index.references), dtype=bool)
    )

    n_docs = len(index.records)
    rec_pair_size = index.rec_pair_size
    target_excluded_per_doc = np.zeros(n_docs, dtype=np.int64)
    kept_count_per_doc = np.zeros(n_docs, dtype=np.int64)
    per_doc_unique = np.zeros(n_docs, dtype=np.int64)
    u_doc_parts: list[np.ndarray] = []
    u_key_parts: list[np.ndarray] = []
    u_count_parts: list[np.ndarray] = []

    for doc_idx, record in enumerate(index.records):
        pair_size = int(record.pair_keys.size)
        if pair_size == 0:
            continue
        if config.target_exclusion == "none":
            keep_count_before_allowed = pair_size
            doc_keys = record.pair_keys
        elif config.target_exclusion == "target_docs_only" and not bool(target_mask_array[doc_idx]):
            keep_count_before_allowed = pair_size
            doc_keys = record.pair_keys
        else:
            pair_hits_target = target_ref_mask[record.left_refs] | target_ref_mask[record.right_refs]
            keep = ~pair_hits_target
            keep_count_before_allowed = int(keep.sum())
            doc_keys = record.pair_keys[keep]
        target_excluded_per_doc[doc_idx] = int(rec_pair_size[doc_idx]) - int(keep_count_before_allowed)

        if allowed_keys is not None and doc_keys.size:
            _, valid = _sorted_membership_positions(allowed_keys, doc_keys)
            doc_keys = doc_keys[valid]

        kept_count_per_doc[doc_idx] = int(doc_keys.size)
        if not doc_keys.size:
            continue
        unique_keys, counts = np.unique(doc_keys, return_counts=True)
        per_doc_unique[doc_idx] = int(unique_keys.size)
        u_doc_parts.append(np.full(unique_keys.size, doc_idx, dtype=np.int32))
        u_key_parts.append(unique_keys.astype(np.uint64, copy=False))
        u_count_parts.append(counts.astype(np.int64, copy=False))

    if u_doc_parts:
        u_doc = np.concatenate(u_doc_parts)
        u_key = np.concatenate(u_key_parts)
        u_counts = np.concatenate(u_count_parts)
    else:
        u_doc = np.array([], dtype=np.int32)
        u_key = np.array([], dtype=np.uint64)
        u_counts = np.array([], dtype=np.int64)
    return _CitationIdentityTargetEventCache(
        mode=config.mode,
        author_scope=config.author_scope,
        target_exclusion=config.target_exclusion,
        remove_self_loops=bool(config.remove_self_loops),
        target_name=str(target_name),
        target_author_uid=str(target_author_uid) if target_author_uid is not None else None,
        target_mask=target_mask_array,
        allowed_pair_keys=allowed_keys,
        u_doc=u_doc,
        u_key=u_key,
        u_counts=u_counts,
        per_doc_unique=per_doc_unique,
        target_excluded_per_doc=target_excluded_per_doc,
        kept_count_per_doc=kept_count_per_doc,
    )


def _validate_target_event_cache(
    cache: _CitationIdentityTargetEventCache,
    *,
    index: CitationIdentityEventIndex,
    config: CitationIdentityConfig,
    target_name: str,
    target_author_uid: Optional[str],
    target_mask: np.ndarray,
    allowed_pair_keys: Optional[np.ndarray],
) -> None:
    if cache.mode != config.mode or cache.author_scope != config.author_scope:
        raise ValueError("target event cache config differs from requested config")
    if cache.target_exclusion != config.target_exclusion:
        raise ValueError("target event cache target-exclusion policy differs from requested config")
    if cache.remove_self_loops != bool(config.remove_self_loops):
        raise ValueError("target event cache self-loop policy differs from requested config")
    if cache.target_name != str(target_name):
        raise ValueError("target event cache target_name differs from requested target")
    requested_uid = str(target_author_uid) if target_author_uid is not None else None
    if cache.target_author_uid != requested_uid:
        raise ValueError("target event cache target_author_uid differs from requested target")
    target_mask_array = np.asarray(target_mask, dtype=bool)
    if cache.target_mask.size != len(index.records) or not np.array_equal(cache.target_mask, target_mask_array):
        raise ValueError("target event cache target_mask differs from requested target")
    allowed_keys = _normalize_allowed_pair_keys(allowed_pair_keys)
    if allowed_keys is None:
        if cache.allowed_pair_keys is not None:
            raise ValueError("target event cache allowed_pair_keys differs from requested keys")
    elif cache.allowed_pair_keys is None or not np.array_equal(cache.allowed_pair_keys, allowed_keys):
        raise ValueError("target event cache allowed_pair_keys differs from requested keys")


class CitationIdentityKLD(FeatureKLDBase):
    """Citation Identity KLD as a thin :class:`FeatureKLDBase` adapter.

    Like ``VocabularyKLD`` and ``ReferencedVocabularyKLD``, this builds its own
    document-feature matrix (the Citation Identity event stream) plus diagnostics,
    then inherits the shared sync/async/Welch engine from ``FeatureKLDBase``. The
    core is *identical by construction* to the former hand-rolled path: the same
    ``DocumentFeatureMatrix.from_weighted_events`` call and the same ``KLDCore``
    kwargs, so the sync/async/Welch frames are byte-identical.

    ``welch_*_doc_counts`` in ``metadata`` are populated only after Welch runs
    (empty ``{}`` otherwise), via the ``perform_welch_tests_all_pairs`` override —
    matching the standalone behaviour when ``run_welch=False``.
    """

    METRIC = "cocit"

    def __init__(
        self,
        index: CitationIdentityEventIndex,
        *,
        config: CitationIdentityConfig,
        target_name: str = "",
        target_author_uid: Optional[str] = None,
        target_mask: np.ndarray,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        window_size: int = 2,
        skip_incomplete_slices: bool = True,
        min_token_global_freq: float = 1.0,
        min_docs_global_freq: int = 1,
        min_tokens_target_slice: float = 1e-12,
        min_tokens_field_slice: float = 1e-12,
        min_docs_target_slice: int = 1,
        min_docs_field_slice: int = 1,
        min_docs_target_test: int = 2,
        min_docs_field_test: int = 2,
        max_vocab_size: Optional[int] = None,
        top_k_kld_terms: Optional[int] = 50,
        lambda_param: float = DEFAULT_LAMBDA_PARAM,
        epsilon: float = DEFAULT_EPSILON,
        precompute_slice_moments: bool | str = False,
        allowed_pair_keys: Optional[np.ndarray] = None,
        _target_event_cache: Optional[_CitationIdentityTargetEventCache] = None,
        show_progress: bool = False,
        verbose: bool = False,
    ) -> None:
        _validate_index_config(index, config)
        if config.target_exclusion != "none" and not (target_author_uid or str(target_name).strip()):
            raise ValueError("target_name or target_author_uid is required")

        years = index.publications["Year"].astype(int).to_numpy(copy=False)
        bibcodes = index.publications["Bibcode"].to_numpy()
        target_mask_array = np.asarray(target_mask, dtype=bool)
        if target_mask_array.ndim != 1 or target_mask_array.size != len(index.records):
            raise ValueError("target_mask must be 1-D with length matching Citation Identity index publications")

        if _target_event_cache is None:
            target_event_cache = _build_citation_identity_target_event_cache(
                index,
                config=config,
                target_name=target_name,
                target_author_uid=target_author_uid,
                target_mask=target_mask_array,
                allowed_pair_keys=allowed_pair_keys,
            )
        else:
            _validate_target_event_cache(
                _target_event_cache,
                index=index,
                config=config,
                target_name=target_name,
                target_author_uid=target_author_uid,
                target_mask=target_mask_array,
                allowed_pair_keys=allowed_pair_keys,
            )
            target_event_cache = _target_event_cache

        records = index.records
        n_docs = len(records)
        rec_pair_size = index.rec_pair_size
        rec_ref_mentions = index.rec_ref_mentions
        rec_analyzable = index.rec_analyzable
        rec_empty_entities = index.rec_empty_entities
        rec_candidate_mass = index.rec_candidate_mass
        rec_self_loop_mass = index.rec_self_loop_mass

        u_doc = target_event_cache.u_doc
        u_key = target_event_cache.u_key
        u_counts = target_event_cache.u_counts
        per_doc_unique = target_event_cache.per_doc_unique
        target_excluded_per_doc = target_event_cache.target_excluded_per_doc
        kept_count_per_doc = target_event_cache.kept_count_per_doc
        if config.counting == "multiplicity":
            weights = u_counts.astype(float)
        elif config.counting == "binary":
            weights = np.ones(u_key.size, dtype=float)
        else:  # document_fractional: 1 / (unique pairs in that document)
            weights = (1.0 / per_doc_unique[u_doc].astype(float)) if u_doc.size else np.array([], dtype=float)
        kept_weighted_per_doc = (
            np.bincount(u_doc, weights=weights, minlength=n_docs) if u_doc.size else np.zeros(n_docs, dtype=float)
        )

        all_keys = u_key
        all_weights = weights
        all_doc_idx = u_doc

        reference_mentions = int(rec_ref_mentions.sum())
        analyzable_reference_mentions = int(rec_analyzable.sum())
        empty_reference_entities = int(rec_empty_entities.sum())
        candidate_pair_mass = float(rec_candidate_mass.sum())
        self_loop_pair_mass = float(rec_self_loop_mass.sum())
        target_excluded_pair_mass = float(target_excluded_per_doc.sum())
        kept_pair_mass_raw = float(kept_count_per_doc.sum())
        kept_pair_mass_weighted = float(weights.sum())
        support_size_total = int(u_key.size)
        without_pairs = ((rec_pair_size == 0) | (kept_count_per_doc == 0)).astype(np.int64)
        empty_documents = int(without_pairs.sum())

        matrix = DocumentFeatureMatrix.from_weighted_events(
            years=years,
            doc_indices=all_doc_idx,
            feature_ids=all_keys,
            weights=all_weights,
            label_for_feature_id=lambda feature_id: _term_from_pair_key(feature_id, index.entity_labels),
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
            min_token_global_freq=min_token_global_freq,
            min_docs_global_freq=min_docs_global_freq,
            max_vocab_size=max_vocab_size,
            precompute_slice_moments=precompute_slice_moments,
        )

        # Inherit the shared KLD engine: FeatureKLDBase builds KLDCore with exactly the
        # same kwargs the standalone used, so sync/async/Welch are identical by construction.
        super().__init__(
            matrix=matrix,
            target_mask=target_mask_array,
            mode="cocit",
            target_label=str(target_author_uid or target_name),
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

        slice_token_counts: dict[int, dict[str, float | int]] = {
            int(label): dict(values) for label, values in self.core.slice_token_counts.items()
        }
        field_entropies: dict[int, float] = {}
        for label, probs in self.core.slice_models_field.items():
            positive = probs[probs > 0]
            if positive.size:
                field_entropies[int(label)] = float(-np.sum(positive * np.log2(positive)))

        diagnostics_summary = {
            "citation_identity_counting": config.counting,
            "citation_author_scope": config.author_scope,
            "target_exclusion": config.target_exclusion,
            "citation_identity_remove_self_loops": bool(config.remove_self_loops),
            "reference_mentions": int(reference_mentions),
            "analyzable_reference_mentions": int(analyzable_reference_mentions),
            "empty_reference_entities": int(empty_reference_entities),
            "candidate_pair_mass": float(candidate_pair_mass),
            "self_loop_pair_mass": float(self_loop_pair_mass),
            "target_excluded_pair_mass": float(target_excluded_pair_mass),
            "kept_pair_mass_raw": float(kept_pair_mass_raw),
            "kept_pair_mass_weighted": float(kept_pair_mass_weighted),
            "support_size_before_filters": int(support_size_total),
            "support_size_after_filters": int(support_size_total),
            "documents_without_pairs_after_filters": int(empty_documents),
        }
        diagnostics_documents = pd.DataFrame(
            {
                "Bibcode": bibcodes,
                "slice": years.astype(np.int64),
                "is_target_document": target_mask_array.astype(bool),
                "reference_mentions": rec_ref_mentions,
                "analyzable_reference_mentions": rec_analyzable,
                "empty_reference_entities": rec_empty_entities,
                "candidate_pair_mass": rec_candidate_mass,
                "self_loop_pair_mass": rec_self_loop_mass,
                "target_excluded_pair_mass": target_excluded_per_doc.astype(np.float64),
                "kept_pair_mass_raw": kept_count_per_doc.astype(np.float64),
                "kept_pair_mass_weighted": kept_weighted_per_doc,
                "support_size_before_filters": per_doc_unique.astype(np.int64),
                "support_size_after_filters": per_doc_unique.astype(np.int64),
                "documents_without_pairs_after_filters": without_pairs,
            }
        )
        diagnostics_by_slice = _diagnostics_by_slice(diagnostics_documents)
        self.metadata = {
            "support_size": int(self.matrix.vocab_size),
            "slices_total": int(len(self.matrix.slices)),
            "slice_token_counts": slice_token_counts,
            "field_entropies": field_entropies,
            "welch_target_doc_counts": dict(self.core.doc_counts_target),
            "welch_field_doc_counts": dict(self.core.doc_counts_field),
            "diagnostics_summary": diagnostics_summary,
            "diagnostics_documents": diagnostics_documents,
            "diagnostics_by_slice": diagnostics_by_slice,
            "support_pair_keys": self.matrix.feature_ids,
        }

    def perform_welch_tests_all_pairs(self, sync_only: bool = False) -> pd.DataFrame:
        welch = super().perform_welch_tests_all_pairs(sync_only=sync_only)
        self.metadata["welch_target_doc_counts"] = dict(self.core.doc_counts_target)
        self.metadata["welch_field_doc_counts"] = dict(self.core.doc_counts_field)
        return welch


def calculate_citation_identity_sync_kld_from_event_index(
    index: CitationIdentityEventIndex,
    *,
    config: CitationIdentityConfig,
    target_name: str = "",
    target_author_uid: Optional[str] = None,
    target_mask: np.ndarray,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    window_size: int = 2,
    skip_incomplete_slices: bool = True,
    min_token_global_freq: float = 1.0,
    min_docs_global_freq: int = 1,
    min_tokens_target_slice: float = 1e-12,
    min_tokens_field_slice: float = 1e-12,
    min_docs_target_slice: int = 1,
    min_docs_field_slice: int = 1,
    min_docs_target_test: int = 2,
    min_docs_field_test: int = 2,
    max_vocab_size: Optional[int] = None,
    top_k_kld_terms: Optional[int] = 50,
    include_async: bool = False,
    run_welch: bool = False,
    welch_sync_only: bool = True,
    lambda_param: float = DEFAULT_LAMBDA_PARAM,
    epsilon: float = DEFAULT_EPSILON,
    allowed_pair_keys: Optional[np.ndarray] = None,
    _target_event_cache: Optional[_CitationIdentityTargetEventCache] = None,
) -> CitationIdentitySyncKLDResult:
    """Thin wrapper over :class:`CitationIdentityKLD` (kept for API/back-compat).

    Builds the model, then runs the inherited sync (+ optional async/Welch) engine.
    ``precompute_slice_moments`` is ``"auto"`` only when Welch runs, exactly as before.
    """
    model = CitationIdentityKLD(
        index,
        config=config,
        target_name=target_name,
        target_author_uid=target_author_uid,
        target_mask=target_mask,
        start_year=start_year,
        end_year=end_year,
        window_size=window_size,
        skip_incomplete_slices=skip_incomplete_slices,
        min_token_global_freq=min_token_global_freq,
        min_docs_global_freq=min_docs_global_freq,
        min_tokens_target_slice=min_tokens_target_slice,
        min_tokens_field_slice=min_tokens_field_slice,
        min_docs_target_slice=min_docs_target_slice,
        min_docs_field_slice=min_docs_field_slice,
        min_docs_target_test=min_docs_target_test,
        min_docs_field_test=min_docs_field_test,
        max_vocab_size=max_vocab_size,
        top_k_kld_terms=top_k_kld_terms,
        lambda_param=lambda_param,
        epsilon=epsilon,
        precompute_slice_moments="auto" if run_welch else False,
        allowed_pair_keys=allowed_pair_keys,
        _target_event_cache=_target_event_cache,
    )
    summed, pointwise = model.calculate_kld_sync()
    async_df = model.calculate_kld_async() if include_async else None
    welch = model.perform_welch_tests_all_pairs(sync_only=welch_sync_only) if run_welch else None
    return MetricResult(
        sync=summed,
        pointwise=pointwise,
        async_df=async_df,
        welch=welch,
        metadata=model.metadata,
        kind="kld",
        metric="citation_identity",
        target_author_uid=target_author_uid,
        target_name=target_name,
        window_size=int(window_size),
        config={
            "start_year": start_year,
            "end_year": end_year,
            "window_size": int(window_size),
            "skip_incomplete_slices": bool(skip_incomplete_slices),
            "include_async": bool(include_async),
            "run_welch": bool(run_welch),
            "lambda_param": float(lambda_param),
            "epsilon": float(epsilon),
            "top_k_kld_terms": top_k_kld_terms,
            "min_token_global_freq": float(min_token_global_freq),
            "min_docs_global_freq": int(min_docs_global_freq),
            "min_tokens_target_slice": float(min_tokens_target_slice),
            "min_tokens_field_slice": float(min_tokens_field_slice),
            "min_docs_target_slice": int(min_docs_target_slice),
            "min_docs_field_slice": int(min_docs_field_slice),
            "min_docs_target_test": int(min_docs_target_test),
            "min_docs_field_test": int(min_docs_field_test),
            "max_vocab_size": max_vocab_size,
            "citation_identity_mode": config.mode,
            "citation_identity_counting": config.counting,
            "citation_author_scope": config.author_scope,
            "target_exclusion": config.target_exclusion,
            "remove_self_loops": bool(config.remove_self_loops),
        },
    )


__all__ = [
    "CitationIdentitySyncKLDResult",
    "CitationIdentityEventIndex",
    "CitationIdentityKLD",
    "calculate_citation_identity_sync_kld_from_event_index",
]
