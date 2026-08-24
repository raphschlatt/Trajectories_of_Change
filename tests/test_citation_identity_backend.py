from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import trajectories_of_change.citation_identity_event as ci_event
from trajectories_of_change.citation_identity import CitationIdentityConfig
from trajectories_of_change.citation_identity_event import (
    CitationIdentityEventIndex,
    _pair_arrays,
    _term_from_pair_key,
    calculate_citation_identity_sync_kld_from_event_index,
)
from trajectories_of_change.contract import build_dataset_bundle, build_target_mask


def _fixture_bundle():
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "target-paper",
                "Year": 2000,
                "Author": ["Target, T."],
                "AuthorUID": ["uid:target"],
                "References": ["target-ref", "einstein-1", "einstein-2", "bohr-1", "empty-ref"],
                "tokens": ["target"],
                "embedding_2d_x": 0.0,
                "embedding_2d_y": 0.0,
            },
            {
                "Bibcode": "field-paper",
                "Year": 2000,
                "Author": ["Field, F."],
                "AuthorUID": ["uid:field"],
                "References": ["target-ref", "einstein-1", "bohr-1", "curie-1"],
                "tokens": ["field"],
                "embedding_2d_x": 1.0,
                "embedding_2d_y": 1.0,
            },
        ]
    )
    references = pd.DataFrame(
        [
            {"Bibcode": "target-ref", "Author": ["Target, T."], "AuthorUID": ["uid:target"]},
            {"Bibcode": "einstein-1", "Author": ["Einstein", "Collaborator"], "AuthorUID": ["uid:einstein", "uid:collab"]},
            {"Bibcode": "einstein-2", "Author": ["Einstein"], "AuthorUID": ["uid:einstein"]},
            {"Bibcode": "bohr-1", "Author": ["Bohr"], "AuthorUID": ["uid:bohr"]},
            {"Bibcode": "curie-1", "Author": ["Curie"], "AuthorUID": ["uid:curie"]},
            {"Bibcode": "empty-ref", "Author": ["No author"], "AuthorUID": []},
        ]
    )
    return build_dataset_bundle(publications, references)


def _run_event(bundle, config: CitationIdentityConfig, *, target_author_uid: str = "uid:target"):
    """Run the production event/core Citation Identity path and return (result, index)."""
    index = CitationIdentityEventIndex.from_frames(bundle.publications, bundle.references, config=config)
    target_mask = build_target_mask(
        bundle.publications,
        target_author_uid=target_author_uid,
        allow_name_fallback=False,
    ).to_numpy(dtype=bool)
    result = calculate_citation_identity_sync_kld_from_event_index(
        index,
        config=config,
        target_author_uid=target_author_uid,
        target_mask=target_mask,
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=1e-12,
        min_docs_global_freq=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        top_k_kld_terms=None,
    )
    return result, index


def _doc_diag(result, bibcode: str) -> pd.Series:
    docs = result.metadata["diagnostics_documents"]
    return docs.set_index("Bibcode").loc[bibcode]


def _support_pairs(result, index) -> set[frozenset[str]]:
    keys = np.asarray(result.metadata["support_pair_keys"], dtype=np.uint64)
    pairs: set[frozenset[str]] = set()
    for key in keys:
        term = _term_from_pair_key(np.uint64(key), index.entity_labels)
        pairs.add(frozenset(part.strip() for part in term.split("|")))
    return pairs


def _assert_ci_results_equal(left, right) -> None:
    pd.testing.assert_frame_equal(left.sync, right.sync)
    pd.testing.assert_frame_equal(left.pointwise, right.pointwise)
    pd.testing.assert_frame_equal(left.async_df, right.async_df)
    for key in ("diagnostics_summary", "slice_token_counts", "field_entropies"):
        assert left.metadata[key] == right.metadata[key]
    pd.testing.assert_frame_equal(
        left.metadata["diagnostics_documents"],
        right.metadata["diagnostics_documents"],
    )
    pd.testing.assert_frame_equal(
        left.metadata["diagnostics_by_slice"],
        right.metadata["diagnostics_by_slice"],
    )
    np.testing.assert_array_equal(left.metadata["support_pair_keys"], right.metadata["support_pair_keys"])


def _assert_target_event_caches_equal(left, right) -> None:
    assert left.mode == right.mode
    assert left.author_scope == right.author_scope
    assert left.target_exclusion == right.target_exclusion
    assert left.remove_self_loops == right.remove_self_loops
    assert left.target_name == right.target_name
    assert left.target_author_uid == right.target_author_uid
    np.testing.assert_array_equal(left.target_mask, right.target_mask)
    np.testing.assert_array_equal(left.allowed_pair_keys, right.allowed_pair_keys)
    np.testing.assert_array_equal(left.u_doc, right.u_doc)
    np.testing.assert_array_equal(left.u_key, right.u_key)
    np.testing.assert_array_equal(left.u_counts, right.u_counts)
    np.testing.assert_array_equal(left.per_doc_unique, right.per_doc_unique)
    np.testing.assert_array_equal(left.target_excluded_per_doc, right.target_excluded_per_doc)
    np.testing.assert_array_equal(left.kept_count_per_doc, right.kept_count_per_doc)


@pytest.mark.parametrize("remove_self_loops", [False, True])
def test_vectorized_pair_arrays_match_scalar_order_and_counts(remove_self_loops: bool) -> None:
    cited_groups = [
        (4, (3,)),
        (7, (1,)),
        (9, (3,)),
        (12, (8,)),
    ]
    expected_keys: list[int] = []
    expected_left: list[int] = []
    expected_right: list[int] = []
    candidate_mass = 0.0
    self_loop_mass = 0.0
    for left_position in range(len(cited_groups) - 1):
        left_reference, (left_entity,) = cited_groups[left_position]
        for right_position in range(left_position + 1, len(cited_groups)):
            right_reference, (right_entity,) = cited_groups[right_position]
            candidate_mass += 1.0
            if left_entity == right_entity:
                self_loop_mass += 1.0
                if remove_self_loops:
                    continue
            low, high = sorted((left_entity, right_entity))
            expected_keys.append((low << 32) | high)
            expected_left.append(left_reference)
            expected_right.append(right_reference)

    keys, left, right, actual_candidate_mass, actual_self_loop_mass = _pair_arrays(
        cited_groups,
        single_entity=True,
        remove_self_loops=remove_self_loops,
    )
    np.testing.assert_array_equal(keys, np.asarray(expected_keys, dtype=np.uint64))
    np.testing.assert_array_equal(left, np.asarray(expected_left, dtype=np.int32))
    np.testing.assert_array_equal(right, np.asarray(expected_right, dtype=np.int32))
    assert actual_candidate_mass == candidate_mass
    assert actual_self_loop_mass == self_loop_mass


def _manual_target_event_arrays(index, *, target_author_uid: str, target_mask: np.ndarray, allowed_pair_keys=None):
    target_reference_ids = {
        str(row["Bibcode"])
        for _, row in index.references.iterrows()
        if target_author_uid in {str(value) for value in row.get("author_uids", [])}
    }
    target_ref_ints = {index.ref_lookup[ref_id] for ref_id in target_reference_ids if ref_id in index.ref_lookup}
    allowed_keys = None if allowed_pair_keys is None else np.unique(np.asarray(allowed_pair_keys, dtype=np.uint64))
    u_doc_parts: list[np.ndarray] = []
    u_key_parts: list[np.ndarray] = []
    u_count_parts: list[np.ndarray] = []
    per_doc_unique = np.zeros(len(index.records), dtype=np.int64)
    target_excluded_per_doc = np.zeros(len(index.records), dtype=np.int64)
    kept_count_per_doc = np.zeros(len(index.records), dtype=np.int64)

    for doc_idx, record in enumerate(index.records):
        pair_hits_target = np.fromiter(
            (
                int(left_ref) in target_ref_ints or int(right_ref) in target_ref_ints
                for left_ref, right_ref in zip(record.left_refs, record.right_refs)
            ),
            dtype=bool,
            count=record.pair_keys.size,
        )
        keep = ~pair_hits_target
        target_excluded_per_doc[doc_idx] = int(record.pair_keys.size - int(keep.sum()))
        keys = record.pair_keys[keep]
        if allowed_keys is not None and keys.size:
            pos = np.searchsorted(allowed_keys, keys)
            valid = pos < allowed_keys.size
            valid_indices = np.flatnonzero(valid)
            valid[valid_indices] = allowed_keys[pos[valid_indices]] == keys[valid_indices]
            keys = keys[valid]
        kept_count_per_doc[doc_idx] = int(keys.size)
        if keys.size:
            unique_keys, counts = np.unique(keys, return_counts=True)
            u_doc_parts.append(np.full(unique_keys.size, doc_idx, dtype=np.int32))
            u_key_parts.append(unique_keys.astype(np.uint64, copy=False))
            u_count_parts.append(counts.astype(np.int64, copy=False))
            per_doc_unique[doc_idx] = int(unique_keys.size)

    return {
        "u_doc": np.concatenate(u_doc_parts) if u_doc_parts else np.array([], dtype=np.int32),
        "u_key": np.concatenate(u_key_parts) if u_key_parts else np.array([], dtype=np.uint64),
        "u_counts": np.concatenate(u_count_parts) if u_count_parts else np.array([], dtype=np.int64),
        "per_doc_unique": per_doc_unique,
        "target_excluded_per_doc": target_excluded_per_doc,
        "kept_count_per_doc": kept_count_per_doc,
    }


def test_document_fractional_identity_diagnostics_match_expected_pair_masses() -> None:
    bundle = _fixture_bundle()
    result, _ = _run_event(
        bundle,
        CitationIdentityConfig(
            counting="document_fractional",
            author_scope="first_author",
            target_exclusion="all_docs",
            remove_self_loops=True,
        ),
    )

    summary = result.metadata["diagnostics_summary"]
    assert summary["self_loop_pair_mass"] == pytest.approx(1.0)
    assert summary["target_excluded_pair_mass"] == pytest.approx(6.0)
    assert summary["empty_reference_entities"] == 1

    target_row = _doc_diag(result, "target-paper")
    # all_docs exclusion drops every target-ref pair; only bohr|einstein survives (x2 raw).
    assert target_row["kept_pair_mass_raw"] == pytest.approx(2.0)
    assert target_row["support_size_after_filters"] == 1
    # document_fractional normalizes each document's retained pair mass to sum to 1.0.
    assert target_row["kept_pair_mass_weighted"] == pytest.approx(1.0)
    assert target_row["documents_without_pairs_after_filters"] == 0
    assert result.metadata["diagnostics_by_slice"].loc[0, "documents_without_pairs_after_filters"] == 0


def test_counting_policies_are_distinct_for_repeated_reference_authors() -> None:
    bundle = _fixture_bundle()
    base = dict(author_scope="first_author", target_exclusion="none", remove_self_loops=True)

    binary, _ = _run_event(bundle, CitationIdentityConfig(counting="binary", **base))
    multiplicity, _ = _run_event(bundle, CitationIdentityConfig(counting="multiplicity", **base))

    binary_row = _doc_diag(binary, "target-paper")
    multiplicity_row = _doc_diag(multiplicity, "target-paper")

    # target-paper retains 5 raw pairs across 3 unique pairs (bohr|einstein x2, target|einstein x2, bohr|target x1).
    assert binary_row["support_size_after_filters"] == 3
    assert binary_row["kept_pair_mass_raw"] == pytest.approx(5.0)
    # binary weights every unique pair once; multiplicity preserves repeat counts.
    assert binary_row["kept_pair_mass_weighted"] == pytest.approx(3.0)
    assert multiplicity_row["kept_pair_mass_weighted"] == pytest.approx(5.0)


def test_counting_models_weight_pairs_distinctly() -> None:
    """All three counting models weight retained pairs in their documented way.

    Locks the package-level guarantee directly (earlier gate runners asserted
    this only as an integration side effect):
    ``document_fractional`` normalises each document to unit mass, ``binary``
    counts each unique pair once, ``multiplicity`` preserves repeated counts.
    """
    bundle = _fixture_bundle()
    base = dict(author_scope="first_author", target_exclusion="none", remove_self_loops=True)

    fractional_row = _doc_diag(
        _run_event(bundle, CitationIdentityConfig(counting="document_fractional", **base))[0],
        "target-paper",
    )
    binary_row = _doc_diag(
        _run_event(bundle, CitationIdentityConfig(counting="binary", **base))[0],
        "target-paper",
    )
    multiplicity_row = _doc_diag(
        _run_event(bundle, CitationIdentityConfig(counting="multiplicity", **base))[0],
        "target-paper",
    )

    # target-paper retains 3 unique pairs over 5 raw co-reference events.
    assert fractional_row["kept_pair_mass_weighted"] == pytest.approx(1.0)
    assert binary_row["kept_pair_mass_weighted"] == pytest.approx(3.0)
    assert multiplicity_row["kept_pair_mass_weighted"] == pytest.approx(5.0)
    # The three counting models are genuinely distinct, not aliases.
    assert (
        fractional_row["kept_pair_mass_weighted"]
        < binary_row["kept_pair_mass_weighted"]
        < multiplicity_row["kept_pair_mass_weighted"]
    )


def test_target_exclusion_target_docs_only_keeps_field_target_pairs() -> None:
    bundle = _fixture_bundle()
    result, _ = _run_event(
        bundle,
        CitationIdentityConfig(
            counting="binary",
            target_exclusion="target_docs_only",
            remove_self_loops=True,
        ),
    )

    target_row = _doc_diag(result, "target-paper")
    field_row = _doc_diag(result, "field-paper")

    # target_docs_only strips target-ref pairs from the target's own document only.
    assert target_row["target_excluded_pair_mass"] == pytest.approx(3.0)
    # the field document retains its target-ref pairs (e.g. curie|target, einstein|target).
    assert field_row["target_excluded_pair_mass"] == pytest.approx(0.0)
    assert field_row["kept_pair_mass_raw"] == pytest.approx(6.0)


def test_all_author_scope_expands_non_first_reference_authors() -> None:
    bundle = _fixture_bundle()
    base = dict(counting="binary", target_exclusion="none", remove_self_loops=True)

    first_result, first_index = _run_event(bundle, CitationIdentityConfig(author_scope="first_author", **base))
    all_result, all_index = _run_event(bundle, CitationIdentityConfig(author_scope="all_authors", **base))

    first_row = _doc_diag(first_result, "target-paper")
    all_row = _doc_diag(all_result, "target-paper")

    # einstein-1 has a second author (collab) that only surfaces under all_authors.
    assert all_row["candidate_pair_mass"] > first_row["candidate_pair_mass"]
    assert all_row["support_size_after_filters"] > first_row["support_size_after_filters"]

    first_pairs = _support_pairs(first_result, first_index)
    all_pairs = _support_pairs(all_result, all_index)
    assert frozenset({"uid:bohr", "uid:collab"}) not in first_pairs
    assert frozenset({"uid:bohr", "uid:collab"}) in all_pairs
    assert frozenset({"uid:collab", "uid:target"}) in all_pairs


def test_target_event_cache_preserves_restricted_counting_outputs() -> None:
    assert hasattr(ci_event, "_build_citation_identity_target_event_cache")
    bundle = _fixture_bundle()
    index = CitationIdentityEventIndex.from_frames(
        bundle.publications,
        bundle.references,
        config=CitationIdentityConfig(
            counting="document_fractional",
            author_scope="first_author",
            target_exclusion="all_docs",
            remove_self_loops=True,
        ),
    )
    target_mask = build_target_mask(
        bundle.publications,
        target_author_uid="uid:target",
        allow_name_fallback=False,
    ).to_numpy(dtype=bool)
    support_config = CitationIdentityConfig(
        counting="document_fractional",
        author_scope="first_author",
        target_exclusion="all_docs",
        remove_self_loops=True,
    )
    support = calculate_citation_identity_sync_kld_from_event_index(
        index,
        config=support_config,
        target_author_uid="uid:target",
        target_mask=target_mask,
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=1e-12,
        min_docs_global_freq=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        top_k_kld_terms=None,
        include_async=True,
    )
    allowed_pair_keys = np.asarray(support.metadata["support_pair_keys"], dtype=np.uint64)
    cache = ci_event._build_citation_identity_target_event_cache(
        index,
        config=support_config,
        target_author_uid="uid:target",
        target_mask=target_mask,
        allowed_pair_keys=allowed_pair_keys,
    )

    for counting in ("binary", "multiplicity"):
        config = CitationIdentityConfig(
            counting=counting,
            author_scope="first_author",
            target_exclusion="all_docs",
            remove_self_loops=True,
        )
        uncached = calculate_citation_identity_sync_kld_from_event_index(
            index,
            config=config,
            target_author_uid="uid:target",
            target_mask=target_mask,
            window_size=1,
            skip_incomplete_slices=False,
            min_token_global_freq=1e-12,
            min_docs_global_freq=1,
            min_docs_target_test=1,
            min_docs_field_test=1,
            top_k_kld_terms=None,
            include_async=True,
            allowed_pair_keys=allowed_pair_keys,
        )
        cached = calculate_citation_identity_sync_kld_from_event_index(
            index,
            config=config,
            target_author_uid="uid:target",
            target_mask=target_mask,
            window_size=1,
            skip_incomplete_slices=False,
            min_token_global_freq=1e-12,
            min_docs_global_freq=1,
            min_docs_target_test=1,
            min_docs_field_test=1,
            top_k_kld_terms=None,
            include_async=True,
            allowed_pair_keys=allowed_pair_keys,
            _target_event_cache=cache,
        )
        _assert_ci_results_equal(cached, uncached)


def test_normalized_event_index_builder_skips_contract_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _fixture_bundle()

    def fail_normalization(*args, **kwargs):
        raise AssertionError("normalized bundle frames must not be normalized again")

    monkeypatch.setattr(ci_event, "normalize_publications_frame", fail_normalization)
    monkeypatch.setattr(ci_event, "normalize_references_frame", fail_normalization)
    index = CitationIdentityEventIndex._from_normalized_frames(
        bundle.publications,
        bundle.references,
        config=CitationIdentityConfig(),
    )
    assert len(index.records) == len(bundle.publications)


def test_target_event_cache_builds_by_document_without_flat_events(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _fixture_bundle()
    index = CitationIdentityEventIndex.from_frames(
        bundle.publications,
        bundle.references,
        config=CitationIdentityConfig(
            counting="document_fractional",
            author_scope="all_authors",
            target_exclusion="all_docs",
            remove_self_loops=True,
        ),
    )
    target_mask = build_target_mask(
        bundle.publications,
        target_author_uid="uid:target",
        allow_name_fallback=False,
    ).to_numpy(dtype=bool)
    config = CitationIdentityConfig(
        counting="document_fractional",
        author_scope="all_authors",
        target_exclusion="all_docs",
        remove_self_loops=True,
    )

    def fail_flat_events():
        raise AssertionError("target cache construction should not materialize global flat events")

    monkeypatch.setattr(index, "_flat_events", fail_flat_events, raising=False)
    unrestricted = ci_event._build_citation_identity_target_event_cache(
        index,
        config=config,
        target_author_uid="uid:target",
        target_mask=target_mask,
        allowed_pair_keys=None,
    )
    expected = _manual_target_event_arrays(
        index,
        target_author_uid="uid:target",
        target_mask=target_mask,
        allowed_pair_keys=None,
    )
    np.testing.assert_array_equal(unrestricted.u_doc, expected["u_doc"])
    np.testing.assert_array_equal(unrestricted.u_key, expected["u_key"])
    np.testing.assert_array_equal(unrestricted.u_counts, expected["u_counts"])
    np.testing.assert_array_equal(unrestricted.per_doc_unique, expected["per_doc_unique"])
    np.testing.assert_array_equal(unrestricted.target_excluded_per_doc, expected["target_excluded_per_doc"])
    np.testing.assert_array_equal(unrestricted.kept_count_per_doc, expected["kept_count_per_doc"])

    allowed_pair_keys = np.unique(unrestricted.u_key)[:2]
    restricted = ci_event._build_citation_identity_target_event_cache(
        index,
        config=config,
        target_author_uid="uid:target",
        target_mask=target_mask,
        allowed_pair_keys=allowed_pair_keys,
    )
    expected_restricted = _manual_target_event_arrays(
        index,
        target_author_uid="uid:target",
        target_mask=target_mask,
        allowed_pair_keys=allowed_pair_keys,
    )
    np.testing.assert_array_equal(restricted.u_doc, expected_restricted["u_doc"])
    np.testing.assert_array_equal(restricted.u_key, expected_restricted["u_key"])
    np.testing.assert_array_equal(restricted.u_counts, expected_restricted["u_counts"])
    np.testing.assert_array_equal(restricted.per_doc_unique, expected_restricted["per_doc_unique"])
    np.testing.assert_array_equal(restricted.target_excluded_per_doc, expected_restricted["target_excluded_per_doc"])
    np.testing.assert_array_equal(restricted.kept_count_per_doc, expected_restricted["kept_count_per_doc"])
