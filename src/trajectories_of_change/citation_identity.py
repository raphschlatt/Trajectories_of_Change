"""Shared Citation Identity configuration and reference-context helpers.

Citation Identity is operationalized as a Reference-Context / Co-Reference
Identity: pairs of cited authors or works in the reference lists of the
target's own publications. The production metric is computed by the
event/core path (:mod:`trajectories_of_change.citation_identity_event`); this
module provides the configuration and the reference-context helpers that path
shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import pandas as pd

from .contract import _coerce_list

CitationCounting = Literal["document_fractional", "binary", "multiplicity"]
CitationAuthorScope = Literal["first_author", "all_authors"]
TargetExclusion = Literal["none", "target_docs_only", "all_docs"]
CitationMode = Literal["authors", "works"]


@dataclass(frozen=True)
class CitationIdentityConfig:
    """Configuration for Citation Identity / co-reference token construction."""

    mode: CitationMode = "authors"
    counting: CitationCounting = "document_fractional"
    author_scope: CitationAuthorScope = "first_author"
    target_exclusion: TargetExclusion = "all_docs"
    remove_self_loops: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"authors", "works"}:
            raise ValueError("mode must be one of 'authors' or 'works'")
        if self.counting not in {"document_fractional", "binary", "multiplicity"}:
            raise ValueError("counting must be one of 'document_fractional', 'binary', 'multiplicity'")
        if self.author_scope not in {"first_author", "all_authors"}:
            raise ValueError("author_scope must be one of 'first_author', 'all_authors'")
        if self.target_exclusion not in {"none", "target_docs_only", "all_docs"}:
            raise ValueError("target_exclusion must be one of 'none', 'target_docs_only', 'all_docs'")


def _reference_entities(
    row: pd.Series | Mapping[str, Any],
    *,
    config: CitationIdentityConfig,
) -> list[str]:
    if config.mode == "works":
        return [str(row["Bibcode"])]

    if "author_uids" in row:
        entities = _coerce_list(row["author_uids"], split_semicolon=True)
    elif "Author" in row:
        entities = _coerce_list(row["Author"], split_semicolon=True)
    else:
        entities = []

    if config.author_scope == "first_author" and entities:
        entities = entities[:1]
    return [str(entity) for entity in entities if str(entity).strip()]


def _diagnostics_by_slice(docs: pd.DataFrame) -> pd.DataFrame:
    if docs.empty:
        return pd.DataFrame(
            columns=[
                "slice",
                "documents",
                "reference_mentions",
                "analyzable_reference_mentions",
                "empty_reference_entities",
                "candidate_pair_mass",
                "self_loop_pair_mass",
                "target_excluded_pair_mass",
                "kept_pair_mass_raw",
                "kept_pair_mass_weighted",
                "support_size_before_filters",
                "support_size_after_filters",
                "documents_without_pairs_after_filters",
            ]
        )
    grouped = (
        docs.groupby("slice", as_index=False)
        .agg(
            documents=("Bibcode", "count"),
            reference_mentions=("reference_mentions", "sum"),
            analyzable_reference_mentions=("analyzable_reference_mentions", "sum"),
            empty_reference_entities=("empty_reference_entities", "sum"),
            candidate_pair_mass=("candidate_pair_mass", "sum"),
            self_loop_pair_mass=("self_loop_pair_mass", "sum"),
            target_excluded_pair_mass=("target_excluded_pair_mass", "sum"),
            kept_pair_mass_raw=("kept_pair_mass_raw", "sum"),
            kept_pair_mass_weighted=("kept_pair_mass_weighted", "sum"),
            support_size_before_filters=("support_size_before_filters", "sum"),
            support_size_after_filters=("support_size_after_filters", "sum"),
            documents_without_pairs_after_filters=("documents_without_pairs_after_filters", "sum"),
        )
        .sort_values("slice")
        .reset_index(drop=True)
    )
    return grouped


__all__ = [
    "CitationAuthorScope",
    "CitationCounting",
    "CitationIdentityConfig",
    "CitationMode",
    "TargetExclusion",
]
