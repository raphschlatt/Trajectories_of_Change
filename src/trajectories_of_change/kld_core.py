"""Shared document-feature KLD core.

The core is intentionally feature-agnostic: rows are documents, columns are
tokens or pair IDs, and values are document feature weights. Vocabulary and
Citation Identity can then share sync, async, and Welch calculations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import ttest_ind_from_stats

from .defaults import DEFAULT_EPSILON, DEFAULT_LAMBDA_PARAM


def _sorted_membership_positions(
    sorted_values: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return insertion positions and an exact-membership mask."""
    positions = np.searchsorted(sorted_values, candidates)
    valid = positions < sorted_values.size
    in_bounds = np.flatnonzero(valid)
    valid[in_bounds] = sorted_values[positions[in_bounds]] == candidates[in_bounds]
    return positions, valid


def create_slices_from_years(
    years: np.ndarray,
    *,
    start_year: Optional[int],
    end_year: Optional[int],
    window_size: int,
    skip_incomplete_slices: bool,
) -> list[tuple[int, int]]:
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if years.size == 0:
        return []
    min_y, max_y = int(years.min()), int(years.max())
    if start_year is not None:
        min_y = max(min_y, int(start_year))
    if end_year is not None:
        max_y = min(max_y, int(end_year))

    slices: list[tuple[int, int]] = []
    start = min_y
    while start <= max_y:
        end = start + window_size - 1
        if end > max_y:
            if skip_incomplete_slices:
                break
            end = max_y
        slices.append((start, end))
        start = end + 1
    return slices


def _aggregate_slices(
    doc_counts: list[dict[int, float]],
    doc_lengths: np.ndarray,
    slice_masks: dict[int, np.ndarray],
    vocab_size: int,
    moments_enabled: bool,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, float],
    dict[int, int],
    dict[int, int],
    dict[int, dict[int, float]],
    dict[int, dict[int, float]],
]:
    """Per-slice token counts, totals, doc counts and optional moments.

    Shared post-vocabulary machinery of both ``DocumentFeatureMatrix`` builders.
    Iterates documents in order 0..n, exactly matching the former inline loops, so
    the (float-sensitive) accumulations stay byte-identical.
    """
    row_slice_labels: list[Optional[int]] = [None] * len(doc_counts)
    for label, mask in slice_masks.items():
        for row_idx in np.flatnonzero(mask):
            row_slice_labels[int(row_idx)] = int(label)

    slice_all_counts = {label: np.zeros(vocab_size, dtype=float) for label in slice_masks}
    slice_all_totals = {label: 0.0 for label in slice_masks}
    slice_all_docs = {label: int(mask.sum()) for label, mask in slice_masks.items()}
    slice_all_analyzable_docs = {label: 0 for label in slice_masks}
    slice_all_moment_sums: dict[int, dict[int, float]] = {label: {} for label in slice_masks}
    slice_all_moment_sum_sq: dict[int, dict[int, float]] = {label: {} for label in slice_masks}

    for row_idx, compact in enumerate(doc_counts):
        label = row_slice_labels[row_idx]
        if label is None:
            continue
        doc_len = float(doc_lengths[row_idx])
        slice_all_totals[label] += doc_len
        for token_idx, count in compact.items():
            slice_all_counts[label][token_idx] += float(count)
        if doc_len <= 0:
            continue
        slice_all_analyzable_docs[label] += 1
        if not moments_enabled:
            continue
        inv_len = 1.0 / doc_len
        sums = slice_all_moment_sums[label]
        sum_sq = slice_all_moment_sum_sq[label]
        for token_idx, count in compact.items():
            rel = float(count) * inv_len
            sums[token_idx] = sums.get(token_idx, 0.0) + rel
            sum_sq[token_idx] = sum_sq.get(token_idx, 0.0) + rel * rel

    return (
        slice_all_counts,
        slice_all_totals,
        slice_all_docs,
        slice_all_analyzable_docs,
        slice_all_moment_sums,
        slice_all_moment_sum_sq,
    )


def token_counts(value: Any) -> Counter:
    if isinstance(value, Counter):
        return Counter({str(token): float(count) for token, count in value.items() if float(count) > 0})
    if isinstance(value, dict):
        return Counter({str(token): float(count) for token, count in value.items() if float(count) > 0})
    if isinstance(value, (list, tuple)):
        return Counter(str(token) for token in value if str(token).strip())
    return Counter()


def smooth_distribution(
    probs: np.ndarray,
    *,
    bg_probs: np.ndarray,
    lambda_param: float,
    epsilon: float,
) -> np.ndarray:
    if probs.size == 0:
        return np.array([], dtype=float)
    smoothed = (1.0 - float(lambda_param)) * probs + float(lambda_param) * bg_probs
    smoothed = np.maximum(smoothed, float(epsilon))
    total = float(smoothed.sum())
    if total <= 0:
        return np.full(probs.size, 1.0 / float(probs.size), dtype=float)
    return smoothed / total


@dataclass(frozen=True)
class DocumentFeatureMatrix:
    """Sparse document-feature matrix plus slice metadata."""

    years: np.ndarray
    slices: list[tuple[int, int]]
    slice_masks: dict[int, np.ndarray]
    vocab_index: pd.Index
    vocab_lookup: dict[str, int]
    doc_term_matrix: csr_matrix
    doc_counts: list[dict[int, float]]
    doc_lengths: np.ndarray
    global_counts: np.ndarray
    slice_all_counts: dict[int, np.ndarray]
    slice_all_totals: dict[int, float]
    slice_all_docs: dict[int, int]
    slice_all_analyzable_docs: dict[int, int]
    slice_all_moment_sums: dict[int, dict[int, float]]
    slice_all_moment_sum_sq: dict[int, dict[int, float]]
    precompute_slice_moments: bool = False
    feature_ids: Optional[np.ndarray] = None

    @property
    def global_vocab(self) -> list[str]:
        return [str(term) for term in self.vocab_index.tolist()]

    @property
    def global_vocab_set(self) -> set[str]:
        return set(self.global_vocab)

    @property
    def vocab_size(self) -> int:
        return int(len(self.vocab_index))

    @classmethod
    def from_token_frame(
        cls,
        corpus: pd.DataFrame,
        *,
        year_col: str,
        token_col: str,
        start_year: Optional[int],
        end_year: Optional[int],
        window_size: int,
        skip_incomplete_slices: bool,
        min_token_global_freq: float,
        min_docs_global_freq: int,
        max_vocab_size: Optional[int],
        precompute_slice_moments: bool | str = False,
        slice_moment_max_doc_terms: int = 500_000,
    ) -> "DocumentFeatureMatrix":
        years = corpus[year_col].astype(int).to_numpy(copy=False)
        slices = create_slices_from_years(
            years,
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
        )
        slice_masks = {int(end): (years >= int(start)) & (years <= int(end)) for start, end in slices}

        counts: Counter = Counter()
        docfreq: Counter = Counter()
        raw_doc_counts: list[Counter] = []
        for tokens in corpus[token_col]:
            doc_counter = token_counts(tokens)
            raw_doc_counts.append(doc_counter)
            counts.update(doc_counter)
            docfreq.update(doc_counter.keys())

        doc_term_total = sum(len(doc_counter) for doc_counter in raw_doc_counts)
        if precompute_slice_moments == "auto":
            moments_enabled = doc_term_total <= int(slice_moment_max_doc_terms)
        else:
            moments_enabled = bool(precompute_slice_moments)

        vocab = [
            term
            for term, count in counts.items()
            if count >= float(min_token_global_freq) and docfreq.get(term, 0) >= int(min_docs_global_freq)
        ]
        if max_vocab_size and len(vocab) > int(max_vocab_size):
            vocab = [term for term, _ in counts.most_common(int(max_vocab_size))]
        vocab_index = pd.Index(vocab)
        vocab_lookup = {str(term): idx for idx, term in enumerate(vocab)}
        vocab_size = len(vocab)

        doc_counts: list[dict[int, float]] = []
        doc_lengths = np.zeros(len(raw_doc_counts), dtype=float)
        global_counts = np.zeros(vocab_size, dtype=float)
        matrix_rows: list[int] = []
        matrix_cols: list[int] = []
        matrix_data: list[float] = []

        for row_idx, doc_counter in enumerate(raw_doc_counts):
            compact: dict[int, float] = {}
            doc_len = 0.0
            for token, count in doc_counter.items():
                idx = vocab_lookup.get(str(token))
                if idx is None:
                    continue
                count_f = float(count)
                compact[idx] = compact.get(idx, 0.0) + count_f
                doc_len += count_f
                global_counts[idx] += count_f
            doc_counts.append(compact)
            doc_lengths[row_idx] = doc_len
            for token_idx, count in compact.items():
                matrix_rows.append(int(row_idx))
                matrix_cols.append(int(token_idx))
                matrix_data.append(float(count))

        doc_term_matrix = csr_matrix(
            (matrix_data, (matrix_rows, matrix_cols)),
            shape=(len(raw_doc_counts), vocab_size),
            dtype=float,
        )
        (
            slice_all_counts,
            slice_all_totals,
            slice_all_docs,
            slice_all_analyzable_docs,
            slice_all_moment_sums,
            slice_all_moment_sum_sq,
        ) = _aggregate_slices(doc_counts, doc_lengths, slice_masks, vocab_size, moments_enabled)
        return cls(
            years=years,
            slices=slices,
            slice_masks=slice_masks,
            vocab_index=vocab_index,
            vocab_lookup=vocab_lookup,
            doc_term_matrix=doc_term_matrix,
            doc_counts=doc_counts,
            doc_lengths=doc_lengths,
            global_counts=global_counts,
            slice_all_counts=slice_all_counts,
            slice_all_totals=slice_all_totals,
            slice_all_docs=slice_all_docs,
            slice_all_analyzable_docs=slice_all_analyzable_docs,
            slice_all_moment_sums=slice_all_moment_sums,
            slice_all_moment_sum_sq=slice_all_moment_sum_sq,
            precompute_slice_moments=moments_enabled,
            feature_ids=None,
        )

    @classmethod
    def from_weighted_events(
        cls,
        *,
        years: np.ndarray,
        doc_indices: np.ndarray,
        feature_ids: np.ndarray,
        weights: np.ndarray,
        label_for_feature_id,
        start_year: Optional[int],
        end_year: Optional[int],
        window_size: int,
        skip_incomplete_slices: bool,
        min_token_global_freq: float,
        min_docs_global_freq: int,
        max_vocab_size: Optional[int],
        precompute_slice_moments: bool | str = False,
        slice_moment_max_doc_terms: int = 500_000,
    ) -> "DocumentFeatureMatrix":
        years = np.asarray(years, dtype=int)
        doc_indices = np.asarray(doc_indices, dtype=np.int32)
        feature_ids = np.asarray(feature_ids)
        weights = np.asarray(weights, dtype=float)
        if not (doc_indices.size == feature_ids.size == weights.size):
            raise ValueError("doc_indices, feature_ids, and weights must have the same length")

        slices = create_slices_from_years(
            years,
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
        )
        slice_masks = {int(end): (years >= int(start)) & (years <= int(end)) for start, end in slices}
        moments_enabled = (
            bool(feature_ids.size <= int(slice_moment_max_doc_terms))
            if precompute_slice_moments == "auto"
            else bool(precompute_slice_moments)
        )

        if feature_ids.size:
            unique_ids, inverse = np.unique(feature_ids, return_inverse=True)
            weighted_counts = np.bincount(inverse, weights=weights).astype(float)
            docfreq = np.bincount(inverse).astype(int)
            vocab_mask = (weighted_counts >= float(min_token_global_freq)) & (
                docfreq >= int(min_docs_global_freq)
            )
            vocab_ids = unique_ids[vocab_mask]
            vocab_weights = weighted_counts[vocab_mask]
            if max_vocab_size and vocab_ids.size > int(max_vocab_size):
                keep = np.argsort(vocab_weights)[::-1][: int(max_vocab_size)]
                vocab_ids = np.sort(vocab_ids[keep])
        else:
            vocab_ids = np.array([], dtype=feature_ids.dtype)

        vocab_size = int(vocab_ids.size)
        vocab_index = pd.Index([str(label_for_feature_id(feature_id)) for feature_id in vocab_ids])
        vocab_lookup = {str(term): idx for idx, term in enumerate(vocab_index)}

        if vocab_size and feature_ids.size:
            positions, valid = _sorted_membership_positions(vocab_ids, feature_ids)
            matrix_rows = doc_indices[valid].astype(np.int32, copy=False)
            matrix_cols = positions[valid].astype(np.int32, copy=False)
            matrix_data = weights[valid].astype(float, copy=False)
        else:
            matrix_rows = np.array([], dtype=np.int32)
            matrix_cols = np.array([], dtype=np.int32)
            matrix_data = np.array([], dtype=float)

        doc_term_matrix = csr_matrix(
            (matrix_data, (matrix_rows, matrix_cols)),
            shape=(len(years), vocab_size),
            dtype=float,
        )
        doc_counts: list[dict[int, float]] = []
        doc_lengths = np.zeros(len(years), dtype=float)
        for row_idx in range(len(years)):
            start, end = doc_term_matrix.indptr[row_idx], doc_term_matrix.indptr[row_idx + 1]
            row_indices = doc_term_matrix.indices[start:end]
            row_data = doc_term_matrix.data[start:end]
            compact = {int(idx): float(value) for idx, value in zip(row_indices, row_data)}
            doc_counts.append(compact)
            doc_lengths[row_idx] = float(row_data.sum()) if row_data.size else 0.0

        global_counts = np.asarray(doc_term_matrix.sum(axis=0)).ravel().astype(float, copy=False)
        (
            slice_all_counts,
            slice_all_totals,
            slice_all_docs,
            slice_all_analyzable_docs,
            slice_all_moment_sums,
            slice_all_moment_sum_sq,
        ) = _aggregate_slices(doc_counts, doc_lengths, slice_masks, vocab_size, moments_enabled)

        return cls(
            years=years,
            slices=slices,
            slice_masks=slice_masks,
            vocab_index=vocab_index,
            vocab_lookup=vocab_lookup,
            doc_term_matrix=doc_term_matrix,
            doc_counts=doc_counts,
            doc_lengths=doc_lengths,
            global_counts=global_counts,
            slice_all_counts=slice_all_counts,
            slice_all_totals=slice_all_totals,
            slice_all_docs=slice_all_docs,
            slice_all_analyzable_docs=slice_all_analyzable_docs,
            slice_all_moment_sums=slice_all_moment_sums,
            slice_all_moment_sum_sq=slice_all_moment_sum_sq,
            precompute_slice_moments=moments_enabled,
            feature_ids=np.asarray(vocab_ids, dtype=feature_ids.dtype) if vocab_size else np.array([], dtype=feature_ids.dtype),
        )

    def counts_for_mask(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        out = np.zeros(self.vocab_size, dtype=float)
        if self.vocab_size == 0:
            return out, 0.0
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.size != self.doc_term_matrix.shape[0]:
            raise ValueError("mask length must match corpus length")
        if not mask_array.any():
            return out, 0.0
        out = np.asarray(self.doc_term_matrix[mask_array].sum(axis=0)).ravel().astype(float, copy=False)
        return out, float(out.sum())

    def slice_counts_for_mask(self, mask: np.ndarray) -> dict[int, tuple[np.ndarray, float, int]]:
        out: dict[int, tuple[np.ndarray, float, int]] = {}
        mask_array = np.asarray(mask, dtype=bool)
        for label, slice_mask in self.slice_masks.items():
            combined = mask_array & slice_mask
            counts, total = self.counts_for_mask(combined)
            out[int(label)] = (counts, total, int(combined.sum()))
        return out

    def moments_for_mask(
        self,
        label: int,
        mask: np.ndarray,
        allowed_indices: Optional[tuple[int, ...]],
    ) -> tuple[dict[int, float], dict[int, float], int]:
        combined = np.asarray(mask, dtype=bool) & self.slice_masks[int(label)]
        rows = np.flatnonzero(combined & (self.doc_lengths > 0))
        n_analyzable = int(rows.size)
        if not n_analyzable or (allowed_indices is not None and not allowed_indices):
            return {}, {}, n_analyzable

        features: np.ndarray | None = None
        values = self.doc_term_matrix[rows]
        if allowed_indices is not None:
            features = np.asarray(
                sorted({int(idx) for idx in allowed_indices if 0 <= int(idx) < self.vocab_size}),
                dtype=np.intp,
            )
            if not features.size:
                return {}, {}, n_analyzable
            values = values[:, features]

        relative = values.multiply(np.reciprocal(self.doc_lengths[rows])[:, None])
        sums_array = np.asarray(relative.sum(axis=0)).ravel()
        sum_sq_array = np.asarray(relative.power(2).sum(axis=0)).ravel()
        present = np.flatnonzero(sums_array)
        keys = present if features is None else features[present]
        return (
            {int(idx): float(value) for idx, value in zip(keys, sums_array[present])},
            {int(idx): float(value) for idx, value in zip(keys, sum_sq_array[present])},
            n_analyzable,
        )

    def all_moments(self, label: int, allowed_indices: tuple[int, ...]) -> tuple[dict[int, float], dict[int, float], int]:
        if not self.precompute_slice_moments:
            return self.moments_for_mask(int(label), self.slice_masks[int(label)], allowed_indices)
        if not allowed_indices:
            return {}, {}, self.slice_all_analyzable_docs[int(label)]
        allowed = set(allowed_indices)
        sums = {idx: value for idx, value in self.slice_all_moment_sums[int(label)].items() if idx in allowed}
        sum_sq = {idx: value for idx, value in self.slice_all_moment_sum_sq[int(label)].items() if idx in allowed}
        return sums, sum_sq, self.slice_all_analyzable_docs[int(label)]


class KLDCore:
    """Feature-agnostic sync/async/Welch KLD calculations."""

    def __init__(
        self,
        matrix: DocumentFeatureMatrix,
        *,
        target_mask: np.ndarray,
        lambda_param: float = DEFAULT_LAMBDA_PARAM,
        epsilon: float = DEFAULT_EPSILON,
        min_tokens_target_slice: float = 10.0,
        min_tokens_field_slice: float = 10.0,
        min_docs_target_slice: int = 1,
        min_docs_field_slice: int = 1,
        min_docs_target_test: int = 2,
        min_docs_field_test: int = 2,
        top_k_kld_terms: Optional[int] = 50,
    ) -> None:
        self.matrix = matrix
        self.target_mask = np.asarray(target_mask, dtype=bool)
        if self.target_mask.size != self.matrix.doc_term_matrix.shape[0]:
            raise ValueError("target_mask length must match feature matrix documents")
        self.lambda_param = float(lambda_param)
        self.epsilon = float(epsilon)
        self.min_tokens_target_slice = float(min_tokens_target_slice)
        self.min_tokens_field_slice = float(min_tokens_field_slice)
        self.min_docs_target_slice = int(min_docs_target_slice)
        self.min_docs_field_slice = int(min_docs_field_slice)
        self.min_docs_target_test = int(min_docs_target_test)
        self.min_docs_field_test = int(min_docs_field_test)
        self.top_k_kld_terms = top_k_kld_terms
        self._sync_contrib_cache: dict[int, np.ndarray] = {}
        self._target_moment_cache: dict[int, tuple[dict[int, float], dict[int, float], int]] = {}
        self._all_moment_cache: dict[int, tuple[dict[int, float], dict[int, float], int]] = {}
        self.doc_counts_target: dict[int, int] = {}
        self.doc_counts_field: dict[int, int] = {}

        target_counts, _ = self.matrix.counts_for_mask(self.target_mask)
        bg_counts = self.matrix.global_counts - target_counts
        bg_total = float(bg_counts.sum())
        if bg_total <= 0 or self.matrix.vocab_size == 0:
            self.bg_probs = np.full(self.matrix.vocab_size, 1.0 / max(self.matrix.vocab_size, 1), dtype=float)
        else:
            self.bg_probs = bg_counts.astype(float) / bg_total
        if self.matrix.vocab_size:
            self.bg_probs = np.maximum(self.bg_probs, self.epsilon)
            self.bg_probs /= float(self.bg_probs.sum())

        self.slice_models_target: dict[int, np.ndarray] = {}
        self.slice_models_field: dict[int, np.ndarray] = {}
        self.slice_token_counts: dict[int, dict[str, float | int]] = {}
        target_slice_counts = self.matrix.slice_counts_for_mask(self.target_mask)
        for _, end in self.matrix.slices:
            label = int(end)
            t_counts, t_total, t_docs = target_slice_counts[label]
            all_counts = self.matrix.slice_all_counts[label]
            all_total = self.matrix.slice_all_totals[label]
            all_docs = self.matrix.slice_all_docs[label]
            f_counts = all_counts - t_counts
            f_total = float(all_total - t_total)
            f_docs = int(all_docs - t_docs)
            self.slice_models_target[label] = (
                t_counts.astype(float) / float(t_total)
                if t_total > 0
                else np.zeros(self.matrix.vocab_size, dtype=float)
            )
            self.slice_models_field[label] = (
                f_counts.astype(float) / float(f_total)
                if f_total > 0
                else np.zeros(self.matrix.vocab_size, dtype=float)
            )
            self.slice_token_counts[label] = {
                "target_tokens": float(t_total),
                "field_tokens": float(f_total),
                "target_docs": int(t_docs),
                "field_docs": int(f_docs),
            }

    def _target_slice_ok(self, label: int) -> bool:
        counts = self.slice_token_counts.get(int(label), {})
        return (
            counts.get("target_tokens", 0) >= self.min_tokens_target_slice
            and counts.get("target_docs", 0) >= self.min_docs_target_slice
        )

    def _field_slice_ok(self, label: int) -> bool:
        counts = self.slice_token_counts.get(int(label), {})
        return (
            counts.get("field_tokens", 0) >= self.min_tokens_field_slice
            and counts.get("field_docs", 0) >= self.min_docs_field_slice
        )

    def _slice_ok(self, label: int) -> bool:
        return self._target_slice_ok(label) and self._field_slice_ok(label)

    def _smooth(self, probs: np.ndarray) -> np.ndarray:
        return smooth_distribution(
            probs,
            bg_probs=self.bg_probs,
            lambda_param=self.lambda_param,
            epsilon=self.epsilon,
        )

    def _sync_contributions(self, label: int) -> np.ndarray:
        label = int(label)
        cached = self._sync_contrib_cache.get(label)
        if cached is None:
            p_t = self._smooth(self.slice_models_target[label])
            p_f = self._smooth(self.slice_models_field[label])
            cached = p_t * np.log2(p_t / p_f)
            self._sync_contrib_cache[label] = cached
        return cached

    def calculate_sync(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per-slice synchronous KLD (diagonal t==t) plus pointwise term contributions.

        sync(t) is the diagonal of :meth:`calculate_async` — ``sync(t) == async(t, t)``
        mathematically. The two are kept as *separate* methods on purpose, not merged
        into one engine:

        * sync uses the elementwise, cached ``_sync_contributions``
          (``sum_i p_t[i] * log2(p_t[i] / p_f[i])``). This is the production hot path;
          ``perform_welch_tests(sync_only=True)`` reuses the same cache on the diagonal.
        * async uses a dot-product reduction
          (``dot(p_t, log2 p_t) - dot(p_t, log2 p_f)``) over all target x field pairs.

        The two formulas are equal up to ~1e-15 (floating-point reduction order differs).
        Routing sync through async's dot form would shift the frozen sync values by that
        amount and break the rtol=1e-9 golden identity gate, so both forms are preserved
        deliberately. Do not "unify" them into a single formula.
        """
        summed_rows: list[dict[str, float | int]] = []
        point_rows: list[dict[str, float | int | str]] = []
        for _, end in self.matrix.slices:
            label = int(end)
            if not self._slice_ok(label):
                continue
            mle_t = self.slice_models_target[label]
            if mle_t.sum() == 0:
                continue
            contribs = self._sync_contributions(label)
            summed_rows.append({"slice": label, "kld_all": float(contribs.sum())})
            mask = mle_t > 0
            for term, value in zip(self.matrix.vocab_index[mask], contribs[mask]):
                point_rows.append({"slice": label, "term": term, "kld_contribution": float(value)})
        return pd.DataFrame(summed_rows), pd.DataFrame(point_rows)

    def calculate_async(self) -> pd.DataFrame:
        """KLD for every target x field slice pair (the off-diagonal lead/lag surface).

        The diagonal (t==t) equals :meth:`calculate_sync` up to ~1e-15; see that method
        for why the two are intentionally separate methods rather than one engine. Async
        is disabled by default in production runs (only the sync diagonal is computed).
        """
        labels = [int(end) for _, end in self.matrix.slices]
        field_labels = [label for label in labels if self._field_slice_ok(label)]
        field_cache = {label: self._smooth(self.slice_models_field[label]) for label in field_labels}
        rows: list[dict[str, float | int]] = []
        for _, end in self.matrix.slices:
            t_label = int(end)
            if not self._target_slice_ok(t_label):
                continue
            mle_t = self.slice_models_target[t_label]
            if mle_t.sum() == 0:
                continue
            p_t = self._smooth(mle_t)
            const_t = float(np.dot(p_t, np.log2(p_t)))
            for f_label in field_labels:
                p_f = field_cache[f_label]
                kld_val = const_t - float(np.dot(p_t, np.log2(p_f)))
                rows.append(
                    {
                        "target_slice": t_label,
                        "field_slice": f_label,
                        "time_diff": f_label - t_label,
                        "kld": float(kld_val),
                    }
                )
        return pd.DataFrame(rows)

    def _moments_to_term_stats(
        self,
        sums: Mapping[int, float],
        sum_sq: Mapping[int, float],
        n_analyzable: int,
        *,
        min_docs_required: int,
        allowed_indices: tuple[int, ...] | None = None,
    ) -> tuple[dict[str, tuple[float, float, int]], int]:
        if n_analyzable < min_docs_required:
            return {}, n_analyzable
        indices = sums.keys() if allowed_indices is None else allowed_indices
        if n_analyzable == 1:
            return {
                str(self.matrix.vocab_index[int(idx)]): (float(sums[int(idx)]), 0.0, 1)
                for idx in indices
                if int(idx) in sums and int(idx) < len(self.matrix.vocab_index)
            }, 1
        stats: dict[str, tuple[float, float, int]] = {}
        n_float = float(n_analyzable)
        for idx in indices:
            idx = int(idx)
            if idx >= len(self.matrix.vocab_index):
                continue
            if idx not in sums:
                continue
            sum_val = float(sums[idx])
            mean_val = float(sum_val) / n_float
            variance = (float(sum_sq.get(idx, 0.0)) - (float(sum_val) ** 2) / n_float) / float(
                max(n_analyzable - 1, 1)
            )
            variance = max(variance, 0.0)
            stats[str(self.matrix.vocab_index[idx])] = (mean_val, float(np.sqrt(variance)), n_analyzable)
        return stats, n_analyzable

    def _target_moments_for_label(self, label: int) -> tuple[dict[int, float], dict[int, float], int]:
        label = int(label)
        cached = self._target_moment_cache.get(label)
        if cached is None:
            cached = self.matrix.moments_for_mask(label, self.target_mask, None)
            self._target_moment_cache[label] = cached
        return cached

    def _all_moments_for_label(self, label: int) -> tuple[dict[int, float], dict[int, float], int]:
        label = int(label)
        cached = self._all_moment_cache.get(label)
        if cached is None:
            if self.matrix.precompute_slice_moments:
                cached = (
                    self.matrix.slice_all_moment_sums[label],
                    self.matrix.slice_all_moment_sum_sq[label],
                    self.matrix.slice_all_analyzable_docs[label],
                )
            else:
                cached = self.matrix.moments_for_mask(label, self.matrix.slice_masks[label], None)
            self._all_moment_cache[label] = cached
        return cached

    def _compute_term_stats(
        self,
        label: int,
        *,
        allowed_indices: tuple[int, ...],
        side: str,
        min_docs_required: int,
    ) -> tuple[dict[str, tuple[float, float, int]], int]:
        target_sums, target_sum_sq, target_n = self._target_moments_for_label(int(label))
        if side == "target":
            return self._moments_to_term_stats(
                target_sums,
                target_sum_sq,
                target_n,
                min_docs_required=min_docs_required,
                allowed_indices=allowed_indices,
            )
        if side != "field":
            raise ValueError("side must be one of 'target' or 'field'")
        all_sums, all_sum_sq, all_n = self._all_moments_for_label(int(label))
        field_n = int(all_n - target_n)
        field_sums: dict[int, float] = {}
        field_sum_sq: dict[int, float] = {}
        for idx in allowed_indices:
            sum_val = float(all_sums.get(idx, 0.0)) - float(target_sums.get(idx, 0.0))
            sq_val = float(all_sum_sq.get(idx, 0.0)) - float(target_sum_sq.get(idx, 0.0))
            if abs(sum_val) > 1e-15:
                field_sums[idx] = sum_val
            if abs(sq_val) > 1e-15:
                field_sum_sq[idx] = sq_val
        return self._moments_to_term_stats(
            field_sums,
            field_sum_sq,
            field_n,
            min_docs_required=min_docs_required,
        )

    def perform_welch_tests(self, sync_only: bool = False) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        labels = [int(end) for _, end in self.matrix.slices]
        for t_label in labels:
            if not self._target_slice_ok(t_label):
                continue
            candidate_field_labels = [t_label] if sync_only else [label for label in labels if self._field_slice_ok(label)]
            for f_label in candidate_field_labels:
                if not self._field_slice_ok(f_label):
                    continue
                if f_label == t_label:
                    contributions = self._sync_contributions(t_label)
                else:
                    p_t = self._smooth(self.slice_models_target[t_label])
                    p_f = self._smooth(self.slice_models_field[f_label])
                    contributions = p_t * np.log2(p_t / p_f)
                if self.top_k_kld_terms is not None and self.top_k_kld_terms > 0:
                    ranking = np.argsort(np.abs(contributions))[::-1][: int(self.top_k_kld_terms)]
                else:
                    ranking = np.flatnonzero(
                        (self.slice_models_target[t_label] > 0) | (self.slice_models_field[f_label] > 0)
                    )
                allowed_indices = tuple(int(idx) for idx in ranking if idx < len(self.matrix.vocab_index))
                stats_t, n_docs_t = self._compute_term_stats(
                    t_label,
                    allowed_indices=allowed_indices,
                    side="target",
                    min_docs_required=self.min_docs_target_test,
                )
                stats_f, n_docs_f = self._compute_term_stats(
                    f_label,
                    allowed_indices=allowed_indices,
                    side="field",
                    min_docs_required=self.min_docs_field_test,
                )
                self.doc_counts_target[t_label] = n_docs_t
                self.doc_counts_field[f_label] = n_docs_f
                if n_docs_t < self.min_docs_target_test or n_docs_f < self.min_docs_field_test:
                    continue
                selected: list[tuple[int, str, float, float, int, float, float, int]] = []
                for idx in ranking:
                    if idx >= len(self.matrix.vocab_index):
                        continue
                    term = str(self.matrix.vocab_index[idx])
                    mean_t, std_t, n_t_val = stats_t.get(term, (0.0, 0.0, n_docs_t))
                    mean_f, std_f, n_f_val = stats_f.get(term, (0.0, 0.0, n_docs_f))
                    if mean_t == 0 and mean_f == 0:
                        continue
                    selected.append(
                        (int(idx), term, mean_t, std_t, n_t_val, mean_f, std_f, n_f_val)
                    )
                if not selected:
                    continue

                pvalues = np.atleast_1d(
                    ttest_ind_from_stats(
                        np.asarray([item[2] for item in selected], dtype=float),
                        np.asarray([item[3] for item in selected], dtype=float),
                        np.asarray([item[4] for item in selected], dtype=int),
                        np.asarray([item[5] for item in selected], dtype=float),
                        np.asarray([item[6] for item in selected], dtype=float),
                        np.asarray([item[7] for item in selected], dtype=int),
                        equal_var=False,
                    ).pvalue
                )
                for item, pval in zip(selected, pvalues):
                    idx, term, mean_t, _, _, mean_f, _, _ = item
                    rows.append(
                        {
                            "target_slice": t_label,
                            "field_slice": f_label,
                            "term": term,
                            "pvalue": float(pval),
                            "kld_contribution": float(contributions[idx]),
                            "mean_target": float(mean_t),
                            "mean_field": float(mean_f),
                        }
                    )
        return pd.DataFrame(
            rows,
            columns=[
                "target_slice",
                "field_slice",
                "term",
                "pvalue",
                "kld_contribution",
                "mean_target",
                "mean_field",
            ],
        )


__all__ = [
    "DocumentFeatureMatrix",
    "KLDCore",
    "create_slices_from_years",
    "smooth_distribution",
    "token_counts",
]
