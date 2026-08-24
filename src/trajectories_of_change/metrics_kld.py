"""
KLD metrics (vocabulary & co-citation) as reusable modules.

This implementation focuses on the package core:
- canonical two-parquet inputs
- AuthorUID-first target selection
- non-overlapping time slices
- optional Welch tests on top-K candidate terms
"""

from __future__ import annotations

from collections import Counter
import logging
import time
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .contract import apply_target_field_split
from .defaults import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_LAMBDA_PARAM,
    DEFAULT_TOP_K_KLD_TERMS,
    DEFAULT_WINDOW_SIZE,
)
from .kld_core import DocumentFeatureMatrix, KLDCore

logger = logging.getLogger(__name__)


def _is_token_container(value: Any) -> bool:
    return isinstance(value, (list, tuple, Counter, dict))


class KLDPrecompute:
    """Reusable corpus-level counts for KLD models with different targets."""

    def __init__(
        self,
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
    ) -> None:
        self.year_col = year_col
        self.token_col = token_col
        self.matrix = DocumentFeatureMatrix.from_token_frame(
            corpus,
            year_col=year_col,
            token_col=token_col,
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
            min_token_global_freq=min_token_global_freq,
            min_docs_global_freq=min_docs_global_freq,
            max_vocab_size=max_vocab_size,
            precompute_slice_moments=precompute_slice_moments,
            slice_moment_max_doc_terms=slice_moment_max_doc_terms,
        )
        self.years = self.matrix.years
        self.slices = self.matrix.slices
        self.slice_masks = self.matrix.slice_masks
        self.precompute_slice_moments = self.matrix.precompute_slice_moments
        self.global_vocab = self.matrix.global_vocab
        self.global_vocab_set = self.matrix.global_vocab_set
        self.vocab_index = self.matrix.vocab_index
        self.vocab_lookup = self.matrix.vocab_lookup
        self.vocab_size = self.matrix.vocab_size
        self.doc_counts = self.matrix.doc_counts
        self.doc_lengths = self.matrix.doc_lengths
        self.global_counts = self.matrix.global_counts
        self.slice_all_counts = self.matrix.slice_all_counts
        self.slice_all_totals = self.matrix.slice_all_totals
        self.slice_all_docs = self.matrix.slice_all_docs
        self.slice_all_analyzable_docs = self.matrix.slice_all_analyzable_docs
        self.slice_all_moment_sums = self.matrix.slice_all_moment_sums
        self.slice_all_moment_sum_sq = self.matrix.slice_all_moment_sum_sq
        self.doc_term_matrix = self.matrix.doc_term_matrix
        self.global_total = float(self.global_counts.sum())

    def counts_for_mask(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        return self.matrix.counts_for_mask(mask)

    def slice_counts_for_mask(self, mask: np.ndarray) -> dict[int, tuple[np.ndarray, float, int]]:
        return self.matrix.slice_counts_for_mask(mask)


class FeatureKLDBase:
    """Shared KLD facade for any document-feature matrix."""

    def __init__(
        self,
        *,
        matrix: DocumentFeatureMatrix,
        target_mask: Sequence[bool],
        mode: str,
        target_label: str,
        lambda_param: float = DEFAULT_LAMBDA_PARAM,
        epsilon: float = DEFAULT_EPSILON,
        min_tokens_target_slice: float = 10,
        min_tokens_field_slice: float = 10,
        min_docs_target_slice: int = 1,
        min_docs_field_slice: int = 1,
        min_docs_target_test: int = 2,
        min_docs_field_test: int = 2,
        top_k_kld_terms: Optional[int] = DEFAULT_TOP_K_KLD_TERMS,
        show_progress: bool = False,
        verbose: bool = False,
    ) -> None:
        target_mask_array = np.asarray(target_mask, dtype=bool)
        if target_mask_array.ndim != 1 or len(target_mask_array) != len(matrix.years):
            raise ValueError("target_mask must be 1-D with length matching the matrix document count")
        self.matrix = matrix
        self.mode = mode
        self.target_label = target_label
        self.lambda_param = float(lambda_param)
        self.epsilon = float(epsilon)
        self.min_tokens_target_slice = float(min_tokens_target_slice)
        self.min_tokens_field_slice = float(min_tokens_field_slice)
        self.min_docs_target_slice = int(min_docs_target_slice)
        self.min_docs_field_slice = int(min_docs_field_slice)
        self.min_docs_target_test = int(min_docs_target_test)
        self.min_docs_field_test = int(min_docs_field_test)
        self.top_k_kld_terms = top_k_kld_terms
        self.show_progress = bool(show_progress)
        self.verbose = bool(verbose)

        self.slices = self.matrix.slices
        self.target_mask = target_mask_array
        self.global_vocab = self.matrix.global_vocab
        self.global_vocab_set = self.matrix.global_vocab_set
        self.vocab_index = self.matrix.vocab_index
        self.vocab_lookup = self.matrix.vocab_lookup
        self.vocab_size = self.matrix.vocab_size
        self.doc_counts_target: dict[int, int] = {}
        self.doc_counts_field: dict[int, int] = {}

        self.core = KLDCore(
            self.matrix,
            target_mask=self.target_mask,
            lambda_param=self.lambda_param,
            epsilon=self.epsilon,
            min_tokens_target_slice=self.min_tokens_target_slice,
            min_tokens_field_slice=self.min_tokens_field_slice,
            min_docs_target_slice=self.min_docs_target_slice,
            min_docs_field_slice=self.min_docs_field_slice,
            min_docs_target_test=self.min_docs_target_test,
            min_docs_field_test=self.min_docs_field_test,
            top_k_kld_terms=self.top_k_kld_terms,
        )
        self.bg_probs = self.core.bg_probs
        self.slice_models_target = self.core.slice_models_target
        self.slice_models_field = self.core.slice_models_field
        self.slice_token_counts = self.core.slice_token_counts
        self._sync_contrib_cache = self.core._sync_contrib_cache

    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        prefix = f"[{self.mode} | {self.target_label}]"
        logger.info("%s %s", prefix, msg)

    def calculate_kld_sync(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._log("Step 1: Sync KLD start")
        start_t = time.perf_counter()
        sync, pointwise = self.core.calculate_sync()
        self._log(f"Step 1 done in {time.perf_counter() - start_t:.1f}s | slices={len(sync)}")
        return sync, pointwise

    def calculate_kld_async(self) -> pd.DataFrame:
        self._log("Step 2: Async KLD start")
        start_t = time.perf_counter()
        async_df = self.core.calculate_async()
        self._log(f"Step 2 done in {time.perf_counter() - start_t:.1f}s | pairs={len(async_df)}")
        return async_df

    def perform_welch_tests_all_pairs(self, sync_only: bool = False) -> pd.DataFrame:
        self._log(f"Step 3: Welch tests start (sync_only={sync_only}) | top_k={self.top_k_kld_terms}")
        start_t = time.perf_counter()
        welch = self.core.perform_welch_tests(sync_only=sync_only)
        self.doc_counts_target = dict(self.core.doc_counts_target)
        self.doc_counts_field = dict(self.core.doc_counts_field)
        self._log(f"Step 3 done in {time.perf_counter() - start_t:.1f}s | rows={len(welch)}")
        return welch

    def result(self, *, include_async: bool = False, run_welch: bool = True):
        """Package the computed KLD tables into a uniform ``MetricResult``."""
        from .metric_result import MetricResult

        sync, pointwise = self.calculate_kld_sync()
        async_df = self.calculate_kld_async() if include_async else None
        welch = self.perform_welch_tests_all_pairs() if run_welch else None
        return MetricResult(
            sync=sync,
            pointwise=pointwise,
            async_df=async_df,
            welch=welch,
            metadata=dict(getattr(self, "metadata", {}) or {}),
            kind="kld",
            metric=getattr(self, "METRIC", "kld"),
            target_author_uid=getattr(self, "target_author_uid", None),
            target_name=self.target_label,
            window_size=int(getattr(self, "window_size", 2)),
            config={
                "start_year": getattr(self, "start_year", None),
                "end_year": getattr(self, "end_year", None),
                "window_size": int(getattr(self, "window_size", DEFAULT_WINDOW_SIZE)),
                "skip_incomplete_slices": bool(getattr(self, "skip_incomplete_slices", True)),
                "include_async": bool(include_async),
                "run_welch": bool(run_welch),
                "lambda_param": self.lambda_param,
                "epsilon": self.epsilon,
                "top_k_kld_terms": self.top_k_kld_terms,
                "min_tokens_target_slice": self.min_tokens_target_slice,
                "min_tokens_field_slice": self.min_tokens_field_slice,
                "min_docs_target_slice": self.min_docs_target_slice,
                "min_docs_field_slice": self.min_docs_field_slice,
                "min_docs_target_test": self.min_docs_target_test,
                "min_docs_field_test": self.min_docs_field_test,
                **{
                    key: getattr(self, key)
                    for key in ("min_token_global_freq", "min_docs_global_freq", "max_vocab_size")
                    if hasattr(self, key)
                },
                **(
                    {"reference_policy": getattr(self, "policy")}
                    if hasattr(self, "policy")
                    else {}
                ),
            },
        )


class BaseKLD(FeatureKLDBase):
    """Token-frame adapter for feature-matrix KLD metrics."""

    def __init__(
        self,
        corpus: pd.DataFrame,
        target_name: str,
        *,
        target_author_uid: Optional[str] = None,
        author_col: str = "Author",
        author_id_col: str = "author_uids",
        year_col: str = "Year",
        token_col: str = "tokens",
        docid_col: str = "Bibcode",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices: bool = True,
        lambda_param: float = DEFAULT_LAMBDA_PARAM,
        epsilon: float = DEFAULT_EPSILON,
        min_token_global_freq: float = 2,
        min_docs_global_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        min_tokens_target_slice: float = 10,
        min_tokens_field_slice: float = 10,
        min_docs_target_slice: int = 1,
        min_docs_field_slice: int = 1,
        top_k_kld_terms: Optional[int] = DEFAULT_TOP_K_KLD_TERMS,
        min_docs_target_test: int = 2,
        min_docs_field_test: int = 2,
        shared_precompute: Optional[KLDPrecompute] = None,
        target_mask: Optional[Sequence[bool]] = None,
        allow_name_fallback: bool = True,
        show_progress: bool = False,
        verbose: bool = False,
    ) -> None:
        self.corpus = corpus.copy(deep=False)
        self.target_name = target_name
        self.target_author_uid = target_author_uid
        self.target_label = target_author_uid or target_name
        self.mode = getattr(self, "mode", self.__class__.__name__.replace("KLD", "").lower() or "kld")
        self.author_col = author_col
        self.author_id_col = author_id_col
        self.year_col = year_col
        self.token_col = token_col
        self.docid_col = docid_col
        self.start_year = start_year
        self.end_year = end_year
        self.window_size = int(window_size)
        self.skip_incomplete_slices = bool(skip_incomplete_slices)
        self.lambda_param = float(lambda_param)
        self.epsilon = float(epsilon)
        self.min_token_global_freq = float(min_token_global_freq)
        self.min_docs_global_freq = int(min_docs_global_freq)
        self.max_vocab_size = max_vocab_size
        self.min_tokens_target_slice = float(min_tokens_target_slice)
        self.min_tokens_field_slice = float(min_tokens_field_slice)
        self.min_docs_target_slice = int(min_docs_target_slice)
        self.min_docs_field_slice = int(min_docs_field_slice)
        self.top_k_kld_terms = top_k_kld_terms
        self.min_docs_target_test = int(min_docs_target_test)
        self.min_docs_field_test = int(min_docs_field_test)
        self.show_progress = bool(show_progress)
        self.verbose = bool(verbose)

        if shared_precompute is None:
            # Only needed when this object builds the matrix from scratch. With a
            # shared precompute the matrix is already built from the prepared corpus,
            # so re-coercing the full corpus on every target is wasted work (the
            # corpus/splits below are not read in the consolidated run path).
            self.corpus[self.year_col] = self.corpus[self.year_col].astype(int)
            self.corpus[self.token_col] = self.corpus[self.token_col].apply(
                lambda value: value if _is_token_container(value) else []
            )
        self.target_corpus, self.field_corpus = apply_target_field_split(
            self.corpus,
            target_mask=target_mask,
            target_name=self.target_name,
            target_author_uid=self.target_author_uid,
            author_col=self.author_col,
            author_id_col=self.author_id_col,
            allow_name_fallback=allow_name_fallback,
        )

        self.precompute = shared_precompute or KLDPrecompute(
            self.corpus,
            year_col=self.year_col,
            token_col=self.token_col,
            start_year=self.start_year,
            end_year=self.end_year,
            window_size=self.window_size,
            skip_incomplete_slices=self.skip_incomplete_slices,
            min_token_global_freq=self.min_token_global_freq,
            min_docs_global_freq=self.min_docs_global_freq,
            max_vocab_size=self.max_vocab_size,
        )
        self.slices = self.precompute.slices
        self.target_mask = self.corpus["__is_target__"].to_numpy(dtype=bool)

        super().__init__(
            matrix=self.precompute.matrix,
            target_mask=self.target_mask,
            mode=self.mode,
            target_label=self.target_label,
            lambda_param=self.lambda_param,
            epsilon=self.epsilon,
            min_tokens_target_slice=self.min_tokens_target_slice,
            min_tokens_field_slice=self.min_tokens_field_slice,
            min_docs_target_slice=self.min_docs_target_slice,
            min_docs_field_slice=self.min_docs_field_slice,
            min_docs_target_test=self.min_docs_target_test,
            min_docs_field_test=self.min_docs_field_test,
            top_k_kld_terms=self.top_k_kld_terms,
            show_progress=self.show_progress,
            verbose=self.verbose,
        )


class VocabularyKLD(BaseKLD):
    """KLD analysis for bag-of-words vocabulary."""

    METRIC = "own_vocab"

    def __init__(self, corpus: pd.DataFrame, target_name: str, **kwargs) -> None:
        self.mode = "vocab"
        super().__init__(corpus, target_name, **kwargs)
