"""
KDE-based density metrics (slice-based, KLD-compatible).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from .contract import apply_target_field_split, resolve_embedding_columns
from .defaults import DEFAULT_DENSITY_EMBEDDING_COLS, DEFAULT_WINDOW_SIZE
from .kld_core import create_slices_from_years
from .stats_utils import _async_min_leadlag, _level_slope


def scott_bandwidth(n_samples: int, dim: int = 2) -> float:
    """Scott's Rule h = n^{-1/(d+4)}."""
    if n_samples <= 0:
        return 1.0
    return n_samples ** (-1.0 / (dim + 4))


class DensityPrecompute:
    """Corpus-level density preparation shared across targets.

    Everything here depends only on the corpus and run parameters, not on the
    target: the prepared corpus (year as int, embedding columns coerced to
    float), the standardization center/scale, the time slices, and the Scott
    bandwidth. Build it once and pass it to every :class:`KDEDensity` via
    ``shared_precompute=`` to avoid recomputing it per target.
    """

    def __init__(
        self,
        corpus: pd.DataFrame,
        *,
        year_col: str = "Year",
        embedding_cols: Sequence[str] = DEFAULT_DENSITY_EMBEDDING_COLS,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices: bool = True,
        bandwidth: Optional[float] = None,
        standardize: bool = True,
    ) -> None:
        prepared = corpus.copy()
        self.year_col = year_col
        self.standardize = bool(standardize)
        self.embedding_cols = resolve_embedding_columns(prepared, embedding_cols)
        prepared[year_col] = prepared[year_col].astype(int)

        coords_df = prepared.loc[:, self.embedding_cols].apply(pd.to_numeric, errors="coerce")
        coords = coords_df.to_numpy(dtype=float)
        if not np.isfinite(coords).all():
            raise ValueError("embedding columns must be finite numeric values")
        prepared.loc[:, self.embedding_cols] = coords_df.astype(float)
        self.prepared_corpus = prepared

        if len(coords) == 0:
            dim = len(self.embedding_cols)
            self.embedding_center_ = np.zeros(dim, dtype=float)
            self.embedding_scale_ = np.ones(dim, dtype=float)
        elif self.standardize:
            center = coords.mean(axis=0)
            scale = coords.std(axis=0, ddof=0)
            scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
            self.embedding_center_ = center.astype(float)
            self.embedding_scale_ = scale.astype(float)
        else:
            self.embedding_center_ = np.zeros(coords.shape[1], dtype=float)
            self.embedding_scale_ = np.ones(coords.shape[1], dtype=float)

        years = prepared[year_col].to_numpy()
        self.slices = create_slices_from_years(
            years,
            start_year=start_year,
            end_year=end_year,
            window_size=window_size,
            skip_incomplete_slices=skip_incomplete_slices,
        )

        if bandwidth is None:
            bandwidth = scott_bandwidth(len(prepared), dim=len(self.embedding_cols))
        self.bandwidth = float(bandwidth)


class KDEDensity:
    """Slice-based KDE density metric with explicit Target-vs-Field separation."""

    def __init__(
        self,
        corpus: pd.DataFrame,
        target_name: str,
        *,
        target_author_uid: Optional[str] = None,
        author_col: str = "Author",
        author_id_col: str = "author_uids",
        year_col: str = "Year",
        docid_col: str = "Bibcode",
        embedding_cols: Sequence[str] = DEFAULT_DENSITY_EMBEDDING_COLS,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        skip_incomplete_slices: bool = True,
        bandwidth: Optional[float] = None,
        kernel: str = "gaussian",
        min_docs_target_slice: int = 1,
        min_docs_field_slice: int = 1,
        allow_name_fallback: bool = True,
        standardize: bool = True,
        shared_precompute: Optional["DensityPrecompute"] = None,
        target_mask: Optional[Sequence[bool]] = None,
    ) -> None:
        if shared_precompute is None:
            shared_precompute = DensityPrecompute(
                corpus,
                year_col=year_col,
                embedding_cols=embedding_cols,
                start_year=start_year,
                end_year=end_year,
                window_size=window_size,
                skip_incomplete_slices=skip_incomplete_slices,
                bandwidth=bandwidth,
                standardize=standardize,
            )
        self._precompute = shared_precompute
        self.target_name = target_name
        self.target_author_uid = target_author_uid
        self.target_label = target_author_uid or target_name
        self.author_col = author_col
        self.author_id_col = author_id_col
        self.year_col = year_col
        self.docid_col = docid_col
        self.start_year = start_year
        self.end_year = end_year
        self.window_size = int(window_size)
        self.skip_incomplete_slices = bool(skip_incomplete_slices)
        self.kernel = kernel
        self.min_docs_target_slice = int(min_docs_target_slice)
        self.min_docs_field_slice = int(min_docs_field_slice)
        # Corpus-level results come from the (possibly shared) precompute.
        self.embedding_cols = shared_precompute.embedding_cols
        self.standardize = shared_precompute.standardize
        self.embedding_center_ = shared_precompute.embedding_center_
        self.embedding_scale_ = shared_precompute.embedding_scale_
        self.slices = shared_precompute.slices
        self.bandwidth = shared_precompute.bandwidth
        self._field_kde_cache: dict[int, tuple[KernelDensity, int]] = {}

        # Per-target view over the already-prepared corpus (no re-coercion/standardisation).
        self.corpus = shared_precompute.prepared_corpus.copy(deep=False)
        self.target_corpus, self.field_corpus = apply_target_field_split(
            self.corpus,
            target_mask=target_mask,
            target_name=self.target_name,
            target_author_uid=self.target_author_uid,
            author_col=self.author_col,
            author_id_col=self.author_id_col,
            allow_name_fallback=allow_name_fallback,
        )

    def _coords(self, df: pd.DataFrame) -> np.ndarray:
        coords = df[self.embedding_cols].to_numpy(dtype=float)
        return (coords - self.embedding_center_) / self.embedding_scale_

    def _fit_kde(self, coords: np.ndarray) -> KernelDensity:
        kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        kde.fit(coords)
        return kde

    def _slice_frames(self, start: int, end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_t = self.target_corpus[
            (self.target_corpus[self.year_col] >= start) & (self.target_corpus[self.year_col] <= end)
        ]
        df_f = self.field_corpus[
            (self.field_corpus[self.year_col] >= start) & (self.field_corpus[self.year_col] <= end)
        ]
        return df_t, df_f

    def _field_kde(self, start: int, end: int) -> tuple[KernelDensity, int] | None:
        label = int(end)
        cached = self._field_kde_cache.get(label)
        if cached is not None:
            return cached
        _, df_f = self._slice_frames(start, end)
        n_f = int(len(df_f))
        if n_f < self.min_docs_field_slice:
            return None
        coords = self._coords(df_f)
        if len(coords) == 0:
            return None
        fitted = (self._fit_kde(coords), n_f)
        self._field_kde_cache[label] = fitted
        return fitted

    def calculate_density_sync(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        slice_rows: list[dict] = []
        point_rows: list[dict] = []

        for start, end in self.slices:
            label = int(end)
            df_t, df_f = self._slice_frames(start, end)
            n_t, n_f = int(len(df_t)), int(len(df_f))
            if n_t < self.min_docs_target_slice or n_f < self.min_docs_field_slice:
                continue

            target_coords = self._coords(df_t)
            field_model = self._field_kde(start, end)
            if field_model is None or len(target_coords) == 0:
                continue

            kde, n_f = field_model
            neg_log_density = -kde.score_samples(target_coords)
            slice_rows.append(
                {
                    "slice": label,
                    "density_neglog_median": float(np.median(neg_log_density)),
                    "target_docs": n_t,
                    "field_docs": n_f,
                }
            )

            years = df_t[self.year_col].to_numpy(dtype=int, copy=False)
            docids = (
                df_t[self.docid_col].to_numpy(copy=False)
                if self.docid_col in df_t.columns
                else np.full(len(df_t), None, dtype=object)
            )
            for year, docid, val in zip(years, docids, neg_log_density):
                point_rows.append(
                    {
                        "slice": label,
                        "year": int(year),
                        self.docid_col: docid,
                        "density_neglog": float(val),
                    }
                )

        return pd.DataFrame(slice_rows), pd.DataFrame(point_rows)

    def calculate_density_async(self) -> pd.DataFrame:
        labels = [int(end) for _, end in self.slices]
        field_kdes: dict[int, KernelDensity] = {}
        field_docs: dict[int, int] = {}

        for start, end in self.slices:
            f_label = int(end)
            field_model = self._field_kde(start, end)
            if field_model is None:
                continue
            field_kdes[f_label], n_f = field_model
            field_docs[f_label] = n_f

        rows: list[dict] = []
        for start, end in self.slices:
            t_label = int(end)
            df_t, _ = self._slice_frames(start, end)
            n_t = int(len(df_t))
            if n_t < self.min_docs_target_slice:
                continue
            target_coords = self._coords(df_t)
            if len(target_coords) == 0:
                continue
            for f_label in labels:
                kde = field_kdes.get(int(f_label))
                if kde is None:
                    continue
                neg_log_density = -kde.score_samples(target_coords)
                rows.append(
                    {
                        "target_slice": t_label,
                        "field_slice": int(f_label),
                        "time_diff": int(f_label) - t_label,
                        "density_neglog_median": float(np.median(neg_log_density)),
                        "target_docs": n_t,
                        "field_docs": int(field_docs.get(int(f_label), 0)),
                    }
                )

        return pd.DataFrame(rows)

    def result(self, *, include_async: bool = False):
        """Package the computed density tables into a uniform ``MetricResult``."""
        from .metric_result import MetricResult

        sync, pointwise = self.calculate_density_sync()
        async_df = self.calculate_density_async() if include_async else None
        return MetricResult(
            sync=sync,
            pointwise=pointwise,
            async_df=async_df,
            welch=None,
            metadata={},
            kind="density",
            metric="density",
            target_author_uid=self.target_author_uid,
            target_name=self.target_label,
            window_size=int(self.window_size),
            config={
                "start_year": self.start_year,
                "end_year": self.end_year,
                "window_size": self.window_size,
                "skip_incomplete_slices": self.skip_incomplete_slices,
                "include_async": bool(include_async),
                "embedding_cols": list(self.embedding_cols),
                "bandwidth": float(self.bandwidth),
                "standardize": bool(self.standardize),
                "min_docs_target_slice": self.min_docs_target_slice,
                "min_docs_field_slice": self.min_docs_field_slice,
            },
        )


def summarize_density_sync(df_sync: pd.DataFrame) -> pd.Series:
    """Return level+slope summaries for sync density."""
    if df_sync.empty:
        return pd.Series({"density_neglog_level": np.nan, "density_neglog_slope": np.nan})
    level, slope = _level_slope(df_sync["slice"].to_numpy(), df_sync["density_neglog_median"].to_numpy())
    return pd.Series({"density_neglog_level": level, "density_neglog_slope": slope})


def summarize_density_async(df_async: pd.DataFrame) -> pd.Series:
    """Async summary (analogous to KLD): mean minimum and mean lead/lag."""
    if df_async.empty:
        return pd.Series({"density_async_min": np.nan, "density_async_leadlag": np.nan})
    min_mean, leadlag_mean = _async_min_leadlag(df_async, "density_neglog_median")
    return pd.Series({"density_async_min": min_mean, "density_async_leadlag": leadlag_mean})
