"""Statistical utilities shared by package metrics and plotting."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd


MULTIPLE_TESTING_METHODS = {
    "none",
    "bonferroni",
    "holm",
    "fdr_bh",
    "fdr_by",
}


def _validate_pvalues(pvalues: np.ndarray) -> None:
    finite = np.isfinite(pvalues)
    if not np.all((0.0 <= pvalues[finite]) & (pvalues[finite] <= 1.0)):
        raise ValueError("p-values must be in [0, 1]")


def _as_array(pvalues: Union[pd.Series, Sequence[float], np.ndarray]) -> np.ndarray:
    if isinstance(pvalues, pd.Series):
        return pvalues.to_numpy(dtype=float)
    if isinstance(pvalues, Iterable):
        return np.asarray(list(pvalues), dtype=float)
    return np.asarray(pvalues, dtype=float)


def adjust_pvalues(
    pvalues: Union[pd.Series, Sequence[float], np.ndarray],
    *,
    method: str = "fdr_bh",
) -> Union[pd.Series, np.ndarray]:
    """Adjust p-values for multiple testing."""
    if method not in MULTIPLE_TESTING_METHODS:
        raise ValueError(f"method must be one of {sorted(MULTIPLE_TESTING_METHODS)}")

    is_series = isinstance(pvalues, pd.Series)
    index = pvalues.index if is_series else None
    arr = _as_array(pvalues)
    _validate_pvalues(arr)

    if method == "none":
        out = arr.copy()
        return pd.Series(out, index=index) if is_series else out

    out = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    p = arr[mask]
    m = int(p.size)
    if m == 0:
        return pd.Series(out, index=index) if is_series else out

    if method == "bonferroni":
        adj = np.minimum(p * m, 1.0)
    elif method == "holm":
        order = np.argsort(p)
        p_sorted = p[order]
        adj_sorted = np.maximum.accumulate(p_sorted * (m - np.arange(m)))
        adj_sorted = np.minimum(adj_sorted, 1.0)
        adj = np.empty_like(adj_sorted)
        adj[order] = adj_sorted
    elif method == "fdr_bh":
        order = np.argsort(p)
        p_sorted = p[order]
        q_sorted = p_sorted * m / np.arange(1, m + 1)
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.minimum(q_sorted, 1.0)
        adj = np.empty_like(q_sorted)
        adj[order] = q_sorted
    elif method == "fdr_by":
        order = np.argsort(p)
        p_sorted = p[order]
        c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
        q_sorted = p_sorted * m * c_m / np.arange(1, m + 1)
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.minimum(q_sorted, 1.0)
        adj = np.empty_like(q_sorted)
        adj[order] = q_sorted
    else:
        raise RuntimeError("unreachable")

    out[mask] = adj
    return pd.Series(out, index=index) if is_series else out


def add_pvalue_adjustments(
    df: pd.DataFrame,
    *,
    p_col: str = "pvalue",
    method: str = "fdr_bh",
    group_cols: Optional[list[str]] = None,
    out_col: str = "p_adj",
) -> pd.DataFrame:
    """Add a p-value adjustment column to a DataFrame."""
    if p_col not in df.columns:
        raise KeyError(f"missing column: {p_col}")
    out = df.copy()
    out[out_col] = np.nan

    if method == "none":
        out[out_col] = out[p_col].astype(float)
        return out

    if not group_cols:
        out[out_col] = adjust_pvalues(out[p_col].astype(float), method=method)
        return out

    group_key = group_cols[0] if len(group_cols) == 1 else group_cols
    for _, idx in out.groupby(group_key, sort=False).groups.items():
        out.loc[idx, out_col] = adjust_pvalues(
            out.loc[idx, p_col].astype(float),
            method=method,
        )
    return out


def _level_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Median level + OLS slope of ``y`` over ``x`` (slope NaN when < 2 points).

    Shared core of the sync ``summarize_*`` functions across the metrics.
    """
    level = float(np.median(y))
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else np.nan
    return level, slope


def _async_min_leadlag(df: pd.DataFrame, value_col: str) -> tuple[float, float]:
    """Mean per-target-slice minimum of ``value_col`` and its mean lead/lag.

    Shared core of the async ``summarize_*`` functions. Assumes a non-empty frame
    with ``target_slice``/``time_diff`` columns; returns ``(nan, nan)`` if no minima.
    """
    mins = df.loc[df.groupby("target_slice")[value_col].idxmin()]
    if mins.empty:
        return np.nan, np.nan
    return float(mins[value_col].mean()), float(mins["time_diff"].mean())
