"""Plotly diagnostic figures (author token ridgeplots, referenced-term lines, and
Citation-Identity robustness scatter/bar), rendered with the shared design system
in ``style.py``. These replace the former ad-hoc matplotlib plots so the whole
project has a single plotting system; output is PDF via ``style.save_figure``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from .style import (
    FONT_COLOR,
    GRID_COLOR,
    PAPER_BG_COLOR,
    PLOT_BG_COLOR,
    PRIMARY_COLOR,
    RIDGE_COLORSCALE,
    apply_standard_layout,
)


def _ridge_density(years: np.ndarray, weights: np.ndarray, grid: np.ndarray, *, bandwidth: float) -> np.ndarray:
    """Gaussian KDE of (year, weight) pairs over a grid, normalised to a max of 1."""
    mask = np.isfinite(years) & np.isfinite(weights) & (weights > 0)
    years = years[mask]
    weights = weights[mask]
    if years.size == 0 or weights.sum() <= 0:
        return np.zeros_like(grid, dtype=float)
    sigma = max(float(bandwidth), 1e-6)
    diff = (grid[:, None] - years[None, :]) / sigma
    density = (np.exp(-0.5 * diff * diff) * weights[None, :]).sum(axis=1) / weights.sum()
    peak = float(np.nanmax(density)) if density.size else 0.0
    return density / peak if peak > 0 else np.zeros_like(grid, dtype=float)


def plot_token_usage_ridgeplot(
    frame: pd.DataFrame,
    *,
    title: str,
    bandwidth: float = 2.5,
    width: int = 900,
) -> go.Figure:
    """Author token-usage ridgeline. ``frame`` columns: ``year_bin``, ``term``,
    ``relative_frequency``, ``token_order``."""
    fig = go.Figure()
    terms = (
        frame.drop_duplicates("term").sort_values("token_order")["term"].tolist()
        if not frame.empty
        else []
    )
    n = len(terms)
    height = max(420, int(34 * max(n, 1) + 150))
    title_x = 160 / width
    if n == 0:
        apply_standard_layout(fig, title, height=height, width=width, title_x=title_x)
        fig.update_layout(font=dict(color=FONT_COLOR), plot_bgcolor=PLOT_BG_COLOR, paper_bgcolor=PAPER_BG_COLOR)
        fig.add_annotation(
            text="No valid token bins after threshold",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font=dict(color=FONT_COLOR),
        )
        return fig

    x_min = float(frame["year_bin"].min())
    x_max = float(frame["year_bin"].max())
    if x_max <= x_min:
        x_max = x_min + 1.0
    x_grid = np.linspace(x_min, x_max, 180)
    fills = sample_colorscale(RIDGE_COLORSCALE, [i / max(1, n - 1) for i in range(n)])
    yticks: list[float] = []
    for idx, term in enumerate(terms):
        y_base = float(n - 1 - idx)
        term_frame = frame[frame["term"].eq(term)].sort_values("year_bin")
        density = _ridge_density(
            term_frame["year_bin"].to_numpy(dtype=float),
            term_frame["relative_frequency"].to_numpy(dtype=float),
            x_grid,
            bandwidth=bandwidth,
        )
        y_top = y_base + density * 0.82
        poly_x = np.concatenate([x_grid, x_grid[::-1]])
        poly_y = np.concatenate([y_top, np.full(x_grid.shape, y_base)])
        fig.add_trace(
            go.Scatter(
                x=poly_x,
                y=poly_y,
                fill="toself",
                fillcolor=fills[idx],
                mode="lines",
                line=dict(color="#303030", width=0.6),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        yticks.append(y_base)

    apply_standard_layout(fig, title, height=height, width=width, title_x=title_x)
    fig.update_layout(
        margin=dict(l=160, r=40, t=80, b=70),
        font=dict(color=FONT_COLOR),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
        showlegend=False,
    )
    # Tight x-range -> no empty background padding on the sides.
    fig.update_xaxes(range=[x_min, x_max], title_text="Year", gridcolor=GRID_COLOR, showgrid=True, zeroline=False)
    fig.update_yaxes(tickvals=yticks, ticktext=terms, range=[-0.75, n - 0.05], showgrid=False, zeroline=False, title_text="Tokens")
    return fig


def plot_ref_term_development(
    data: pd.DataFrame,
    *,
    title: str,
    top_terms: int = 12,
    width: int = 1000,
    height: int = 520,
) -> go.Figure:
    """Referenced-term development lines. ``data`` columns: ``term``, ``slice``,
    ``target_share``, ``target_count``."""
    fig = go.Figure()
    title_x = 90 / width
    if data.empty:
        apply_standard_layout(fig, title, height=height, width=width, title_x=title_x)
        fig.update_layout(font=dict(color=FONT_COLOR), plot_bgcolor=PLOT_BG_COLOR, paper_bgcolor=PAPER_BG_COLOR)
        return fig
    totals = data.groupby("term")["target_count"].sum().sort_values(ascending=False)
    terms = totals.head(top_terms).index.tolist()
    view = data[data["term"].isin(terms)]
    for term in terms:
        term_rows = view[view["term"].eq(term)].sort_values("slice")
        fig.add_trace(
            go.Scatter(
                x=term_rows["slice"],
                y=term_rows["target_share"],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=1.6),
                name=str(term),
            )
        )
    apply_standard_layout(fig, title, height=height, width=width, title_x=title_x, legend_x=0.0, legend_y=1.0)
    fig.update_layout(
        font=dict(color=FONT_COLOR),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
        legend=dict(orientation="h", y=1.0),
    )
    fig.update_xaxes(title_text="Slice (end year)", gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(title_text="Target share", gridcolor=GRID_COLOR, zeroline=False)
    return fig


def plot_slope_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str = "main clean slope",
    y_label: str = "variant slope",
    width: int = 760,
    height: int = 560,
) -> go.Figure:
    sub = df.dropna(subset=[x_col, y_col])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=sub[x_col], y=sub[y_col], mode="markers", marker=dict(size=9, color=PRIMARY_COLOR), showlegend=False)
    )
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.25)", width=1))
    fig.add_vline(x=0, line=dict(color="rgba(0,0,0,0.25)", width=1))
    apply_standard_layout(fig, title, height=height, width=width, title_x=80 / width)
    fig.update_layout(font=dict(color=FONT_COLOR), plot_bgcolor=PLOT_BG_COLOR, paper_bgcolor=PAPER_BG_COLOR)
    fig.update_xaxes(title_text=x_label, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(title_text=y_label, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def plot_slope_delta_bar(
    df: pd.DataFrame,
    *,
    value_col: str,
    label_col: str,
    title: str,
    width: int = 900,
) -> go.Figure:
    work = df[[label_col, value_col]].copy().dropna().sort_values(value_col)
    height = max(300, int(26 * len(work) + 150))
    values = work[value_col].astype(float).tolist()
    labels = [str(v)[:34] for v in work[label_col]]
    colors = ["#9d3b3b" if v < 0 else PRIMARY_COLOR for v in values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors), showlegend=False))
    fig.add_vline(x=0, line=dict(color="rgba(0,0,0,0.4)", width=1))
    apply_standard_layout(fig, title, height=height, width=width, title_x=240 / width)
    fig.update_layout(
        margin=dict(l=240, r=40, t=80, b=60),
        font=dict(color=FONT_COLOR),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(automargin=True)
    return fig
