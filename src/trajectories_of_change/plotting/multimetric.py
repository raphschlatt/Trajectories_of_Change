"""Plotting and table helpers for multimetric comparison outputs."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .style import (
    FONT_COLOR,
    BASE_PLOT_WIDTH,
    BASE_PANEL_HEIGHT,
    BASE_TWO_ROW_HEIGHT,
    PRIMARY_COLOR,
    apply_standard_layout,
    make_colorbar,
    make_colorbar_for_axis,
)


DEFAULT_PLOT_WIDTH = BASE_PLOT_WIDTH
DEFAULT_SYNC_PLOT_HEIGHT = BASE_PANEL_HEIGHT
DEFAULT_2ROW_HEIGHT = BASE_TWO_ROW_HEIGHT


def default_label_map() -> Dict[str, str]:
    """Human-friendly plot labels for common multimetric columns."""
    return {
        # Density (KDE on standardized map coordinates by default)
        "density_neglog_level": "Field density (−log KDE)",
        "density_neglog_slope": "Field-density slope (−log KDE/yr)",
        # Vocabulary KLD
        "vocab_kld_all_level": "Vocab KLD (bits)",
        "vocab_kld_all_slope": "Vocab KLD slope (bits/yr)",
        "ref_vocab_kld_all_level": "Referenced Vocab KLD (bits)",
        "ref_vocab_kld_all_slope": "Referenced Vocab KLD slope (bits/yr)",
        # Co-citation KLD
        "cocit_kld_all_level": "Co-cit KLD (bits)",
        "cocit_kld_all_slope": "Co-cit KLD slope (bits/yr)",
    }


def _label(col: str, label_map: Mapping[str, str]) -> str:
    return label_map.get(col, col)


def _axis_ref(prefix: str, index: int) -> str:
    return prefix if index == 1 else f"{prefix}{index}"


def _linear_slope(x: pd.Series, y: pd.Series) -> float:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(values) < 2 or values["x"].nunique() < 2:
        return 0.0
    return float(np.polyfit(values["x"].to_numpy(dtype=float), values["y"].to_numpy(dtype=float), 1)[0])


def _metric_range(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return (0.0, 1.0)
    low = float(numeric.min())
    high = float(numeric.max())
    if high == low:
        pad = abs(high) * 0.08 or 1.0
    else:
        pad = (high - low) * 0.08
    return (low - pad, high + pad)


def plot_sync_trend_small_multiples(
    df: pd.DataFrame,
    *,
    metric_order: Sequence[str],
    author_col: str = "author_display_name",
    metric_col: str = "metric",
    slice_col: str = "slice",
    value_col: str = "value",
    author_order: Sequence[str] | None = None,
    title: str = "Synchronous Trajectories: All-Term Layers",
    width: int = DEFAULT_PLOT_WIDTH,
    height: int | None = None,
    sort_by_trend_profile: bool = True,
    slope_zero_eps: float = 1e-12,
    show_slope_labels: bool = True,
    color_points: bool = True,
) -> go.Figure:
    """Plot compact sync trajectories for several authors and metrics.

    This is an overview companion to the large legacy sync dashboards. It shows
    the all-term synchronous value trajectory and its linear trend per
    author-metric cell; it does not show Welch layers, pointwise terms, or
    colorbars.
    """

    if not metric_order:
        raise ValueError("metric_order must not be empty")
    needed = [author_col, metric_col, slice_col, value_col]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for sync trend small multiples: {missing}")

    sub = df[needed].copy()
    sub[author_col] = sub[author_col].astype(str)
    sub[metric_col] = sub[metric_col].astype(str)
    sub[slice_col] = pd.to_numeric(sub[slice_col], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[slice_col, value_col])
    sub = sub[sub[metric_col].isin(metric_order)]
    if sub.empty:
        raise ValueError("No finite sync trajectory values to plot")

    metrics = list(metric_order)
    authors = list(author_order) if author_order is not None else list(dict.fromkeys(sub[author_col]))
    authors = [author for author in authors if author in set(sub[author_col])]
    if not authors:
        raise ValueError("No requested authors have sync trajectory values")

    slopes: dict[tuple[str, str], float] = {}
    for author in authors:
        for metric in metrics:
            group = sub[(sub[author_col] == author) & (sub[metric_col] == metric)]
            slopes[(author, metric)] = _linear_slope(group[slice_col], group[value_col])

    if sort_by_trend_profile:
        def sort_key(author: str) -> tuple[float, float, str]:
            signs = []
            strengths = []
            for metric in metrics:
                slope = slopes[(author, metric)]
                sign = 0.0 if abs(slope) <= slope_zero_eps else float(np.sign(slope))
                signs.append(sign)
                strengths.append(slope)
            return (-sum(signs), -sum(strengths), author)

        authors = sorted(authors, key=sort_key)

    n_rows = len(authors)
    n_cols = len(metrics)
    plot_height = height or max(900, 170 + 118 * n_rows)
    vertical_spacing = min(0.025, 0.18 / max(n_rows, 1))
    # Make the horizontal panel gap the same *pixel* size as the vertical one.
    # Spacing fractions are relative to the plot area, so convert via the final
    # plot-area aspect ratio (margins applied in _normalize_sync_trend_layout:
    # l=195, r=45, t=165, b=74).
    plot_area_h = max(plot_height - 165 - 74, 1)
    plot_area_w = max(width - 195 - 45, 1)
    horizontal_spacing = min(0.035, vertical_spacing * plot_area_h / plot_area_w)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=False,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    x_min = float(sub[slice_col].min())
    x_max = float(sub[slice_col].max())
    y_ranges = {metric: _metric_range(sub.loc[sub[metric_col] == metric, value_col]) for metric in metrics}

    for row_idx, author in enumerate(authors, start=1):
        for col_idx, metric in enumerate(metrics, start=1):
            cell = sub[(sub[author_col] == author) & (sub[metric_col] == metric)].sort_values(slice_col)
            axis_index = (row_idx - 1) * n_cols + col_idx
            xref = _axis_ref("x", axis_index)
            yref = _axis_ref("y", axis_index)

            fig.add_trace(
                go.Scatter(
                    x=cell[slice_col],
                    y=cell[value_col],
                    mode="lines+markers",
                    line=dict(color="rgba(120,130,140,0.55)", width=1.35),
                    marker=(
                        dict(
                            size=5,
                            color=cell[value_col],
                            colorscale="turbo",
                            cmin=y_ranges[metric][0],
                            cmax=y_ranges[metric][1],
                            showscale=False,
                            line=dict(width=0.25, color="rgba(45,55,65,0.35)"),
                        )
                        if color_points
                        else dict(size=5, color="rgba(80,90,100,0.65)")
                    ),
                    name="Sync value",
                    legendgroup="sync",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{author}</b><br>{metric}<br>"
                        + "Slice: %{x}<br>Value: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=col_idx,
            )

            slope = slopes[(author, metric)]
            intercept = 0.0
            if len(cell) >= 2 and cell[slice_col].nunique() >= 2:
                slope, intercept = np.polyfit(cell[slice_col].to_numpy(dtype=float), cell[value_col].to_numpy(dtype=float), 1)
                trend_x = np.array([float(cell[slice_col].min()), float(cell[slice_col].max())])
                trend_y = slope * trend_x + intercept
            else:
                trend_x = cell[slice_col].to_numpy(dtype=float)
                trend_y = cell[value_col].to_numpy(dtype=float)

            fig.add_trace(
                go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode="lines",
                    line=dict(color=PRIMARY_COLOR, width=2.8),
                    name="Trend",
                    legendgroup="trend",
                    showlegend=False,
                    hovertemplate=f"<b>{author}</b><br>{metric}<br>Slope: {float(slope):+.5f}/yr<extra></extra>",
                ),
                row=row_idx,
                col=col_idx,
            )

            fig.update_xaxes(
                range=[x_min, x_max],
                showticklabels=row_idx == n_rows,
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.8)",
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                range=y_ranges[metric],
                showticklabels=False,
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.8)",
                row=row_idx,
                col=col_idx,
            )
            if show_slope_labels:
                fig.add_annotation(
                    x=x_max,
                    y=y_ranges[metric][1],
                    xref=xref,
                    yref=yref,
                    text=f"{float(slopes[(author, metric)]):+.3f}/yr",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(size=9, color="rgba(35,55,85,0.75)"),
                )

    apply_standard_layout(fig, "", height=plot_height, width=width)
    fig.update_layout(
        margin=dict(l=195, r=45, t=165, b=74),
        font=dict(size=12, color=FONT_COLOR),
        showlegend=True,
        legend=dict(
            orientation="h",
            traceorder="normal",
            x=0.0,
            y=1.037,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color=FONT_COLOR),
        ),
    )
    # Legend samples that *show* the line styles (like the author dashboards),
    # instead of describing them in text.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines+markers",
            line=dict(color="rgba(120,130,140,0.85)", width=1.35),
            marker=dict(size=7, color="rgba(95,105,115,0.9)", line=dict(width=0.3, color="rgba(45,55,65,0.5)")),
            name="Sync value",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=PRIMARY_COLOR, width=2.8),
            name="Trend",
            showlegend=True,
        )
    )
    fig.add_annotation(
        x=0.0,
        y=1.04,
        xref="paper",
        yref="paper",
        text=title,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=17, color=FONT_COLOR),
    )

    for col_idx, metric in enumerate(metrics, start=1):
        axis_index = col_idx
        domain = fig.layout[_axis_ref("xaxis", axis_index)].domain
        fig.add_annotation(
            x=sum(domain) / 2,
            y=1.006,
            xref="paper",
            yref="paper",
            text=metric,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=14, color=FONT_COLOR),
        )
    for row_idx, author in enumerate(authors, start=1):
        axis_index = (row_idx - 1) * n_cols + 1
        domain = fig.layout[_axis_ref("yaxis", axis_index)].domain
        fig.add_annotation(
            x=-0.008,
            y=sum(domain) / 2,
            xref="paper",
            yref="paper",
            text=author,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            align="right",
            font=dict(size=12, color=FONT_COLOR),
        )
    fig.add_annotation(
        x=0.5,
        y=-0.045,
        xref="paper",
        yref="paper",
        text="Timeslice (End Year)",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=14, color=FONT_COLOR),
    )
    return fig


def plot_slope_agreement(
    df: pd.DataFrame,
    *,
    author_col: str = "author",
    density_slope_col: str = "density_neglog_slope",
    vocab_slope_col: str = "vocab_kld_all_slope",
    cocit_slope_col: str = "cocit_kld_all_slope",
    density_cov_col: str = "density_slices_sync",
    vocab_cov_col: str = "vocab_slices_kld",
    cocit_cov_col: str = "cocit_slices_kld",
    label_map: Optional[Mapping[str, str]] = None,
    width: int = DEFAULT_PLOT_WIDTH,
    height: int = DEFAULT_2ROW_HEIGHT,
    title: str = "Slope Agreement (All-Term Metrics)",
) -> go.Figure:
    """
    2D scatter: x=density slope, y=vocab slope, color=cocit slope.

    This is designed for:
    - interactive exploration (hover shows all slopes + coverage)
    - publication export (axes + colorbar + legend)
    """
    needed = [author_col, density_slope_col, vocab_slope_col, cocit_slope_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for slope plot: {missing}")

    sub = df[[c for c in [author_col, density_slope_col, vocab_slope_col, cocit_slope_col, density_cov_col, vocab_cov_col, cocit_cov_col] if c in df.columns]].copy()
    sub = sub.dropna(subset=[density_slope_col, vocab_slope_col, cocit_slope_col])
    sub[author_col] = sub[author_col].astype(str)

    label_map = dict(label_map or default_label_map())
    x_label = _label(density_slope_col, label_map)
    y_label = _label(vocab_slope_col, label_map)
    c_label = _label(cocit_slope_col, label_map)

    custom_cols = [author_col]
    if density_cov_col in sub.columns:
        custom_cols.append(density_cov_col)
    if vocab_cov_col in sub.columns:
        custom_cols.append(vocab_cov_col)
    if cocit_cov_col in sub.columns:
        custom_cols.append(cocit_cov_col)
    customdata = sub[custom_cols].to_numpy()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=sub[density_slope_col],
            y=sub[vocab_slope_col],
            mode="markers",
            marker=dict(
                size=10,
                opacity=0.9,
                color=sub[cocit_slope_col],
                colorscale="turbo",
                showscale=True,
                colorbar=make_colorbar(c_label, y=0.5, length=1.0),
                line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + f"{x_label}: %{{x:.4f}}<br>"
                + f"{y_label}: %{{y:.4f}}<br>"
                + f"{c_label}: %{{marker.color:.4f}}<br>"
                + (
                    (f"{density_cov_col}: %{{customdata[{custom_cols.index(density_cov_col)}]}}<br>" if density_cov_col in custom_cols else "")
                    + (f"{vocab_cov_col}: %{{customdata[{custom_cols.index(vocab_cov_col)}]}}<br>" if vocab_cov_col in custom_cols else "")
                    + (f"{cocit_cov_col}: %{{customdata[{custom_cols.index(cocit_cov_col)}]}}<br>" if cocit_cov_col in custom_cols else "")
                )
                + "<extra></extra>"
            ),
            name="Authors",
            showlegend=False,
        )
    )

    # Zero reference lines (helpful for sign agreement in publications).
    x_min = float(min(sub[density_slope_col].min(), 0))
    x_max = float(max(sub[density_slope_col].max(), 0))
    y_min = float(min(sub[vocab_slope_col].min(), 0))
    y_max = float(max(sub[vocab_slope_col].max(), 0))
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.5
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.5
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    fig.add_shape(type="line", x0=0, x1=0, y0=y_min, y1=y_max, line=dict(color="rgba(0,0,0,0.35)", dash="dash"))
    fig.add_shape(type="line", x0=x_min, x1=x_max, y0=0, y1=0, line=dict(color="rgba(0,0,0,0.35)", dash="dash"))
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.35)", dash="dash"),
            name="Zero lines (sign reference)",
            showlegend=True,
        )
    )

    apply_standard_layout(fig, title, height=height, width=width)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return fig


def plot_level_agreement(
    df: pd.DataFrame,
    *,
    author_col: str = "author",
    density_level_col: str = "density_neglog_level",
    vocab_level_col: str = "vocab_kld_all_level",
    cocit_level_col: str = "cocit_kld_all_level",
    label_map: Optional[Mapping[str, str]] = None,
    width: int = DEFAULT_PLOT_WIDTH,
    height: int = DEFAULT_2ROW_HEIGHT,
    title: str = "Level Agreement (All-Term Metrics)",
) -> go.Figure:
    """2D scatter: x=density level, y=vocab level, color=cocit level."""
    needed = [author_col, density_level_col, vocab_level_col, cocit_level_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for level plot: {missing}")

    sub = df[[author_col, density_level_col, vocab_level_col, cocit_level_col]].copy()
    sub = sub.dropna(subset=[density_level_col, vocab_level_col, cocit_level_col])
    sub[author_col] = sub[author_col].astype(str)
    customdata = sub[[author_col]].to_numpy()

    label_map = dict(label_map or default_label_map())
    x_label = _label(density_level_col, label_map)
    y_label = _label(vocab_level_col, label_map)
    c_label = _label(cocit_level_col, label_map)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=sub[density_level_col],
            y=sub[vocab_level_col],
            mode="markers",
            marker=dict(
                size=10,
                opacity=0.9,
                color=sub[cocit_level_col],
                colorscale="turbo",
                showscale=True,
                colorbar=make_colorbar(c_label, y=0.5, length=1.0),
                line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + f"{x_label}: %{{x:.4f}}<br>"
                + f"{y_label}: %{{y:.4f}}<br>"
                + f"{c_label}: %{{marker.color:.4f}}<br>"
                + "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    apply_standard_layout(fig, title, height=height, width=width)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return fig


def plot_correlation_heatmaps(
    df: pd.DataFrame,
    *,
    level_cols: Sequence[str] = (
        "density_neglog_level",
        "vocab_kld_all_level",
        "cocit_kld_all_level",
    ),
    slope_cols: Sequence[str] = (
        "density_neglog_slope",
        "vocab_kld_all_slope",
        "cocit_kld_all_slope",
    ),
    method: str = "spearman",
    label_map: Optional[Mapping[str, str]] = None,
    width: int = DEFAULT_PLOT_WIDTH,
    height: int = DEFAULT_2ROW_HEIGHT,
    title: str = "Correlations (Across Authors)",
) -> go.Figure:
    """
    Two-row heatmap: correlations for levels (row 1) and slopes (row 2).

    Uses a diverging color scale because correlations can be negative.
    """
    level_cols = [c for c in level_cols if c in df.columns]
    slope_cols = [c for c in slope_cols if c in df.columns]

    corr_levels = df[level_cols].corr(method=method) if len(level_cols) >= 2 else None
    corr_slopes = df[slope_cols].corr(method=method) if len(slope_cols) >= 2 else None
    label_map = dict(label_map or default_label_map())

    def _corr_tick_labels(cols: Sequence[str]) -> list[str]:
        # Keep correlation heatmaps compact: the subplot title already tells you
        # whether you're looking at levels or slopes.
        labels: list[str] = []
        for c in cols:
            if c.startswith("density_"):
                labels.append("Density")
            elif c.startswith("vocab_"):
                labels.append("Vocab KLD")
            elif c.startswith("cocit_"):
                labels.append("Co-cit KLD")
            else:
                labels.append(_label(c, label_map))
        if len(set(labels)) != len(labels):
            labels = [_label(c, label_map) for c in cols]
        return labels

    if corr_levels is not None:
        labels = _corr_tick_labels(level_cols)
        corr_levels = corr_levels.copy()
        corr_levels.columns = labels
        corr_levels.index = labels
    if corr_slopes is not None:
        labels = _corr_tick_labels(slope_cols)
        corr_slopes = corr_slopes.copy()
        corr_slopes.columns = labels
        corr_slopes.index = labels

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=(f"Levels ({method.title()})", f"Slopes ({method.title()})"),
    )

    def _add_corr_heatmap(corr: Optional[pd.DataFrame], row: int, *, coloraxis: str) -> None:
        if corr is None or corr.empty:
            fig.add_annotation(
                text="Not enough columns for correlation.",
                x=0.5,
                y=0.5,
                xref=f"x{row}",
                yref=f"y{row}",
                showarrow=False,
            )
            return
        z = corr.to_numpy()
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=list(corr.columns),
                y=list(corr.index),
                zmin=-1,
                zmax=1,
                coloraxis=coloraxis,
                text=np.round(z, 2),
                texttemplate="%{text:.2f}",
                hovertemplate="x=%{x}<br>y=%{y}<br>ρ=%{z:.3f}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    _add_corr_heatmap(corr_levels, row=1, coloraxis="coloraxis1")
    _add_corr_heatmap(corr_slopes, row=2, coloraxis="coloraxis2")

    cb_title = f"{method.title()} ρ"
    fig.update_layout(
        coloraxis1=dict(
            colorscale="turbo",
            reversescale=False,
            cmin=-1,
            cmax=1,
            colorbar=make_colorbar_for_axis(fig, "yaxis", cb_title),
        ),
        coloraxis2=dict(
            colorscale="turbo",
            reversescale=False,
            cmin=-1,
            cmax=1,
            colorbar=make_colorbar_for_axis(fig, "yaxis2", cb_title),
        ),
    )
    apply_standard_layout(fig, title, height=height, width=width)
    fig.update_xaxes(tickangle=0, automargin=True, row=1, col=1)
    fig.update_xaxes(tickangle=0, automargin=True, row=2, col=1)
    fig.update_yaxes(automargin=True, row=1, col=1)
    fig.update_yaxes(automargin=True, row=2, col=1)
    return fig
