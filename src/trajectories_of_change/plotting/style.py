"""Shared Plotly styling helpers for Trajectories of Change plots."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# =============================================================================
# Plot constants
# =============================================================================

LINE_COLOR_ALL = 'lightblue'
LINE_COLOR_SIG = 'lightgreen'
TREND_COLOR = 'firebrick'
LEG_BG = 'rgba(255,255,255,0.85)'

# =============================================================================
# Figure sizing — single source of truth. Every paper/dashboard figure size is
# derived from this spec so nothing can silently fall back to the Kaleido
# 700x500 default (the "squeeze"). Values are byte-identical to the historic
# LEGACY_* sizes, so figure appearance is unchanged.
# =============================================================================
BASE_PLOT_WIDTH = 1900
BASE_PANEL_HEIGHT = 600
TWO_ROW_FACTOR = 1.3


@dataclass(frozen=True)
class FigureSize:
    width: int
    height: int


FIGURE_SIZES = {
    "dashboard_one_row": FigureSize(BASE_PLOT_WIDTH, BASE_PANEL_HEIGHT),                       # 1900x600
    "dashboard_two_row": FigureSize(BASE_PLOT_WIDTH, int(BASE_PANEL_HEIGHT * TWO_ROW_FACTOR)),  # 1900x780
    "sync_trend": FigureSize(int(BASE_PLOT_WIDTH * 0.88), int(BASE_PANEL_HEIGHT * TWO_ROW_FACTOR)),  # 1672x780
    "relationship": FigureSize(BASE_PLOT_WIDTH // 2, BASE_PLOT_WIDTH // 2),                    # 950x950
    "citation_counts": FigureSize(int(BASE_PLOT_WIDTH * 0.72), BASE_PANEL_HEIGHT),             # 1368x600
}

# Backwards-compatible aliases (removed in the final cleanup phase).
BASE_TWO_ROW_HEIGHT = FIGURE_SIZES["dashboard_two_row"].height
SIGNED_LEADLAG_COLORSCALE = [
    [0.0, "#30123b"],
    [0.25, "#3b82c4"],
    [0.5, "#f1f3f4"],
    [0.75, "#f37651"],
    [1.0, "#b40426"],
]

# Shared text/data colors used across the paper figures (single source of truth;
# previously hardcoded in both the renderer and multimetric).
FONT_COLOR = "#1f3557"     # shared dark-navy text color
PRIMARY_COLOR = "#345995"  # primary data/trend/bar color
SLATE_COLOR = "#4f6d8c"    # secondary muted-slate bar color
PLOT_BG_COLOR = "#e5ecf6"  # plotting-area background
PAPER_BG_COLOR = "white"   # figure (paper) background
GRID_COLOR = "rgba(255,255,255,0.85)"   # white grid on the light-blue plot area
RIDGE_COLORSCALE = ["#9b179e", "#cf4568", "#ff9f2d"]  # author token-usage ridgeplots

EXPORT_FORMATS = {"pdf", "png", "svg", "html"}

# Plotly colorbar defaults (shared across the project)
COLORBAR_X = 1.0  # anchor at the plot edge; use xpad for pixel-stable spacing
COLORBAR_XPAD = 15  # fixed pixel gap between plot area and colorbar
COLORBAR_OUTLINEWIDTH = 0
COLORBAR_TITLE_SIDE = 'right'  # vertical title next to the bar
COLORBAR_YANCHOR = 'middle'

def make_colorbar(
    title: str,
    *,
    x: float = COLORBAR_X,
    xpad: int = COLORBAR_XPAD,
    y: float | None = None,
    length: float | None = None,
    yanchor: str = COLORBAR_YANCHOR,
):
    """Create a consistent Plotly `colorbar` dict (scatter markers + heatmaps).

    Keep this minimal: one place to define x-offset, outline, and vertical title.
    """
    cb = dict(
        title=dict(text=title, side=COLORBAR_TITLE_SIDE),
        outlinewidth=COLORBAR_OUTLINEWIDTH,
        x=x,
        xpad=xpad,
    )
    if y is not None:
        cb["y"] = y
        cb["yanchor"] = yanchor
    if length is not None:
        cb["len"] = length
    return cb


def _get_axis_domain(fig, axis_name: str) -> tuple[float, float] | None:
    """Return (start, end) domain for a layout axis like 'yaxis' / 'yaxis2'."""
    try:
        axis = fig.layout[axis_name]
    except Exception:
        return None
    domain = getattr(axis, "domain", None)
    if not domain or len(domain) != 2:
        return None
    return (float(domain[0]), float(domain[1]))


def make_colorbar_for_axis(
    fig,
    axis_name: str,
    title: str,
    *,
    x: float = COLORBAR_X,
    xpad: int = COLORBAR_XPAD,
    yanchor: str = COLORBAR_YANCHOR,
    length: float | None = None,
):
    """Create a colorbar aligned to the given subplot axis domain.

    This avoids hard-coded y/len values (e.g. 0.77/0.23, 0.5) and keeps
    colorbars visually consistent even if subplot spacing/margins change.
    """
    domain = _get_axis_domain(fig, axis_name)
    if domain is None:
        y = 0.5
        domain_len = 1.0
    else:
        y = (domain[0] + domain[1]) / 2.0
        domain_len = domain[1] - domain[0]

    return make_colorbar(
        title,
        x=x,
        xpad=xpad,
        y=y,
        length=domain_len if length is None else length,
        yanchor=yanchor,
    )


# =============================================================================
# Plot helpers
# =============================================================================

def add_trendline(
    fig,
    x_values,
    y_values,
    name: str,
    row: int,
    col: int = 1,
    color: str = None
):
    """Add the canonical OLS trendline to a Plotly subplot."""
    x_arr = pd.Series(x_values).to_numpy()
    y_arr = pd.Series(y_values).to_numpy()
    mask = ~np.isnan(y_arr)

    if mask.sum() < 2:
        return

    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    x_trend = np.linspace(x_arr.min(), x_arr.max(), len(x_arr))
    y_trend = slope * x_trend + intercept

    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode='lines',
        line=dict(color=color or TREND_COLOR, width=2, dash='dash'),
        name=name,
        showlegend=True
    ), row=row, col=col)


def apply_standard_layout(
    fig,
    title: str,
    height: int = 600,
    width: int = 1900,
    title_x: float = None,
    title_y: float = 0.96,
    legend_x: float = 0.0,
    legend_y: float = 1.0
):
    """
    Apply standard layout settings to a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure
    title : str
        Figure title
    height : int
        Height in pixels
    width : int
        Width in pixels
    title_x : float
        Title x position (default: computed from width)
    title_y : float
        Title y position
    legend_x : float
        Legend x position
    legend_y : float
        Legend y position
    """
    if title_x is None:
        title_x = 80 / width

    fig.update_layout(
        # Slightly wider right margin to keep vertical colorbar titles un-clipped.
        margin=dict(r=170, t=80, b=80, l=80),
        height=height,
        width=width,
        title=dict(
            text=title,
            x=title_x,
            xanchor='left',
            y=title_y
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=legend_y,
            xanchor='left',
            x=legend_x,
            bgcolor=LEG_BG,
            borderwidth=0
        )
    )


def apply_coloraxis_layout(
    fig,
    colorbar_len: float | None = None,
    *,
    title_top: str = "KLD_all (bits)",
    title_bottom: str = "KLD_sig (bits)",
):
    """
    Apply consistent colorbar configuration for 2-row plots.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure
    colorbar_len : float
        Colorbar length (0–1)
    """
    cb_top = make_colorbar_for_axis(fig, "yaxis", title_top, length=colorbar_len)
    cb_bot = make_colorbar_for_axis(fig, "yaxis2", title_bottom, length=colorbar_len)
    fig.update_layout(
        coloraxis1=dict(
            colorscale='turbo',
            colorbar=cb_top,
        ),
        coloraxis2=dict(
            colorscale='turbo',
            colorbar=cb_bot,
        ),
    )


def add_heatmap_decorations(
    fig,
    pivot,
    row: int,
    window_size: int,
    min_year: int = None,
    max_year: int = None,
    *,
    show_minima_line: bool = False,
    minima_line_color: str = "red",
    minima_line_width: float = 1.2,
):
    """
    Add diagonal line and minimum rectangles to a (KLD-style) heatmap.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure
    pivot : pd.DataFrame
        Pivot table with values (index=target_slice, columns=field_slice)
    row : int
        Subplot row (1-based)
    window_size : int
        Window size for offset computation
    min_year : int
        Minimum year for diagonal (default: inferred from pivot)
    max_year : int
        Maximum year for diagonal (default: inferred from pivot)
    show_minima_line : bool
        If true, add a thin line through the row minima. This complements the
        red rectangles without replacing them.
    """
    if pivot.empty:
        return

    # Determine year bounds
    all_years = list(pivot.index) + list(pivot.columns)
    if min_year is None:
        min_year = min(all_years) if all_years else 1900
    if max_year is None:
        max_year = max(all_years) if all_years else 2000

    # Add diagonal
    fig.add_shape(
        type="line",
        x0=min_year, y0=min_year,
        x1=max_year, y1=max_year,
        line=dict(color="#e377c2", width=2, dash="dash"),
        row=row, col=1
    )

    # Add minima rectangles and collect their centers for an optional path.
    offset = window_size / 2.0
    minima_x: list[float] = []
    minima_y: list[float] = []
    for idx, target_slice in enumerate(pivot.index):
        row_values = pivot.iloc[idx].values.astype(float)
        valid_values = row_values[~np.isnan(row_values)]

        if len(valid_values) == 0:
            continue

        min_col_idx = np.nanargmin(row_values)
        field_slice = pivot.columns[min_col_idx]
        minima_x.append(float(field_slice))
        minima_y.append(float(target_slice))

        fig.add_shape(
            type="rect",
            x0=field_slice - offset, x1=field_slice + offset,
            y0=target_slice - offset, y1=target_slice + offset,
            line=dict(color="red", width=2),
            fillcolor="rgba(0,0,0,0)",
            row=row, col=1
        )

    if show_minima_line and minima_x:
        fig.add_trace(
            go.Scatter(
                x=minima_x,
                y=minima_y,
                mode="lines",
                line=dict(color=minima_line_color, width=minima_line_width),
                name="Minimum path",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )


# =============================================================================
# Shared save + small data/format helpers (single source of truth for all plots)
# =============================================================================

# Optional batch export: kaleido spins up a headless browser per image, which is
# expensive (~seconds each). Inside figure_batch(), save_figure() buffers figures
# and exports them in chunks through ONE browser session via plotly.io.write_images,
# which is several times faster for many-figure runs (e.g. all-50 dashboards).
_FIGURE_BATCH: dict | None = None


@contextmanager
def figure_batch(chunk_size: int = 40):
    """Buffer save_figure() exports and flush them in batched kaleido sessions."""
    global _FIGURE_BATCH
    previous = _FIGURE_BATCH
    _FIGURE_BATCH = {"items": [], "chunk": max(1, int(chunk_size))}
    try:
        yield
    finally:
        _flush_figure_batch()
        _FIGURE_BATCH = previous


def _flush_figure_batch() -> None:
    if not _FIGURE_BATCH or not _FIGURE_BATCH["items"]:
        return

    import plotly.io as pio
    from collections import defaultdict

    items = _FIGURE_BATCH["items"]
    # plotly.io.write_images runs ONE kaleido session per call but IGNORES each
    # figure's layout width/height. So group buffered figures by (width, height)
    # and pass the size EXPLICITLY per group: that keeps the batched speed
    # (~4x faster than per-figure) AND honors the bespoke sizes — no 700x500
    # squeeze. Items with an unknown size fall back to a single fig.write_image.
    groups: dict = defaultdict(list)
    singles: list = []
    for fig, path, w, h in items:
        (groups[(w, h)] if (w and h) else singles).append((fig, path))
    for (w, h), grp in groups.items():
        figs = [g[0] for g in grp]
        files = [str(g[1]) for g in grp]
        try:
            pio.write_images(figs, files, width=w, height=h)
        except subprocess.TimeoutExpired:
            if not all(Path(f).exists() and Path(f).stat().st_size > 0 for f in files):
                raise
            print(f"[WARN] Kaleido cleanup timed out after writing {len(files)} batched figure(s)")
        for f in files:
            print(f"[SAVED] {f}")
    for fig, path in singles:
        try:
            fig.write_image(str(path))
        except subprocess.TimeoutExpired:
            if not (Path(path).exists() and Path(path).stat().st_size > 0):
                raise
            print(f"[WARN] Kaleido cleanup timed out after writing {path}")
        print(f"[SAVED] {path}")
    items.clear()


def save_figure(fig, path, *, width: int | None = None, height: int | None = None, fmt: str = "pdf") -> None:
    """Write a Plotly figure to disk. PDF by default; the one save path for every plot."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported figure format: {fmt}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Squeeze guard: never export a non-HTML figure with no size anywhere — that
    # silently renders at the Kaleido 700x500 default. Builders set the layout
    # size or callers pass width/height; this makes the squeeze unreproducible.
    if fmt != "html" and width is None and height is None:
        layout = fig.layout
        if getattr(layout, "width", None) is None or getattr(layout, "height", None) is None:
            raise ValueError(
                f"save_figure: refusing to export {path} with no width/height and an "
                "underspecified layout size (would fall back to the Kaleido 700x500 "
                "default). Pass width/height or set fig.layout.width/height."
            )
    if _FIGURE_BATCH is not None and fmt != "html":
        if width:
            fig.update_layout(width=width)
        if height:
            fig.update_layout(height=height)
        # Record the explicit size so the batched flush can pass it per group
        # (pio.write_images ignores per-figure layout size — see _flush_figure_batch).
        w = int(width) if width else (int(fig.layout.width) if getattr(fig.layout, "width", None) else None)
        h = int(height) if height else (int(fig.layout.height) if getattr(fig.layout, "height", None) else None)
        _FIGURE_BATCH["items"].append((fig, path, w, h))
        if len(_FIGURE_BATCH["items"]) >= _FIGURE_BATCH["chunk"]:
            _flush_figure_batch()
        return
    if fmt == "html":
        fig.write_html(str(path))
    else:
        try:
            fig.write_image(str(path), width=width, height=height)
        except subprocess.TimeoutExpired:
            if not path.exists() or path.stat().st_size == 0:
                raise
            print(f"[WARN] Kaleido cleanup timed out after writing {path}")
    print(f"[SAVED] {path}")
