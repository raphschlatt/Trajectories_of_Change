"""
Reusable dashboard plots for slice-based density analyses.

This module provides a single function to build and optionally export the
standard density plots:
1) sync trajectory (per-slice median −log KDE under the field KDE)
2) pointwise scatter (per-paper −log KDE, with optional median overlay)
3) async heatmap (slice×slice median −log KDE)

Density values are whatever the upstream `KDEDensity` run produced: by
default KDE is evaluated in globally standardized canonical 2D map coordinates;
explicit nD/5D coordinate columns can be used in the metric before plotting.
Lower values mean the target lies in a locally denser field region. Higher
values mean lower local field density around the target, not necessarily
geometric map periphery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .style import (
    LINE_COLOR_ALL,
    TREND_COLOR,
    add_heatmap_decorations,
    add_trendline,
    apply_standard_layout,
    make_colorbar,
    save_figure,
)
from .._filenames import safe_filename_component


def plot_density_dashboards(
    d_sync: pd.DataFrame,
    d_pt: pd.DataFrame,
    d_async: Optional[pd.DataFrame],
    *,
    target_name: str,
    docid_col: str = "Bibcode",
    window_size: int,
    sync_plot_width: int,
    sync_plot_height: int,
    show: bool = False,
    export: bool = True,
    export_dir: Optional[Union[str, Path]] = None,
    export_name: Optional[str] = None,
    show_minima_line: bool = False,
) -> dict[str, Optional[go.Figure]]:
    """
    Build/show/export the standard density plots (sync, pointwise, async).

    Notes
    -----
    - Export filenames follow the historic density dashboard convention:
      `density_sync_{name}.png`, `density_pointwise_{name}.png`,
      `density_async_{name}.png`.
    - Plot appearance is intentionally kept stable (same titles/axes/colors).
    """

    export_dir_path = Path(export_dir) if export_dir is not None else None
    name_slug = safe_filename_component(export_name or target_name).lower()

    fig_sync: Optional[go.Figure] = None
    fig_pt: Optional[go.Figure] = None
    fig_async: Optional[go.Figure] = None

    # 1) Sync trajectory
    if d_sync is not None and not d_sync.empty:
        fig_sync = make_subplots(rows=1, cols=1)
        sync_colorbar = make_colorbar("−log field KDE", y=0.5, length=1.0)
        fig_sync.add_trace(
            go.Scatter(
                x=d_sync["slice"],
                y=d_sync["density_neglog_median"],
                mode="lines+markers",
                line=dict(color=LINE_COLOR_ALL, width=2),
                marker=dict(
                    size=8,
                    opacity=0.95,
                    color=d_sync["density_neglog_median"],
                    colorscale="turbo",
                    showscale=True,
                    colorbar=sync_colorbar,
                ),
                name="Median −log field KDE",
            ),
            row=1,
            col=1,
        )
        add_trendline(
            fig_sync,
            d_sync["slice"],
            d_sync["density_neglog_median"],
            "Trend",
            1,
            color=TREND_COLOR,
        )
        apply_standard_layout(
            fig_sync,
            f"Synchronous Density: EDE - {target_name}",
            height=sync_plot_height,
            width=sync_plot_width,
        )
        fig_sync.update_xaxes(title_text="Timeslice (End Year)", row=1, col=1)
        fig_sync.update_yaxes(title_text="−log KDE under field model", row=1, col=1)
        if show:
            fig_sync.show()
        if export and export_dir_path is not None:
            out_path = str(export_dir_path / f"density_sync_{name_slug}.png")
            save_figure(
                fig_sync,
                out_path,
                width=sync_plot_width,
                height=sync_plot_height,
                fmt="png",
            )

    # 2) Pointwise (per paper)
    if d_pt is not None and not d_pt.empty:
        fig_pt = make_subplots(rows=1, cols=1)
        pt_colorbar = make_colorbar("−log field KDE", y=0.5, length=1.0)
        fig_pt.add_trace(
            go.Scatter(
                x=d_pt["slice"],
                y=d_pt["density_neglog"],
                mode="markers",
                marker=dict(
                    size=5,
                    opacity=0.35,
                    color=d_pt["density_neglog"],
                    colorscale="turbo",
                    showscale=True,
                    colorbar=pt_colorbar,
                ),
                text=d_pt[docid_col] if docid_col in d_pt.columns else None,
                hovertemplate="%{text}<br>Slice: %{x}<br>−log field KDE: %{y:.4f}<extra></extra>",
                name="Papers",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        if d_sync is not None and not d_sync.empty:
            fig_pt.add_trace(
                go.Scatter(
                    x=d_sync["slice"],
                    y=d_sync["density_neglog_median"],
                    mode="lines+markers",
                    line=dict(color="black", width=3),
                    marker=dict(size=7, opacity=0.9),
                    name="Median per slice",
                ),
                row=1,
                col=1,
            )
        apply_standard_layout(
            fig_pt,
            f"Pointwise Density: EDE - {target_name}",
            height=sync_plot_height,
            width=sync_plot_width,
        )
        fig_pt.update_xaxes(title_text="Timeslice (End Year)", row=1, col=1)
        fig_pt.update_yaxes(title_text="−log KDE under field model", row=1, col=1)
        if show:
            fig_pt.show()
        if export and export_dir_path is not None:
            out_path = str(export_dir_path / f"density_pointwise_{name_slug}.png")
            save_figure(
                fig_pt,
                out_path,
                width=sync_plot_width,
                height=sync_plot_height,
                fmt="png",
            )

    # 3) Async heatmap (slice×slice)
    if d_async is not None and not d_async.empty:
        pivot = d_async.pivot(index="target_slice", columns="field_slice", values="density_neglog_median")
        fig_async = make_subplots(rows=1, cols=1)
        async_colorbar = make_colorbar("−log field KDE", y=0.5, length=1.0)
        fig_async.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="turbo",
                colorbar=async_colorbar,
                connectgaps=False,
                hoverongaps=False,
                name="−log field KDE",
            ),
            row=1,
            col=1,
        )
        # Dummy traces for the legend (match KLD dashboards)
        fig_async.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="#e377c2", width=2, dash="dash"),
                name="Diagonal (Sync)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig_async.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=12,
                    color="rgba(0,0,0,0)",
                    line=dict(color="red", width=2),
                ),
                name="Minimum (Lead/Lag)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        add_heatmap_decorations(
            fig_async,
            pivot,
            row=1,
            window_size=window_size,
            show_minima_line=show_minima_line,
        )
        # Match KLD dashboards: fix y-range, keep x autorange
        offset = window_size / 2.0
        y_range = [float(pivot.index.min()) - offset, float(pivot.index.max()) + offset]
        fig_async.update_yaxes(range=y_range, row=1, col=1)
        apply_standard_layout(
            fig_async,
            f"Asynchronous Density: EDE - {target_name}",
            height=sync_plot_height,
            width=sync_plot_width,
        )
        fig_async.update_xaxes(title_text="Field slice (End Year)", row=1, col=1)
        fig_async.update_yaxes(title_text="Target slice (End Year)", row=1, col=1)
        if show:
            fig_async.show()
        if export and export_dir_path is not None:
            out_path = str(export_dir_path / f"density_async_{name_slug}.png")
            save_figure(
                fig_async,
                out_path,
                width=sync_plot_width,
                height=sync_plot_height,
                fmt="png",
            )

    return {"sync": fig_sync, "pointwise": fig_pt, "async": fig_async}
