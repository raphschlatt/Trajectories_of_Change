from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import trajectories_of_change.plotting.style as plotting_style
import trajectories_of_change.plotting as plotting
from trajectories_of_change.plotting.density import plot_density_dashboards
from trajectories_of_change.plotting.kld import plot_kld_dashboards
from trajectories_of_change.plotting.multimetric import plot_sync_trend_small_multiples
from trajectories_of_change.plotting.style import (
    SIGNED_LEADLAG_COLORSCALE,
    add_heatmap_decorations,
    figure_batch,
    save_figure,
)


def test_style_module_keeps_single_public_save_helper() -> None:
    assert plotting_style.save_figure is save_figure
    assert not hasattr(plotting_style, "save_plot")
    assert not hasattr(plotting_style, "save_plot_upper_crop")


def test_plotting_facade_has_two_public_entry_points() -> None:
    assert plotting.__all__ == ["plot_metric", "plot_multimetric"]
    assert not hasattr(plotting, "plot_kld_dashboards")
    assert not hasattr(plotting, "plot_density_dashboards")
    assert not hasattr(plotting, "plot_dashboard")


def test_heatmap_decorations_add_rectangles_and_optional_minima_line() -> None:
    pivot = pd.DataFrame(
        {
            1998: [3.0, 0.5],
            2000: [2.0, 2.0],
            2002: [1.0, 4.0],
        },
        index=[2000, 2002],
    )
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index), row=1, col=1)

    add_heatmap_decorations(fig, pivot, row=1, window_size=2, show_minima_line=True)

    shapes = list(fig.layout.shapes)
    assert sum(shape.type == "rect" for shape in shapes) == 2
    assert any(shape.type == "line" and shape.line.dash == "dash" for shape in shapes)
    minima_lines = [trace for trace in fig.data if trace.name == "Minimum path"]
    assert len(minima_lines) == 1
    assert list(minima_lines[0].x) == [2002, 1998]
    assert list(minima_lines[0].y) == [2000, 2002]


def test_signed_leadlag_colorscale_is_not_turbo_or_half_rainbow() -> None:
    colors = [stop[1].lower() for stop in SIGNED_LEADLAG_COLORSCALE]

    assert len(colors) == 5
    assert "#f1f3f4" in colors
    assert not any(color in colors for color in ["green", "#00ff00", "#f6c445"])


def test_save_figure_accepts_kaleido_cleanup_timeout_after_file_write(tmp_path) -> None:
    class FakeFigure:
        def write_image(self, filepath, width=None, height=None):
            Path(filepath).write_bytes(b"%PDF-1.4\n")
            raise subprocess.TimeoutExpired(["taskkill", "/F"], timeout=6)

    path = tmp_path / "ridgeplot.pdf"

    save_figure(FakeFigure(), path, width=900, height=500, fmt="pdf")

    assert path.read_bytes().startswith(b"%PDF")


def test_figure_batch_accepts_kaleido_cleanup_timeout_after_file_writes(tmp_path, monkeypatch) -> None:
    import plotly.io as pio

    def fake_write_images(figs, files, **kwargs):  # kwargs: width/height per size group
        for file in files:
            Path(file).write_bytes(b"%PDF-1.4\n")
        raise subprocess.TimeoutExpired(["taskkill", "/F"], timeout=6)

    monkeypatch.setattr(pio, "write_images", fake_write_images)

    paths = [tmp_path / f"ridgeplot_{idx}.pdf" for idx in range(2)]
    with figure_batch(chunk_size=10):
        for path in paths:
            save_figure(go.Figure(), path, width=900, height=500, fmt="pdf")

    assert all(path.read_bytes().startswith(b"%PDF") for path in paths)


def test_kld_dashboard_titles_keep_legacy_title_and_legend_spacing() -> None:
    df_res = pd.DataFrame(
        [
            {"slice": 2000, "kld_all": 1.0, "kld_sig": 0.2},
            {"slice": 2002, "kld_all": 1.5, "kld_sig": 0.4},
        ]
    )
    terms = pd.DataFrame(
        [
            {"slice": 2000, "term": "alpha", "pvalue": 0.01, "p_used": 0.02, "kld_contribution": 0.2},
            {"slice": 2002, "term": "beta", "pvalue": 0.02, "p_used": 0.03, "kld_contribution": 0.4},
        ]
    )
    async_df = pd.DataFrame(
        [
            {"target_slice": 2000, "field_slice": 2000, "kld": 1.0, "kld_sig": 0.2},
            {"target_slice": 2000, "field_slice": 2002, "kld": 0.5, "kld_sig": 0.1},
            {"target_slice": 2002, "field_slice": 2002, "kld": 1.5, "kld_sig": 0.4},
        ]
    )

    figures = plot_kld_dashboards(
        df_res,
        terms,
        async_df,
        target_name="Treder",
        alpha=0.05,
        pointwise_alpha=0.05,
        window_size=2,
        sync_plot_width=900,
        sync_plot_height=400,
        export=False,
        show=False,
        export_prefix="kld_ref_vocab",
        multiple_testing="fdr_bh",
        multiple_testing_scope="slice",
    )

    sync_title = figures["sync"].layout.title.text
    async_title = figures["async"].layout.title.text
    assert "Synchronous Divergence: All vs. Significant (FDR-BH, slice scope, q<0.05) - Treder" == sync_title
    assert "Asynchronous Divergence: All vs. Significant (FDR-BH, slice scope, q<0.05) - Treder" == async_title
    assert "<sup>" not in sync_title
    assert figures["sync"].layout.legend.orientation == "h"
    assert figures["sync"].layout.legend.y == 1.0


def test_density_dashboard_titles_follow_ede_grammar() -> None:
    sync = pd.DataFrame(
        [
            {"slice": 2000, "density_neglog_median": 1.0},
            {"slice": 2002, "density_neglog_median": 1.5},
        ]
    )
    async_df = pd.DataFrame(
        [
            {"target_slice": 2000, "field_slice": 2000, "density_neglog_median": 1.0},
            {"target_slice": 2000, "field_slice": 2002, "density_neglog_median": 0.5},
            {"target_slice": 2002, "field_slice": 2002, "density_neglog_median": 1.5},
        ]
    )

    figures = plot_density_dashboards(
        sync,
        pd.DataFrame(),
        async_df,
        target_name="Treder",
        window_size=2,
        sync_plot_width=900,
        sync_plot_height=400,
        export=False,
        show=False,
    )

    assert "Synchronous Density: EDE - Treder" in figures["sync"].layout.title.text
    assert "<sup>" not in figures["sync"].layout.title.text
    assert figures["sync"].layout.legend.orientation == "h"
    assert figures["sync"].layout.legend.y == 1.0
    assert figures["sync"].layout.height == 400
    assert "Asynchronous Density: EDE - Treder" in figures["async"].layout.title.text


def test_sync_trend_small_multiples_uses_authors_by_metrics_lines() -> None:
    series = pd.DataFrame(
        [
            {"author_display_name": "Mostly Rising", "metric": "Own Vocab", "slice": 2000, "value": 1.0},
            {"author_display_name": "Mostly Rising", "metric": "Own Vocab", "slice": 2002, "value": 2.0},
            {"author_display_name": "Mostly Rising", "metric": "Ref Vocab", "slice": 2000, "value": 1.0},
            {"author_display_name": "Mostly Rising", "metric": "Ref Vocab", "slice": 2002, "value": 2.0},
            {"author_display_name": "Mostly Rising", "metric": "Citation Identity", "slice": 2000, "value": 1.0},
            {"author_display_name": "Mostly Rising", "metric": "Citation Identity", "slice": 2002, "value": 2.0},
            {"author_display_name": "Mostly Rising", "metric": "EDE", "slice": 2000, "value": 1.0},
            {"author_display_name": "Mostly Rising", "metric": "EDE", "slice": 2002, "value": 2.0},
            {"author_display_name": "Falling", "metric": "Own Vocab", "slice": 2000, "value": 2.0},
            {"author_display_name": "Falling", "metric": "Own Vocab", "slice": 2002, "value": 1.0},
            {"author_display_name": "Falling", "metric": "Ref Vocab", "slice": 2000, "value": 2.0},
            {"author_display_name": "Falling", "metric": "Ref Vocab", "slice": 2002, "value": 1.0},
            {"author_display_name": "Falling", "metric": "Citation Identity", "slice": 2000, "value": 2.0},
            {"author_display_name": "Falling", "metric": "Citation Identity", "slice": 2002, "value": 1.0},
            {"author_display_name": "Falling", "metric": "EDE", "slice": 2000, "value": 2.0},
            {"author_display_name": "Falling", "metric": "EDE", "slice": 2002, "value": 1.0},
        ]
    )

    fig = plot_sync_trend_small_multiples(
        series,
        metric_order=["Own Vocab", "Ref Vocab", "Citation Identity", "EDE"],
        title="Synchronous Trajectories: All-Term Layers",
    )

    scatter_traces = [trace for trace in fig.data if isinstance(trace, go.Scatter)]
    # 2 authors x 4 metrics x (sync line + trend) = 16 data traces, plus 2
    # legend-only glyph traces that *show* the line styles ("Sync value" / "Trend").
    data_traces = [trace for trace in scatter_traces if not trace.showlegend]
    legend_traces = [trace for trace in scatter_traces if trace.showlegend]
    assert len(data_traces) == 16
    assert [trace.name for trace in legend_traces] == ["Sync value", "Trend"]
    assert fig.layout.showlegend is True
    annotations = list(fig.layout.annotations)
    title = next(annotation for annotation in annotations if annotation.text == "Synchronous Trajectories: All-Term Layers")
    header = next(annotation for annotation in annotations if annotation.text == "Own Vocab")
    assert "FIG-" not in title.text
    # Title sits above the line-glyph legend, which sits above the column headers.
    assert title.y > fig.layout.legend.y > header.y
    assert any(annotation.text == "Mostly Rising" for annotation in annotations)
    assert all(getattr(axis, "showticklabels", None) is False for axis in fig.select_yaxes())
    assert data_traces[0].marker.colorscale is not None
    assert data_traces[0].marker.showscale is False
    assert fig.layout.height >= 900
