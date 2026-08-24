"""Optional plotting API.

Install with `trajectories-of-change[plotting]` when Plotly/Matplotlib
dependencies are not already available.
"""

from ..defaults import (
    DEFAULT_ALPHA,
    DEFAULT_MULTIPLE_TESTING,
    DEFAULT_MULTIPLE_TESTING_SCOPE,
)


def _write_figure_dict(figures: dict, export_dir, *, prefix: str, target_name: str, fmt: str) -> None:
    from pathlib import Path

    from .._filenames import safe_filename_component
    from .style import figure_batch, save_figure

    target_slug = safe_filename_component(target_name)
    with figure_batch():
        for name, fig in figures.items():
            if fig is None:
                continue
            save_figure(
                fig,
                Path(export_dir) / f"{prefix}_{name}_{target_slug}.{fmt}",
                fmt=fmt,
            )


def _plot_dashboard(
    result,
    *,
    export_dir=None,
    alpha: float = DEFAULT_ALPHA,
    multiple_testing: str = DEFAULT_MULTIPLE_TESTING,
    multiple_testing_scope: str = DEFAULT_MULTIPLE_TESTING_SCOPE,
    show: bool = False,
    format: str = "html",
    **kwargs,
):
    """Render the standard 3-panel dashboard for a :class:`MetricResult`.

    The per-metric plotting entry point: pair it with a metric's ``.result()`` to
    go from "compute one metric" to "plot just that metric". Dispatches on
    ``result.kind`` to the KLD or density builder. Sizing always comes from
    ``FIGURE_SIZES`` so the export can never fall back to the Kaleido default.

    For ``kind="kld"`` the result must carry ``async_df`` and ``welch`` — compute
    with ``.result(include_async=True, run_welch=True)``.
    """
    from .style import FIGURE_SIZES
    from .density import plot_density_dashboards
    from .kld import plot_kld_dashboards, prepare_kld_dashboard_inputs

    size = FIGURE_SIZES["dashboard_one_row"]
    if result.kind == "kld":
        if result.async_df is None or result.welch is None:
            raise ValueError(
                "plot_dashboard(kind='kld') needs result.async_df and result.welch; "
                "compute with .result(include_async=True, run_welch=True)."
            )
        df_res, welch_sync, async_full = prepare_kld_dashboard_inputs(
            result.sync,
            result.async_df,
            result.welch,
            alpha=alpha,
            multiple_testing=multiple_testing,
            multiple_testing_scope=multiple_testing_scope,
        )
        figures = plot_kld_dashboards(
            df_res,
            welch_sync,
            async_full,
            target_name=result.target_name,
            alpha=alpha,
            pointwise_alpha=alpha,
            window_size=result.window_size,
            sync_plot_width=size.width,
            sync_plot_height=size.height,
            export_prefix=f"kld_{result.metric}",
            export=False,
            show=False,
            multiple_testing=multiple_testing,
            multiple_testing_scope=multiple_testing_scope,
            **kwargs,
        )
        if export_dir is not None:
            _write_figure_dict(
                figures,
                export_dir,
                prefix=f"kld_{result.metric}",
                target_name=result.target_name,
                fmt=format,
            )
        if show:
            for figure in figures.values():
                if figure is not None:
                    figure.show()
        return figures
    if result.kind == "density":
        figures = plot_density_dashboards(
            result.sync,
            result.pointwise,
            result.async_df,
            target_name=result.target_name,
            window_size=result.window_size,
            sync_plot_width=size.width,
            sync_plot_height=size.height,
            export=False,
            show=False,
            **kwargs,
        )
        if export_dir is not None:
            _write_figure_dict(
                figures,
                export_dir,
                prefix="density",
                target_name=result.target_name,
                fmt=format,
            )
        if show:
            for figure in figures.values():
                if figure is not None:
                    figure.show()
        return figures
    raise ValueError(f"Unknown MetricResult.kind: {result.kind!r}")


def plot_metric(
    result,
    *,
    export_dir=None,
    format: str | None = None,
    show: bool | None = None,
    alpha: float | None = None,
    multiple_testing: str | None = None,
    multiple_testing_scope: str | None = None,
    **kwargs,
):
    """Plot one :class:`MetricResult` using the appropriate per-metric visual."""
    significance = dict(result.config or {})
    should_show = export_dir is None if show is None else bool(show)
    return _plot_dashboard(
        result,
        export_dir=export_dir,
        format=format or "html",
        show=should_show,
        alpha=float(alpha if alpha is not None else significance.get("alpha", DEFAULT_ALPHA)),
        multiple_testing=str(
            multiple_testing
            if multiple_testing is not None
            else significance.get("multiple_testing", DEFAULT_MULTIPLE_TESTING)
        ),
        multiple_testing_scope=str(
            multiple_testing_scope
            if multiple_testing_scope is not None
            else significance.get("multiple_testing_scope", DEFAULT_MULTIPLE_TESTING_SCOPE)
        ),
        **kwargs,
    )


def plot_multimetric(metrics_df, *, export_dir=None, format: str = "html", show: bool | None = None, **kwargs):
    """Build standard multimetric summary plots and optionally export them."""
    from pathlib import Path

    from .multimetric import (
        plot_correlation_heatmaps,
        plot_level_agreement,
        plot_slope_agreement,
    )
    from .style import figure_batch, save_figure

    out_dir = Path(export_dir) if export_dir is not None else None
    specs = {
        "multimetric_slope_agreement": lambda: plot_slope_agreement(metrics_df, **kwargs),
        "multimetric_level_agreement": lambda: plot_level_agreement(metrics_df, **kwargs),
        "multimetric_correlations": lambda: plot_correlation_heatmaps(metrics_df, **kwargs),
    }
    figures = {}
    should_show = export_dir is None if show is None else bool(show)
    with figure_batch():
        for stem, factory in specs.items():
            fig = factory()
            if out_dir is not None:
                save_figure(fig, out_dir / f"{stem}.{format}", fmt=format)
            if should_show:
                fig.show()
            figures[stem] = fig
    return figures


__all__ = ["plot_metric", "plot_multimetric"]
