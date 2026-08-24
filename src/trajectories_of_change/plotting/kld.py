"""Reusable KLD dashboard plots for vocabulary and co-citation analyses."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .style import (
    add_heatmap_decorations,
    add_trendline,
    apply_standard_layout,
    apply_coloraxis_layout,
    save_figure,
    TWO_ROW_FACTOR,
    LINE_COLOR_ALL,
    LINE_COLOR_SIG,
)
from ..stats_utils import add_pvalue_adjustments
from ..defaults import DEFAULT_MULTIPLE_TESTING, DEFAULT_MULTIPLE_TESTING_SCOPE
from .._filenames import safe_filename_component


def _significance_label(alpha: float, multiple_testing: str, multiple_testing_scope: str) -> str:
    if multiple_testing == "none":
        return f"Significant (raw p<{alpha})"
    method = multiple_testing.upper().replace("_", "-")
    return f"Significant ({method}, {multiple_testing_scope} scope, q<{alpha})"


def _fdr_group_cols(scope: str, *, sync: bool) -> list[str] | None:
    """FDR grouping columns for a correction scope (shared by every KLD view)."""
    if scope == "global":
        return None
    if scope == "pair":
        return ["slice"] if sync else ["target_slice", "field_slice"]
    if scope == "slice":
        return ["slice"] if sync else ["target_slice"]
    raise ValueError(f"Unsupported correction scope: {scope}")


def prepare_kld_dashboard_inputs(
    df_sync: pd.DataFrame,
    df_async: pd.DataFrame,
    welch_all_df: pd.DataFrame,
    *,
    alpha: float,
    multiple_testing: str = DEFAULT_MULTIPLE_TESTING,
    multiple_testing_scope: str = DEFAULT_MULTIPLE_TESTING_SCOPE,
):
    """Prepare standard inputs for `plot_kld_dashboards()`.

    This removes duplication between vocabulary and co-citation KLD views.

    Returns
    -------
    df_res : pd.DataFrame
        Per-slice aggregated KLD results with columns: slice, kld_all, kld_sig.
        `kld_sig` uses `multiple_testing` and `multiple_testing_scope`; with
        `multiple_testing="none"`, raw p-values are used.
    welch_sync_df : pd.DataFrame
        Pointwise terms for synchronous slice pairs with columns incl. slice, term, pvalue, kld_contribution
    df_async_full : pd.DataFrame
        Async matrix with added kld_sig column (summed significant contributions)
    """
    welch_sync_df = welch_all_df[welch_all_df["target_slice"] == welch_all_df["field_slice"]].copy()
    welch_sync_df.rename(columns={"target_slice": "slice"}, inplace=True)
    welch_sync_df.drop(columns=["field_slice"], inplace=True)

    if multiple_testing != "none":
        welch_sync_df = add_pvalue_adjustments(
            welch_sync_df,
            p_col="pvalue",
            method=multiple_testing,
            group_cols=_fdr_group_cols(multiple_testing_scope, sync=True),
            out_col="p_adj",
        )
        welch_sync_df["p_used"] = welch_sync_df["p_adj"]
    else:
        welch_sync_df["p_used"] = welch_sync_df["pvalue"]
    welch_sync_df.attrs["multiple_testing"] = multiple_testing
    welch_sync_df.attrs["multiple_testing_scope"] = multiple_testing_scope

    sig_sync_terms = welch_sync_df[welch_sync_df["p_used"] < alpha].copy()
    kld_sig_df = (
        sig_sync_terms.groupby("slice", as_index=False)["kld_contribution"]
        .sum()
        .rename(columns={"kld_contribution": "kld_sig"})
    )
    kld_sig_abs_df = (
        sig_sync_terms.assign(kld_contribution_abs=sig_sync_terms["kld_contribution"].abs())
        .groupby("slice", as_index=False)["kld_contribution_abs"]
        .sum()
        .rename(columns={"kld_contribution_abs": "kld_sig_abs"})
    )
    df_res = df_sync.merge(kld_sig_df, on="slice", how="left").fillna(0)
    df_res = df_res.merge(kld_sig_abs_df, on="slice", how="left").fillna({"kld_sig_abs": 0})

    if multiple_testing != "none":
        welch_all_adj = add_pvalue_adjustments(
            welch_all_df,
            p_col="pvalue",
            method=multiple_testing,
            group_cols=_fdr_group_cols(multiple_testing_scope, sync=False),
            out_col="p_adj",
        )
        sig_async_terms = welch_all_adj[welch_all_adj["p_adj"] < alpha].copy()
    else:
        sig_async_terms = welch_all_df[welch_all_df["pvalue"] < alpha].copy()

    async_sig_summed = sig_async_terms.groupby(["target_slice", "field_slice"], as_index=False)[
        "kld_contribution"
    ].sum()
    async_sig_summed.rename(columns={"kld_contribution": "kld_sig"}, inplace=True)
    async_sig_abs_summed = sig_async_terms.assign(
        kld_contribution_abs=sig_async_terms["kld_contribution"].abs()
    ).groupby(["target_slice", "field_slice"], as_index=False)["kld_contribution_abs"].sum()
    async_sig_abs_summed.rename(columns={"kld_contribution_abs": "kld_sig_abs"}, inplace=True)

    df_async_full = df_async.merge(async_sig_summed, on=["target_slice", "field_slice"], how="left")
    df_async_full = df_async_full.merge(
        async_sig_abs_summed, on=["target_slice", "field_slice"], how="left"
    ).fillna({"kld_sig": 0, "kld_sig_abs": 0})
    df_async_full.attrs["multiple_testing"] = multiple_testing
    df_async_full.attrs["multiple_testing_scope"] = multiple_testing_scope

    return df_res, welch_sync_df, df_async_full


def plot_kld_dashboards(
    df_res,
    welch_sync_df,
    df_async_full,
    *,
    target_name: str,
    alpha: float,
    pointwise_alpha: float,
    window_size: int,
    sync_plot_width: int,
    sync_plot_height: int,
    export_dir: Optional[str] = None,
    export_prefix: str = "kld",
    export: bool = True,
    show: bool = False,
    highlight_terms: Optional[dict] = None,
    trend_color: Optional[str] = None,
    multiple_testing: Optional[str] = None,
    multiple_testing_scope: Optional[str] = None,
    show_minima_line: bool = False,
):
    """Build the three standard plots and optionally export them as PNGs."""
    if export and export_dir is None:
        raise ValueError("export_dir is required when export=True")

    fig_sync = None
    fig_pt = None
    fig_async = None

    plot_height_2row = int(sync_plot_height * TWO_ROW_FACTOR)
    target_lower = safe_filename_component(target_name).lower()
    multiple_testing = multiple_testing or welch_sync_df.attrs.get(
        "multiple_testing", DEFAULT_MULTIPLE_TESTING
    )
    multiple_testing_scope = multiple_testing_scope or welch_sync_df.attrs.get(
        "multiple_testing_scope", DEFAULT_MULTIPLE_TESTING_SCOPE
    )
    sig_label = _significance_label(alpha, multiple_testing, multiple_testing_scope)
    pointwise_sig_label = _significance_label(pointwise_alpha, multiple_testing, multiple_testing_scope)

    if not df_res.empty:
        fig_sync = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        fig_sync.add_trace(
            go.Scatter(
                x=df_res["slice"],
                y=df_res["kld_all"],
                mode="lines+markers",
                line=dict(color=LINE_COLOR_ALL, width=2),
                marker=dict(size=8, opacity=0.95, color=df_res["kld_all"], coloraxis="coloraxis1"),
                name="KLD_all",
            ),
            row=1,
            col=1,
        )
        add_trendline(
            fig_sync,
            df_res["slice"],
            df_res["kld_all"],
            "Trend (KLD_all)",
            1,
            color=trend_color,
        )
        fig_sync.add_trace(
            go.Scatter(
                x=df_res["slice"],
                y=df_res["kld_sig"],
                mode="lines+markers",
                line=dict(color=LINE_COLOR_SIG, width=3),
                marker=dict(size=8, opacity=1.0, color=df_res["kld_sig"], coloraxis="coloraxis2"),
                name="KLD_sig",
            ),
            row=2,
            col=1,
        )
        add_trendline(
            fig_sync,
            df_res["slice"],
            df_res["kld_sig"],
            "Trend (KLD_sig)",
            2,
            color=trend_color,
        )
        apply_standard_layout(
            fig_sync,
            f"Synchronous Divergence: All vs. {sig_label} - {target_name}",
            height=plot_height_2row,
            width=sync_plot_width,
        )
        apply_coloraxis_layout(fig_sync)
        fig_sync.update_xaxes(showticklabels=False, row=1, col=1)
        fig_sync.update_xaxes(title_text="Timeslice (End Year)", row=2, col=1)
        fig_sync.update_yaxes(title_text="KLD_all (Bits)", row=1, col=1)
        fig_sync.update_yaxes(title_text="KLD_sig (Bits)", row=2, col=1)
        if show:
            fig_sync.show()
        if export:
            save_figure(
                fig_sync,
                Path(export_dir) / f"{export_prefix}_sync_summed_{target_lower}.png",
                width=sync_plot_width,
                height=plot_height_2row,
                fmt="png",
            )

        pointwise_all = welch_sync_df.copy()
        p_col = "p_used" if "p_used" in pointwise_all.columns else "pvalue"
        pointwise_sync = pointwise_all[pointwise_all[p_col] < pointwise_alpha].copy()
        if not pointwise_all.empty:
            fig_pt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
            fig_pt.add_trace(
                go.Scatter(
                    x=pointwise_all["slice"],
                    y=pointwise_all["kld_contribution"],
                    mode="markers",
                    marker=dict(size=5, opacity=0.4, color=pointwise_all["kld_contribution"], coloraxis="coloraxis1"),
                    text=pointwise_all["term"],
                    hovertemplate="%{text}<br>Slice: %{x}<br>KLD: %{y:.4f}<extra></extra>",
                    name="All Terms",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            if not pointwise_sync.empty:
                fig_pt.add_trace(
                    go.Scatter(
                        x=pointwise_sync["slice"],
                        y=pointwise_sync["kld_contribution"],
                        mode="markers",
                        marker=dict(
                            size=5,
                            opacity=0.6,
                            color=pointwise_sync["kld_contribution"],
                            coloraxis="coloraxis2",
                        ),
                        text=pointwise_sync["term"],
                        hovertemplate="%{text}<br>Slice: %{x}<br>KLD: %{y:.4f}<extra></extra>",
                        name=pointwise_sig_label,
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )
            apply_standard_layout(
                fig_pt,
                f"Pointwise KLD Contributions (tested terms): All vs. {sig_label} - {target_name}",
                height=plot_height_2row,
                width=sync_plot_width,
            )
            apply_coloraxis_layout(
                fig_pt,
                title_top="KLD contribution (tested terms, bits)",
                title_bottom="KLD contribution (significant terms, bits)",
            )
            fig_pt.update_xaxes(showticklabels=False, row=1, col=1)
            fig_pt.update_xaxes(title_text="Timeslice (End Year)", row=2, col=1)
            fig_pt.update_yaxes(title_text="KLD_all Contribution (Bits)", row=1, col=1)
            fig_pt.update_yaxes(title_text="KLD_sig Contribution (Bits)", row=2, col=1)
            # Highlight configured terms (optional)
            if highlight_terms:
                for term, color in highlight_terms.items():
                    subset_all = pointwise_all[pointwise_all["term"].str.lower() == term.lower()]
                    subset_sig = pointwise_sync[pointwise_sync["term"].str.lower() == term.lower()]
                    if not subset_all.empty:
                        fig_pt.add_trace(
                            go.Scatter(
                                x=subset_all["slice"],
                                y=subset_all["kld_contribution"],
                                mode="lines+markers+text",
                                text=[term] * len(subset_all),
                                textposition="top center",
                                marker=dict(color=color, size=10, line=dict(width=2, color="black")),
                                line=dict(color=color, width=2),
                                name=term,
                                showlegend=True,
                            ),
                            row=1,
                            col=1,
                        )
                    if not subset_sig.empty:
                        fig_pt.add_trace(
                            go.Scatter(
                                x=subset_sig["slice"],
                                y=subset_sig["kld_contribution"],
                                mode="lines+markers+text",
                                text=[term] * len(subset_sig),
                                textposition="top center",
                                marker=dict(color=color, size=10, line=dict(width=2, color="black")),
                                line=dict(color=color, width=2),
                                name=term,
                                showlegend=False,
                            ),
                            row=2,
                            col=1,
                        )
            if show:
                fig_pt.show()
            if export:
                save_figure(
                    fig_pt,
                    Path(export_dir) / f"{export_prefix}_sync_pointwise_{target_lower}.png",
                    width=sync_plot_width,
                    height=plot_height_2row,
                    fmt="png",
                )

    if df_async_full is not None and not df_async_full.empty:
        df_all = df_async_full.copy()
        pivot_all = df_all.pivot(index="target_slice", columns="field_slice", values="kld")
        pivot_sig = df_all.pivot(index="target_slice", columns="field_slice", values="kld_sig")
        pivot_sig = pivot_sig.replace(0, np.nan)
        fig_async = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        fig_async.add_trace(
            go.Heatmap(
                z=pivot_all.values,
                x=pivot_all.columns,
                y=pivot_all.index,
                coloraxis="coloraxis1",
                zmin=0,
                connectgaps=False,
                hoverongaps=False,
                name="KLD_all",
            ),
            row=1,
            col=1,
        )
        fig_async.add_trace(
            go.Heatmap(
                z=pivot_sig.values,
                x=pivot_sig.columns,
                y=pivot_sig.index,
                coloraxis="coloraxis2",
                zmin=0,
                connectgaps=False,
                hoverongaps=False,
                name="KLD_sig",
            ),
            row=2,
            col=1,
        )
        all_years = list(pivot_all.index) + list(pivot_all.columns)
        min_year = min(all_years) if all_years else 1900
        max_year = max(all_years) if all_years else 2000
        # Dummy traces for legend
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
                marker=dict(symbol="square", size=12, color="rgba(0,0,0,0)", line=dict(color="red", width=2)),
                name="Minimum (Lead/Lag)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        offset = window_size / 2.0
        add_heatmap_decorations(
            fig_async,
            pivot_all,
            row=1,
            window_size=window_size,
            min_year=min_year,
            max_year=max_year,
            show_minima_line=show_minima_line,
        )
        add_heatmap_decorations(
            fig_async,
            pivot_sig,
            row=2,
            window_size=window_size,
            min_year=min_year,
            max_year=max_year,
            show_minima_line=show_minima_line,
        )
        apply_standard_layout(
            fig_async,
            f"Asynchronous Divergence: All vs. {sig_label} - {target_name}",
            height=plot_height_2row,
            width=sync_plot_width,
        )
        apply_coloraxis_layout(fig_async)
        min_target_year = df_all["target_slice"].min()
        max_target_year = df_all["target_slice"].max()
        y_range = [min_target_year - offset, max_target_year + offset]
        fig_async.update_yaxes(title_text="Target Timeslice (End Year)", range=y_range, row=1, col=1)
        fig_async.update_yaxes(title_text="Target Timeslice (End Year)", range=y_range, row=2, col=1)
        fig_async.update_xaxes(showticklabels=False, row=1, col=1)
        fig_async.update_xaxes(title_text="Field Timeslice (End Year)", row=2, col=1)
        if show:
            fig_async.show()
        if export:
            save_figure(
                fig_async,
                Path(export_dir) / f"{export_prefix}_async_heatmap_{target_lower}.png",
                width=sync_plot_width,
                height=plot_height_2row,
                fmt="png",
            )

    return {"sync": fig_sync, "pointwise": fig_pt, "async": fig_async}
