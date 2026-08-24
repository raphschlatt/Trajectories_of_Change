"""Per-metric ``plot_dashboard`` renders at the FIGURE_SIZES size, never the default."""

from __future__ import annotations

import itertools

import pandas as pd
import plotly.graph_objects as go
import pytest

from trajectories_of_change import MetricResult
from trajectories_of_change.plotting import plot_metric, plot_multimetric

SLICES = [1990, 1992, 1994, 1996]
_PAIRS = list(itertools.product(SLICES, SLICES))


def _kld_result() -> MetricResult:
    async_df = pd.DataFrame(
        [{"target_slice": t, "field_slice": f, "time_diff": f - t, "kld": 0.2 + 0.05 * abs(f - t)} for t, f in _PAIRS]
    )
    welch = pd.DataFrame(
        [
            {"target_slice": t, "field_slice": f, "term": term, "pvalue": p, "kld_contribution": c}
            for (t, f), (term, p, c) in itertools.product(
                _PAIRS, [("alpha", 0.01, 0.05), ("beta", 0.5, -0.02), ("gamma", 0.03, 0.04)]
            )
        ]
    )
    sync = (
        async_df[async_df.target_slice == async_df.field_slice]
        .rename(columns={"target_slice": "slice", "kld": "kld_all"})[["slice", "kld_all"]]
        .reset_index(drop=True)
    )
    return MetricResult(
        sync=sync, pointwise=pd.DataFrame(), async_df=async_df, welch=welch,
        kind="kld", metric="own_vocab", target_name="Test", window_size=2,
    )


def _density_result() -> MetricResult:
    async_df = pd.DataFrame(
        [
            {"target_slice": t, "field_slice": f, "time_diff": f - t,
             "density_neglog_median": 1.0 + 0.1 * abs(f - t), "target_docs": 5, "field_docs": 20}
            for t, f in _PAIRS
        ]
    )
    sync = (
        async_df[async_df.target_slice == async_df.field_slice]
        .rename(columns={"target_slice": "slice"})[["slice", "density_neglog_median"]]
        .reset_index(drop=True)
    )
    return MetricResult(
        sync=sync, pointwise=pd.DataFrame(), async_df=async_df,
        kind="density", metric="density", target_name="Test", window_size=2,
    )


def test_plot_metric_kld_export_does_not_show(tmp_path, monkeypatch):
    monkeypatch.setattr(go.Figure, "show", lambda self: pytest.fail("export opened a browser"))

    figures = plot_metric(_kld_result(), export_dir=tmp_path, alpha=0.2)
    assert len(list(tmp_path.glob("*.html"))) >= 2
    assert all(fig.layout.width == 1900 for fig in figures.values() if fig is not None)


def test_plot_metric_resolves_implicit_and_explicit_show(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(self))

    plot_metric(_density_result())
    assert shown

    shown.clear()
    plot_metric(_density_result(), show=False)
    assert not shown

    plot_metric(_density_result(), export_dir=tmp_path, show=True)
    assert shown


def test_plot_metric_density(tmp_path):
    plot_metric(_density_result(), export_dir=tmp_path, show=False)
    assert list(tmp_path.glob("*.html"))


def test_plot_metric_requires_async_and_welch(tmp_path):
    bare = MetricResult(
        sync=pd.DataFrame({"slice": SLICES, "kld_all": [0.1, 0.1, 0.1, 0.1]}),
        pointwise=pd.DataFrame(), kind="kld", metric="own_vocab",
    )
    with pytest.raises(ValueError, match="async_df"):
        plot_metric(bare, export_dir=tmp_path, show=False)


def test_plot_metric_alias_dispatches_dashboard(tmp_path):
    plot_metric(_density_result(), export_dir=tmp_path, show=False)
    assert list(tmp_path.glob("*.html"))


def test_plot_metric_uses_windows_safe_filename(tmp_path):
    result = _density_result()
    result = MetricResult(
        sync=result.sync,
        pointwise=result.pointwise,
        async_df=result.async_df,
        kind=result.kind,
        metric=result.metric,
        target_name="uid:author/name?*",
        window_size=result.window_size,
    )

    plot_metric(result, export_dir=tmp_path, show=False)

    names = [path.name for path in tmp_path.glob("*.html")]
    assert names
    assert all(not any(char in name for char in ':?*') for name in names)


def test_plot_multimetric_writes_summary_html_without_showing(tmp_path, monkeypatch):
    monkeypatch.setattr(go.Figure, "show", lambda self: pytest.fail("export opened a browser"))
    metrics = pd.DataFrame(
        [
            {
                "author": "A",
                "density_neglog_level": 1.0,
                "density_neglog_slope": 0.1,
                "vocab_kld_all_level": 0.2,
                "vocab_kld_all_slope": 0.02,
                "cocit_kld_all_level": 0.3,
                "cocit_kld_all_slope": 0.03,
            },
            {
                "author": "B",
                "density_neglog_level": 2.0,
                "density_neglog_slope": -0.1,
                "vocab_kld_all_level": 0.4,
                "vocab_kld_all_slope": -0.02,
                "cocit_kld_all_level": 0.1,
                "cocit_kld_all_slope": -0.01,
            },
        ]
    )

    figures = plot_multimetric(metrics, export_dir=tmp_path)

    assert set(figures) == {
        "multimetric_slope_agreement",
        "multimetric_level_agreement",
        "multimetric_correlations",
    }
    assert (tmp_path / "multimetric_slope_agreement.html").exists()
