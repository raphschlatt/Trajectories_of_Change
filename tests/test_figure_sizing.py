"""Size-regression guard for the plot engine.

Every figure must export at its declared ``FIGURE_SIZES`` size, and
``save_figure`` must refuse to export a figure with no size anywhere — that
would silently fall back to the Kaleido 700x500 default (the historic
"squeeze" bug). These two assertions make the squeeze structurally
impossible to reintroduce.
"""

from __future__ import annotations

import re

import plotly.graph_objects as go
import pytest

from trajectories_of_change.plotting.style import FIGURE_SIZES, save_figure

# Kaleido/Skia writes PDF points as px * 72/96 = px * 0.75, plus a sub-point
# rounding offset (observed up to ~0.6 pt). A 1.5 pt tolerance absorbs that while
# still catching any real regression (the squeeze is off by hundreds of points).
PT_PER_PX = 0.75
TOL_PT = 1.5


def _mediabox_size_pt(path) -> tuple[float, float]:
    data = path.read_bytes()
    match = re.search(
        rb"/MediaBox\s*\[\s*([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)",
        data,
    )
    assert match is not None, f"no /MediaBox found in {path}"
    x0, y0, x1, y1 = (float(v) for v in match.groups())
    return x1 - x0, y1 - y0


@pytest.mark.parametrize("key, size", list(FIGURE_SIZES.items()))
def test_figure_exports_at_declared_size(tmp_path, key, size):
    fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[0, 1, 2]))
    fig.update_layout(width=size.width, height=size.height)
    out = tmp_path / f"{key}.pdf"
    save_figure(fig, out, width=size.width, height=size.height, fmt="pdf")

    w_pt, h_pt = _mediabox_size_pt(out)
    assert abs(w_pt - size.width * PT_PER_PX) <= TOL_PT, (
        f"{key}: width {w_pt}pt != {size.width}px"
    )
    assert abs(h_pt - size.height * PT_PER_PX) <= TOL_PT, (
        f"{key}: height {h_pt}pt != {size.height}px"
    )
    # Never the Kaleido default canvas.
    assert not (abs(w_pt - 525.12) <= TOL_PT and abs(h_pt - 375.12) <= TOL_PT), (
        f"{key} exported at the 700x500 Kaleido default (the squeeze)"
    )


def test_save_figure_rejects_unsized_export(tmp_path):
    fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[0, 1, 2]))  # no layout width/height
    with pytest.raises(ValueError, match="Kaleido"):
        save_figure(fig, tmp_path / "unsized.pdf", fmt="pdf")
