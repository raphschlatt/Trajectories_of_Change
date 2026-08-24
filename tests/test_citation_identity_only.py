"""Citation-Identity-only mode must yield byte-identical cocit_* columns.

The canonical ``include=(...)`` selector lets a caller skip metrics it does
not consume.
The Citation-Identity policy-robustness runner only reads ``cocit_*`` columns,
so it disables OV/RV/Density. This test pins the contract that doing so:

* leaves every ``cocit_*`` column (and ``author_uid``) numerically identical to
  the full multimetric run (the shared cocit assembly path is untouched), and
* drops the OV/RV/Density columns entirely (no empty/NaN residue).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from trajectories_of_change.multimetric import run_top_authors_metrics_from_parquets

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA = ROOT / "examples" / "data"
PUBLICATIONS = EXAMPLE_DATA / "publications.parquet"
REFERENCES = EXAMPLE_DATA / "references.parquet"

TARGETS = [
    "uid:field_like",
    "uid:stable_vocab_distinct",
    "uid:spiky_vocab_distinct",
    "uid:citation_distinct",
    "uid:density_shift",
    "uid:correlated_distinct",
    "uid:converging_distinct",
    "uid:geometry_trap",
]

# include_async + run_welch exercise the full cocit assembly: sync
# summary, Welch sync, async summary and the diagnostics-derived columns.
_RUN_KW = dict(
    auto_discover_sidecars=True,
    targets=TARGETS,
    select_by="uid",
    window_size=2,
    skip_incomplete_slices=False,
    top_k_kld_terms=20,
    alpha=0.2,
    multiple_testing="fdr_bh",
    multiple_testing_scope="slice",
    citation_identity_counting="multiplicity",
    target_exclusion="none",
    remove_self_loops=False,
    run_welch=True,
    include_async=True,
    show_progress=False,
    n_jobs=1,
)


@pytest.fixture(scope="module")
def full_metrics() -> pd.DataFrame:
    """Full run: OV + RV + Citation-Identity + Density (all toggles default True)."""
    return run_top_authors_metrics_from_parquets(PUBLICATIONS, REFERENCES, **_RUN_KW)


@pytest.fixture(scope="module")
def citation_identity_only_metrics() -> pd.DataFrame:
    """Citation-Identity-only run: OV/RV/Density disabled."""
    return run_top_authors_metrics_from_parquets(
        PUBLICATIONS,
        REFERENCES,
        include=("citation_identity",),
        **_RUN_KW,
    )


def test_cocit_columns_are_byte_identical(
    full_metrics: pd.DataFrame, citation_identity_only_metrics: pd.DataFrame
) -> None:
    cocit_cols = [c for c in full_metrics.columns if c.startswith("cocit_")]
    # Sanity: the full run actually produced cocit columns to compare.
    assert cocit_cols, "expected cocit_* columns in the full multimetric output"
    key_and_cocit = ["author_uid", *cocit_cols]

    full_view = full_metrics[key_and_cocit].reset_index(drop=True)
    ci_only_view = citation_identity_only_metrics[key_and_cocit].reset_index(drop=True)
    assert_frame_equal(ci_only_view, full_view, check_exact=False, rtol=1e-9, atol=1e-12)


def test_other_metric_columns_are_dropped(
    citation_identity_only_metrics: pd.DataFrame,
) -> None:
    cols = list(citation_identity_only_metrics.columns)
    assert not [c for c in cols if c.startswith("vocab_")], "OV columns should be absent"
    assert not [c for c in cols if c.startswith("ref_vocab_")], "RV columns should be absent"
    assert not [c for c in cols if c.startswith("density_")], "density columns should be absent"
    # Citation-Identity columns and the author key must remain.
    assert "author_uid" in cols
    assert [c for c in cols if c.startswith("cocit_")]
