"""Thread-based parallel engine: results must equal the serial path, and the
RAM-aware worker-count resolver must behave predictably.

Targets are independent and threads share the read-only base, so parallel output
must be identical to serial (row order preserved by ``executor.map``). The
identity test at ``n_jobs=8`` doubles as a thread-safety/race stress test on the
shared frames, event index and precomputes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from trajectories_of_change._parallel import resolve_n_jobs
from trajectories_of_change.cli import _parse_jobs
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
    show_progress=False,
)


@pytest.fixture(scope="module")
def serial_metrics() -> pd.DataFrame:
    return run_top_authors_metrics_from_parquets(PUBLICATIONS, REFERENCES, n_jobs=1, **_RUN_KW)


@pytest.mark.parametrize("n_jobs", [2, 4, 8])
def test_parallel_matches_serial(serial_metrics: pd.DataFrame, n_jobs: int) -> None:
    parallel = run_top_authors_metrics_from_parquets(PUBLICATIONS, REFERENCES, n_jobs=n_jobs, **_RUN_KW)
    # Targets are independent and row order is preserved -> results are identical.
    assert_frame_equal(parallel, serial_metrics, check_exact=False, rtol=1e-9, atol=1e-12)


def test_auto_jobs_runs_and_matches_serial(serial_metrics: pd.DataFrame) -> None:
    parallel = run_top_authors_metrics_from_parquets(PUBLICATIONS, REFERENCES, n_jobs="auto", **_RUN_KW)
    assert_frame_equal(parallel, serial_metrics, check_exact=False, rtol=1e-9, atol=1e-12)


def test_resolve_n_jobs_explicit_int_is_respected() -> None:
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(3) == 3
    assert resolve_n_jobs(0) == 1  # floored to a usable minimum


def test_resolve_n_jobs_negative_counts_from_cores() -> None:
    cores = os.cpu_count() or 1
    assert resolve_n_jobs(-1) == cores
    assert resolve_n_jobs(-2) == max(1, cores - 1)


def test_resolve_n_jobs_auto_is_bounded_by_cores() -> None:
    cores = os.cpu_count() or 1
    resolved = resolve_n_jobs("auto")
    assert 1 <= resolved <= cores


def test_resolve_n_jobs_auto_falls_back_when_ram_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trajectories_of_change._parallel._free_ram_bytes", lambda: None)
    assert resolve_n_jobs("auto") == 1


def test_resolve_n_jobs_auto_caps_to_one_under_tight_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only ~1 GB free, 1.5 GB per worker -> RAM cap drives it to a single worker.
    monkeypatch.setattr("trajectories_of_change._parallel._free_ram_bytes", lambda: 1_000_000_000)
    assert resolve_n_jobs("auto") == 1


def test_resolve_n_jobs_rejects_bad_input() -> None:
    with pytest.raises(TypeError):
        resolve_n_jobs(True)
    with pytest.raises(ValueError):
        resolve_n_jobs("lots")


def test_parse_jobs_cli_helper() -> None:
    assert _parse_jobs("auto") == "auto"
    assert _parse_jobs("AUTO") == "auto"
    assert _parse_jobs("4") == 4
    with pytest.raises(Exception):
        _parse_jobs("many")
