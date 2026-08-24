"""Per-metric compute toggles: runner kwargs + CLI flags."""

from __future__ import annotations

from pathlib import Path

from trajectories_of_change import run_top_authors_metrics_from_parquets as run
from trajectories_of_change.cli import build_parser
from trajectories_of_change.citation_identity_event import CitationIdentityEventIndex

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "examples" / "data" / "publications.parquet"
REF = ROOT / "examples" / "data" / "references.parquet"
KW = dict(top_n=2, window_size=2, skip_incomplete_slices=False, show_progress=False)


def test_runner_excluding_citation_identity_skips_index_build(monkeypatch):
    def unexpected_build(*args, **kwargs):
        raise AssertionError("Citation Identity index must not be built")

    monkeypatch.setattr(CitationIdentityEventIndex, "from_frames", unexpected_build)
    df = run(PUB, REF, include=("density",), **KW)
    assert len(df) == 2
    assert not [column for column in df if column.startswith("cocit_")]


def test_runner_compute_subset_off_own_and_density():
    df = run(PUB, REF, include=("ref_vocab", "citation_identity"), **KW)
    assert len(df) == 2
    assert not [column for column in df if column.startswith("vocab_")]
    assert not [column for column in df if column.startswith("density_")]


def test_cli_metric_toggle_flags_parse():
    parser = build_parser()
    args = parser.parse_args(
        ["metrics", "p.parquet", "r.parquet", "--out", "o.parquet", "--metrics", "ref_vocab", "density"]
    )
    assert args.metrics == ["ref_vocab", "density"]
    default = parser.parse_args(["metrics", "p.parquet", "r.parquet", "--out", "o.parquet"])
    assert tuple(default.metrics) == ("own_vocab", "ref_vocab", "density", "citation_identity")
