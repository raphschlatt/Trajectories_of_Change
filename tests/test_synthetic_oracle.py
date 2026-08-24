from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trajectories_of_change import KDEDensity, VocabularyKLD, load_dataset_bundle
from trajectories_of_change.multimetric import _prepare_sync_welch, run_top_authors_metrics_from_parquets


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


@pytest.fixture(scope="module")
def oracle_metrics() -> pd.DataFrame:
    df = run_top_authors_metrics_from_parquets(
        PUBLICATIONS,
        REFERENCES,
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
    return df.set_index("author_uid")


def test_synthetic_oracle_generator_writes_valid_bundle(tmp_path: Path) -> None:
    out_dir = tmp_path / "oracle"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_synthetic_oracle_data.py"),
            "--out-dir",
            str(out_dir),
            "--profile",
            "test",
        ],
        cwd=ROOT,
        check=True,
    )

    bundle = load_dataset_bundle(
        out_dir / "publications.parquet",
        out_dir / "references.parquet",
        auto_discover_sidecars=True,
    )

    assert len(bundle.publications) == 340
    assert len(bundle.references) == 32
    assert bundle.validation is not None
    assert bundle.validation.errors == []
    assert bundle.manifest["oracle"]["expected"]["vocabulary_spiky"] == "uid:spiky_vocab_distinct"


def test_oracle_vocab_and_cocitation_rankings(oracle_metrics: pd.DataFrame) -> None:
    assert oracle_metrics.loc["uid:stable_vocab_distinct", "vocab_kld_all_level"] > 10.0
    assert oracle_metrics.loc["uid:stable_vocab_distinct", "vocab_kld_all_level"] > (
        oracle_metrics.loc["uid:field_like", "vocab_kld_all_level"] * 100
    )
    assert oracle_metrics.loc["uid:stable_vocab_distinct", "vocab_kld_all_level"] > (
        oracle_metrics.loc["uid:spiky_vocab_distinct", "vocab_kld_all_level"] * 100
    )

    assert oracle_metrics.loc["uid:citation_distinct", "cocit_kld_all_level"] > 10.0
    assert oracle_metrics.loc["uid:citation_distinct", "cocit_kld_all_level"] > (
        oracle_metrics.loc["uid:field_like", "cocit_kld_all_level"] * 50
    )
    assert oracle_metrics.loc["uid:correlated_distinct", "vocab_kld_all_level"] > 5.0
    assert oracle_metrics.loc["uid:correlated_distinct", "cocit_kld_all_level"] > (
        oracle_metrics.loc["uid:field_like", "cocit_kld_all_level"] * 5
    )


def test_oracle_welch_distinguishes_stable_terms_from_spike_terms() -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES)

    stable = VocabularyKLD(
        bundle.publications,
        "",
        target_author_uid="uid:stable_vocab_distinct",
        window_size=2,
        skip_incomplete_slices=False,
        top_k_kld_terms=20,
        allow_name_fallback=False,
    )
    stable_adj = _prepare_sync_welch(
        stable.perform_welch_tests_all_pairs(sync_only=True),
        method="fdr_bh",
        scope="slice",
    )
    stable_sig = stable_adj[stable_adj["p_adj"] <= 0.2]
    stable_terms = set(stable_sig.loc[stable_sig["kld_contribution"] > 0, "term"])

    assert {"tetrad", "torsion", "gauge", "frame"}.issubset(stable_terms)

    spiky = VocabularyKLD(
        bundle.publications,
        "",
        target_author_uid="uid:spiky_vocab_distinct",
        window_size=2,
        skip_incomplete_slices=False,
        top_k_kld_terms=20,
        allow_name_fallback=False,
    )
    spiky_adj = _prepare_sync_welch(
        spiky.perform_welch_tests_all_pairs(sync_only=True),
        method="fdr_bh",
        scope="slice",
    )
    spiky_sig_terms = set(spiky_adj.loc[spiky_adj["p_adj"] <= 0.2, "term"])

    assert {"singularity_spike", "brane_spike", "instant_spike", "burst_spike"}.isdisjoint(spiky_sig_terms)
    assert len(stable_sig) > len(spiky_adj[spiky_adj["p_adj"] <= 0.2])


def test_oracle_density_slope_and_geometry_trap(oracle_metrics: pd.DataFrame) -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES)

    assert oracle_metrics.loc["uid:density_shift", "density_neglog_slope"] > 0.15
    assert oracle_metrics.loc["uid:density_shift", "density_neglog_level"] > (
        oracle_metrics.loc["uid:geometry_trap", "density_neglog_level"] * 2
    )

    pubs = bundle.publications
    field_like_radius = np.hypot(
        pubs.loc[pubs["oracle_scenario"] == "field_like", "embedding_2d_x"],
        pubs.loc[pubs["oracle_scenario"] == "field_like", "embedding_2d_y"],
    ).median()
    trap_radius = np.hypot(
        pubs.loc[pubs["oracle_scenario"] == "geometry_trap", "embedding_2d_x"],
        pubs.loc[pubs["oracle_scenario"] == "geometry_trap", "embedding_2d_y"],
    ).median()

    assert trap_radius > field_like_radius + 5.0


def test_oracle_slopes_and_correlations_are_recoverable(oracle_metrics: pd.DataFrame) -> None:
    assert oracle_metrics.loc["uid:correlated_distinct", "vocab_kld_all_slope"] > 0.5
    assert oracle_metrics.loc["uid:converging_distinct", "vocab_kld_all_slope"] < -0.5
    assert oracle_metrics.loc["uid:converging_distinct", "cocit_kld_all_slope"] < -1.0

    level_corr = oracle_metrics[
        ["vocab_kld_all_level", "cocit_kld_all_level"]
    ].corr(method="spearman").iloc[0, 1]
    slope_corr = oracle_metrics[
        ["vocab_kld_all_slope", "cocit_kld_all_slope"]
    ].corr(method="spearman").iloc[0, 1]

    assert level_corr > 0.5
    assert slope_corr > 0.4


def test_oracle_density_supports_2d_5d_and_10d() -> None:
    bundle = load_dataset_bundle(PUBLICATIONS, REFERENCES)
    for embedding_cols in [
        ("embedding_2d_x", "embedding_2d_y"),
        tuple(f"embedding_5d_{idx}" for idx in range(5)),
        tuple(f"embedding_10d_{idx}" for idx in range(10)),
    ]:
        density = KDEDensity(
            bundle.publications,
            "",
            target_author_uid="uid:density_shift",
            embedding_cols=embedding_cols,
            window_size=2,
            skip_incomplete_slices=False,
            allow_name_fallback=False,
        )
        sync_df, point_df = density.calculate_density_sync()
        assert len(sync_df) == 5
        assert np.isfinite(sync_df["density_neglog_median"]).all()
        assert np.isfinite(point_df["density_neglog"]).all()
