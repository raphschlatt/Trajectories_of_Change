"""Referenced Vocabulary integrated into the consolidated metrics run.

Guards three things:
1. Equivalence: ``ref_vocab_*`` in the consolidated run matches a standalone
   ``ReferencedVocabularyKLD`` computation (same numbers) -> wiring is correct.
2. Graceful skip: when references carry no ``tokens`` column, RV is silently
   omitted (no ``ref_vocab_*`` columns, no error) while OV/CI/Density still run.
3. Availability flag + warning reflect whether RV can run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trajectories_of_change import (
    ReferencedVocabularyKLD,
    load_dataset_bundle,
    run_top_authors_metrics_from_parquets,
)
from trajectories_of_change.multimetric import _empty_welch_sync, summarize_kld_sync

TARGET_UID = "uid:target"


def _publications() -> pd.DataFrame:
    rows = []
    plan = {
        "uid:target": ([("R1", "R2"), ("R1", "R2"), ("R1", "R3"), ("R1", "R2")], (0.0, 0.0)),
        "uid:field": ([("R3", "R4"), ("R3", "R4"), ("R3", "R4"), ("R2", "R4")], (1.0, 1.0)),
    }
    own_tokens = {
        "uid:target": ["alpha", "alpha", "beta"],
        "uid:field": ["gamma", "delta", "delta"],
    }
    for uid, (ref_lists, base) in plan.items():
        who = uid.split(":")[1]
        for offset, refs in enumerate(ref_lists):
            year = 2000 + offset
            rows.append(
                {
                    "Bibcode": f"{who}-{year}",
                    "Year": year,
                    "Author": [who.title()],
                    "author_uids": [uid],
                    "author_display_names": [who.title()],
                    "References": list(refs),
                    "tokens": own_tokens[uid] + ["shared", "common"],
                    "embedding_2d_x": base[0] + 0.01 * offset,
                    "embedding_2d_y": base[1] + 0.02 * offset,
                }
            )
    return pd.DataFrame(rows)


def _references(*, include_tokens: bool = True) -> pd.DataFrame:
    spec = {
        "R1": (["alpha", "alpha", "beta"], "uid:r1"),
        "R2": (["beta", "gamma"], "uid:r2"),
        "R3": (["gamma", "delta"], "uid:r3"),
        "R4": (["delta", "alpha"], "uid:r4"),
    }
    rows = []
    for bibcode, (tokens, uid) in spec.items():
        row = {
            "Bibcode": bibcode,
            "Author": [bibcode],
            "author_uids": [uid],
            "author_display_names": [bibcode],
            "Title": " ".join(tokens),
            "Title_en": " ".join(tokens),
            "Title_lang": "en",
            "Abstract": "",
            "Abstract_en": "",
            "Abstract_lang": "en",
        }
        if include_tokens:
            row["tokens"] = list(tokens)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_bundle(tmp_path: Path, *, include_ref_tokens: bool = True) -> tuple[Path, Path]:
    pubs_path = tmp_path / "publications.parquet"
    refs_path = tmp_path / "references.parquet"
    _publications().to_parquet(pubs_path, index=False)
    _references(include_tokens=include_ref_tokens).to_parquet(refs_path, index=False)
    return pubs_path, refs_path


_RUN_KW = dict(
    targets=[TARGET_UID],
    select_by="uid",
    window_size=1,
    skip_incomplete_slices=False,
    top_k_kld_terms=50,
    run_welch=False,
    include_async=False,
    show_progress=False,
)


def test_referenced_vocab_in_core_matches_standalone(tmp_path: Path) -> None:
    pubs_path, refs_path = _write_bundle(tmp_path, include_ref_tokens=True)
    df = run_top_authors_metrics_from_parquets(pubs_path, refs_path, **_RUN_KW)

    assert len(df) == 1
    row = df.iloc[0]
    # RV columns are present and finite; the other three metrics still run
    # (their numeric behaviour is covered by test_metrics_and_cocitation).
    assert "ref_vocab_kld_all_level" in df.columns
    assert "ref_vocab_kld_all_slope" in df.columns
    assert np.isfinite(float(row["ref_vocab_kld_all_level"]))
    for col in ("vocab_kld_all_level", "cocit_kld_all_level", "density_neglog_level"):
        assert col in df.columns
    for suffix in (
        "slices_total",
        "slices_kld",
        "slices_welch",
        "welch_rows",
        "target_docs_median_kld",
        "field_docs_median_kld",
        "target_tokens_median_kld",
        "field_tokens_median_kld",
        "target_docs_median_welch",
        "field_docs_median_welch",
        "sig_terms_total",
        "sig_terms_median_per_slice",
    ):
        assert f"vocab_{suffix}" in df.columns
        assert f"ref_vocab_{suffix}" in df.columns

    # Standalone reference computation with the same parameters.
    standalone = ReferencedVocabularyKLD(
        _publications(),
        _references(include_tokens=True),
        target_author_uid=TARGET_UID,
        policy="inclusive",
        window_size=1,
        skip_incomplete_slices=False,
        lambda_param=0.5,
        epsilon=1e-12,
        top_k_kld_terms=50,
    )
    rv_sync, _ = standalone.calculate_kld_sync()
    expected = summarize_kld_sync(rv_sync, _empty_welch_sync(), 0.2, welch_enabled=False)

    assert float(row["ref_vocab_kld_all_level"]) == float(expected["kld_all_level"])
    assert float(row["ref_vocab_kld_all_slope"]) == float(expected["kld_all_slope"])


def test_referenced_vocab_skipped_without_reference_tokens(tmp_path: Path) -> None:
    pubs_path, refs_path = _write_bundle(tmp_path, include_ref_tokens=False)

    bundle = load_dataset_bundle(pubs_path, refs_path)
    assert bundle.validation.metric_availability["referenced_vocabulary"] is False

    df = run_top_authors_metrics_from_parquets(pubs_path, refs_path, **_RUN_KW)
    # No RV columns at all, but the other three metrics still ran.
    assert not any(col.startswith("ref_vocab_") for col in df.columns)
    assert "vocab_kld_all_level" in df.columns
    assert "cocit_kld_all_level" in df.columns
    assert "density_neglog_level" in df.columns
