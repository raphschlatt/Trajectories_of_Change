from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _canonical_publications() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Bibcode": "2000A&A...000..001S",
                "Year": 2000,
                "Author": ["Smith, A.", "Jones, B."],
                "AuthorUID": ["uid:smith", "uid:jones"],
                "AuthorDisplayName": ["Smith, Alice", "Jones, Bob"],
                "References": ["1990Ref....001A", "1990Ref....002B"],
                "tokens": ["alpha", "beta", "gamma"],
                "UMAP-1": 0.0,
                "UMAP-2": 0.2,
            },
            {
                "Bibcode": "2000A&A...000..003J",
                "Year": 2000,
                "Author": ["Jones, B."],
                "AuthorUID": ["uid:jones"],
                "AuthorDisplayName": ["Jones, Bob"],
                "References": ["1990Ref....002B", "1990Ref....003C"],
                "tokens": ["delta", "epsilon", "zeta"],
                "UMAP-1": 1.0,
                "UMAP-2": 1.2,
            },
            {
                "Bibcode": "2001A&A...000..002S",
                "Year": 2001,
                "Author": ["Smith A."],
                "AuthorUID": ["uid:smith"],
                "AuthorDisplayName": ["Smith, Alice"],
                "References": ["1990Ref....001A", "1990Ref....003C"],
                "tokens": ["alpha", "entropy", "black"],
                "UMAP-1": 0.1,
                "UMAP-2": 0.3,
            },
            {
                "Bibcode": "2001A&A...000..004E",
                "Year": 2001,
                "Author": ["Ellis, G. F. R."],
                "AuthorUID": ["uid:ellis"],
                "AuthorDisplayName": ["Ellis, George F. R."],
                "References": ["1990Ref....003C", "1990Ref....004D"],
                "tokens": ["cosmology", "field", "equation"],
                "UMAP-1": 1.3,
                "UMAP-2": 1.5,
            },
        ]
    )


def _canonical_references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Bibcode": "1990Ref....001A",
                "Author": ["Smith, A."],
                "AuthorUID": ["uid:smith"],
                "AuthorDisplayName": ["Smith, Alice"],
            },
            {
                "Bibcode": "1990Ref....002B",
                "Author": ["Jones, B."],
                "AuthorUID": ["uid:jones"],
                "AuthorDisplayName": ["Jones, Bob"],
            },
            {
                "Bibcode": "1990Ref....003C",
                "Author": ["Ellis, G. F. R."],
                "AuthorUID": ["uid:ellis"],
                "AuthorDisplayName": ["Ellis, George F. R."],
            },
            {
                "Bibcode": "1990Ref....004D",
                "Author": ["Carter, B."],
                "AuthorUID": ["uid:carter"],
                "AuthorDisplayName": ["Carter, Brandon"],
            },
        ]
    )


@pytest.fixture
def canonical_bundle_paths(tmp_path: Path) -> tuple[Path, Path]:
    publications_path = tmp_path / "publications.parquet"
    references_path = tmp_path / "references.parquet"
    _canonical_publications().to_parquet(publications_path, index=False)
    _canonical_references().to_parquet(references_path, index=False)
    return publications_path, references_path


@pytest.fixture
def alias_publications_df() -> pd.DataFrame:
    return _canonical_publications().copy()
