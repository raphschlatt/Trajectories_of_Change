"""Generate the deterministic synthetic oracle bundle used by examples/tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = 1729
YEARS = tuple(range(2000, 2010))

FIELD_TERMS = (
    "gravity",
    "relativity",
    "spacetime",
    "metric",
    "field",
    "equation",
    "cosmos",
    "mass",
)
STABLE_TERMS = ("tetrad", "torsion", "gauge", "frame")
SPIKE_TERMS = ("singularity_spike", "brane_spike", "instant_spike", "burst_spike")
CORRELATED_TERMS = ("holography", "brane", "duality", "string")
CONVERGING_TERMS = ("ether", "machian", "scalar", "inertia")

TARGET_AUTHORS = {
    "uid:field_like": ("Field-Like, F.", "field_like"),
    "uid:stable_vocab_distinct": ("Stable Vocabulary, V.", "stable_vocab_distinct"),
    "uid:spiky_vocab_distinct": ("Spiky Vocabulary, S.", "spiky_vocab_distinct"),
    "uid:citation_distinct": ("Citation Distinct, C.", "citation_distinct"),
    "uid:density_shift": ("Density Shift, D.", "density_shift"),
    "uid:correlated_distinct": ("Correlated Distinct, K.", "correlated_distinct"),
    "uid:converging_distinct": ("Converging Distinct, N.", "converging_distinct"),
    "uid:geometry_trap": ("Geometry Trap, G.", "geometry_trap"),
}

PROFILES = {
    "test": {"field_docs_per_year": 18, "target_docs_per_author_year": 2},
    "demo": {"field_docs_per_year": 80, "target_docs_per_author_year": 3},
}

CENTERS = {
    "core": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "outer_dense": np.array([6.0, 6.0, 3.0, 3.0, 2.0, 1.0, -1.0, 0.5, -0.5, 1.5]),
    "sparse": np.array([3.0, -4.0, 5.0, -2.0, 1.0, 4.0, -3.0, 2.0, -1.0, 0.0]),
}


def _repeat(tokens: Iterable[str], n: int) -> list[str]:
    out: list[str] = []
    for token in tokens:
        out.extend([token] * int(n))
    return out


def _field_tokens(year: int, serial: int) -> list[str]:
    variant = (f"field_variant_{(year + serial) % 4}",)
    return _repeat(FIELD_TERMS, 3) + _repeat(variant, 2)


def _target_tokens(scenario: str, year: int, doc_no: int) -> list[str]:
    if scenario == "stable_vocab_distinct":
        return _repeat(FIELD_TERMS, 1) + _repeat(STABLE_TERMS, 7)
    if scenario == "spiky_vocab_distinct":
        if year == 2004 and doc_no == 0:
            return _repeat(FIELD_TERMS, 1) + _repeat(SPIKE_TERMS, 16)
        return _field_tokens(year, doc_no)
    if scenario == "correlated_distinct":
        repeats = 1 + max(0, year - 2000) // 2
        return _repeat(FIELD_TERMS, 2) + _repeat(CORRELATED_TERMS, repeats)
    if scenario == "converging_distinct":
        repeats = max(0, 6 - 2 * max(0, year - 2000))
        return _repeat(FIELD_TERMS, 2) + _repeat(CONVERGING_TERMS, repeats)
    return _field_tokens(year, doc_no)


def _reference_tokens(idx: int, group: str) -> list[str]:
    """Deterministic tokenized reference text so Referenced Vocabulary can run.

    Each reference carries two rotating field terms plus a group marker; the
    rotation gives enough cross-reference variety for a non-degenerate KLD signal.
    """
    first = FIELD_TERMS[idx % len(FIELD_TERMS)]
    second = FIELD_TERMS[(idx + 3) % len(FIELD_TERMS)]
    return _repeat((first, second), 3) + [f"ref_{group}"]


def _coords(region: str, year: int, serial: int) -> np.ndarray:
    center = CENTERS[region]
    phase = (year - YEARS[0] + 1) * 0.71 + serial * 0.37
    scale = {"core": 1.0, "outer_dense": 0.25, "sparse": 0.8}[region]
    jitter = np.array(
        [
            scale * (np.sin(phase + dim * 0.43) * 0.055 + np.cos(phase * 0.7 + dim) * 0.025)
            for dim in range(10)
        ]
    )
    return center + jitter


def _add_embedding_columns(row: dict, coords: np.ndarray) -> dict:
    row["embedding_2d_x"] = float(coords[0])
    row["embedding_2d_y"] = float(coords[1])
    for dim in range(5):
        row[f"embedding_5d_{dim}"] = float(coords[dim])
    for dim in range(10):
        row[f"embedding_10d_{dim}"] = float(coords[dim])
    return row


def _reference_ids(prefix: str, n: int = 8) -> list[str]:
    return [f"SYNREF-{prefix}-{idx:02d}" for idx in range(n)]


COMMON_REFS = _reference_ids("COM")
CITATION_REFS = _reference_ids("CIT")
CORRELATED_REFS = _reference_ids("COR")
CONVERGING_REFS = _reference_ids("CNV")
ALL_REFS = COMMON_REFS + CITATION_REFS + CORRELATED_REFS + CONVERGING_REFS


def _rotating_refs(pool: list[str], year: int, serial: int, n: int = 4) -> list[str]:
    start = (year - YEARS[0] + serial) % len(pool)
    return [pool[(start + offset) % len(pool)] for offset in range(n)]


def _target_refs(scenario: str, year: int, doc_no: int) -> list[str]:
    if scenario == "citation_distinct":
        return _rotating_refs(CITATION_REFS, year, doc_no)
    if scenario == "stable_vocab_distinct":
        return _rotating_refs(CORRELATED_REFS, year, doc_no)
    if scenario == "correlated_distinct":
        pool = COMMON_REFS if year <= 2001 else CORRELATED_REFS
        return _rotating_refs(pool, year, doc_no)
    if scenario == "converging_distinct":
        pool = CONVERGING_REFS if year <= 2003 else COMMON_REFS
        return _rotating_refs(pool, year, doc_no)
    return _rotating_refs(COMMON_REFS, year, doc_no)


def _target_region(scenario: str, year: int) -> str:
    if scenario == "density_shift":
        if year <= 2001:
            return "core"
        return "sparse"
    if scenario == "geometry_trap":
        return "outer_dense"
    return "core"


def _citation_count(scenario: str, year: int, serial: int) -> int:
    age = max(0, YEARS[-1] - year + 1)
    scenario_weight = {
        "citation_distinct": 9,
        "correlated_distinct": 7,
        "stable_vocab_distinct": 5,
        "density_shift": 4,
        "spiky_vocab_distinct": 3,
        "converging_distinct": 3,
        "geometry_trap": 2,
        "field_background": 2,
    }.get(scenario, 1)
    return int(age * scenario_weight + (serial % 5))


def _publication_row(
    *,
    bibcode: str,
    year: int,
    author_uid: str,
    author_name: str,
    tokens: list[str],
    references: list[str],
    scenario: str,
    region: str,
    serial: int,
) -> dict:
    row = {
        "Bibcode": bibcode,
        "Year": int(year),
        "Title": f"Synthetic oracle paper: {scenario} {year}",
        "Abstract": f"Deterministic synthetic document for scenario {scenario}.",
        "Author": [author_name],
        "author_uids": [author_uid],
        "author_display_names": [author_name],
        "References": references,
        "Citation Count": _citation_count(scenario, year, serial),
        "tokens": tokens,
        "oracle_scenario": scenario,
        "oracle_region": region,
    }
    return _add_embedding_columns(row, _coords(region, year, serial))


def build_bundle(profile: str = "test") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    cfg = PROFILES[profile]
    rows: list[dict] = []
    serial = 0

    field_author_count = 20
    for year in YEARS:
        for idx in range(cfg["field_docs_per_year"]):
            if idx < int(cfg["field_docs_per_year"] * 0.42):
                region = "core"
            elif idx < int(cfg["field_docs_per_year"] * 0.94):
                region = "outer_dense"
            else:
                region = "sparse"
            author_idx = (year + idx) % field_author_count
            rows.append(
                _publication_row(
                    bibcode=f"SYN{year}FIELD{idx:03d}",
                    year=year,
                    author_uid=f"uid:field_{author_idx:02d}",
                    author_name=f"Field Author {author_idx:02d}",
                    tokens=_field_tokens(year, idx),
                    references=_rotating_refs(COMMON_REFS, year, idx),
                    scenario="field_background",
                    region=region,
                    serial=serial,
                )
            )
            serial += 1

        for author_uid, (author_name, scenario) in TARGET_AUTHORS.items():
            for doc_no in range(cfg["target_docs_per_author_year"]):
                region = _target_region(scenario, year)
                rows.append(
                    _publication_row(
                        bibcode=f"SYN{year}{scenario.upper()[:5]}{doc_no:02d}",
                        year=year,
                        author_uid=author_uid,
                        author_name=author_name,
                        tokens=_target_tokens(scenario, year, doc_no),
                        references=_target_refs(scenario, year, doc_no),
                        scenario=scenario,
                        region=region,
                        serial=serial,
                    )
                )
                serial += 1

    ref_rows: list[dict] = []
    for idx, bibcode in enumerate(ALL_REFS):
        group = bibcode.split("-")[1].lower()
        ref_rows.append(
            {
                "Bibcode": bibcode,
                "Author": [f"Reference {group.upper()} {idx % 8:02d}"],
                "author_uids": [f"uid:ref_{group}_{idx % 8:02d}"],
                "author_display_names": [f"Reference {group.upper()} {idx % 8:02d}"],
                "tokens": _reference_tokens(idx, group),
            }
        )

    publications = pd.DataFrame(rows)
    references = pd.DataFrame(ref_rows)
    oracle = {
        "profile": profile,
        "seed": SEED,
        "years": [int(year) for year in YEARS],
        "target_authors": {
            uid: {"display_name": name, "scenario": scenario}
            for uid, (name, scenario) in TARGET_AUTHORS.items()
        },
        "expected": {
            "vocabulary_high": ["uid:stable_vocab_distinct", "uid:correlated_distinct"],
            "vocabulary_spiky": "uid:spiky_vocab_distinct",
            "cocitation_high": ["uid:citation_distinct", "uid:correlated_distinct"],
            "density_increasing_neglog": "uid:density_shift",
            "density_geometric_trap": "uid:geometry_trap",
            "converging_negative_kld_slope": "uid:converging_distinct",
        },
    }
    return publications, references, oracle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(out_dir: Path, *, profile: str = "test") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    publications, references, oracle = build_bundle(profile)

    publications_path = out_dir / "publications.parquet"
    references_path = out_dir / "references.parquet"
    publications.to_parquet(publications_path, index=False)
    references.to_parquet(references_path, index=False)

    manifest = {
        "schema_version": 1,
        "run_id": f"synthetic-oracle-v1-{profile}",
        "producer": "trajectories-of-change-synthetic-oracle",
        "producer_version": "0.1.0",
        "and_enabled": False,
        "counts": {
            "publications": int(len(publications)),
            "references": int(len(references)),
        },
        "artifacts": [
            {
                "path": "publications.parquet",
                "sha256": _sha256(publications_path),
            },
            {
                "path": "references.parquet",
                "sha256": _sha256(references_path),
            },
        ],
        "oracle": oracle,
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_summary.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "run:",
                f"  run_id: synthetic-oracle-v1-{profile}",
                "producer:",
                "  name: trajectories-of-change-synthetic-oracle",
                "  version: 0.1.0",
                "reproducibility:",
                "  config_file: config_used.yaml",
                "  git_commit: synthetic-oracle",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "config_used.yaml").write_text(
        "\n".join(
            [
                "run:",
                f"  random_seed: {SEED}",
                "search:",
                "  query: synthetic oracle corpus for trajectories-of-change",
                "topic_model:",
                "  embedding_provider: synthetic",
                "  embedding_model: hand-authored-oracle",
                "  reduction_method: synthetic_2d_5d_10d",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="examples/data",
        type=Path,
        help="Output directory for the synthetic oracle bundle.",
    )
    parser.add_argument(
        "--profile",
        default="test",
        choices=sorted(PROFILES),
        help="Corpus size profile.",
    )
    args = parser.parse_args()
    write_bundle(args.out_dir, profile=args.profile)
    print(f"Wrote synthetic oracle bundle to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
