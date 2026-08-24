from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


PUBLICATIONS_REQUIRED_COLUMNS = ("Bibcode", "Year", "Author", "References")
REFERENCES_REQUIRED_COLUMNS = ("Bibcode", "Author")

_COLUMN_ALIASES = {
    "author_uids": ("AuthorUID", "author_ids"),
    "author_display_names": ("AuthorDisplayName",),
    "embedding_2d_x": ("UMAP-1",),
    "embedding_2d_y": ("UMAP-2",),
    "Title_en": ("Title",),
    "Abstract_en": ("Abstract",),
}

_LIST_COLUMNS = {
    "Author",
    "References",
    "author_uids",
    "author_display_names",
    "tokens",
}

_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in _COLUMN_ALIASES.items()
    for alias in aliases
}
COLUMN_ALIASES = dict(_ALIAS_TO_CANONICAL)

PLACEHOLDER_AUTHOR_UID_MARKERS = (
    "::n.author::",
    "::unknown::",
    "no author",
    "unknown",
)


class DatasetValidationError(ValueError):
    """Raised when an input dataset does not satisfy the package contract."""


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metric_availability: dict[str, bool] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise DatasetValidationError("; ".join(self.errors))


@dataclass(slots=True)
class DatasetBundle:
    publications: pd.DataFrame
    references: pd.DataFrame
    manifest: Optional[dict[str, Any]] = None
    provenance: Optional[dict[str, Any]] = None
    validation: Optional[ValidationReport] = None
    cleaning_report: Optional[dict[str, Any]] = None


def _literal_list(value: str) -> Optional[list[str]]:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed if item is not None and str(item) != ""]
    return None


def _coerce_list(
    value: Any,
    *,
    split_semicolon: bool = False,
    allow_scalar_string: bool = True,
    preserve_empty: bool = False,
) -> list[str]:
    """Coerce a list-like/scalar value to ``list[str]``.

    Default: strip items, drop ``None``/empty, scalar strings gated by
    ``allow_scalar_string``. With ``preserve_empty=True`` empty strings are kept,
    ``None`` becomes ``""``, and scalar strings always pass through — i.e. the exact
    behaviour formerly in the separate ``_coerce_list_preserve_empty``.
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        if preserve_empty:
            return ["" if item is None else str(item).strip() for item in value]
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        as_list = value.tolist()
        if isinstance(as_list, list):
            if preserve_empty:
                return ["" if item is None else str(item).strip() for item in as_list]
            return [str(item).strip() for item in as_list if item is not None and str(item).strip()]

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        if preserve_empty:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return ["" if item is None else str(item).strip() for item in parsed]
        else:
            parsed = _literal_list(text)
            if parsed is not None:
                return [item.strip() for item in parsed if item.strip()]

    if split_semicolon and ";" in text:
        if preserve_empty:
            return [part.strip() for part in text.split(";")]
        return [part.strip() for part in text.split(";") if part.strip()]

    if preserve_empty or allow_scalar_string:
        return [text]
    return []


def _coerce_list_column(series: pd.Series, *, split_semicolon: bool = False) -> list[list[str]]:
    """Vectorized equivalent of ``series.apply(lambda v: _coerce_list(v, split_semicolon=...))``.

    Fast path for the overwhelmingly common case where a cell is already a
    container (list/tuple/ndarray) of plain strings: strip + drop-empty, matching
    ``_coerce_list`` lines 97-102 exactly. Any irregular cell (scalar string,
    ``set``, NaN, ``"[...]"`` literal, mixed types) is delegated to the unchanged
    scalar ``_coerce_list`` so the result stays byte-identical.
    """
    result: list[list[str]] = []
    append = result.append
    for value in series.to_numpy():
        if type(value) is list or type(value) is tuple:
            items = value
        elif isinstance(value, np.ndarray):
            items = value.tolist()
        else:
            append(_coerce_list(value, split_semicolon=split_semicolon))
            continue
        for item in items:
            if type(item) is not str:
                append(_coerce_list(value, split_semicolon=split_semicolon))
                break
        else:
            append([s for s in (it.strip() for it in items) if s])
    return result


def _coerce_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"Invalid year value: {value!r}") from exc


def _coerce_bibcode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none"}:
        return ""
    return text


def _rename_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in renamed.columns:
            continue
        for alias in aliases:
            if alias in renamed.columns:
                renamed = renamed.rename(columns={alias: canonical})
                break
    return renamed


def canonicalize_column_name(name: str) -> str:
    return _ALIAS_TO_CANONICAL.get(name, name)


def _normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _rename_alias_columns(df)
    for column in _LIST_COLUMNS.intersection(normalized.columns):
        split_semicolon = column in {"References", "tokens", "author_uids", "author_display_names"}
        normalized[column] = _coerce_list_column(normalized[column], split_semicolon=split_semicolon)
    if "Year" in normalized.columns:
        normalized["Year"] = normalized["Year"].apply(_coerce_year)
    if "Bibcode" in normalized.columns:
        normalized["Bibcode"] = normalized["Bibcode"].apply(_coerce_bibcode)
    return normalized


def normalize_publications_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_common_columns(df)
    if "References" in normalized.columns:
        normalized["References"] = normalized["References"].apply(
            lambda refs: [ref for ref in refs if ref]
        )
    return normalized


def normalize_references_frame(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_common_columns(df)


def _missing_columns(df: pd.DataFrame, required: Sequence[str]) -> list[str]:
    return [column for column in required if column not in df.columns]


def _flatten_string_lists(series: pd.Series) -> set[str]:
    values: set[str] = set()
    for items in series.dropna():
        for item in _coerce_list(items, split_semicolon=True):
            if item:
                values.add(item)
    return values


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().casefold() not in {"nan", "none"}
    if isinstance(value, (list, tuple, set)):
        return any(_is_non_empty(item) for item in value)
    if hasattr(value, "tolist") and not isinstance(value, str):
        return _is_non_empty(value.tolist())
    return True


def is_placeholder_author_uid(value: Any) -> bool:
    """Return True for non-person placeholder author identities."""
    text = str(value).strip().casefold()
    if not text:
        return True
    return any(marker in text for marker in PLACEHOLDER_AUTHOR_UID_MARKERS)


def _metric_availability(publications: pd.DataFrame, references: pd.DataFrame) -> dict[str, bool]:
    return {
        "vocabulary_kld": "tokens" in publications.columns,
        "density": {"embedding_2d_x", "embedding_2d_y"}.issubset(publications.columns),
        "cocitation_works": "References" in publications.columns,
        "cocitation_authors": (
            "References" in publications.columns
            and ("author_uids" in references.columns or "Author" in references.columns)
        ),
        "author_identity": "author_uids" in publications.columns,
        "referenced_vocabulary": "tokens" in references.columns,
    }


def _warn_duplicate_list_values(
    report: ValidationReport,
    df: pd.DataFrame,
    *,
    frame_name: str,
    column: str,
) -> None:
    if column not in df.columns:
        return

    duplicate_rows: list[str] = []
    bibcodes = (
        df["Bibcode"].to_numpy(copy=False)
        if "Bibcode" in df.columns
        else df.index.to_numpy(copy=False)
    )
    for bibcode, values in zip(bibcodes, df[column].to_numpy(copy=False)):
        if not isinstance(values, list):
            continue
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if len(cleaned) != len(set(cleaned)):
            duplicate_rows.append(str(bibcode))

    if duplicate_rows:
        preview = ", ".join(duplicate_rows[:5])
        report.warnings.append(
            f"{frame_name}.{column} contains duplicate values within "
            f"{len(duplicate_rows)} row(s), e.g. {preview}"
        )


def validate_dataset_bundle(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    strict: bool = True,
    drop_missing_references: bool = False,
) -> ValidationReport:
    report = ValidationReport()

    missing_publication_columns = _missing_columns(publications, PUBLICATIONS_REQUIRED_COLUMNS)
    missing_reference_columns = _missing_columns(references, REFERENCES_REQUIRED_COLUMNS)
    if missing_publication_columns:
        report.errors.append(
            f"publications missing required columns: {', '.join(missing_publication_columns)}"
        )
    if missing_reference_columns:
        report.errors.append(
            f"references missing required columns: {', '.join(missing_reference_columns)}"
        )

    if "Bibcode" in publications.columns and publications["Bibcode"].duplicated().any():
        report.errors.append("publications.Bibcode must be unique")
    if "Bibcode" in references.columns and references["Bibcode"].duplicated().any():
        report.errors.append("references.Bibcode must be unique")
    if "Bibcode" in publications.columns and (publications["Bibcode"].astype(str).str.strip() == "").any():
        report.errors.append("publications.Bibcode must not be empty")
    if "Bibcode" in references.columns and (references["Bibcode"].astype(str).str.strip() == "").any():
        report.errors.append("references.Bibcode must not be empty")

    if "References" in publications.columns:
        non_list_mask = ~publications["References"].map(lambda value: isinstance(value, list))
        if non_list_mask.any():
            report.errors.append("publications.References must be a list column")

    if "Author" in publications.columns:
        non_list_mask = ~publications["Author"].map(lambda value: isinstance(value, list))
        if non_list_mask.any():
            report.errors.append("publications.Author must be a list column after normalization")

    if "Author" in references.columns:
        non_list_mask = ~references["Author"].map(lambda value: isinstance(value, list))
        if non_list_mask.any():
            report.errors.append("references.Author must be a list column after normalization")

    if "author_uids" in publications.columns:
        _warn_duplicate_list_values(
            report,
            publications,
            frame_name="publications",
            column="author_uids",
        )

    if "author_uids" in references.columns:
        _warn_duplicate_list_values(
            report,
            references,
            frame_name="references",
            column="author_uids",
        )

    for frame_name, df in (("publications", publications), ("references", references)):
        if "author_uids" in df.columns and "author_display_names" in df.columns:
            mismatch = df.apply(
                lambda row: len(row["author_display_names"]) not in {0, len(row["author_uids"])},
                axis=1,
            )
            if mismatch.any():
                report.errors.append(
                    f"{frame_name}.author_display_names must be empty or positionally aligned "
                    f"with {frame_name}.author_uids"
                )

    if "References" in publications.columns and "Bibcode" in references.columns:
        known_references = set(references["Bibcode"].dropna().astype(str))
        referenced_bibcodes = _flatten_string_lists(publications["References"])
        missing_references = sorted(referenced_bibcodes - known_references)
        if missing_references:
            preview = ", ".join(missing_references[:5])
            message = f"{len(missing_references)} referenced Bibcodes are missing in references: {preview}"
            if strict and not drop_missing_references:
                report.errors.append(message)
            else:
                report.warnings.append(message)

    report.metric_availability = _metric_availability(publications, references)
    if not report.metric_availability["vocabulary_kld"]:
        report.warnings.append("tokens missing in publications: vocabulary KLD unavailable")
    if not report.metric_availability["density"]:
        report.warnings.append(
            "embedding_2d_x/embedding_2d_y missing in publications: density metric unavailable"
        )
    if not report.metric_availability["referenced_vocabulary"]:
        report.warnings.append(
            "tokens missing in references: referenced vocabulary metric unavailable"
        )
    if not report.metric_availability["author_identity"]:
        report.warnings.append(
            "author_uids missing in publications: author selection falls back to author names"
        )
    if strict:
        report.raise_for_errors()
    return report


def _drop_unknown_references(publications: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    known = set(references["Bibcode"].dropna().astype(str))
    trimmed = publications.copy()
    trimmed["References"] = trimmed["References"].apply(
        lambda refs: [ref for ref in refs if ref in known]
    )
    return trimmed


def _empty_cleaning_report(input_publications: int, input_references: int) -> dict[str, Any]:
    return {
        "input_rows": {"publications": int(input_publications), "references": int(input_references)},
        "output_rows": {"publications": int(input_publications), "references": int(input_references)},
        "bibcode": {
            "publications_empty_dropped": 0,
            "references_empty_dropped": 0,
            "publications_duplicate_bibcodes": 0,
            "publications_duplicate_rows_removed": 0,
            "references_duplicate_bibcodes": 0,
            "references_duplicate_rows_removed": 0,
            "publications_duplicate_examples": [],
            "references_duplicate_examples": [],
        },
        "references": {
            "empty_ids_removed": 0,
            "duplicate_ids_removed": 0,
            "missing_ids_removed": 0,
            "missing_unique_ids_removed": 0,
            "missing_examples": [],
        },
        "author_identities": {
            "publications_duplicate_uids_removed": 0,
            "references_duplicate_uids_removed": 0,
            "publications_placeholder_uids_removed": 0,
            "references_placeholder_uids_removed": 0,
            "publications_rows_all_uids_removed": 0,
            "references_rows_all_uids_removed": 0,
        },
    }


def _reference_len(value: Any) -> int:
    return len(_coerce_list(value, split_semicolon=True))


def _non_empty_field_count(row: pd.Series) -> int:
    return sum(1 for value in row.values if _is_non_empty(value))


def _drop_empty_bibcodes(df: pd.DataFrame, *, frame_name: str, report: dict[str, Any]) -> pd.DataFrame:
    if "Bibcode" not in df.columns:
        return df
    empty = df["Bibcode"].astype(str).str.strip() == ""
    count = int(empty.sum())
    report["bibcode"][f"{frame_name}_empty_dropped"] = count
    if count == 0:
        return df
    return df.loc[~empty].copy()


def _deduplicate_bibcodes(df: pd.DataFrame, *, frame_name: str, report: dict[str, Any]) -> pd.DataFrame:
    if "Bibcode" not in df.columns or df.empty:
        return df
    duplicate_mask = df["Bibcode"].duplicated(keep=False)
    duplicate_codes = sorted(df.loc[duplicate_mask, "Bibcode"].astype(str).unique())
    report["bibcode"][f"{frame_name}_duplicate_bibcodes"] = len(duplicate_codes)
    report["bibcode"][f"{frame_name}_duplicate_rows_removed"] = int(duplicate_mask.sum() - len(duplicate_codes))
    report["bibcode"][f"{frame_name}_duplicate_examples"] = duplicate_codes[:10]
    if not duplicate_codes:
        return df

    ranked = df.copy()
    ranked["__toc_original_order__"] = range(len(ranked))
    ranked["__toc_reference_len__"] = (
        ranked["References"].apply(_reference_len) if "References" in ranked.columns else 0
    )
    data_cols = [column for column in ranked.columns if not column.startswith("__toc_")]
    ranked["__toc_non_empty_fields__"] = ranked[data_cols].apply(_non_empty_field_count, axis=1)
    ranked = ranked.sort_values(
        ["Bibcode", "__toc_reference_len__", "__toc_non_empty_fields__", "__toc_original_order__"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    selected = ranked.drop_duplicates("Bibcode", keep="first")
    selected = selected.sort_values("__toc_original_order__", kind="mergesort")
    return selected.loc[:, data_cols].reset_index(drop=True)


def _clean_references_column(publications: pd.DataFrame, references: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    if "References" not in publications.columns or "Bibcode" not in references.columns:
        return publications
    known = set(references["Bibcode"].dropna().astype(str).str.strip())
    missing_unique: set[str] = set()
    examples: list[dict[str, Any]] = []

    def clean_row(row: pd.Series) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        removed_missing: list[str] = []
        for raw in _coerce_list(row.get("References"), split_semicolon=True, preserve_empty=True):
            ref = raw.strip()
            if not ref:
                report["references"]["empty_ids_removed"] += 1
                continue
            if ref in seen:
                report["references"]["duplicate_ids_removed"] += 1
                continue
            seen.add(ref)
            if ref not in known:
                report["references"]["missing_ids_removed"] += 1
                missing_unique.add(ref)
                removed_missing.append(ref)
                continue
            cleaned.append(ref)
        if removed_missing and len(examples) < 10:
            examples.append(
                {
                    "publication_bibcode": str(row.get("Bibcode", "")),
                    "removed": removed_missing[:10],
                }
            )
        return cleaned

    out = publications.copy()
    out["References"] = out.apply(clean_row, axis=1)
    report["references"]["missing_unique_ids_removed"] = len(missing_unique)
    report["references"]["missing_examples"] = examples
    return out


def _clean_author_identities(df: pd.DataFrame, *, frame_name: str, report: dict[str, Any]) -> pd.DataFrame:
    if "author_uids" not in df.columns:
        return df
    out = df.copy()
    has_display = "author_display_names" in out.columns
    duplicate_key = f"{frame_name}_duplicate_uids_removed"
    placeholder_key = f"{frame_name}_placeholder_uids_removed"
    all_removed_key = f"{frame_name}_rows_all_uids_removed"

    def clean_row(row: pd.Series) -> tuple[list[str], list[str]]:
        uids = _coerce_list(row.get("author_uids"), split_semicolon=True, preserve_empty=True)
        displays = _coerce_list(row.get("author_display_names"), split_semicolon=True) if has_display else []
        cleaned_uids: list[str] = []
        cleaned_displays: list[str] = []
        seen: set[str] = set()
        had_uid = bool(uids)
        for idx, uid in enumerate(uids):
            if is_placeholder_author_uid(uid):
                report["author_identities"][placeholder_key] += 1
                continue
            if uid in seen:
                report["author_identities"][duplicate_key] += 1
                continue
            seen.add(uid)
            cleaned_uids.append(uid)
            if has_display:
                display = displays[idx] if idx < len(displays) and displays[idx] else uid
                cleaned_displays.append(display)
        if had_uid and not cleaned_uids:
            report["author_identities"][all_removed_key] += 1
        return cleaned_uids, cleaned_displays

    cleaned = out.apply(clean_row, axis=1)
    out["author_uids"] = [uids for uids, _ in cleaned]
    if has_display:
        out["author_display_names"] = [displays for _, displays in cleaned]
    return out


def _count_empty_reference_entries(publications: pd.DataFrame) -> int:
    if "References" not in publications.columns:
        return 0
    count = 0
    for value in publications["References"]:
        for ref in _coerce_list(value, split_semicolon=True, preserve_empty=True):
            if not ref.strip():
                count += 1
    return count


def _load_json(path: Optional[Path], *, required: bool = False) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        if required:
            raise DatasetValidationError(f"Required sidecar file does not exist: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Optional[Path], *, required: bool = False) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        if required:
            raise DatasetValidationError(f"Required sidecar file does not exist: {path}")
        return None
    try:
        import yaml
    except ImportError as exc:
        raise DatasetValidationError(
            "PyYAML is required to load YAML provenance sidecars. "
            "It is a core dependency; reinstall trajectories-of-change."
        ) from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a local file."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_artifact(path: Path | str, *, artifact_path: str | None = None, include_sha256: bool = True) -> dict[str, Any]:
    """Return a small manifest entry for an artifact path."""
    resolved = Path(path)
    payload: dict[str, Any] = {
        "path": artifact_path or str(resolved),
        "bytes": int(resolved.stat().st_size),
    }
    if include_sha256:
        payload["sha256"] = sha256_file(resolved)
    return payload


def _discover_sidecar(base_dir: Path, filename: str) -> Optional[Path]:
    candidate = base_dir / filename
    return candidate if candidate.exists() else None


def _read_parquet_selected(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    selected = list(dict.fromkeys(columns))
    try:
        return pd.read_parquet(path, columns=selected)
    except Exception:
        df = pd.read_parquet(path)
        selected_set = set(selected)
        available = [
            column
            for column in df.columns
            if column in selected_set or canonicalize_column_name(column) in selected_set
        ]
        return df.loc[:, available].copy()


def _build_provenance(
    *,
    run_summary: Optional[dict[str, Any]],
    config: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if run_summary is None and config is None:
        return None
    return {
        "run_summary": run_summary or {},
        "config": config or {},
    }


def _sidecar_consistency_warnings(
    *,
    manifest: Optional[dict[str, Any]],
    run_summary: Optional[dict[str, Any]],
    config: Optional[dict[str, Any]],
    config_path: Optional[Path],
) -> list[str]:
    warnings: list[str] = []
    if manifest and run_summary:
        manifest_run_id = manifest.get("run_id")
        summary_run_id = run_summary.get("run", {}).get("run_id")
        if manifest_run_id and summary_run_id and str(manifest_run_id) != str(summary_run_id):
            warnings.append(
                "provenance run_id mismatch: "
                f"manifest.run_id={manifest_run_id!r}, run_summary.run.run_id={summary_run_id!r}"
            )

    if run_summary:
        config_file = run_summary.get("reproducibility", {}).get("config_file")
        if config_file and config is None:
            expected = config_path if config_path is not None else config_file
            warnings.append(
                "provenance config missing: "
                f"run_summary.reproducibility.config_file points to {expected!s}"
            )
    return warnings


def _manifest_artifact_warnings(
    manifest: Optional[dict[str, Any]],
    *,
    publications_path: Path,
    references_path: Path,
    check_sha256: bool = False,
) -> list[str]:
    if not manifest or not isinstance(manifest.get("artifacts"), dict):
        return []
    warnings: list[str] = []
    paths = {
        "publications": publications_path,
        "references": references_path,
    }
    for key, path in paths.items():
        entry = manifest["artifacts"].get(key)
        if not isinstance(entry, dict) or not path.exists():
            continue
        if "bytes" in entry and int(entry["bytes"]) != int(path.stat().st_size):
            warnings.append(
                f"manifest artifact bytes mismatch for {key}: "
                f"manifest={entry['bytes']}, loaded={path.stat().st_size}"
            )
        if check_sha256 and entry.get("sha256"):
            digest = sha256_file(path)
            if str(entry["sha256"]) != digest:
                warnings.append(
                    f"manifest artifact sha256 mismatch for {key}: "
                    f"manifest={entry['sha256']}, loaded={digest}"
                )
    return warnings


def build_dataset_bundle(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    manifest: Optional[dict[str, Any]] = None,
    provenance: Optional[dict[str, Any]] = None,
    strict: bool = True,
    drop_missing_references: bool = False,
    cleaning_report: Optional[dict[str, Any]] = None,
    provenance_warnings: Optional[Sequence[str]] = None,
    strict_provenance: bool = False,
) -> DatasetBundle:
    return _finalize_dataset_bundle(
        normalize_publications_frame(publications),
        normalize_references_frame(references),
        manifest=manifest,
        provenance=provenance,
        strict=strict,
        drop_missing_references=drop_missing_references,
        cleaning_report=cleaning_report,
        provenance_warnings=provenance_warnings,
        strict_provenance=strict_provenance,
    )


def _finalize_dataset_bundle(
    publications_norm: pd.DataFrame,
    references_norm: pd.DataFrame,
    *,
    manifest: Optional[dict[str, Any]] = None,
    provenance: Optional[dict[str, Any]] = None,
    strict: bool = True,
    drop_missing_references: bool = False,
    cleaning_report: Optional[dict[str, Any]] = None,
    provenance_warnings: Optional[Sequence[str]] = None,
    strict_provenance: bool = False,
) -> DatasetBundle:
    """Validate and package frames that already satisfy the canonical schema."""
    report = validate_dataset_bundle(
        publications_norm,
        references_norm,
        strict=False if drop_missing_references else strict,
        drop_missing_references=drop_missing_references,
    )
    if manifest and isinstance(manifest.get("counts"), dict):
        counts = manifest["counts"]
        expected_counts = {
            "publications": len(publications_norm),
            "references": len(references_norm),
        }
        for key, actual in expected_counts.items():
            if key in counts and int(counts[key]) != actual:
                report.warnings.append(
                    f"manifest counts mismatch for {key}: manifest={counts[key]}, loaded={actual}"
                )
    provenance_warning_list = list(provenance_warnings or [])
    report.warnings.extend(provenance_warning_list)
    if strict_provenance and provenance_warning_list:
        report.errors.extend(provenance_warning_list)
        raise DatasetValidationError("; ".join(provenance_warning_list))
    if drop_missing_references and "References" in publications_norm.columns and "Bibcode" in references_norm.columns:
        publications_norm = _drop_unknown_references(publications_norm, references_norm)
        report = validate_dataset_bundle(
            publications_norm,
            references_norm,
            strict=False,
            drop_missing_references=drop_missing_references,
        )
        if manifest and isinstance(manifest.get("counts"), dict):
            counts = manifest["counts"]
            if "publications" in counts and int(counts["publications"]) != len(publications_norm):
                report.warnings.append(
                    "manifest counts mismatch for publications after dropping missing references: "
                    f"manifest={counts['publications']}, loaded={len(publications_norm)}"
                )
    if strict:
        report.raise_for_errors()
    return DatasetBundle(
        publications=publications_norm,
        references=references_norm,
        manifest=manifest,
        provenance=provenance,
        validation=report,
        cleaning_report=cleaning_report,
    )


def _prepare_dataset_frames(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    manifest: Optional[dict[str, Any]] = None,
    provenance: Optional[dict[str, Any]] = None,
    strict: bool = True,
    provenance_warnings: Optional[Sequence[str]] = None,
    strict_provenance: bool = False,
) -> DatasetBundle:
    report = _empty_cleaning_report(len(publications), len(references))
    report["references"]["empty_ids_removed"] = _count_empty_reference_entries(publications)
    publications_norm = normalize_publications_frame(publications)
    references_norm = normalize_references_frame(references)

    publications_norm = _drop_empty_bibcodes(publications_norm, frame_name="publications", report=report)
    references_norm = _drop_empty_bibcodes(references_norm, frame_name="references", report=report)
    publications_norm = _deduplicate_bibcodes(publications_norm, frame_name="publications", report=report)
    references_norm = _deduplicate_bibcodes(references_norm, frame_name="references", report=report)
    publications_norm = _clean_references_column(publications_norm, references_norm, report)
    publications_norm = _clean_author_identities(publications_norm, frame_name="publications", report=report)
    references_norm = _clean_author_identities(references_norm, frame_name="references", report=report)

    report["output_rows"] = {
        "publications": int(len(publications_norm)),
        "references": int(len(references_norm)),
    }
    prepared_manifest = dict(manifest or {})
    prepared_manifest["counts"] = {
        "publications": int(len(publications_norm)),
        "references": int(len(references_norm)),
    }
    prepared_manifest["cleaning"] = report

    return _finalize_dataset_bundle(
        publications_norm,
        references_norm,
        manifest=prepared_manifest,
        provenance=provenance,
        strict=strict,
        cleaning_report=report,
        provenance_warnings=provenance_warnings,
        strict_provenance=strict_provenance,
    )


def _load_dataset_files(
    publications_path: Path | str,
    references_path: Path | str,
    *,
    manifest_path: Optional[Path | str] = None,
    run_summary_path: Optional[Path | str] = None,
    config_path: Optional[Path | str] = None,
    auto_discover_sidecars: bool = False,
    publication_columns: Optional[Sequence[str]] = None,
    reference_columns: Optional[Sequence[str]] = None,
    strict_provenance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    publications_path = Path(publications_path)
    references_path = Path(references_path)
    for label, path in (
        ("publications", publications_path),
        ("references", references_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} parquet not found: {path}")
    base_dir = publications_path.parent
    manifest_requested = manifest_path is not None
    run_summary_requested = run_summary_path is not None
    config_requested = config_path is not None
    if auto_discover_sidecars:
        manifest_path = manifest_path or _discover_sidecar(base_dir, "dataset_manifest.json")
        run_summary_path = run_summary_path or _discover_sidecar(base_dir, "run_summary.yaml")
        config_path = config_path or _discover_sidecar(base_dir, "config_used.yaml")

    publications = _read_parquet_selected(publications_path, publication_columns)
    references = _read_parquet_selected(references_path, reference_columns)
    manifest = _load_json(Path(manifest_path) if manifest_path is not None else None, required=manifest_requested)
    run_summary = _load_yaml(
        Path(run_summary_path) if run_summary_path is not None else None,
        required=run_summary_requested,
    )
    if config_path is None and run_summary_path is not None and run_summary:
        config_file = run_summary.get("reproducibility", {}).get("config_file")
        if config_file:
            config_path = Path(run_summary_path).parent / str(config_file)
    config = _load_yaml(Path(config_path) if config_path is not None else None, required=config_requested)
    provenance = _build_provenance(run_summary=run_summary, config=config)
    provenance_warnings = _sidecar_consistency_warnings(
        manifest=manifest,
        run_summary=run_summary,
        config=config,
        config_path=Path(config_path) if config_path is not None else None,
    )
    provenance_warnings.extend(
        _manifest_artifact_warnings(
            manifest,
            publications_path=publications_path,
            references_path=references_path,
            check_sha256=strict_provenance,
        )
    )
    return publications, references, manifest, provenance, provenance_warnings


def load_dataset_bundle(
    publications_path: Path | str,
    references_path: Path | str,
    *,
    manifest_path: Optional[Path | str] = None,
    run_summary_path: Optional[Path | str] = None,
    config_path: Optional[Path | str] = None,
    auto_discover_sidecars: bool = False,
    strict: bool = True,
    drop_missing_references: bool = False,
    validate: Optional[bool] = None,
    publication_columns: Optional[Sequence[str]] = None,
    reference_columns: Optional[Sequence[str]] = None,
    strict_provenance: bool = False,
) -> DatasetBundle:
    if validate is not None:
        strict = validate
    publications, references, manifest, provenance, provenance_warnings = _load_dataset_files(
        publications_path,
        references_path,
        manifest_path=manifest_path,
        run_summary_path=run_summary_path,
        config_path=config_path,
        auto_discover_sidecars=auto_discover_sidecars,
        publication_columns=publication_columns,
        reference_columns=reference_columns,
        strict_provenance=strict_provenance,
    )
    return build_dataset_bundle(
        publications,
        references,
        manifest=manifest,
        provenance=provenance,
        strict=strict,
        drop_missing_references=drop_missing_references,
        provenance_warnings=provenance_warnings,
        strict_provenance=strict_provenance,
    )


def _load_bundle_arg(
    bundle_or_publications_path: DatasetBundle | str | Path,
    references_path: str | Path | None,
    *,
    manifest_path: str | Path | None = None,
    run_summary_path: str | Path | None = None,
    config_path: str | Path | None = None,
    auto_discover_sidecars: bool = False,
    strict_provenance: bool = False,
    assume_valid: bool = False,
    publication_columns: Optional[Sequence[str]] = None,
    reference_columns: Optional[Sequence[str]] = None,
    combined_bundle_error: bool = False,
) -> DatasetBundle:
    """Resolve the shared path-or-bundle input contract for metric facades."""
    if isinstance(bundle_or_publications_path, DatasetBundle):
        if references_path is not None:
            raise ValueError("references_path must be omitted when passing a DatasetBundle")
        sidecar_paths = (manifest_path, run_summary_path, config_path)
        if combined_bundle_error and (
            any(value is not None for value in sidecar_paths)
            or auto_discover_sidecars
            or strict_provenance
            or assume_valid
        ):
            raise ValueError("sidecar, provenance, and assume_valid options apply only to path inputs")
        if any(value is not None for value in sidecar_paths):
            raise ValueError("sidecar paths apply only to path inputs")
        if auto_discover_sidecars or strict_provenance or assume_valid:
            raise ValueError(
                "auto_discover_sidecars, strict_provenance, and assume_valid apply only to path inputs"
            )
        return bundle_or_publications_path
    if references_path is None:
        message = (
            "references_path is required for path inputs"
            if combined_bundle_error
            else "references_path is required when the first argument is not a DatasetBundle"
        )
        raise ValueError(message)
    return load_dataset_bundle(
        bundle_or_publications_path,
        references_path,
        manifest_path=manifest_path,
        run_summary_path=run_summary_path,
        config_path=config_path,
        auto_discover_sidecars=auto_discover_sidecars,
        strict_provenance=strict_provenance,
        validate=not assume_valid,
        publication_columns=publication_columns if assume_valid else None,
        reference_columns=reference_columns if assume_valid else None,
    )


def prepare_dataset_bundle(
    publications_path: Path | str,
    references_path: Path | str,
    *,
    manifest_path: Optional[Path | str] = None,
    run_summary_path: Optional[Path | str] = None,
    config_path: Optional[Path | str] = None,
    auto_discover_sidecars: bool = False,
    strict: bool = True,
    strict_provenance: bool = False,
) -> DatasetBundle:
    """Load and deterministically prepare a raw two-parquet dataset for analysis."""
    publications, references, manifest, provenance, provenance_warnings = _load_dataset_files(
        publications_path,
        references_path,
        manifest_path=manifest_path,
        run_summary_path=run_summary_path,
        config_path=config_path,
        auto_discover_sidecars=auto_discover_sidecars,
        strict_provenance=strict_provenance,
    )
    return _prepare_dataset_frames(
        publications,
        references,
        manifest=manifest,
        provenance=provenance,
        strict=strict,
        provenance_warnings=provenance_warnings,
        strict_provenance=strict_provenance,
    )


def _row_contains_target(values: Any, target_value: str) -> bool:
    normalized_values = _coerce_list(values, split_semicolon=True)
    if not normalized_values:
        return False
    target_norm = str(target_value).strip().casefold()
    return any(value.casefold() == target_norm for value in normalized_values)


def build_target_mask(
    df: pd.DataFrame,
    *,
    target_name: Optional[str] = None,
    target_author_uid: Optional[str] = None,
    author_ids_col: str = "author_uids",
    author_col: str = "Author",
    allow_name_fallback: bool = True,
) -> pd.Series:
    candidate_uid = target_author_uid or None
    candidate_name = target_name or None
    candidate_value = candidate_uid or candidate_name
    if not candidate_value:
        raise ValueError("target_author_uid or target_name must be a non-empty string")

    if candidate_uid and author_ids_col in df.columns:
        mask = df[author_ids_col].map(lambda values: _row_contains_target(values, candidate_uid))
        if mask.any():
            return mask
        if not allow_name_fallback:
            raise DatasetValidationError(
                f"Target {candidate_uid!r} not found in {author_ids_col}; name fallback disabled"
            )
    elif candidate_name and author_ids_col in df.columns:
        # Metric constructors historically accepted an author UID in their
        # positional target-name slot. Keep that behavior without retaining
        # duplicate parameter names in this helper.
        mask = df[author_ids_col].map(lambda values: _row_contains_target(values, candidate_name))
        if mask.any():
            return mask

    if author_col not in df.columns:
        raise DatasetValidationError(
            f"Target {candidate_value!r} not found via IDs and {author_col} is unavailable"
        )
    if not allow_name_fallback:
        raise DatasetValidationError(
            f"Target {candidate_value!r} not found via IDs; name fallback disabled"
        )

    mask = df[author_col].map(lambda values: _row_contains_target(values, candidate_name or candidate_value))
    return mask


def apply_target_field_split(
    corpus: pd.DataFrame,
    *,
    target_mask: Optional[Sequence[bool]] = None,
    target_name: str = "",
    target_author_uid: Optional[str] = None,
    author_col: str = "Author",
    author_id_col: str = "author_uids",
    allow_name_fallback: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tag ``corpus`` with ``__is_target__`` and return ``(target, field)`` views.

    Shared by :class:`BaseKLD` and :class:`KDEDensity` (formerly a byte-copied block
    in each). Resolves the target mask — built via :func:`build_target_mask` or taken
    from a provided ``target_mask`` — adds the ``__is_target__`` column in place, and
    returns the two ``.loc`` splits as ``copy(deep=False)`` views. Behaviour is
    identical to the former inline code (same mask, same error message, same views).
    """
    if target_mask is None:
        target_mask_series = build_target_mask(
            corpus,
            target_name=target_name,
            target_author_uid=target_author_uid,
            author_col=author_col,
            author_ids_col=author_id_col,
            allow_name_fallback=allow_name_fallback,
        )
    else:
        mask_array = np.asarray(target_mask, dtype=bool)
        if mask_array.ndim != 1 or mask_array.size != len(corpus):
            raise ValueError("target_mask must be 1-D with length matching corpus length")
        target_mask_series = pd.Series(mask_array, index=corpus.index)
    corpus["__is_target__"] = target_mask_series
    target_corpus = corpus.loc[corpus["__is_target__"]].copy(deep=False)
    field_corpus = corpus.loc[~corpus["__is_target__"]].copy(deep=False)
    return target_corpus, field_corpus


def _build_uid_display_name_map(
    df: pd.DataFrame,
    *,
    author_ids_col: str = "author_uids",
    author_display_col: str = "author_display_names",
) -> dict[str, Optional[str]]:
    mapping: dict[str, Optional[str]] = {}
    if author_ids_col not in df.columns or author_display_col not in df.columns:
        return mapping
    for raw_ids, raw_labels in zip(
        df[author_ids_col].to_numpy(copy=False),
        df[author_display_col].to_numpy(copy=False),
    ):
        ids = _coerce_list(raw_ids, split_semicolon=True)
        labels = _coerce_list(raw_labels, split_semicolon=True)
        for idx, author_uid in enumerate(ids):
            if idx < len(labels):
                uid = str(author_uid)
                mapping.setdefault(uid, str(labels[idx]).strip() or None)
    return mapping


def resolve_target_label(
    df: pd.DataFrame,
    target_value: str,
    *,
    author_ids_col: str = "author_uids",
    author_display_col: str = "author_display_names",
    author_col: str = "Author",
) -> str:
    uid_label = _build_uid_display_name_map(
        df,
        author_ids_col=author_ids_col,
        author_display_col=author_display_col,
    ).get(target_value)
    if uid_label:
        return uid_label
    if author_col in df.columns:
        for values in df[author_col]:
            authors = _coerce_list(values, split_semicolon=False)
            for author in authors:
                if author.casefold() == target_value.casefold():
                    return author
    return target_value


def resolve_embedding_columns(
    df: pd.DataFrame,
    requested: Sequence[str] = ("embedding_2d_x", "embedding_2d_y"),
) -> list[str]:
    resolved: list[str] = []
    missing: list[str] = []
    for column in requested:
        canonical = canonicalize_column_name(column)
        if canonical in df.columns:
            resolved.append(canonical)
        elif column in df.columns:
            resolved.append(column)
        else:
            missing.append(canonical)
    if missing:
        raise DatasetValidationError(f"missing embedding columns: {missing}")
    return resolved


def _ensure_citation_identity_columns(
    publications: pd.DataFrame,
    *,
    references: Optional[pd.DataFrame],
    mode: str = "authors",
) -> None:
    if "References" not in publications.columns:
        raise DatasetValidationError("References required for Citation Identity")
    if references is None:
        raise DatasetValidationError("references table required for Citation Identity")
    if mode == "authors" and not {"author_uids", "Author"}.intersection(references.columns):
        raise DatasetValidationError(
            "references must provide author_uids or Author for author-mode Citation Identity"
        )
