"""Uniform per-metric result object.

Metric ``.result()`` accessors and the simple facade return
:class:`MetricResult` so own vocabulary, referenced vocabulary, Citation
Identity and density share one transport shape. ``kind`` selects the plotting
path; ``metric`` names the measure. Schema-v1 Citation Image folders remain
loadable as legacy data. This
subsumes the legacy ``CitationIdentitySyncKLDResult`` (now an alias).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .defaults import DEFAULT_WINDOW_SIZE


_RESULT_SCHEMA_VERSION = 2


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class MetricResult:
    """Sync/pointwise/async/Welch tables for one metric on one target.

    ``sync`` and ``pointwise`` are always present; ``async_df`` and ``welch`` are
    optional (density has no Welch). Field order keeps ``sync``/``pointwise`` first (required)
    and everything else defaulted, so keyword construction of the legacy alias
    keeps working.
    """

    sync: pd.DataFrame
    pointwise: pd.DataFrame
    async_df: pd.DataFrame | None = None
    welch: pd.DataFrame | None = None
    kind: str = "kld"          # "kld" | "density" (legacy folders may say "citation_image")
    metric: str = "kld"        # "own_vocab" | "ref_vocab" | "citation_identity" | "density"
    target_author_uid: str | None = None
    target_name: str = ""
    window_size: int = DEFAULT_WINDOW_SIZE
    config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        """Write this result using the canonical versioned folder format."""
        result_dir = Path(directory)
        result_dir.mkdir(parents=True, exist_ok=True)
        table_specs = {
            "sync": ("sync.parquet", self.sync),
            "pointwise": ("pointwise.parquet", self.pointwise),
            "async": ("async.parquet", self.async_df),
            "welch": ("welch.parquet", self.welch),
        }
        tables: dict[str, str] = {}
        for name, (filename, table) in table_specs.items():
            if table is None:
                continue
            table.to_parquet(result_dir / filename, index=False)
            tables[name] = filename

        manifest = {
            "schema_version": _RESULT_SCHEMA_VERSION,
            "metric": self.metric,
            "kind": self.kind,
            "target_author_uid": self.target_author_uid,
            "target_name": self.target_name,
            "window_size": int(self.window_size),
            "config": _jsonable(self.config),
            "provenance": _jsonable(self.provenance),
            "metadata": _jsonable(self.metadata),
            "tables": tables,
        }
        (result_dir / "metric_result.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result_dir

    @classmethod
    def load(cls, directory: str | Path) -> "MetricResult":
        """Load schema-v1 or schema-v2 result folders."""
        result_dir = Path(directory)
        manifest_path = result_dir / "metric_result.json"
        if not manifest_path.exists():
            raise ValueError(f"metric_result.json not found in {result_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_version = int(manifest.get("schema_version", 1))
        if schema_version not in {1, _RESULT_SCHEMA_VERSION}:
            raise ValueError(f"unsupported MetricResult schema_version: {schema_version}")
        tables = dict(manifest.get("tables") or {})

        def read_table(name: str) -> pd.DataFrame | None:
            filename = tables.get(name)
            if not filename:
                return None
            path = result_dir / str(filename)
            if not path.exists():
                raise ValueError(f"metric result table not found: {path}")
            return pd.read_parquet(path)

        sync = read_table("sync")
        if sync is None:
            raise ValueError(f"metric result in {result_dir} is missing sync table")
        pointwise = read_table("pointwise")
        return cls(
            sync=sync,
            pointwise=pointwise if pointwise is not None else pd.DataFrame(),
            async_df=read_table("async"),
            welch=read_table("welch"),
            kind=str(manifest.get("kind") or "kld"),
            metric=str(manifest.get("metric") or "kld"),
            target_author_uid=(
                str(manifest["target_author_uid"])
                if manifest.get("target_author_uid") is not None
                else None
            ),
            target_name=str(manifest.get("target_name") or ""),
            window_size=int(manifest.get("window_size") or DEFAULT_WINDOW_SIZE),
            config=dict(manifest.get("config") or {}),
            provenance=dict(manifest.get("provenance") or {}),
            metadata=dict(manifest.get("metadata") or {}),
        )
