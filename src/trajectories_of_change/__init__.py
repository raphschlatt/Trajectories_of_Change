"""Public package API for Trajectories of Change."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


__version__ = "0.2.0"

__all__ = [
    "CitationIdentityKLD",
    "KDEDensity",
    "MetricResult",
    "ReferencedVocabularyKLD",
    "VocabularyKLD",
    "iter_top_authors_metrics_from_parquets",
    "load_dataset_bundle",
    "prepare_dataset_bundle",
    "run_metric",
    "run_metrics",
    "run_top_authors_metrics_from_parquets",
]

_EXPORTS = {
    "CitationIdentityKLD": ("citation_identity_event", "CitationIdentityKLD"),
    "KDEDensity": ("metrics_density", "KDEDensity"),
    "MetricResult": ("metric_result", "MetricResult"),
    "ReferencedVocabularyKLD": ("referenced_vocabulary", "ReferencedVocabularyKLD"),
    "VocabularyKLD": ("metrics_kld", "VocabularyKLD"),
    "iter_top_authors_metrics_from_parquets": ("multimetric", "iter_top_authors_metrics_from_parquets"),
    "load_dataset_bundle": ("contract", "load_dataset_bundle"),
    "prepare_dataset_bundle": ("contract", "prepare_dataset_bundle"),
    "run_metric": ("api", "run_metric"),
    "run_metrics": ("api", "run_metrics"),
    "run_top_authors_metrics_from_parquets": ("multimetric", "run_top_authors_metrics_from_parquets"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .api import run_metric, run_metrics
    from .citation_identity_event import CitationIdentityKLD
    from .contract import load_dataset_bundle, prepare_dataset_bundle
    from .metric_result import MetricResult
    from .metrics_density import KDEDensity
    from .metrics_kld import VocabularyKLD
    from .multimetric import (
        iter_top_authors_metrics_from_parquets,
        run_top_authors_metrics_from_parquets,
    )
    from .referenced_vocabulary import ReferencedVocabularyKLD
