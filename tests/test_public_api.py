from __future__ import annotations

import subprocess
import sys


def test_public_api_imports_work_outside_repo_root(tmp_path) -> None:
    script = """
import trajectories_of_change
from trajectories_of_change import (
    CitationIdentityKLD,
    KDEDensity,
    MetricResult,
    ReferencedVocabularyKLD,
    VocabularyKLD,
    load_dataset_bundle,
    prepare_dataset_bundle,
    run_metric,
    run_metrics,
    run_top_authors_metrics_from_parquets,
    iter_top_authors_metrics_from_parquets,
)
from trajectories_of_change.citation_identity import CitationIdentityConfig
from trajectories_of_change.contract import (
    COLUMN_ALIASES,
    PUBLICATIONS_REQUIRED_COLUMNS,
    REFERENCES_REQUIRED_COLUMNS,
    build_target_mask,
    canonicalize_column_name,
    is_placeholder_author_uid,
    normalize_publications_frame,
    normalize_references_frame,
    resolve_embedding_columns,
    validate_dataset_bundle,
)
from trajectories_of_change.defaults import (
    DEFAULT_ALPHA,
    DEFAULT_CITATION_AUTHOR_SCOPE,
    DEFAULT_CITATION_IDENTITY_COUNTING,
    DEFAULT_COCIT_MODE,
    DEFAULT_DENSITY_EMBEDDING_COLS,
    DEFAULT_EPSILON,
    DEFAULT_LAMBDA_PARAM,
    DEFAULT_MULTIPLE_TESTING,
    DEFAULT_MULTIPLE_TESTING_SCOPE,
    DEFAULT_TARGET_EXCLUSION,
    DEFAULT_TOP_K_KLD_TERMS,
    DEFAULT_WINDOW_SIZE,
)
from trajectories_of_change.metrics_density import summarize_density_sync
from trajectories_of_change.referenced_vocabulary import build_reference_token_cache
assert trajectories_of_change.__file__
assert set(trajectories_of_change.__all__) == {
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
}
for name in (
    "COLUMN_ALIASES",
    "DEFAULT_ALPHA",
    "DEFAULT_TOP_K_KLD_TERMS",
    "build_reference_token_cache",
    "build_target_mask",
    "canonicalize_column_name",
    "is_placeholder_author_uid",
    "normalize_publications_frame",
    "normalize_references_frame",
    "resolve_embedding_columns",
    "summarize_density_sync",
    "validate_dataset_bundle",
):
    assert not hasattr(trajectories_of_change, name), name
assert load_dataset_bundle.__name__ == "load_dataset_bundle"
assert prepare_dataset_bundle.__name__ == "prepare_dataset_bundle"
assert CitationIdentityConfig.__name__ == "CitationIdentityConfig"
assert CitationIdentityKLD.__name__ == "CitationIdentityKLD"
assert MetricResult.__name__ == "MetricResult"
assert DEFAULT_ALPHA == 0.2
assert DEFAULT_CITATION_AUTHOR_SCOPE == "first_author"
assert DEFAULT_CITATION_IDENTITY_COUNTING == "document_fractional"
assert DEFAULT_COCIT_MODE == "authors"
assert DEFAULT_DENSITY_EMBEDDING_COLS == ("embedding_2d_x", "embedding_2d_y")
assert DEFAULT_EPSILON == 1e-12
assert DEFAULT_LAMBDA_PARAM == 0.5
assert DEFAULT_MULTIPLE_TESTING == "fdr_bh"
assert DEFAULT_MULTIPLE_TESTING_SCOPE == "slice"
assert DEFAULT_TARGET_EXCLUSION == "all_docs"
assert DEFAULT_TOP_K_KLD_TERMS == 50
assert DEFAULT_WINDOW_SIZE == 2
assert COLUMN_ALIASES["AuthorUID"] == "author_uids"
assert PUBLICATIONS_REQUIRED_COLUMNS == ("Bibcode", "Year", "Author", "References")
assert REFERENCES_REQUIRED_COLUMNS == ("Bibcode", "Author")
assert build_target_mask.__name__ == "build_target_mask"
assert canonicalize_column_name("UMAP-1") == "embedding_2d_x"
assert is_placeholder_author_uid("unknown") is True
assert normalize_publications_frame.__name__ == "normalize_publications_frame"
assert normalize_references_frame.__name__ == "normalize_references_frame"
assert resolve_embedding_columns.__name__ == "resolve_embedding_columns"
assert validate_dataset_bundle.__name__ == "validate_dataset_bundle"
assert build_reference_token_cache.__name__ == "build_reference_token_cache"
assert summarize_density_sync.__name__ == "summarize_density_sync"
assert VocabularyKLD.__name__ == "VocabularyKLD"
assert KDEDensity.__name__ == "KDEDensity"
assert ReferencedVocabularyKLD.__name__ == "ReferencedVocabularyKLD"
assert run_metric.__name__ == "run_metric"
assert run_metrics.__name__ == "run_metrics"
assert run_top_authors_metrics_from_parquets.__name__ == "run_top_authors_metrics_from_parquets"
assert iter_top_authors_metrics_from_parquets.__name__ == "iter_top_authors_metrics_from_parquets"
print("ok")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
