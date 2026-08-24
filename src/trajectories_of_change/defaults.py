"""Public defaults shared by the Python API, CLI, docs, and examples."""

from __future__ import annotations

DEFAULT_ALPHA = 0.2
DEFAULT_EPSILON = 1e-12
DEFAULT_LAMBDA_PARAM = 0.5
DEFAULT_CITATION_AUTHOR_SCOPE = "first_author"
DEFAULT_CITATION_IDENTITY_COUNTING = "document_fractional"
DEFAULT_TARGET_EXCLUSION = "all_docs"
DEFAULT_COCIT_MODE = "authors"
DEFAULT_DENSITY_EMBEDDING_COLS = ("embedding_2d_x", "embedding_2d_y")
DEFAULT_MULTIPLE_TESTING = "fdr_bh"
DEFAULT_MULTIPLE_TESTING_SCOPE = "slice"
DEFAULT_TOP_K_KLD_TERMS = 50
DEFAULT_WINDOW_SIZE = 2
DEFAULT_REFERENCE_POLICY = "inclusive"
METRIC_KEYS = ("own_vocab", "ref_vocab", "density", "citation_identity")
