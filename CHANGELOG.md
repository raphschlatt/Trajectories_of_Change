# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [0.2.1] - 2026-11-23

### Changed

- README, citation metadata, and `docs/methods.md` now refer to the accepted
  follow-up study in *Frontiers in Complex Systems*: its title and status,
  the settings of its analysis run, its reproduction package, and a caution
  against reading raw lead/lag and density values as ahead of or behind the
  field.
- Documentation corrections: the level columns are medians; the follow-up
  study uses PaCMAP; `validate_dataset_bundle` takes the two tables; a run
  without embedding columns stops unless density is deselected;
  `references.author_uids` is listed as an input of Citation Identity.

### Fixed

- `max_vocab_size` for Own Vocabulary now keeps the most frequent terms among
  those that pass `min_token_global_freq` and `min_docs_global_freq`.
  Previously the cap was taken over all terms, so a term repeated within a
  single document could bypass `min_docs_global_freq`. Runs without
  `max_vocab_size` or with `min_docs_global_freq=1` are unchanged.
- Citation Identity diagnostics now follow the computed time slices:
  `cocit_diagnostics_by_slice` aggregates per slice (window end year) instead
  of per publication year, and the `slice` column of `cocit_diagnostics`
  holds each document's slice label, empty for documents outside all slices.
  The values change only for windows wider than one year or when documents
  fall outside the analysed years.
- `run_metric(metric="citation_identity")` now stores the requested
  `window_size`, `start_year`, `end_year`, and `skip_incomplete_slices` in
  the result instead of the defaults.

## [0.2.0] - 2026-08-24

### Added

- Added versioned `MetricResult` persistence with resolved configuration,
  target UID, provenance, and schema-v1 loading compatibility.
- Added PEP 561 typing metadata and CI coverage for Python 3.11–3.13 on Linux
  and Windows.

### Changed

- Consolidated the package around four production metrics: Own Vocabulary,
  Referenced Vocabulary, Density/EDE, and outgoing Citation Identity.
- Made bundle-based multimetric runs operate directly in memory, preserving
  provenance without a temporary Parquet round-trip.
- Unified significance configuration from computation through plotting,
  including explicit multiple-testing method and scope.
- Simplified the CLI, plotting facade, defaults, filenames, and advanced
  option routing while retaining strict validation and reproducible outputs.
- Centralized multimetric option resolution and per-target state, eliminated
  duplicate CLI bundle loads, and avoided repeated contract normalization.
- Required a Plotly version compatible with current Kaleido releases so a
  fresh plotting-extra install also supports static image export.
- Made plotting display conditional: calls without an export directory display
  figures, while export and CLI calls no longer open browser tabs implicitly.
- Accelerated startup, repeated density calculations, KLD slice moments, and
  per-pair Welch tests without changing significance decisions.
- Vectorized the dominant Citation-Identity pair-construction path, reducing
  controlled GRG Top-4 median wall time by a further 13.1% with exact output
  and significance-decision parity.

### Removed

- Removed package-level Citation Image; incoming reception diagnostics remain
  part of the authors' paper-side analysis and are deliberately not a package
  metric.
- Removed obsolete loader, tooltip, Lowess/statsmodels, legacy crop/Pillow,
  and duplicate orchestration paths.

## [0.1.0] - 2026-05-04

- Prepared the repository for an initial PyPI-oriented release.
- Added a canonical package surface, `toc` CLI, optional plotting API, and Colab quickstart.
- Added the two-Parquet data contract with optional provenance sidecars.
- Added deterministic synthetic oracle data for tests, examples, and paper-facing sanity checks.
- Added real-input preparation for Bibcodes, references, and analytical author identities.
- Added UV-based release infrastructure, CI, metadata checks, and package hygiene tests.
