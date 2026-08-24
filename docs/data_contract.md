# Data Contract

The package's input contract is intentionally small. A valid run needs exactly
two canonical tables:

- `publications.parquet`
- `references.parquet`

Optional sidecars can be supplied for research transparency, but they are not
required for package use:

- `dataset_manifest.json`
- `run_summary.yaml`
- `config_used.yaml`

`prepare_dataset_bundle(...)` turns a raw export (for example an ADS export
with author-name disambiguation applied) into an analysis-ready bundle.
`load_dataset_bundle(...)` validates an already prepared bundle and
normalizes accepted aliases. If sidecars are supplied, their metadata is
carried into result rows where it is useful for later interpretation.

## Required Columns

`publications.parquet` must contain:

- `Bibcode`
- `Year`
- `Author`
- `References`

`references.parquet` must contain:

- `Bibcode`
- `Author`

A row in `references.parquet` is one citable work — the works that the IDs in
`publications.References` point to, with their own metadata (`Author`, and
optionally `tokens`). After preparation, every ID kept in
`publications.References` resolves to exactly one row here; unresolved IDs
are removed from `publications.References` by `toc prepare`.

## Metric-Specific Columns

The vocabulary measures need tokenized text:

- `tokens` holds the lemmatised terms of each document. The package does not
  tokenize or lemmatise raw text, that step belongs to your upstream
  pipeline. In the papers' corpus these are lemmatised English terms from
  titles and abstracts, produced with the public
  [ads-bib](https://github.com/raphschlatt/ads-bib) pipeline. Any consistent
  pipeline is valid, because the measures compare target and field within
  one corpus. Use the same pipeline for every document and for
  `references.tokens`, and record it alongside the run, since KLD levels are
  only comparable across studies when the tokenization matches.

Default density needs the canonical 2D coordinates:

- `embedding_2d_x` and `embedding_2d_y` are document coordinates from an
  embedding and projection pipeline of your choice. The papers use
  transformer document embeddings reduced with UMAP, which is where the
  accepted `UMAP-1` and `UMAP-2` aliases come from. The package standardizes
  the coordinates and treats them as given. EDE values are only comparable
  within one coordinate space.

Explicit higher-dimensional (nD) density is supported only when finite
numeric coordinate columns are passed through `density_embedding_cols`.

Author-centered runs should use stable IDs:

- `author_uids`

Name matching is retained as fallback only. The preferred selection path is
`author_uids` first, because display names are not a stable identity contract.
In prepared bundles, `Author` remains the raw display layer, while
`author_uids` is the cleaned analytical identity layer. In the papers' corpus
these identifiers come from the public disambiguation pipeline
[ads-and](https://github.com/raphschlatt/ads-and). `author_display_names`
is kept positionally aligned to the cleaned `author_uids`.

## Preparation

Raw bibliographic exports may contain small upstream artifacts. Preparation is
deterministic and reported in `dataset_manifest.json` plus
`cleaning_report.json` when run through the CLI.

- `Bibcode` values are stripped; empty Bibcodes are dropped and reported.
- Duplicate Bibcodes are reduced to one row by longest `References` list, then
  most non-empty fields, then original order.
- `publications.References` is stripped, deduplicated per publication, and
  unresolved reference IDs are removed while keeping the publication row.
- Duplicate and placeholder `author_uids` are removed per row; placeholders
  include `::n.author::`, `::unknown::`, `unknown`, and `no author`.
- Raw `Author` values are not rewritten.

After `toc prepare`, `dataset_manifest.json` describes the prepared analysis
bundle. `artifacts` points to the prepared `publications.parquet` and
`references.parquet` with bytes and SHA256 checksums. `source_artifacts`
describes the raw inputs used to create that bundle. `counts` always refers to
the prepared outputs, and `cleaning` matches `cleaning_report.json`.

If optional sidecars are present, loaders check for obvious provenance
mismatches such as different run IDs between `dataset_manifest.json` and
`run_summary.yaml`, missing referenced configs, or manifest artifact sizes that
do not match the loaded Parquets. These are warnings by default and can be made
strict in the CLI with `--strict-provenance`.

## Accepted Aliases

The loader accepts and normalizes common producer aliases, including:

- `AuthorUID`
- `AuthorDisplayName`
- `author_ids`
- `UMAP-1`
- `UMAP-2`
- `Title`
- `Abstract`

## Invariants

- Co-citation is built from `publications.References` plus `references`.
- Prepared `publications.Bibcode` and `references.Bibcode` are unique and
  non-empty.
- Prepared `publications.References` contains only Bibcodes present in
  `references`.
- Time slices are non-overlapping.
- Default density uses `embedding_2d_x`, `embedding_2d_y`.
- Higher-dimensional density must be requested explicitly with numeric
  coordinate columns.
- Optional provenance may describe how the bundle was produced, but metrics must
  also work with the two Parquet files alone.
