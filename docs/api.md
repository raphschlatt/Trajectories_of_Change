# Python API

Core imports:

```python
from trajectories_of_change import (
    prepare_dataset_bundle,
    load_dataset_bundle,
    run_metric,
    run_metrics,
    VocabularyKLD,
    ReferencedVocabularyKLD,
    KDEDensity,
    CitationIdentityKLD,
    run_top_authors_metrics_from_parquets,
    iter_top_authors_metrics_from_parquets,
)
```

Most users need only three paths.

Load a prepared bundle:

```python
bundle = load_dataset_bundle(
    "data/prepared/grg/publications.parquet",
    "data/prepared/grg/references.parquet",
    auto_discover_sidecars=True,
)
```

Run one metric and plot it:

```python
from trajectories_of_change.plotting import plot_metric

uid = "AUTHOR_UID_HERE"
result = run_metric(bundle, metric="own_vocab", target_author_uid=uid)
plot_metric(result, export_dir="outputs/author", format="html")
```

If you do not yet know an author UID, the cheapest way is to look at the
data directly:

```python
bundle.publications["author_uids"].explode().value_counts().head(20)
```

Alternatively, the lower-level selector picks the most productive authors
without computing any metrics:

```python
from trajectories_of_change.multimetric import pick_top_authors

uids = pick_top_authors(bundle.publications, "Author", top_n=20)
uid = uids[0]
```

(It prefers the `author_uids` ID column internally; the name column passed
here is only the display fallback.)

For cohort work, `run_metrics(bundle, top_n=20)` performs this UID-first
selection automatically and returns each selected `author_uid`.

Run the consolidated top-author metrics and plot the overview:

```python
from trajectories_of_change.plotting import plot_multimetric

metrics = run_metrics(bundle, top_n=50)
plot_multimetric(metrics, export_dir="outputs/top50")
```

`run_metric(...)` accepts exactly `own_vocab`, `ref_vocab`, `density`, and
`citation_identity`. `target_author_uid` is required; the simple API does not
perform author-name fallback.
The matching CLI path is `toc metric ... --target-author-uid ... --out-dir ...`;
`toc plot metric RESULT_DIR --out-dir ...` plots the exported `MetricResult`
folder.

Citation Identity is the author's outgoing co-reference divergence; see
[methods](methods.md) for the full definition and its distinction from
incoming reception (Citation Image).

Use `prepare_dataset_bundle(...)` for raw exports (for example ADS exports
with author-name disambiguation applied). It returns a
validated `DatasetBundle` with a `cleaning_report` and a manifest entry that
records deterministic Bibcode, Reference, and author-identity cleaning. The
function returns data in memory; it does not write prepared Parquets unless your
script writes `bundle.publications` and `bundle.references` to disk.

`run_metrics(...)` accepts either a loaded `DatasetBundle` (as in the
examples above) or two Parquet paths as its leading arguments. It is the
friendly name for the consolidated production path implemented by
`run_top_authors_metrics_from_parquets(...)` and returns the same compact
`pandas.DataFrame`. `iter_top_authors_metrics_from_parquets(...)` yields
one result dictionary per target and is useful for CLI-style streaming. These
metric helpers expect the paths passed to them to already be canonical. If your
source handoff is raw, duplicated, or has unresolved references, run
`prepare_dataset_bundle(...)` and write the prepared Parquets first, or use
`uv run toc prepare ...`.
The matching CLI path is `toc metrics ...`; `toc plot multimetric ...` plots the
exported summary table.

Use `include` as the single metric selector:

```python
metrics = run_metrics(
    bundle,
    top_n=20,
    include=("own_vocab", "ref_vocab", "density"),
)
```

Excluded metrics are not initialized or computed, and their columns are absent
from the result.
The shared defaults are `window_size=2`, `top_k_kld_terms=50`,
`alpha=0.2`, `multiple_testing="fdr_bh"`,
`multiple_testing_scope="slice"`, author co-citation mode, standardized 2D
density coordinates, and clean Citation-Identity settings:
`citation_identity_counting="document_fractional"`,
`citation_author_scope="first_author"`, `target_exclusion="all_docs"`, and
self-loop removal. Import constants from `trajectories_of_change.defaults` when
you want to reference those values by name.

| Setting | Default | Change it when |
|---|---:|---|
| `window_size` | `2` | the time resolution is scientifically justified |
| `top_k_kld_terms` | `50` | testing all observed terms or a different screening budget is intended |
| `lambda_param` | `0.5` | reproducing a run with an explicitly pinned smoothing value |
| `epsilon` | `1e-12` | numerical diagnostics justify a different floor |
| `alpha` | `0.2` | the inferential design specifies another threshold |
| `multiple_testing` | `"fdr_bh"` | another named correction is part of the analysis plan |
| `multiple_testing_scope` | `"slice"` | the hypothesis family is explicitly pairwise or global |

To reproduce a specific analysis, always take `lambda_param` (and every other
parameter) from that run's recorded configuration — a pinned study
configuration can differ from the current package defaults.
For prepared bundles that have already passed validation, pass
`assume_valid=True` to skip strict validation during the metrics run. This is a
performance option only; metric definitions and output columns are unchanged.
Pass `details_out_dir="outputs/details"` when you want per-target KLD and
Density detail tables for dashboard or deep-dive plots. The summary return value
stays compact.
With Citation-Identity runs, the details directory also receives
`cocit_diagnostics.parquet` and `cocit_diagnostics_by_slice.parquet`; these
tables report removed self-loop mass, target-excluded mass, empty reference
identities, and documents left without analyzable pairs.

The compact multimetric table groups its headline columns by prefix:

- identity and run context: `author_uid`, `author_display_name`, `alpha`,
  `multiple_testing`, `multiple_testing_scope`, and resolved method options;
- Own Vocabulary: `vocab_*`;
- Referenced Vocabulary: `ref_vocab_*`;
- density (Embedding Density Estimation, EDE): `density_*`;
- Citation Identity: `cocit_*`.

Level and slope columns are the primary cross-author comparison outputs. The
level is the target's average across time slices and the slope is its trend,
so a positive `vocab_kld_all_slope` means the target's vocabulary moves away
from the field over time. Coverage, document-count, and configuration
columns are retained for interpretation and reproducibility. The headline columns are systematic —
`{prefix}kld_all|kld_sig|kld_sig_abs_{level|slope}` for the KLD metrics and
`density_neglog_{level|slope}` for density — and the remaining columns echo
coverage counts and the resolved run options; `metrics.columns` lists the
full set for your configuration.

For multimetric runs, Citation Identity has one production path: the package
builds a compact `CitationIdentityEventIndex`, converts weighted co-reference
events to `DocumentFeatureMatrix`, and computes sync, async, pointwise, and
optional Welch outputs through `KLDCore`.
With document-fractional Citation Identity, the multimetric helpers default the
Citation-Identity global pair-mass threshold to `1.0` and the target/field
slice thresholds to positive mass (`1e-12`). Each citing document contributes
total pair mass `1.0`, but KLD slices may retain less than one document's mass
after global support filtering; vocabulary thresholds are unchanged.

`ReferencedVocabularyKLD(...)` is the core metric for cited-literature
vocabulary. It requires a `references.parquet` with a `tokens` column
produced by the same tokenization pipeline as `publications.tokens`; the
package does not tokenize raw text and deliberately does not fall back to
regex tokenization. The
metric builds weighted reference-token profiles and then uses the same
`DocumentFeatureMatrix + KLDCore` layer as `VocabularyKLD`. It also runs
automatically inside `run_top_authors_metrics_from_parquets(...)` / `toc
metrics` when references are tokenized, contributing `ref_vocab_` columns to the
consolidated output (skipped when `references.tokens` is absent).

Welch tests are enabled by default, so `KLD_sig` summaries are always present.
Pass `run_welch=False` for faster descriptive runs. In that mode `KLD_all`
levels/slopes are still computed, while significant-summary columns are `NaN`
because no document-level test was run.

The single-metric API defaults to `include_async=True`; multimetric runs default
to `include_async=False` because the pairwise matrices are substantially more
expensive. Both layers use the same option name. The library runner defaults to
`n_jobs=1`; the CLI defaults to `--jobs auto`.

On real corpora, plan for runtime. The bundled 340-document example computes
a single author in well under a second. A corpus of 180k publications with 50
targets and all four measures including the Welch tests is an overnight job
on a workstation. The two big levers are disabling the Welch tests with
`run_welch=False` and leaving the asynchronous comparison off, which is the
multimetric default. The CLI's `--jobs auto` distributes targets across
workers and is memory aware, while the Python API stays serial unless
`n_jobs` is set. `show_progress=False` silences the progress bar of
multimetric runs.

Sparse authors produce short trajectories. Measures are computed per time
slice, slices without analyzable target documents contribute no value, and
density enforces a minimum document count per slice
(`min_docs_target_slice` and `min_docs_field_slice`, both default `1`).
Slopes for authors with only a handful of publications rest on very few
points and should be read accordingly.

Every `run_metric(...)` call returns a `MetricResult` with the target UID,
resolved metric configuration, and dataset provenance. `result.save(path)` and
`MetricResult.load(path)` persist that context alongside the Parquet tables in
a versioned folder format.

The Python API returns data structures only. For a reproducible folder with
metrics, logs, plots, config, and a compact report, use the CLI `--run-dir`
mode; it wraps the same package functions without defining a separate pipeline.

## Dataset preparation utilities

These helpers support preparing, validating, and inspecting custom bundles.
Most users only need the top-level `prepare_dataset_bundle` /
`load_dataset_bundle` imports above. Lower-level contract helpers live under
their implementation modules so the package top level stays focused on the
metric API.

```python
from trajectories_of_change.contract import (
    validate_dataset_bundle,
    normalize_publications_frame,
    normalize_references_frame,
    canonicalize_column_name,
    build_target_mask,
    is_placeholder_author_uid,
    resolve_embedding_columns,
    COLUMN_ALIASES,
    PUBLICATIONS_REQUIRED_COLUMNS,
    REFERENCES_REQUIRED_COLUMNS,
)
from trajectories_of_change.referenced_vocabulary import build_reference_token_cache
```

- `validate_dataset_bundle(bundle)` — check a `DatasetBundle` against the contract
  (required columns, types, reference integrity); raises with a precise message on
  the first violation.
- `normalize_publications_frame(df)` / `normalize_references_frame(df)` — canonicalize
  column names and coerce list/scalar types on a raw publications/references frame,
  returning the normalized frame.
- `canonicalize_column_name(name)` — map an accepted alias to its canonical column
  name (the alias table is `COLUMN_ALIASES`).
- `build_target_mask(df, target_name=..., target_author_uid=...)` — boolean `Series`
  selecting a target author's documents (by author UID, with optional name fallback).
- `is_placeholder_author_uid(uid)` — `True` for synthetic/placeholder author IDs that
  should be skipped during target selection.
- `resolve_embedding_columns(df, requested=...)` — resolve requested embedding column
  names to the canonical columns actually present (used by the density metric).
- `build_reference_token_cache(references)` — precompute a reusable reference-token
  cache for `ReferencedVocabularyKLD` so repeated targets do not re-tokenize.
- `COLUMN_ALIASES`, `PUBLICATIONS_REQUIRED_COLUMNS`, `REFERENCES_REQUIRED_COLUMNS` —
  the alias map and required-column sets the contract enforces.

Optional plotting imports:

```python
from trajectories_of_change.plotting import plot_metric, plot_multimetric
```

Install the package into a project with `uv add
"trajectories-of-change[plotting]"` (plain `pip install` works as well). For
standalone CLI use:

```bash
uv tool install "trajectories-of-change[plotting]"
```

In a local checkout of this repository, `uv sync --all-extras --group dev`
installs everything and makes `uv run toc ...` available.

Use `plot_metric(...)` for a single `MetricResult` and
`plot_multimetric(...)` for a `run_metrics(...)` summary table. Both display
figures by default, accept `export_dir=None`, and export HTML by default when an
output directory is supplied. Specialized figure builders remain private
implementation submodules rather than additional top-level plotting API.
