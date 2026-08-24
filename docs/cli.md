# CLI

In a uv project that has the package installed, run the `toc` entry point
through uv:

```bash
uv run toc --help
uv run toc --version
```

After `uv tool install "trajectories-of-change[plotting]"` or a plain pip
install, the bare `toc` command works the same way.

The examples below use `raw/` for your unprocessed export and `prepared/`
for the cleaned bundle; substitute your own paths. To try commands without
your own data, the repository's synthetic example bundle works everywhere a
prepared bundle is expected, e.g.:

```bash
uv run toc validate examples/data/publications.parquet examples/data/references.parquet --auto-discover-sidecars
```

Prepare a raw export (deduplication, reference cleaning, author-identity
cleaning — see the [data contract](data_contract.md)):

```bash
uv run toc prepare raw/publications.parquet raw/references.parquet \
  --auto-discover-sidecars \
  --out-dir prepared/
```

Validate a prepared bundle:

```bash
uv run toc validate prepared/publications.parquet prepared/references.parquet --auto-discover-sidecars
```

Run one metric for one author and plot it:

```bash
uv run toc metric citation_identity prepared/publications.parquet prepared/references.parquet \
  --target-author-uid TARGET_AUTHOR_UID \
  --out-dir outputs/citation_identity

uv run toc plot metric outputs/citation_identity --out-dir outputs/citation_identity/figures
```

`toc metric` accepts exactly `own_vocab`, `ref_vocab`, `density`,
and `citation_identity`. It writes a versioned `MetricResult` folder
with `metric_result.json`, `sync.parquet`, optional `pointwise.parquet`,
optional `async.parquet`, and optional `welch.parquet` (the four output
tables are defined in [methods](methods.md)). The manifest records the
target UID, resolved metric configuration, and input provenance.

For reproducible runs, prefer `--run-dir`. It writes config, summary,
report, logs, metrics, and optional detail tables together:

```bash
uv run toc metrics prepared/publications.parquet prepared/references.parquet \
  --auto-discover-sidecars \
  --assume-valid \
  --top-n 20 \
  --window-size 2 \
  --top-k-kld-terms 50 \
  --details-out-dir details \
  --run-dir runs/top20
```

These explicit values mirror the Python API defaults: `window_size=2`,
`top_k_kld_terms=50`, `alpha=0.2`, `multiple_testing="fdr_bh"`,
`multiple_testing_scope="slice"`, author co-citation mode, standardized
2D density coordinates, and the clean Citation-Identity defaults:
document-fractional counting, first cited author, all-document target
exclusion, and self-loop removal.

`--assume-valid` skips strict bundle validation during the metrics run. Use it
only after `toc prepare` and `toc validate`; it does not change any metric
definition. `toc metrics` validates and loads the bundle passed to it, but it
does not run the deterministic cleaning step from `toc prepare`.
`--details-out-dir` writes per-target KLD and Density detail tables for
dashboard and deep-dive analysis while keeping the main metrics table compact.
For Citation-Identity runs it also writes co-reference diagnostics by document
and slice.

Citation-Identity policy flags:

```bash
--citation-identity-counting document-fractional|binary|multiplicity
--citation-author-scope first-author|all-authors
--target-exclusion none|target-docs-only|all-docs
--reference-policy inclusive|external_only
--no-welch
```

`--no-welch` skips document-level Welch tests; `KLD_all` is still computed, but
significant-summary columns are written as missing values.

Referenced Vocabulary is computed automatically when `references.parquet`
carries a `tokens` column. `--reference-policy inclusive` (default) keeps
target-authored references; `external_only` drops them as a self-citation
sensitivity check. Select the exact set of metrics with the canonical selector:

```bash
--metrics own_vocab ref_vocab density citation_identity
```

Excluded metrics are not initialized and emit no placeholder columns. Outputs
for Referenced Vocabulary use the `ref_vocab_` prefix; Citation Identity keeps
the established `cocit_` output prefix.

Target selection: `--select-by uid` (default) treats `--target` values as author
UIDs; `--select-by name` treats them as display names.

Parallelism: `toc metrics` distributes targets across workers with `--jobs`
(default `auto`, which is memory-aware); the Python API is serial unless you
pass `n_jobs`.

Advanced tuning (rarely needed; defaults match the Python API): `--lambda-param`
(KLD smoothing), `--epsilon` (numerical floor), `--density-bandwidth` (KDE
bandwidth; default Scott's rule), `--density-min-docs-target` /
`--density-min-docs-field` (both default `1`), and `--density-cols` (Python
API: `density_embedding_cols`). Every CLI flag has a matching keyword
argument in the Python API facade: `toc metric` maps to `run_metric(...)`,
and `toc metrics` maps to `run_metrics(...)`; `toc metrics --help` lists the
complete flag set.

Run explicit higher-dimensional density by naming the coordinate columns
(the bundled example data carries `embedding_5d_*` columns to try this):

```bash
uv run toc metrics examples/data/publications.parquet examples/data/references.parquet \
  --target uid:density_shift \
  --density-cols embedding_5d_0 embedding_5d_1 embedding_5d_2 embedding_5d_3 embedding_5d_4 \
  --out outputs/metrics_5d.parquet
```

Plot multimetric summaries:

```bash
uv run toc plot multimetric --run-dir runs/top20
```

For quick ad hoc exports, use `--out` instead of `--run-dir`:

```bash
uv run toc metrics prepared/publications.parquet prepared/references.parquet \
  --auto-discover-sidecars \
  --assume-valid \
  --top-n 5 \
  --out outputs/metrics.parquet

uv run toc plot multimetric outputs/metrics.parquet --out-dir outputs/plots
```

Run mode writes `config_used.yaml`, `run_summary.yaml`, `report.md`, logs, and
`results/multimetric.parquet`. It records input paths, bytes, checksums,
sidecars, runtime, output artifacts, and validation/provenance warnings. Large
input Parquets are not copied into the run directory.

`toc validate` reports provenance mismatches as warnings by default. Add
`--strict-provenance` for final publication-facing runs when sidecar
mismatches should fail early.

The CLI is a thin wrapper around the Python API. It does not define a separate
pipeline or a different data contract.
