# Local Data

This directory is a convenient place for your own local bundles. Real data
files are gitignored and are not part of the package or source distribution.

Suggested layout:

- `data/incoming/<source-run>/`: raw producer handoffs before `toc prepare`.
- `data/prepared/<bundle-name>/`: canonical package-ready bundles.
- `runs/<run-name>/`: all metric outputs, detail tables, logs, reports, and
  plots from `toc metrics --run-dir`.

Prefer named bundles under `data/prepared/` over a single root-level
`data/publications.parquet` — an unnamed "current" bundle quickly becomes
ambiguous.

Canonical bundle files are:

- `publications.parquet`
- `references.parquet`
- optional `dataset_manifest.json`
- optional `run_summary.yaml`
- optional `config_used.yaml`

Caches, embedding experiments, and plot exports are generated artifacts. Keep
them outside the data directory or regenerate them from named bundles/runs when
needed.
