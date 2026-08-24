# Examples

This directory contains the quickstart for the package API. Method details
are in [docs/methods.md](../docs/methods.md).

- `quickstart_colab.ipynb` runs the README quick start (the five-author
  overview) and then a single-author deep dive down to the term level.
- `data/` is a deterministic synthetic oracle bundle, so the quickstart works
  without any private data and its main metric expectations stay testable.

Locally and in Colab the quickstart defaults to:

- `examples/data/publications.parquet`
- `examples/data/references.parquet`
- optional standard provenance sidecars in the same directory

It runs with the shared package defaults (`window_size=2`,
`top_k_kld_terms=50`, `alpha=0.2`, `multiple_testing="fdr_bh"`,
`multiple_testing_scope="slice"`).

Regenerate the bundle reproducibly with:

```bash
uv run python scripts/generate_synthetic_oracle_data.py --out-dir examples/data
```

For your own data, point the notebook at a prepared
`publications.parquet`/`references.parquet` bundle. Prepare a raw export
first:

```bash
uv run toc prepare data/incoming/<bundle>/publications.parquet data/incoming/<bundle>/references.parquet \
  --auto-discover-sidecars \
  --out-dir data/prepared/<bundle>
```

Then use `data/prepared/<bundle>/publications.parquet` and
`data/prepared/<bundle>/references.parquet`. The notebook uses no local
`sys.path` hacks and is designed to work after a normal package installation.
