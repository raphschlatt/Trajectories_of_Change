# Trajectories of Change Docs

Start here, depending on what you want to do:

- **I want to try the example.** Run the
  [quickstart notebook](../examples/quickstart_colab.ipynb) (also on
  [Colab](https://colab.research.google.com/github/raphschlatt/Trajectories_of_Change/blob/main/examples/quickstart_colab.ipynb)),
  or copy the ten-line quickstart from the [README](../README.md).
- **I have my own data.** Read the [data contract](data_contract.md) for the
  two-Parquet input format, then use `toc prepare` / `toc validate` from the
  [CLI guide](cli.md) or `prepare_dataset_bundle(...)` from the
  [Python API](api.md).
- **I want to understand the methods.** Read [methods](methods.md) for the
  four measures (Own Vocabulary, Referenced Vocabulary, Citation Identity,
  Embedding Density Estimation), significance testing, and interpretation
  notes.

Reference material:

- [Python API](api.md) — functions, defaults, and advanced options.
- [CLI](cli.md) — the `toc` commands and their main flags (`--help` has the
  complete set).
- [Data contract](data_contract.md) — required columns, aliases, sidecars.
- [Methods](methods.md) — metric definitions and interpretation.
- For maintainers: [release checklist](developer-notes/release.md).

These docs describe what the package accepts, computes, and returns.
