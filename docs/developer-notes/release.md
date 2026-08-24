# Release Checklist

This project is released from a clean Git tag through GitHub Actions, not from
a local working state. Local commands are only preflight checks.

## Local Preflight

For ordinary feature work, do not use this full preflight as the default
verification path. Prefer targeted tests for the touched API/CLI/docs surface.
Run the full suite only for explicit release preflight, because the full local
checks can be expensive.

```bash
uv sync --all-extras --group dev
uv run python scripts/check_env.py
uv run pytest -q
uv run python -m compileall src tests examples
uv build
uv run python -m twine check dist/*
uv run python scripts/ops/check_release_version.py v0.2.0
```

Also smoke-test the CLI against the bundled synthetic oracle data:

```bash
uv run toc --version
uv run toc validate examples/data/publications.parquet examples/data/references.parquet --auto-discover-sidecars
uv run toc metrics examples/data/publications.parquet examples/data/references.parquet \
  --auto-discover-sidecars \
  --assume-valid \
  --target uid:stable_vocab_distinct \
  --no-progress \
  --out runs/release_smoke/toc_metrics_smoke.parquet
```

## Release Workflow Structure

`.github/workflows/release.yml` separates duties into three jobs:

1. `build`: runs the metadata gate, environment check, tests, build, and twine
   check with no write or OIDC permissions; uploads the checked `dist/`
   artifacts.
2. `publish-pypi` (tag push) or `publish-testpypi` (manual `workflow_dispatch`):
   downloads the artifacts and publishes with `id-token: write` only, gated by
   the `pypi` / `testpypi` GitHub environment.
3. `github-release` (tag push only): creates the GitHub Release with
   `contents: write` only, after the PyPI publish succeeded.

Use the manual `workflow_dispatch` run (TestPyPI) as a full rehearsal before
the first real tag.

## One-Time External Setup

Before the first real tag:

1. PyPI: add a pending Trusted Publisher for `trajectories-of-change`
   (owner `raphschlatt`, repo `Trajectories_of_Change`, workflow `release.yml`,
   environment `pypi`).
2. TestPyPI: same, with environment `testpypi`.
3. Zenodo: connect the GitHub account **and enable the repository toggle before
   pushing the tag**. Zenodo archives on GitHub Release creation via webhook;
   the repository must be public and enabled at that moment, otherwise no DOI
   is minted for the release.

## Tag-Based Release

1. Update `CHANGELOG.md`, `CITATION.cff`, `.zenodo.json`, `pyproject.toml`,
   and `src/trajectories_of_change/__init__.py`. Set the real release date in
   `CHANGELOG.md`, `CITATION.cff`, and `.zenodo.json` together; the metadata
   gate checks their mutual consistency. Bump the version pin in the
   quickstart notebook's install cell and its `raw.githubusercontent.com`
   data URL (`vX.Y.Z` tag) to the new release.
2. Run the local preflight from a clean working tree.
3. Run the TestPyPI rehearsal via `workflow_dispatch` and verify an isolated
   install from TestPyPI.
4. Confirm the Zenodo repository toggle is ON (public repo).
5. Push a tag matching `vX.Y.Z`.
6. Approve the `pypi` environment gate after the unprivileged build job is
   green.
7. Verify: package on PyPI, GitHub Release with artifacts, Zenodo DOI minted.
8. Follow-up commit: add the minted DOI to `CITATION.cff` and the README badge.
   Zenodo DOI metadata should only be added after Zenodo has minted the DOI
   for an actual release.
