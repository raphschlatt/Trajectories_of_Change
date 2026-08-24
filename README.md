# Trajectories of Change

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/LICENSE)
[![CI](https://github.com/raphschlatt/Trajectories_of_Change/actions/workflows/ci.yml/badge.svg)](https://github.com/raphschlatt/Trajectories_of_Change/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22084878.svg)](https://doi.org/10.5281/zenodo.22084878)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphschlatt/Trajectories_of_Change/blob/main/examples/quickstart_colab.ipynb)

`Trajectories_of_Change` measures how individual authors move relative to their
research field over time. It takes a corpus of publications and their cited
references and computes trajectories for a target author or a group of authors
across four measures. The package is domain agnostic and runs on any publication dataset that follows
its input format. It implements the measures introduced in
[Trajectories of Change: Approaches for Tracking Knowledge Evolution](https://arxiv.org/abs/2501.00391)
and the complete pipeline of a follow-up study (see [Citing](#citing)).

## The four measures

| Measure | Output prefix | Features |
|---|---|---|
| Own Vocabulary | `vocab_*` | lemmatised terms of the target's own titles and abstracts |
| Referenced Vocabulary | `ref_vocab_*` | terms of the works the target cites |
| Citation Identity | `cocit_*` | unordered pairs of co-cited authors in the target's reference lists |
| Embedding Density Estimation (EDE) | `density_*` | document embedding coordinates |

The three distributional measures are Kullback–Leibler divergences between
the target's distribution and the field's, computed in bits per time slice,
where the field is the corpus with the target removed. EDE is a kernel
density estimate at the target's positions in a document embedding.
Definitions and interpretation guidance are in
[docs/methods.md](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/docs/methods.md).

## Installation

Use [uv](https://docs.astral.sh/uv/). Requires Python 3.11 or newer.

```bash
uv add "trajectories-of-change[plotting]"
```

Without the Plotly plotting extra, if you only need the numbers:

```bash
uv add trajectories-of-change
```

For the `toc` command line outside a project:

```bash
uv tool install "trajectories-of-change[plotting]"
```

Plain `pip install "trajectories-of-change[plotting]"` works as well.

## Quick start

The repository ships a small synthetic example dataset whose authors behave
in known ways by construction, so your first run is verifiable. Clone the
repository or download the
[examples/data](https://github.com/raphschlatt/Trajectories_of_Change/tree/main/examples/data)
folder, or run the same flow in the browser via the
[Colab notebook](https://colab.research.google.com/github/raphschlatt/Trajectories_of_Change/blob/main/examples/quickstart_colab.ipynb).

```python
from trajectories_of_change import load_dataset_bundle, run_metrics
from trajectories_of_change.plotting import plot_multimetric

bundle = load_dataset_bundle(
    "examples/data/publications.parquet",
    "examples/data/references.parquet",
)
metrics = run_metrics(bundle, top_n=5, show_progress=False)

levels = ["vocab_kld_all_level", "ref_vocab_kld_all_level", "cocit_kld_all_level", "density_neglog_level"]
print(metrics[["author_display_name"] + levels].round(2).to_string())

plot_multimetric(metrics, export_dir="outputs/quickstart", show=False)
```

```text
     author_display_name  vocab_kld_all_level  ref_vocab_kld_all_level  cocit_kld_all_level  density_neglog_level
0         Field-Like, F.                 0.03                     0.01                 0.15                  0.53
1  Stable Vocabulary, V.                13.77                     0.18                 1.44                  0.53
2   Spiky Vocabulary, S.                 0.03                     0.01                 0.15                  0.53
3  Citation Distinct, C.                 0.03                     2.51                18.09                  0.53
4      Density Shift, D.                 0.03                     0.01                 0.15                  2.67
```

This is the expected pattern, and the outliers are correct results, not a
broken install. Each synthetic author carries one built-in behavior, visible
in its own column. "Stable Vocabulary, V." holds a stably divergent
vocabulary (13.77 bits against a 0.03 baseline; `kld_all` sums all feature
contributions, so values above 1 bit are normal). "Citation Distinct, C."
co-cites differently (18.09) and consequently also cites differently worded
literature (2.51). "Density Shift, D." moves into a sparser field region
(2.67, where lower density values mean denser neighbourhoods). "Spiky
Vocabulary, S." is deliberately unremarkable here, its divergence comes as
single-slice spikes that the trajectory plots reveal rather than the level.
The plot call then lists its exported files and writes the overview figures
to `outputs/quickstart`.

## Input data

Two Parquet tables. The examples are real rows from the bundled dataset,
whose names are synthetic.

`publications.parquet`, one row per publication of your corpus:

| Column | Needed for | Type | Example |
|---|---|---|---|
| `Bibcode` | all measures | `str` | `"SYN2000STABL00"` |
| `Year` | all measures | `int` | `2000` |
| `Author` | all measures | `list[str]` | `["Stable Vocabulary, V."]` |
| `author_uids` | author selection, strongly recommended | `list[str]` | `["uid:stable_vocab_distinct"]` |
| `References` | Citation Identity | `list[str]` | `["SYNREF-COR-00", ...]` |
| `tokens` | Own Vocabulary | `list[str]` | `["gravity", "relativity", "spacetime", ...]` |
| `embedding_2d_x`, `embedding_2d_y` | EDE | `float` | `0.0738`, `0.0656` |

`references.parquet`, one row per cited work:

| Column | Needed for | Type | Example |
|---|---|---|---|
| `Bibcode` | Citation Identity, Referenced Vocabulary | `str` | `"SYNREF-COM-00"` |
| `Author` | Citation Identity | `list[str]` | `["Reference COM 00"]` |
| `tokens` | Referenced Vocabulary | `list[str]` | `["gravity", "gravity", "gravity", ...]` |

Repeated tokens are meaningful, term frequency is encoded by repetition.
Produce `tokens`, the embedding coordinates and `author_uids` with your own
upstream pipeline. The pipeline behind the papers is public in
[ads-bib](https://github.com/raphschlatt/ads-bib) and
[ads-and](https://github.com/raphschlatt/ads-and). Clean raw exports with
duplicates or unresolved references once with `toc prepare`. Accepted column
aliases and the full contract are in
[docs/data_contract.md](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/docs/data_contract.md).

## Usage

### Python API

The quick start used `run_metrics`, the entry point for author cohorts. For
one measure of one author in full per-slice detail, use `run_metric`:

```python
from trajectories_of_change import run_metric
from trajectories_of_change.plotting import plot_metric

# bundle as loaded in the quick start
result = run_metric(bundle, metric="citation_identity", target_author_uid="uid:stable_vocab_distinct")
plot_metric(result, export_dir="outputs/identity", format="html", show=False)
```

All functions, defaults, output columns and advanced options are documented
in
[docs/api.md](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/docs/api.md).

### Command line

The CLI wraps the same functions with the same defaults and produces the
same numbers.

```bash
uv run toc metrics prepared/publications.parquet prepared/references.parquet \
  --top-n 20 \
  --run-dir runs/top20

uv run toc plot multimetric --run-dir runs/top20
```

The `--run-dir` mode keeps results, plots, logs, the run configuration and a
short report together in one folder. All commands and flags are documented
in
[docs/cli.md](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/docs/cli.md).

## Documentation

- [Start here](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/docs/index.md)
- [Data contract](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/docs/data_contract.md)
- [Methods](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/docs/methods.md)
- [Python API](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/docs/api.md)
- [CLI](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/docs/cli.md)

## Citing

Cite `trajectories-of-change` as software via
[CITATION.cff](https://github.com/raphschlatt/Trajectories_of_Change/blob/main/CITATION.cff)
(GitHub renders it through the "Cite this repository" button) or the Zenodo
concept DOI
[https://doi.org/10.5281/zenodo.22084878](https://doi.org/10.5281/zenodo.22084878).
Cite the framework paper if you discuss the approach or the measures:

> Raphael Schlattmann and Malte Vogl (2024). *Trajectories of Change:
> Approaches for Tracking Knowledge Evolution*. BHDC 2023: Big Historical
> Data Conference, Jena.
> [https://doi.org/10.48550/arXiv.2501.00391](https://doi.org/10.48550/arXiv.2501.00391)

The complete pipeline implemented here is specified in the follow-up study,
whose supplementary material serves as the formal specification of this
package:

> Raphael Schlattmann and Malte Vogl. *Tracing individual knowledge
> trajectories within a changing scientific field: language, citation, and
> semantic measures in general relativity and gravitation*. In preparation.

Resources:

- Framework paper: [https://doi.org/10.48550/arXiv.2501.00391](https://doi.org/10.48550/arXiv.2501.00391)
- Upstream corpus pipeline: [https://github.com/raphschlatt/ads-bib](https://github.com/raphschlatt/ads-bib)
- Author disambiguation: [https://github.com/raphschlatt/ads-and](https://github.com/raphschlatt/ads-and)

MIT licensed.
