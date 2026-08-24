# Methods

## Time Slices And Output Tables

All measures are computed per time slice: consecutive, non-overlapping windows
of `window_size` years (default `2`) over the `Year` column, from the first
to the last year in the corpus. A trailing window that would extend past the
last year is dropped by default; `skip_incomplete_slices=False` keeps it as a
shortened final window. Within each slice, the target author's documents are
compared against the field baseline: all documents in the corpus except the
target's own.

Each measure produces up to four tables, named the same way everywhere
(`MetricResult` attributes, exported Parquet files, CLI outputs):

- `sync`: the author's slice compared with the field's *same* slice — the
  main trajectory.
- `async`: every author slice compared with every *other* field slice — a
  pairwise matrix, useful as a lead/lag diagnostic. Off by default in
  multimetric runs because it is substantially more expensive.
- `pointwise`: per-term (or per-pair) contributions behind each slice value —
  which terms carry the divergence.
- `welch`: per-term Welch test results behind the significance summaries.

## KLD And Welch

`KLD_all` is the full Kullback–Leibler divergence, computed in bits, between
the target's and the field's distribution over the shared feature set of a
time slice. It is the descriptive effect size used throughout the papers.

All distributional estimates are formed at slice level, not per document, and
are smoothed with Jelinek–Mercer interpolation against a time-pooled field
background, floored at `epsilon`. The papers' main analysis pins the
interpolation weight at `lambda_param=0.05`. The package default of `0.5` is
their robustness axis, so pass the value explicitly when reproducing a run.

A Welch two-sample t-test over documents filters candidate features for
stability. By default the tested family is the Top-K features with the
largest absolute KLD contributions in a slice pair, and the resulting
p-values are adjusted with the Benjamini–Hochberg procedure
(`multiple_testing="fdr_bh"`). `top_k_kld_terms=None` switches to the
explicit full-test mode over all observed slice-pair features.

`KLD_sig` is the signed sum of the document-stable contributions. It can be
small when positive and negative stable contributions cancel, so it is read
as a direction. `KLD_sig_abs` sums the absolute stable contributions and is
read as the magnitude carried by stable distinguishing features.

Whenever a result is called significant, interpret it together with:

- `alpha`
- `multiple_testing`
- `multiple_testing_scope`
- `top_k_kld_terms`

The default `top_k_kld_terms=50`, `alpha=0.2`,
`multiple_testing="fdr_bh"`, and `multiple_testing_scope="slice"` define an
exploratory candidate screen, not final confirmatory inference.

## Referenced Vocabulary

Referenced Vocabulary applies the same KLD machinery to the vocabulary of the
works a target *cites*, rather than the target's own lemmatised terms. The
features carry a two-level weighting. Within a cited work a term counts by
its relative frequency, and within a citing publication the cited works share
the mass equally, so each citing publication contributes total mass 1 and
neither a long bibliography nor a long cited text dominates. The slice axis
is the citing publication's year.

The measure runs automatically in the consolidated metrics run when
`references` carry a `tokens` column (otherwise it is skipped, like density
without embedding columns). The default `--reference-policy inclusive` keeps
target-authored references in the reference vocabulary; `external_only` drops
them as a self-citation sensitivity check. Outputs use the `ref_vocab_`
prefix (e.g. `ref_vocab_kld_all_level`).

## Density

Density is the package's operationalization of Embedding Density Estimation
(EDE) — figure titles use that name. It asks how densely the field occupies
the semantic neighborhood of the target's publications, based on document
embedding coordinates and kernel density estimation.

Density is reported as negative log KDE:

- lower `density_neglog_*` means the target lies in a locally denser field
  region;
- higher `density_neglog_*` means the local field density around the target is
  lower.

This is a density statement, not a direct center/periphery statement. A cluster
can be geometrically near the edge of a 2D map and still be locally dense; a
geometrically central region can become relatively less dense if other
subfields grow more strongly. The metric therefore asks how densely the field
occupies the target's semantic neighborhood in the evaluated coordinate space.

Default density runs in the standardized canonical 2D map coordinates
(`embedding_2d_x`, `embedding_2d_y`). Explicit nD/5D runs are supported through
`density_embedding_cols`, but 2D and nD values are not the same numerical
scale. Compare trends and rankings only within the same coordinate space and
run configuration.

The KDE is probabilistically normalized. It does not directly measure absolute
publication mass. If the whole field grows while its relative spatial
distribution stays similar, `density_neglog_*` need not change much. Absolute
regional growth would require an additional intensity-style analysis.

Synchronous density evaluates target documents under the field KDE of the same
time slice. Asynchronous density evaluates target slices against other field
slices and can be used as a lead/lag or adoption diagnostic.

## Reporting And Robustness Checks

If you run sensitivity analyses on your own data, keep the target question
explicit:

- level robustness: do author rankings remain similar?
- slope robustness: do trend directions remain similar?
- correlation robustness: do the vocabulary, citation, and density metrics
  keep similar cross-author relationships?
- significant-term robustness: do significant KLD candidates remain stable
  under the reported `alpha`, `multiple_testing`, `multiple_testing_scope`, and
  `top_k_kld_terms`?

When you report results, state at least: corpus/query, time span,
`window_size` (default `2`), density coordinate columns and how they were
produced, tokenization pipeline, `top_k_kld_terms` (default `50`), `alpha`
(default `0.2`), multiple-testing method and scope, and author count. KLD
levels are comparable across authors within one run; across corpora,
tokenizations, or configurations, compare rankings and trends rather than raw
values.

## Synthetic Oracle Corpus

The bundled `examples/data` corpus is synthetic by design. It is not a model of
real scientific history; it is an oracle for expected metric behavior. The
scenarios encode field-like authors, stable vocabulary divergence, one-document
vocabulary spikes, citation divergence, density shifts, converging divergence,
and a geometrically external but locally dense cluster.

This corpus is useful for tests and tutorials because the expected signals
are known before running the package: stable terms such as `tetrad`,
`torsion`, `gauge`, and `frame` should survive Welch/FDR filtering; spike terms
should not become stable markers; citation-distinct authors should separate
Citation Identity from Own Vocabulary; and Density should track local field
density rather than geometric map periphery.

## Citation Identity

Citation Identity measures the target author's own outgoing reference
structure: which cited authors or works are combined in the target's
reference lists, compared with the field. One metric, several historical
names — Co-Citation-KLD, Reference-Context Identity, Co-Reference Identity —
and the output column prefix `cocit_` all refer to this same metric; the API
name is `citation_identity`.

Its mirror concept, Citation Image (how the field cites or co-cites the
target — incoming reception), is a different analysis and not a package
metric.

The main Citation-Identity run uses:

- cited `author_uids` where available;
- first cited author per reference by default;
- unordered author pairs;
- one unique pair per citing document;
- document-fractional weighting, so each citing document contributes total
  pair mass `1.0`;
- self-loop removal;
- target exclusion from both target and field reference contexts.

Because document-fractional weighting makes pair mass comparable to analyzable
document count, the multimetric helpers use a Citation-Identity global
pair-mass threshold of `min_token_global_freq=1.0` and require positive retained
pair mass per target/field slice (`min_tokens_target_slice=1e-12` and
`min_tokens_field_slice=1e-12`) unless the caller explicitly overrides those
thresholds. This matters because global support filtering can leave a real
target slice with less than one document-equivalent of retained pair mass. The
vocabulary KLD defaults remain token-count based.

Target exclusion creates a re-normalized baseline: the KLD compares the
target's non-self reference context with the field's non-target reference
context. This intentionally removes the target's own received citation image
from the baseline. It is not the raw field structure and should be interpreted
together with diagnostics.

Sensitivity runs should vary the assumptions explicitly:

- `target_exclusion="none"` for raw/no target exclusion;
- `target_exclusion="target_docs_only"` for target-document-only filtering;
- `target_exclusion="all_docs"` for the main target-free baseline;
- `citation_identity_counting="binary"` or `"multiplicity"`;
- `citation_author_scope="all_authors"`.

Dropped-mass diagnostics are part of the method, not just logs. High
self-loop mass, target-excluded mass, empty reference identities, or many
documents without remaining pairs after filtering can signal self-referential
traditions, unresolved identities, thin support, or a mismatch between the
dataset and the chosen policy.

KLD on co-reference pairs has the same zero-probability problem as vocabulary
KLD, often more strongly because pair spaces are sparse. Jelinek-Mercer
smoothing and the configured epsilon floor keep KLD finite when a target uses
pairs that are absent from the cleaned field slice. Report `lambda_param`,
`epsilon`, support size, field entropy, and dropped mass when interpreting
levels across authors.

Implementation note: all KLD-based metrics share one document-feature core,
so sync, async, pointwise, and Welch outputs are defined identically across
Vocabulary and Citation Identity; the Citation-Identity path additionally
writes the document/slice diagnostic exports described above.
