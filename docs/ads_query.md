# ADS query of the follow-up study

This is the exact NASA/ADS search query behind the corpus of the follow-up
study on general relativity and gravitation (see the
[README](../README.md#the-follow-up-study)): the canonical ADS-GRG corpus
(prepared dataset `grg_ads_toponymy_20260430_explore_ref_tokens`), as recorded
in the producer run configuration (`ADS_Pipeline`, run
`grg_openrouter_toponymy_explore`, stage `search`). It is the set-based
construction of the earlier study (Schlattmann & Vogl 2024, arXiv:2501.00391):
two curated ADS libraries as seeds, expanded by one and two citation
generations, unioned with a multilingual keyword/abstract search. The study's
supplementary material points to this file.

```text
((docs(library/P0hyiLw0T8qsyHSBpWigJA))
  OR (citations(docs(library/P0hyiLw0T8qsyHSBpWigJA)) AND year:1915-2000)
  OR (citations(citations(docs(library/P0hyiLw0T8qsyHSBpWigJA))) AND year:1915-2000))
OR ((docs(library/_-RCjJotRKCZC1PP03MqwA) AND year:1915-2000))
OR (abs:((general AND relativi*) OR gravit* OR (allgemein* AND relativi*)
    OR "relativité générale" OR "teoria della relatività" OR "Gravité quantique"
    OR "Gravità quantistica" OR (einheitlich* AND feld*) OR Quantengravitation
    OR "champ unifié" OR (unified AND field*) OR "quantum gravity"
    OR cosmo* OR Kosmo*) AND year:1915-2000)
```

Notes:

- The two `library/...` identifiers are curated ADS libraries seeding the set-based
  construction; the `citations(...)` clauses add the first and second citation
  generations of the primary seed within the year window.
- The year filter of the query clauses is `1915-2000`; the prepared corpus contains
  records from 1911 onward because seed-library items are included without the year
  clause. The analysed span reported in the paper is 1911-2000.
- Retrieval used the ADS API with a personal token (not part of this repository).
  The corpus is not part of this repository either. The study's reproduction
  package, [https://doi.org/10.5281/zenodo.22726749](https://doi.org/10.5281/zenodo.22726749),
  documents it under `corpus/` with the bibcode lists and the manifest of the
  prepared bundle (183,680 publications, 445,832 reference records after
  preparation); the ADS records themselves are not redistributed.

## Release

The study cites release 0.2.0 of this package: tag `v0.2.0`, commit
`232851aa47582c11bdc0789ef6721130ab1b3b2f`, archived at
[https://doi.org/10.5281/zenodo.22084879](https://doi.org/10.5281/zenodo.22084879).
This file was added to the repository after release 0.2.1; the query itself
was first published in the reproduction package (`corpus/ads_query.md`).
