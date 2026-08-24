from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trajectories_of_change import ReferencedVocabularyKLD, VocabularyKLD
from trajectories_of_change.metrics_kld import FeatureKLDBase
from trajectories_of_change.referenced_vocabulary import (
    _build_referenced_vocab_event_cache,
    build_reference_token_cache,
    build_referenced_vocab_events,
)


def _publications() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Bibcode": "T2000",
                "Year": 2000,
                "Author": ["Target"],
                "author_uids": ["uid:target"],
                "author_display_names": ["Target, T."],
                "References": ["SELF", "OTHER"],
                "tokens": ["own", "target"],
            },
            {
                "Bibcode": "F1998",
                "Year": 1998,
                "Author": ["Field"],
                "author_uids": ["uid:field"],
                "author_display_names": ["Field, F."],
                "References": ["OTHER"],
                "tokens": ["own", "field"],
            },
            {
                "Bibcode": "T2002",
                "Year": 2002,
                "Author": ["Target"],
                "author_uids": ["uid:target"],
                "author_display_names": ["Target, T."],
                "References": ["OTHER"],
                "tokens": ["target", "later"],
            },
            {
                "Bibcode": "F2004",
                "Year": 2004,
                "Author": ["Field"],
                "author_uids": ["uid:field"],
                "author_display_names": ["Field, F."],
                "References": ["SELF", "OTHER"],
                "tokens": ["field", "later"],
            },
        ]
    )


def _references(*, include_tokens: bool = True) -> pd.DataFrame:
    rows = [
        {
            "Bibcode": "SELF",
            "Author": ["Target"],
            "author_uids": ["uid:target"],
            "author_display_names": ["Target, T."],
            "Title": "alpha alpha",
            "Title_en": "alpha alpha",
            "Title_lang": "en",
            "Abstract": "",
            "Abstract_en": "",
            "Abstract_lang": "en",
        },
        {
            "Bibcode": "OTHER",
            "Author": ["Other"],
            "author_uids": ["uid:other"],
            "author_display_names": ["Other, O."],
            "Title": "beta beta",
            "Title_en": "beta beta",
            "Title_lang": "en",
            "Abstract": "",
            "Abstract_en": "",
            "Abstract_lang": "en",
        },
    ]
    if include_tokens:
        rows[0]["tokens"] = ["alpha", "alpha"]
        rows[1]["tokens"] = ["beta", "beta"]
    return pd.DataFrame(rows)


def test_referenced_vocabulary_and_own_vocab_share_feature_kld_base() -> None:
    publications = _publications()

    own = VocabularyKLD(
        publications,
        "",
        target_author_uid="uid:target",
        window_size=2,
        skip_incomplete_slices=False,
        allow_name_fallback=False,
        min_token_global_freq=0,
        min_docs_global_freq=1,
    )
    referenced = ReferencedVocabularyKLD(
        publications,
        _references(),
        target_author_uid="uid:target",
        policy="inclusive",
        window_size=2,
        skip_incomplete_slices=False,
        min_token_global_freq=0,
        min_docs_global_freq=1,
    )

    assert isinstance(own, FeatureKLDBase)
    assert isinstance(referenced, FeatureKLDBase)
    assert own.core.__class__ is referenced.core.__class__


def test_referenced_vocabulary_requires_reference_tokens() -> None:
    with pytest.raises(ValueError, match="references.tokens"):
        ReferencedVocabularyKLD(
            _publications(),
            _references(include_tokens=False),
            target_author_uid="uid:target",
            policy="inclusive",
        )


def test_reference_token_cache_preserves_record_semantics() -> None:
    references = _references()
    references.loc[1, "Title_lang"] = "de"
    cache = build_reference_token_cache(references)

    assert cache.ref_lookup == {"SELF": 0, "OTHER": 1}
    assert cache.ref_tokens[0] == {"alpha": 2}
    assert cache.ref_tokens[1] == {"beta": 2}
    assert cache.ref_author_uids == [{"uid:target"}, {"uid:other"}]
    assert cache.ref_language_flags == [
        {
            "title_nonenglish": False,
            "abstract_nonenglish": False,
            "title_untranslated": False,
            "abstract_untranslated": False,
        },
        {
            "title_nonenglish": True,
            "abstract_nonenglish": False,
            "title_untranslated": True,
            "abstract_untranslated": False,
        },
    ]


def test_referenced_vocabulary_policies_and_document_mass() -> None:
    publications = _publications().head(1).copy()
    references = _references()

    inclusive = ReferencedVocabularyKLD(
        publications,
        references,
        target_author_uid="uid:target",
        policy="inclusive",
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=0,
        min_docs_global_freq=1,
    )
    external = ReferencedVocabularyKLD(
        publications,
        references,
        target_author_uid="uid:target",
        policy="external_only",
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=0,
        min_docs_global_freq=1,
    )

    inclusive_weights = {
        inclusive.events.label_for_feature_id(feature): weight
        for feature, weight in zip(inclusive.events.feature_ids, inclusive.events.weights)
    }
    external_weights = {
        external.events.label_for_feature_id(feature): weight
        for feature, weight in zip(external.events.feature_ids, external.events.weights)
    }

    assert inclusive_weights == {"alpha": pytest.approx(0.5), "beta": pytest.approx(0.5)}
    assert external_weights == {"beta": pytest.approx(1.0)}
    assert inclusive.events.weights.sum() == pytest.approx(1.0)
    assert external.events.weights.sum() == pytest.approx(1.0)
    assert inclusive.diagnostics["target_authored_kept_reference_mass"].sum() == pytest.approx(0.5)
    assert external.diagnostics["removed_target_authored_reference_mentions"].sum() == 1


def test_referenced_vocabulary_diagnostics_are_target_dependent_and_skippable() -> None:
    # Guards A4: RV diagnostics are PER TARGET, so a probe target's diagnostics must not
    # be shared across targets (that was a latent bug). The consolidated/hoisted path can
    # skip building them entirely via build_diagnostics=False, which must not affect KLD.
    publications = _publications()
    references = _references()
    common = dict(
        policy="inclusive",
        window_size=2,
        skip_incomplete_slices=False,
        min_token_global_freq=0,
        min_docs_global_freq=1,
    )
    target = ReferencedVocabularyKLD(publications, references, target_author_uid="uid:target", **common)
    field = ReferencedVocabularyKLD(publications, references, target_author_uid="uid:field", **common)

    # Different targets mark different documents -> diagnostics are target-dependent.
    assert (
        target.diagnostics["is_target_document"].tolist()
        != field.diagnostics["is_target_document"].tolist()
    )

    # The hoisted path reuses the (target-independent) matrix but skips diagnostics.
    skipped = ReferencedVocabularyKLD(
        publications,
        references,
        target_author_uid="uid:target",
        prebuilt_matrix=target.matrix,
        build_diagnostics=False,
        **common,
    )
    assert skipped.diagnostics is None
    pd.testing.assert_frame_equal(
        skipped.calculate_kld_sync()[0],
        target.calculate_kld_sync()[0],
    )


def test_referenced_vocab_event_cache_matches_uncached_events_and_models() -> None:
    publications = _publications()
    references = _references()
    reference_cache = build_reference_token_cache(references)
    event_cache = _build_referenced_vocab_event_cache(
        publications,
        references,
        reference_cache=reference_cache,
    )
    common = dict(
        window_size=2,
        skip_incomplete_slices=False,
        min_token_global_freq=0,
        min_docs_global_freq=1,
        min_tokens_target_slice=1e-12,
        min_tokens_field_slice=1e-12,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
    )

    for target_author_uid in ("uid:target", "uid:field"):
        for policy in ("inclusive", "external_only"):
            uncached_events = build_referenced_vocab_events(
                publications,
                references,
                target_author_uid=target_author_uid,
                policy=policy,
                reference_cache=reference_cache,
            )
            cached_events = build_referenced_vocab_events(
                publications,
                references,
                target_author_uid=target_author_uid,
                policy=policy,
                reference_cache=reference_cache,
                _event_cache=event_cache,
            )
            assert cached_events.token_labels == uncached_events.token_labels
            assert cached_events.doc_indices.tolist() == uncached_events.doc_indices.tolist()
            assert cached_events.feature_ids.tolist() == uncached_events.feature_ids.tolist()
            assert cached_events.weights.tolist() == pytest.approx(uncached_events.weights.tolist())
            pd.testing.assert_frame_equal(cached_events.diagnostics, uncached_events.diagnostics)

            uncached_model = ReferencedVocabularyKLD(
                publications,
                references,
                target_author_uid=target_author_uid,
                policy=policy,
                reference_cache=reference_cache,
                **common,
            )
            cached_model = ReferencedVocabularyKLD(
                publications,
                references,
                target_author_uid=target_author_uid,
                policy=policy,
                reference_cache=reference_cache,
                _event_cache=event_cache,
                **common,
            )
            pd.testing.assert_frame_equal(cached_model.diagnostics, uncached_model.diagnostics)
            pd.testing.assert_frame_equal(cached_model.calculate_kld_sync()[0], uncached_model.calculate_kld_sync()[0])
            pd.testing.assert_frame_equal(cached_model.calculate_kld_sync()[1], uncached_model.calculate_kld_sync()[1])
            pd.testing.assert_frame_equal(cached_model.calculate_kld_async(), uncached_model.calculate_kld_async())


def test_referenced_vocab_event_cache_precomputes_token_weights_in_first_seen_order() -> None:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "P2000",
                "Year": 2000,
                "Author": ["Target"],
                "author_uids": ["uid:target"],
                "References": ["WEIGHTED"],
                "tokens": ["own"],
            }
        ]
    )
    references = pd.DataFrame(
        [
            {
                "Bibcode": "WEIGHTED",
                "Author": ["Other"],
                "author_uids": ["uid:other"],
                "Title": "alpha beta alpha gamma",
                "Title_en": "alpha beta alpha gamma",
                "Title_lang": "en",
                "Abstract": "",
                "Abstract_en": "",
                "Abstract_lang": "en",
                "tokens": ["alpha", "beta", "alpha", "gamma"],
            }
        ]
    )

    event_cache = _build_referenced_vocab_event_cache(publications, references)
    ref_int = event_cache.reference_cache.ref_lookup["WEIGHTED"]

    assert event_cache.reference_token_weights[ref_int] == (
        ("alpha", pytest.approx(0.5)),
        ("beta", pytest.approx(0.25)),
        ("gamma", pytest.approx(0.25)),
    )


def test_referenced_vocabulary_async_uses_field_minus_target_sign() -> None:
    model = ReferencedVocabularyKLD(
        _publications(),
        _references(),
        target_author_uid="uid:target",
        policy="inclusive",
        window_size=2,
        skip_incomplete_slices=False,
        min_token_global_freq=0,
        min_docs_global_freq=1,
        min_tokens_target_slice=1e-12,
        min_tokens_field_slice=1e-12,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
    )

    async_df = model.calculate_kld_async()

    assert {"target_slice", "field_slice", "time_diff", "kld"}.issubset(async_df.columns)
    assert (async_df["time_diff"] == async_df["field_slice"] - async_df["target_slice"]).all()
