from __future__ import annotations

import json

import pandas as pd

from trajectories_of_change import MetricResult


def test_metric_result_v2_roundtrip_preserves_config_and_provenance(tmp_path) -> None:
    result = MetricResult(
        sync=pd.DataFrame({"slice": [2000], "kld_all": [0.25]}),
        pointwise=pd.DataFrame({"slice": [2000], "term": ["alpha"]}),
        async_df=pd.DataFrame(
            {"target_slice": [2000], "field_slice": [2000], "kld": [0.25]}
        ),
        welch=pd.DataFrame({"target_slice": [2000], "pvalue": [0.01]}),
        kind="kld",
        metric="own_vocab",
        target_author_uid="uid:author",
        target_name="Author, A.",
        window_size=2,
        config={
            "lambda_param": 0.05,
            "epsilon": 1e-12,
            "alpha": 0.05,
            "multiple_testing": "holm",
            "multiple_testing_scope": "global",
        },
        provenance={"dataset_manifest": {"run_id": "run-1"}, "validation_warnings": []},
        metadata={"target_doc_counts": {2000: 3}},
    )

    saved_dir = result.save(tmp_path / "result")
    loaded = MetricResult.load(saved_dir)

    assert loaded.target_author_uid == result.target_author_uid
    assert loaded.config == result.config
    assert loaded.provenance == result.provenance
    assert loaded.metadata == {"target_doc_counts": {"2000": 3}}
    pd.testing.assert_frame_equal(loaded.sync, result.sync)
    pd.testing.assert_frame_equal(loaded.pointwise, result.pointwise)
    pd.testing.assert_frame_equal(loaded.async_df, result.async_df)
    pd.testing.assert_frame_equal(loaded.welch, result.welch)
    manifest = json.loads((saved_dir / "metric_result.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2


def test_metric_result_loads_legacy_v1_folder(tmp_path) -> None:
    result_dir = tmp_path / "legacy"
    result_dir.mkdir()
    pd.DataFrame({"slice": [2000], "kld_all": [0.5]}).to_parquet(
        result_dir / "sync.parquet", index=False
    )
    (result_dir / "metric_result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "kld",
                "metric": "own_vocab",
                "target_name": "Legacy",
                "window_size": 2,
                "tables": {"sync": "sync.parquet"},
            }
        ),
        encoding="utf-8",
    )

    loaded = MetricResult.load(result_dir)

    assert loaded.metric == "own_vocab"
    assert loaded.target_author_uid is None
    assert loaded.config == {}
    assert loaded.provenance == {}
    assert loaded.pointwise.empty
