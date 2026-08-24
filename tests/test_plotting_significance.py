from __future__ import annotations

import pandas as pd

from trajectories_of_change.plotting.kld import prepare_kld_dashboard_inputs


def test_pair_slice_and_global_corrections_produce_distinct_async_results() -> None:
    pairs = [(2000, 2000), (2000, 2002), (2002, 2000), (2002, 2002)]
    async_df = pd.DataFrame(
        [
            {"target_slice": target, "field_slice": field, "kld": 1.0}
            for target, field in pairs
        ]
    )
    sync_df = pd.DataFrame(
        [
            {"slice": 2000, "kld_all": 1.0},
            {"slice": 2002, "kld_all": 1.0},
        ]
    )
    pvalues = {
        (2000, 2000): 0.020,
        (2000, 2002): 0.008,
        (2002, 2000): 0.500,
        (2002, 2002): 0.500,
    }
    welch = pd.DataFrame(
        [
            {
                "target_slice": target,
                "field_slice": field,
                "term": term,
                "pvalue": pvalues[(target, field)] if term == "signal" else 0.900,
                "kld_contribution": contribution if term == "signal" else 0.0,
            }
            for target, field in pairs
            for term, contribution in (("signal", 1.0 if field == 2000 else 2.0), ("noise", 0.0))
        ]
    )

    totals = {}
    for scope in ("pair", "slice", "global"):
        _, prepared_welch, prepared_async = prepare_kld_dashboard_inputs(
            sync_df,
            async_df,
            welch,
            alpha=0.05,
            multiple_testing="bonferroni",
            multiple_testing_scope=scope,
        )
        totals[scope] = float(prepared_async["kld_sig"].sum())
        assert prepared_welch.attrs["multiple_testing"] == "bonferroni"
        assert prepared_welch.attrs["multiple_testing_scope"] == scope
        assert prepared_async.attrs["multiple_testing_scope"] == scope

    assert totals == {"pair": 3.0, "slice": 2.0, "global": 0.0}
