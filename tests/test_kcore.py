"""Iterative k-core tests (Phase 1, T6).

Uses the shared tmp-warehouse Spark session (``tests/conftest.py``). Toy graphs
are built with k=2 so a small hand-checkable edge set exercises the multi-pass
cascade, the already-converged short circuit, and the non-convergence guard.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from batch_recsys_lab.features.kcore import build_summary, run_kcore

pytestmark = pytest.mark.spark

_SRC_DDL = (
    "user_id string, parent_asin string, ts timestamp, rating double, "
    "asin string, helpful_vote long, verified_purchase boolean"
)


def _write_edges(spark, table, edges):
    """Materialize a toy interaction table from (user_id, parent_asin) edges.

    Non-graph columns are filled with fixed values; the k-core is a pure function
    of the edge set, so their values are immaterial to the assertions.
    """
    ns = table.rsplit(".", 1)[0]
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {ns}")
    rows = [
        (u, p, datetime(2022, 1, 1), 5.0, f"{p}-a", 0, True) for (u, p) in edges
    ]
    spark.createDataFrame(rows, _SRC_DDL).writeTo(table).createOrReplace()


def _funnel_view(funnel):
    """Drop nondeterministic/irrelevant keys for exact per-iteration comparison."""
    return [
        {
            "iteration": r["iteration"],
            "rows": r["rows"],
            "users": r["users"],
            "items": r["items"],
            "converged": r["converged"],
        }
        for r in funnel
    ]


def _edge_set(spark, table):
    return {
        (r["user_id"], r["parent_asin"])
        for r in spark.table(table).select("user_id", "parent_asin").collect()
    }


# Cascade toy (k=2): removing the degree-1 item P3 orphans U3's remaining edge,
# which must be pruned in a *second* pass — so convergence needs >= 2 real
# filtering iterations.
#   U1-P1 U1-P2 | U2-P1 U2-P2 | U3-P2 U3-P3
# Stable 2-core is exactly {U1,U2} x {P1,P2}.
_CASCADE_EDGES = [
    ("U1", "P1"),
    ("U1", "P2"),
    ("U2", "P1"),
    ("U2", "P2"),
    ("U3", "P2"),
    ("U3", "P3"),
]
_CASCADE_CORE = {("U1", "P1"), ("U1", "P2"), ("U2", "P1"), ("U2", "P2")}


def test_cascade_needs_two_iterations(spark):
    src = "local.tst_kcore.cascade_src"
    tgt = "local.tst_kcore.cascade_core"
    funnel_tbl = "local.tst_kcore.funnel_cascade"
    _write_edges(spark, src, _CASCADE_EDGES)

    funnel = run_kcore(
        spark, src, tgt, funnel_tbl, k=2, max_iters=50, run_id="RUN_CASCADE"
    )

    assert _funnel_view(funnel) == [
        {"iteration": 0, "rows": 6, "users": 3, "items": 3, "converged": False},
        {"iteration": 1, "rows": 5, "users": 3, "items": 2, "converged": False},
        {"iteration": 2, "rows": 4, "users": 2, "items": 2, "converged": False},
        {"iteration": 3, "rows": 4, "users": 2, "items": 2, "converged": True},
    ]
    assert _edge_set(spark, tgt) == _CASCADE_CORE

    summary = build_summary(funnel, src, tgt, k=2)
    assert summary["input_rows"] == 6
    assert summary["output_rows"] == 4
    assert summary["kcore_pruned"] == 2
    assert summary["output_users"] == 2
    assert summary["output_items"] == 2
    assert summary["iterations"] == 3
    assert summary["converged"] is True


def test_already_converged_immediately(spark):
    src = "local.tst_kcore.stable_src"
    tgt = "local.tst_kcore.stable_core"
    funnel_tbl = "local.tst_kcore.funnel_stable"
    _write_edges(spark, src, sorted(_CASCADE_CORE))

    funnel = run_kcore(
        spark, src, tgt, funnel_tbl, k=2, max_iters=50, run_id="RUN_STABLE"
    )

    # Input is already a 2-core: one confirming pass leaves the row count
    # unchanged, so it converges on iteration 1.
    assert _funnel_view(funnel) == [
        {"iteration": 0, "rows": 4, "users": 2, "items": 2, "converged": False},
        {"iteration": 1, "rows": 4, "users": 2, "items": 2, "converged": True},
    ]
    assert funnel[-1]["converged"] is True
    assert _edge_set(spark, tgt) == _CASCADE_CORE
    assert build_summary(funnel, src, tgt, k=2)["iterations"] == 1


def test_max_iters_guard_raises(spark):
    src = "local.tst_kcore.guard_src"
    tgt = "local.tst_kcore.guard_core"
    funnel_tbl = "local.tst_kcore.funnel_guard"
    _write_edges(spark, src, _CASCADE_EDGES)

    # The cascade needs 3 iterations to converge; capping at 1 must fail hard.
    with pytest.raises(RuntimeError, match="did not converge"):
        run_kcore(spark, src, tgt, funnel_tbl, k=2, max_iters=1, run_id="RUN_GUARD")

    # Funnel gathered so far (iteration 0 snapshot + iteration 1) was persisted
    # for inspection even though the run failed.
    persisted = (
        spark.table(funnel_tbl).where("run_id = 'RUN_GUARD'").count()
    )
    assert persisted == 2
    # The non-converged table must NOT have been written.
    assert not spark.catalog.tableExists(tgt)


def test_funnel_lands_in_iceberg_with_run_id(spark):
    src = "local.tst_kcore.land_src"
    tgt = "local.tst_kcore.land_core"
    funnel_tbl = "local.dq.kcore_funnel"  # exercise the real default funnel table
    _write_edges(spark, src, _CASCADE_EDGES)

    run_kcore(spark, src, tgt, funnel_tbl, k=2, max_iters=50, run_id="RUN_LAND")

    persisted = (
        spark.table(funnel_tbl)
        .where("run_id = 'RUN_LAND'")
        .orderBy("iteration")
        .collect()
    )
    assert [r["iteration"] for r in persisted] == [0, 1, 2, 3]
    assert [r["rows"] for r in persisted] == [6, 5, 4, 4]
    assert persisted[-1]["converged"] is True
    # Schema/column names land as declared.
    assert set(spark.table(funnel_tbl).columns) == {
        "run_id",
        "iteration",
        "rows",
        "users",
        "items",
        "converged",
        "wall_clock_s",
    }
