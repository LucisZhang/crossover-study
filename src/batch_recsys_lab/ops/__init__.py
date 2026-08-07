"""Lakehouse ops exhibits (Phase 5, T20; UPGRADE_PLAN.md §8 Phase 5).

Monthly-partition incremental append, late-data MERGE upsert, compaction and
snapshot expiry — every one of them against a **new, disposable**
``local.ops.*`` namespace.

ABSOLUTE rule for this package: the published lakehouse
(``local.bronze.*``/``local.silver.*``/``local.gold.*``/``local.dq.*``/
``local.quarantine.*``) is **never** written, compacted or expired. Those tables
back every recorded eval number; rewriting their files would invalidate the
snapshot IDs carried in ``results/runs.jsonl`` (CLAUDE.md invariant #3). The
enforcement is :func:`require_ops_table`, called as the unconditional first
statement of every mutator in this package, plus a JVM-free post-step assertion
in ``run_scenario`` that the protected tables' snapshot IDs did not move.
"""

from batch_recsys_lab.ops.maintenance import OPS_NAMESPACE, require_ops_table

__all__ = ["OPS_NAMESPACE", "require_ops_table"]
