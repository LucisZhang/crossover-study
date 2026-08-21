"""Static-demo data export (Phase 6; docs/engineering-log/UPGRADE_PLAN.md §9).

``demo/data/`` is a pure projection of committed evidence: a number may appear
in a demo JSON only if it is byte-identical (full precision — display rounding
lives in the site's JS, never here) to a value reachable from

1. a ``results/runs.jsonl`` record,
2. a results artifact whose SHA-256 a record carries, or
3. a per-user parquet a record names in ``per_user_artifact``.

:mod:`batch_recsys_lab.demo.export_core` makes the untraceable number
impossible by construction: every leaf is written through ``TracedWriter``,
which records a ``demo/data/trace_manifest.json`` entry for it.
:mod:`batch_recsys_lab.demo.verify_traceability` re-checks that claim from the
outside, sharing no code with the writing path.
"""
