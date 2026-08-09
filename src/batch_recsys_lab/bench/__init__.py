"""Engine benchmarks (Phase 7 stretch item 1).

Read-only reality checks that re-run a pipeline stage on a different engine and
reconcile it against the recorded Spark waterfall. Nothing in this package ever
writes to ``data/warehouse/`` — outputs live under ``data/bench/<engine>/``.

Nothing here ever starts a JVM: the point of a single-node reality check is that
it runs without the Spark session or the JDK pin. (``pyspark`` is still imported
transitively — ``batch_recsys_lab.contracts.__init__`` re-exports the Spark-based
engine alongside the pure-YAML loader — but it is never used and no gateway is
launched.)
"""
