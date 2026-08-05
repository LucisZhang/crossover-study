# Experiment Log

Dated entries tracking pipeline runs, acceptance checks, and notable findings for the
batch-recsys-lab silver/gold/contracts pipeline.

---

## Phase 1 acceptance — silver+gold+contracts (2026-08-05)

Full-scale acceptance run (`run_id 20260805T143256Z-7406fc1`, second of two independent
`make data` rebuilds; first build `run_id 20260805T135836Z-7406fc1` same day).

### Reconciliation waterfall — exact on all edges

**Reviews:**

| stage | rows | delta |
|---|---:|---:|
| raw | 43,886,944 | — |
| silver (post-gate) | 43,365,424 | −521,520 (2 quarantined rating_domain, 477,968 exact duplicates, 43,550 superseded) |
| gold 5-core | 15,473,536 | −27,891,888 (k-core filtering) |

**Items:** 1,610,012 clean through (raw → silver, no quarantine loss).

k-core converged at **iteration 16**: users 18,286,190 → 1,641,026; items 1,609,860 →
368,228. Convergence has a long tail — iterations 8–16 shave fewer than 200 rows total
across both sides, i.e. nearly all of the reduction happens in the first ~7 iterations.

### Determinism

8/8 tables content-identical across two independent `make data` rebuilds (~27.5 min
each).

### Contracts

All contracts pass; `run_audit` exit code 0. Querying `local.dq.dq_results` for the
latest run (`20260805T143256Z-7406fc1`, latest row per `(table_name, check_id)`) shows
**zero rows with `status == 'fail'`** — the only `fail` rows in the ledger belong to
earlier/intermediate audit passes on the same run_id and on the prior run_id
(`20260805T135836Z-7406fc1`, `20260805T121746Z-7406fc1`), superseded by later
re-assertions with `status` downgraded to `measured` (T8 nullable-marking note on
Iceberg `createOrReplace`).

### DQ metrics (latest run, `local.dq.dq_results`)

| metric | value |
|---|---|
| orphan_rate (interactions → items FK, `item_fk`) | 0.0 (0 / 43,365,424) |
| brand_unknown_share | 4.545% (73,178 / 1,610,012) |
| price_unparseable | 316 rows, 0.0196% (316 / 1,610,012) |
| rating_domain violations (gate, silver ingest) | 2 (out of 43,886,944 raw rows; matches the 2 quarantined rows in the waterfall above) |
| rating_domain violations (post-gate, silver/gold) | 0 |

**Brand source share** (`brand_source_share` details JSON, `local.silver.items`):

| source | rows | share of 1,610,012 |
|---|---:|---:|
| Brand field | 1,153,897 | 71.67% |
| Manufacturer fallback | 384,785 | 23.90% |
| Neither (none) | 71,330 | 4.43% |

**Measured vs expected unknown-brand share:** UPGRADE_PLAN.md §7 (and the raw-hazards
note in §"Known raw-data hazards") flags ~18% unknown-brand share as the expectation for
this category. Measured `brand_unknown_share` is **4.545%**, well below the ~18%
expectation — the Manufacturer fallback (23.90% of rows) recovers most of the brand
signal that would otherwise show up as unknown, so the *residual* unknown share after
fallback is much lower than the raw/no-fallback expectation.

### Gold table sizes

| table | rows |
|---|---:|
| user_stats | 1,641,026 |
| item_features | 368,228 |
| popularity | 1,223,106 |

### MANIFEST

`data/MANIFEST.md` contains the `## Reconciliation waterfall` section with the funnel
(verified present, not rewritten here).

### Splits

Frozen splits decision: defaults, per `docs/volume_by_month.md`.
