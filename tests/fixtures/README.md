# Bundled bronze fixtures

Deterministic ~50k-row sample of `local.bronze.reviews` plus the matching items slice from `local.bronze.items`, used as the CI substrate (docs/engineering-log/UPGRADE_PLAN.md repo map, `tests/fixtures/`).

## Provenance

- Source tables: `local.bronze.reviews`, `local.bronze.items`
- Sampling rule (reviews): rows where `pmod(xxhash64(user_id, asin, timestamp), 800) == 0`, then `ORDER BY xxhash64(user_id, asin, timestamp) LIMIT 50000`.
- Sampling rule (items): distinct `parent_asin` from the reviews sample, joined to `local.bronze.items`, `ORDER BY xxhash64(parent_asin) LIMIT 5000`.
- Generated: 2026-08-05T08:47:25Z (filled at runtime by `make fixture`)
- Reviews fixture rows: 50000
- Items fixture rows: 5000
- No columns dropped (combined fixture size <= 50MB).

## Files

- `bronze_reviews_50k.parquet` — single-file parquet, 50000 rows, bronze reviews schema.
- `bronze_items_fixture.parquet` — single-file parquet, up to 5000 rows, bronze items schema.

## Regeneration

Regeneration is deterministic given the same bronze snapshot: `make fixture` (`python -m batch_recsys_lab.ingest.make_fixture`) re-derives byte-identical output because the sampling predicate is a pure content hash of `(user_id, asin, timestamp)` / `parent_asin`, not an arbitrary scan order.
