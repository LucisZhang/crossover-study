# Amazon Reviews 2023 — Electronics — Data Manifest

Dataset: Amazon Reviews 2023 (McAuley-Lab), Electronics review category and metadata.

## Source URLs

- https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz
- https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz

## Citation

Hou, Yupeng, et al. "Bridging Language and Items for Retrieval and Recommendation." arXiv preprint arXiv:2403.03952 (2024).

## License

Research use only per the dataset's release terms; raw data is never redistributed and data/ is gitignored (only this manifest is committed).

## Download date

2026-08-05

## Files

### Electronics.jsonl.gz

- URL: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz
- Size (bytes): 6474438619
- Server Last-Modified: Thu, 16 Jan 2025 23:28:26 GMT
- SHA-256 (computed locally — ours is ground truth): 17b2c5f3736d4c0cb874859076436ebd4513f4c2396c528440a63204084e6a28

### meta_Electronics.jsonl.gz

- URL: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz
- Size (bytes): 1312900427
- Server Last-Modified: Thu, 16 Jan 2025 22:22:45 GMT
- SHA-256 (computed locally — ours is ground truth): a4a196a1c8e443e0942d8a5c79b5a5d2d68e29d483f2badec1126690b4b2790d

## Published counts

43.9M reviews / 1.61M items (per the Amazon Reviews 2023 / Hou et al. 2024 release site). Observed bronze counts and the delta against these published counts are recorded in the "Bronze reconciliation" section below.

## Bronze layer notes

bronze.reviews projects out the `text` and `images` columns per UPGRADE_PLAN.md §5 (the lab never uses review text; item text comes from metadata).

## Bronze reconciliation

| Table | Published (rounded) | Observed | Delta vs rounded |
|---|---|---|---|
| reviews | 43.9M | 43,886,944 | -13,056 |
| items | 1.61M (rounded) | 1,610,012 | +12 |

Note: published figures are rounded (per the Amazon Reviews 2023 / Hou et al. 2024 release site); the observed bronze count is canonical. Delta is observed minus the literal rounded published number, not a measure of ingestion correctness.

Last verified: 2026-08-17T15:35:45Z

## Reconciliation waterfall

Run ID: `20260817T154801Z-327c417` · generated 2026-08-17T15:56:55.922854+00:00

Every dropped row carries a reason; per edge, Σ reason-rows == the source count AND the `kept` count == the live Iceberg table count (re-read at publish time). Assertions are enforced in code (non-zero exit on drift).

### reviews  (raw → bronze → silver → gold)

| stage_from | stage_to | reason | rows |
|---|---|---|---|
| raw | bronze | kept | 43,886,944 |
| raw | bronze | corrupt | 0 |
| bronze | silver | kept | 43,365,424 |
| bronze | silver | quarantine:rating_domain | 2 |
| bronze | silver | exact_duplicate | 477,968 |
| bronze | silver | superseded_by_later_review | 43,550 |
| silver | gold | kept | 15,473,536 |
| silver | gold | kcore_pruned | 27,891,888 |

Reconciliation checks:
- raw → bronze: Σ = 43,886,944 = source 43,886,944 ✓; target `local.bronze.reviews` count = 43,886,944 ✓
- bronze → silver: Σ = 43,886,944 = source 43,886,944 ✓; target `local.silver.interactions` count = 43,365,424 ✓
- silver → gold: Σ = 43,365,424 = source 43,365,424 ✓; target `local.gold.interactions_5core` count = 15,473,536 ✓

#### k-core funnel (reviews, run `20260817T154801Z-327c417`)

| iteration | rows | users | items | converged | wall_clock_s |
|---|---|---|---|---|---|
| 0 | 43,365,424 | 18,286,190 | 1,609,860 | False | 20.239 |
| 1 | 16,883,417 | 1,847,620 | 603,239 | False | 28.727 |
| 2 | 15,929,364 | 1,743,137 | 383,259 | False | 15.397 |
| 3 | 15,541,920 | 1,650,879 | 375,640 | False | 18.095 |
| 4 | 15,497,125 | 1,646,183 | 368,980 | False | 19.351 |
| 5 | 15,477,434 | 1,641,604 | 368,626 | False | 15.676 |
| 6 | 15,475,048 | 1,641,345 | 368,288 | False | 17.941 |
| 7 | 15,473,873 | 1,641,077 | 368,262 | False | 16.646 |
| 8 | 15,473,705 | 1,641,061 | 368,236 | False | 15.403 |
| 9 | 15,473,597 | 1,641,037 | 368,233 | False | 15.994 |
| 10 | 15,473,576 | 1,641,034 | 368,230 | False | 16.294 |
| 11 | 15,473,556 | 1,641,029 | 368,230 | False | 15.915 |
| 12 | 15,473,552 | 1,641,029 | 368,229 | False | 16.766 |
| 13 | 15,473,544 | 1,641,027 | 368,229 | False | 16.724 |
| 14 | 15,473,540 | 1,641,027 | 368,228 | False | 15.234 |
| 15 | 15,473,536 | 1,641,026 | 368,228 | False | 13.331 |
| 16 | 15,473,536 | 1,641,026 | 368,228 | True | 13.848 |

### items  (raw → bronze → silver)

| stage_from | stage_to | reason | rows |
|---|---|---|---|
| raw | bronze | kept | 1,610,012 |
| raw | bronze | corrupt | 0 |
| bronze | silver | kept | 1,610,012 |
| bronze | silver | exact_duplicate | 0 |
| bronze | silver | superseded_by_later_review | 0 |

Reconciliation checks:
- raw → bronze: Σ = 1,610,012 = source 1,610,012 ✓; target `local.bronze.items` count = 1,610,012 ✓
- bronze → silver: Σ = 1,610,012 = source 1,610,012 ✓; target `local.silver.items` count = 1,610,012 ✓
