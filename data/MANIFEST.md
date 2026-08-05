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

Ingest wall-clock: reviews=509s, items=926s

Last verified: 2026-08-05T08:38:24Z
