# MovieLens 32M (ML-32M) — Data Manifest

Regime-contrast dataset for UPGRADE_PLAN.md §8c (Phase 9, T9-3a/T9-3b). Separate from `data/MANIFEST.md` on purpose: run records hash their whole dataset manifest, and `make reproduce-headline` compares that hash for the pinned Amazon headline. One manifest per dataset keeps the two independent.

## Source URL

- https://files.grouplens.org/datasets/movielens/ml-32m.zip

## Citation

Harper, F. Maxwell, and Joseph A. Konstan. "The MovieLens Datasets: History and Context." ACM Transactions on Interactive Intelligent Systems 5, no. 4 (2015): 19:1-19:19. https://doi.org/10.1145/2827872

## License

Research use only per the GroupLens usage license; raw data is never redistributed and data/ is gitignored (only this manifest is committed).

## Download date

2026-08-20

## Files

### ml-32m.zip

- URL: https://files.grouplens.org/datasets/movielens/ml-32m.zip
- Size (bytes): 238950008
- SHA-256 (computed locally — ours is ground truth): e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee

### ratings.csv

- Extracted from: ml-32m.zip (ml-32m/ratings.csv)
- Size (bytes): 877076222
- Data rows (excl. header): 32000204
- SHA-256 (computed locally — ours is ground truth): 91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee

### movies.csv

- Extracted from: ml-32m.zip (ml-32m/movies.csv)
- Size (bytes): 4242926
- Data rows (excl. header): 87585
- SHA-256 (computed locally — ours is ground truth): b37ca1abc7798de741138ed252b62f69f7e37c84b8a8fab1b82d409b4c6c5cc2

### tags.csv

- Extracted from: ml-32m.zip (ml-32m/tags.csv)
- Size (bytes): 72353890
- Data rows (excl. header): 2000072
- SHA-256 (computed locally — ours is ground truth): c52e458a89b9deee410f50813e7a64491fbe92a6f07b5c4b58e64c0b3cc95b89

## Not extracted

- ml-32m/links.csv: IMDb/TMDb foreign keys only; no module in this lab reads them
- ml-32m/README.txt: documentation; the archive itself is the hashed artifact of record

## Published counts

32000204 ratings / 200948 users / 87585 movies, timestamps 1995 → Oct 2023 (ML-32M README); no verified published count for tag applications, so the observed tags.csv row count above is ground truth. Observed extracted row counts: movies.csv=87585, ratings.csv=32000204, tags.csv=2000072. These per-file row counts are enforced against the live bronze tables by `make bronze-verify-ml32m`.

## Timestamp caveat (UPGRADE_PLAN.md §8b)

MovieLens timestamps are rating-ENTRY times, not consumption times, and the catalog is backfilled — an item's first rating is only a proxy for its release. Disclosed wherever the regime contrast is reported (cite Sun et al., arXiv:2307.09985).
