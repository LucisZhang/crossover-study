# Volume by Month

This document is the split-boundary inspection required by `docs/engineering-log/UPGRADE_PLAN.md` §6.1 before freezing `configs/splits.yaml`. It summarizes interaction volume over time to inform the choice of TRAIN/VAL/TEST boundaries.

- **Source table:** `local.silver.interactions` (43,365,424 rows, post-contract, post-dedup)
- **Run date:** 2026-08-05
- **Query provenance:** `groupBy month` aggregation over `silver.interactions`, counting interactions and distinct users per month, plus cumulative row counts through candidate boundary dates.
- **Snapshot coverage:** 1996-11 → 2023-09

## Yearly Totals

| Year | Interactions |
|---|---:|
| 1996 | 1 |
| 1998 | 8 |
| 1999 | 412 |
| 2000 | 3,696 |
| 2001 | 6,752 |
| 2002 | 8,982 |
| 2003 | 12,176 |
| 2004 | 17,392 |
| 2005 | 32,672 |
| 2006 | 53,564 |
| 2007 | 123,975 |
| 2008 | 155,318 |
| 2009 | 207,214 |
| 2010 | 305,411 |
| 2011 | 507,258 |
| 2012 | 790,861 |
| 2013 | 1,693,857 |
| 2014 | 2,448,635 |
| 2015 | 3,367,404 |
| 2016 | 3,715,191 |
| 2017 | 3,750,210 |
| 2018 | 3,866,981 |
| 2019 | 4,910,456 |
| 2020 | 5,659,386 |
| 2021 | 5,371,684 |
| 2022 | 4,472,269 |
| 2023 | 1,883,659 |

## Monthly Volume — Last 40 Months

| Month | Interactions | Distinct Users |
|---|---:|---:|
| 2020-06 | 412,644 | 349,247 |
| 2020-07 | 477,765 | 405,093 |
| 2020-08 | 485,366 | 413,178 |
| 2020-09 | 455,649 | 390,730 |
| 2020-10 | 473,351 | 403,708 |
| 2020-11 | 422,230 | 364,294 |
| 2020-12 | 526,915 | 455,368 |
| 2021-01 | 580,385 | 498,343 |
| 2021-02 | 493,875 | 426,392 |
| 2021-03 | 575,480 | 493,236 |
| 2021-04 | 523,653 | 448,267 |
| 2021-05 | 458,035 | 392,742 |
| 2021-06 | 449,342 | 386,405 |
| 2021-07 | 475,733 | 408,243 |
| 2021-08 | 413,612 | 357,219 |
| 2021-09 | 354,888 | 308,064 |
| 2021-10 | 339,295 | 293,265 |
| 2021-11 | 316,969 | 275,706 |
| 2021-12 | 390,417 | 335,970 |
| 2022-01 | 428,206 | 367,128 |
| 2022-02 | 333,100 | 286,636 |
| 2022-03 | 364,415 | 308,218 |
| 2022-04 | 339,393 | 288,249 |
| 2022-05 | 329,253 | 281,627 |
| 2022-06 | 314,612 | 268,812 |
| 2022-07 | 385,155 | 324,385 |
| 2022-08 | 423,359 | 347,971 |
| 2022-09 | 375,431 | 309,278 |
| 2022-10 | 384,065 | 305,416 |
| 2022-11 | 370,110 | 286,355 |
| 2022-12 | 425,170 | 331,436 |
| 2023-01 | 463,426 | 362,237 |
| 2023-02 | 372,363 | 298,192 |
| 2023-03 | 419,611 | 326,960 |
| 2023-04 | 252,473 | 199,305 |
| 2023-05 | 144,889 | 120,867 |
| 2023-06 | 93,827 | 78,990 |
| 2023-07 | 78,710 | 64,659 |
| 2023-08 | 54,737 | 43,688 |
| 2023-09 | 3,623 | 2,993 |

## Cumulative Share at Candidate Boundaries

| Boundary (through) | Cumulative Interactions | Cumulative Share |
|---|---:|---:|
| 2022-03-31 | 38,135,217 | 87.94% |
| 2022-06-30 | 39,118,475 | 90.21% |
| 2022-09-30 | 40,302,420 | 92.94% |
| 2022-12-31 | 41,481,765 | 95.66% |
| 2023-03-31 | 42,737,165 | 98.55% |

## Decision

The owner froze the §6.1 defaults on 2026-08-05:

| Split | Date Range | Rows | Share |
|---|---|---:|---:|
| TRAIN | ≤ 2022-06-30 | 39,118,475 | 90.21% |
| VAL | 2022-07-01 → 2022-12-31 | 2,363,290 | — |
| TEST | 2023-01-01 → snapshot end (2023-09) | 1,883,659 | — |

**Rationale:** The snapshot's crawl-cutoff taper (2023-04 onward collapses from ~420k/month down to 3.6k by 2023-09) means later boundaries would put most of TEST inside the taper, understating true volume. Earlier boundaries would stretch TEST to 15 months across multiple seasons, complicating evaluation. The chosen defaults keep TEST's first three months (2023-01 through 2023-03) at full, pre-taper volume, giving a representative evaluation window before the crawl cutoff distorts the data.
