# Seed archive (read-only reference)

Copied 2026-08-05 from the CISC3018 course-project archive in OneDrive
(`.../Cloud Computing and Big Data System/Project/`). These files are **reference
material only** — never imported, executed, or treated as this repo's implementation.

| File | Why it's here (UPGRADE_PLAN.md §3.4) |
|---|---|
| `jsonl_to_parquet.ipynb` | Schema seed: Amazon metadata StructType + `details`-map extraction + text-field assembly. Port the field list into `bronze.items` / `silver.items`. **Fixes required:** no −1.0 sentinel fills (NULL stays NULL), case-insensitive `details` key matching, keep units. |
| `Model Training and Bot - Metadata only.ipynb` | Design seed: MiniLM (`all-MiniLM-L6-v2`) item-text recipe (title + brand + category + features). **Fix required:** the `0.92·cosine + 0.08·popularity` blend uses unnormalized popularity — normalize before blending. |
| `PROJECT_..._KNOWLEDGE_BASE.md` | Accurate 1,477-line index of the whole course archive. Consult this before going back to OneDrive. |

Everything else in the course archive is discarded by the plan. Notably
`final_demo_data.csv` (1M rows) is **disqualified as a data substrate**: no timestamp
column and a first-1M-lines biased sample. Raw data is downloaded fresh in Phase 0.

The course project was five-person team work; see UPGRADE_PLAN.md §11 for the required
provenance language.
