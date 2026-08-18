# `demo/` — the static exhibit site

Six exhibits (UPGRADE_PLAN §9) over the evidence in `results/runs.jsonl`. No build
step, no bundler, no framework, no CDN: `index.html` plus vanilla ES modules plus
hand-rolled SVG. Everything it loads is a relative path inside this directory.

```
demo/
  index.html          six sections in §9 order + the receipts drawer
  css/site.css        light default, system font stack
  js/                 ES modules (no build step)
    app.js            boot + section init; one exhibit failing never takes the page down
    data.js           404-tolerant fetch, JSON-pointer resolution, getTraced()
    fmt.js            ALL display rounding (data files carry full precision)
    charts.js         SVG chart primitives; port of eval/crossover_chart.py geometry
    receipts.js       the drawer + the traced-number affordance
    crossover.js      exhibit 1
    regime.js         exhibit 1b (Phase 8 regime map)
    shoppers.js  search.js  dq.js  lineage.js    exhibits 2–5 (stubs in T31)
  data/               JSON projections of the results log (written by src/batch_recsys_lab/demo/)
  vendor/             MiniLM + transformers.js, not committed (see "Search assets")
```

## Serving it

```
make demo-serve          # python -m http.server 8000 --bind 127.0.0.1 -d demo
```

then open `127.0.0.1:8000`. Any static server works:

```
python3 -m http.server 8000 --bind 127.0.0.1 -d demo
```

**`file://` caveat.** Opening `demo/index.html` directly from disk will not work.
ES modules and `fetch()` are subject to the same-origin policy, and a `file://`
document has an opaque origin, so the browser blocks both the module graph and
the `data/*.json` reads. This is a browser security rule, not a network
dependency — an HTTP origin is required, `127.0.0.1` is one.

## What "offline" means here

**Zero non-loopback network.** Loading and using this site issues no request that
leaves the machine: no CDN script, no webfont, no analytics beacon, no favicon
fetch, no remote model. The only origin involved is the loopback server above.

This is checked mechanically, not asserted:

- `make demo-offline-check` scans the assembled tree for external URLs.
- The Phase 6 acceptance run drives all six exhibits in a DNS-black-holed
  browser (`--host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE 127.0.0.1"`) and
  requires zero non-loopback requests and a clean console.

The semantic-search exhibit is the interesting case: the MiniLM model runs
in-browser with `env.allowRemoteModels = false` against a **local vendored copy**
of the weights. It is lazy-loaded behind a size warning, and if the assets are
absent the exhibit degrades to the precomputed example queries rather than
reaching for the network.

URLs do appear in this README as prose and citations. They appear nowhere in
`index.html`, `css/`, `js/`, or `data/` — not in a `src`, `href`, `url()`, or
`fetch()` position. `charts.js` builds SVG through `innerHTML` specifically so
that even the SVG namespace URI never has to be written down.

## The traceability rule

> Every number displayed on this site traces to a record in
> `results/runs.jsonl` — directly, or via a record-anchored artifact hash.

Concretely:

1. `demo/data/*.json` is a **projection**, never a computation. A value may
   appear there only if it is byte-identical to something reachable from (a) a
   `runs.jsonl` record, (b) a results artifact whose SHA-256 a record carries, or
   (c) a per-user parquet a record names in `per_user_artifact`. The transitive
   (b)/(c) forms are what "or via a record-anchored artifact hash" means: the
   record pins the artifact's hash, so the artifact's contents are as anchored as
   the record's own fields.
2. Every numeric leaf in those documents has an entry in
   `data/trace_manifest.json` giving its value and its source. The exporter
   writes leaves through a `TracedWriter`, so an untraced number is impossible by
   construction rather than by review.
3. The site reads those leaves through `data.js::getTraced(file, pointer)`, which
   returns the value **and** the `run_id`. `receipts.js` renders it with a
   `data-run-id` attribute and a dotted underline; clicking opens the record.
4. A number the page computes in the browser from traced components — currently
   only the "% of users routed to blend" share, summed over traced per-segment
   user counts — is drawn with a **dashed** underline instead of dotted, and its
   receipt names the record its components came from. The distinction is
   deliberate: dotted means "copied from the log", dashed means "arithmetic on
   things copied from the log".

   Relatedly, the case study labels every claim with the site's evidence-class
   convention — **measured** / **estimated** / **projected** — plus a fourth
   label this lab adds: **derived**, for values recomposed from recorded runs
   without any new measurement (the n\* TEST grid: per-user outputs of the
   recorded one-shot TEST runs, re-aggregated under a different routing
   threshold; no re-scoring, no refitting, no new ground-truth consultation).
   "Derived" is deliberately not "measured": a badge-rendering site must never
   present a recomposition as a fresh TEST measurement.
5. `make demo-verify` re-resolves every manifest entry independently of the
   writer (coverage, exact match — same type, no epsilon — document hashes,
   artifact-hash-vs-record agreement, and a staleness guard on the log's own
   hash). `make demo-verify-record` is the CI mode; it skips only the per-user
   parquet reads.

Full precision lives in the data files. All rounding lives in `js/fmt.js` — if
you find rounding anywhere else, that is a bug.

Documents that have not been exported yet (`policy_grid.json`, `shoppers.json`,
`dq.json`, `lineage.json`, `timetravel.json`, `search/`) simply 404, and the
owning exhibit renders an "exhibit data not yet exported" panel. That is a
supported state, not a failure mode.

### Phase 8 data (`phase8.json`)

`data/phase8.json` carries the §8b follow-on exhibits: the two T8-2
recency-matched TEST arms (item-kNN-t12m and ALS-decay hl365, incl. the 3-seed
spread and the four paired deltas), the T8-3 exploratory deep depth buckets,
and the three regime maps (T8-1 static-ALS baseline plus the two T8-2
recompositions) with the churn gate. It obeys rules 1–5 above unchanged: every
leaf was written through `TracedWriter` (`kind="runs_record"` sources only) and
`make demo-verify` re-resolves all of them. Caveat: its exporter was run as a
one-off script against the pinned records rather than from
`src/batch_recsys_lab/demo/` — `make demo-export` therefore does not yet
regenerate this file (it leaves it, and its manifest entries, intact).
Promoting that script to `export_phase8.py` beside the other exporters is the
open follow-up.

## Search assets

Not committed, and measured rather than estimated: **79.5MB** total —
31.9MB of int8 payload and item metadata in `demo/data/search/`, 47.6MB of
vendored browser runtime and MiniLM weights in `demo/vendor/`.

| | bytes | what |
|---|---|---|
| `data/search/embeddings_int8.bin` | 19,200,000 | 50,000 × 384 int8, C-order |
| `data/search/items_meta.json` | 12,434,794 | parallel descriptive arrays, 50,000 rows |
| `data/search/scales_f32.bin` | 200,000 | one LE float32 per row |
| `data/search/example_queries.json` | 58,579 | 12 canned queries — the whole of fallback mode |
| `data/search/embeddings_meta.json` | 2,890 | shape, quantization, provenance |
| `vendor/…/onnx/model_quantized.onnx` | 22,972,370 | q8 MiniLM |
| `vendor/transformers/ort-wasm-simd-threaded.jsep.wasm` | 21,596,019 | ONNX Runtime Web |
| `vendor/transformers/*` (js, map, mjs, LICENSE) | 2,323,876 | transformers.js 3.8.1 dist subset |
| `vendor/…/tokenizer*.json`, `config.json` | 712,677 | tokenizer + model config |

That is above the ~50MB the Phase 6 plan anticipated; the gap is the ONNX
Runtime WASM binary, which is the price of running the model in the tab at all.
Both trees are regenerated locally by one command:

```
make demo-assets      # export_search.py + the SHA-256-verified model download
```

**The hash, not the URL, is ground truth.** The download script verifies each
fetched file against the SHA-256 recorded below and fails loudly — non-zero exit,
no partial install, an explicit "hash mismatch — URL content has drifted"
message — on any mismatch. If a mismatch happens, the correct response is to
investigate the drift, not to update the hash.

The table below is machine-parsed by
`src/batch_recsys_lab/demo/fetch_search_assets.py`: the four columns and their
headers are load-bearing, and a file the script wants but the table does not
name (or vice versa) is itself a hard failure. It was generated by
`--record-hashes` and pasted verbatim; regenerate it the same way if a pin ever
moves deliberately.

| File | SHA-256 | Approx size | Source URL |
|---|---|---|---|
| `transformers-3.8.1.tgz` | `207714c36765b87accfd9b7b0672c3505805af97140990e0d9f8ac6e3cd5471e` | 10.5 MB | `https://registry.npmjs.org/@huggingface/transformers/-/transformers-3.8.1.tgz` |
| `models/Xenova/all-MiniLM-L6-v2/config.json` | `7135149f7cffa1a573466c6e4d8423ed73b62fd2332c575bf738a0d033f70df7` | 650 B | `https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/config.json` |
| `models/Xenova/all-MiniLM-L6-v2/tokenizer.json` | `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0` | 712 kB | `https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/tokenizer.json` |
| `models/Xenova/all-MiniLM-L6-v2/tokenizer_config.json` | `9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3` | 366 B | `https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/tokenizer_config.json` |
| `models/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx` | `afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1` | 23.0 MB | `https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/onnx/model_quantized.onnx` |

Both model pins are commit-pinned, never `main`. The npm tarball is hashed as
one artifact: that hash covers every extracted byte, which is stronger than
per-file hashes over a hand-listed subset, since a file nobody thought to list
cannot slip in. `huggingface.co` redirects large files to a CDN host; the script
prints the URL the bytes actually arrived from, and the hash gate makes the
delivery path irrelevant.

Deleting `demo/data/search/` and `demo/vendor/` must leave every other exhibit
green. This is cut-order item #2 in UPGRADE_PLAN §12 and it is kept genuinely
decoupled — the search exhibit itself degrades in two steps:

| present | exhibit 3 shows |
|---|---|
| `data/search/` + `vendor/` | live in-browser search, behind an explicit click with the real MB count, plus the parity receipt |
| `data/search/example_queries.json` only (58kB) | **fallback mode**: the 12 canned queries with their precomputed results, labeled as precomputed |
| nothing | the standard "exhibit data not yet exported" panel |

`example_queries.json` carries each hit's title, brand and price inline for
exactly that reason: fallback mode costs 58kB and needs neither `.bin` nor the
12.4MB `items_meta.json`.

**Evidence class — search scores are not evaluation evidence.** The cosine
similarities this exhibit renders are a capability demonstration: item-text
embeddings, no held-out interactions, no full-catalog ranking protocol, no
bootstrap CI. They are the only numbers on the site with no results-log record
behind them, and they are deliberately *not* drawn with the dotted traced-number
affordance. What is anchored is the embeddings' provenance: `export_search.py`
re-hashes `embeddings.npy` and refuses to run unless it matches both the MiniLM
manifest and `source_embeddings_sha256` on the `kind="ann_receipt"` record
`20260807T090857Z-97af81f`, which the exhibit footer links.

## Notes for whoever touches this next

- No external libraries and no build step is a hard constraint, not a
  preference. If an exhibit seems to need one, it needs less exhibit.
- `charts.js` ports the palette, geometry and CI-band treatment of
  `src/batch_recsys_lab/eval/crossover_chart.py`. The on-site chart is meant to
  be visually the same object as the committed
  `results/figures/crossover_test.svg`: same slot colours in the same
  `model_order`, y-axis always from 0, bands at alpha 0.13, direct labels at the
  line ends in ink. If you change one, change the other.
- Dataset: Amazon Reviews 2023, Electronics category (Hou et al. 2024,
  https://amazon-reviews-2023.github.io/) — research license; the raw data is
  never redistributed from here, and the demo ships only aggregates and
  re-hashed user identifiers.
