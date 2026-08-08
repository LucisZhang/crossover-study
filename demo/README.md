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

## Search assets

Not committed: ~50MB between the int8 embedding payload and the MiniLM ONNX
weights. Both are regenerated locally by one command:

```
make demo-assets      # export_search.py + the SHA-256-verified model download
```

**The hash, not the URL, is ground truth.** The download script verifies each
fetched file against the SHA-256 recorded below and fails loudly — non-zero exit,
no partial install, an explicit "hash mismatch — URL content has drifted"
message — on any mismatch. If a mismatch happens, the correct response is to
investigate the drift, not to update the hash.

| File | Source | SHA-256 |
|---|---|---|
| _(T35 fills this table)_ | | |

Deleting `demo/data/search/` and `demo/vendor/` must leave every other exhibit
green; the site falls back to the precomputed example queries. This is cut-order
item #2 in UPGRADE_PLAN §12 and it is kept genuinely decoupled.

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
