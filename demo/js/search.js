// search.js -- exhibit 3, live in-browser semantic search (T35).
//
// WHAT THIS IS. A query is embedded *in your browser* by a vendored quantized
// MiniLM (transformers.js, env.allowRemoteModels = false) and scored against a
// precomputed int8 payload of the 50k most-popular catalog items. No request
// leaves the machine; nothing here talks to a server that is not the loopback
// one serving this page.
//
// EVIDENCE CLASS -- read this before reading the numbers. The similarity scores
// this exhibit renders are a CAPABILITY DEMONSTRATION, not evaluation evidence.
// They are cosine similarities between a query and item *text* embeddings: no
// held-out interactions, no full-catalog ranking protocol, no bootstrap CI, no
// results-log record to click. They are therefore deliberately NOT drawn with
// the dotted traced-number affordance the rest of the site uses -- a dotted
// number on this site means "copied from results/runs.jsonl", and none of these
// are. What *is* anchored is the provenance of the embeddings, shown in the
// exhibit footer with a run chip to the kind="ann_receipt" record.
//
// DEGRADATION (cut order #2). demo/data/search/ and demo/vendor/ are ~32MB and
// are not committed. This module probes for what is actually on disk and picks
// the richest mode it can support:
//
//   live      example_queries + payload bins + vendored model
//             -> free-text queries, lazily after an explicit click
//   fallback   example_queries.json only
//             -> the 12 canned queries with their precomputed results, labeled
//   absent     nothing
//             -> the standard "not yet exported" panel
//
// example_queries.json carries its hits' titles/brands/prices inline precisely
// so that FALLBACK MODE costs 58kB and needs neither .bin nor items_meta.json.

import { renderPlaceholder, loadJSON } from './data.js';
import * as fmt from './fmt.js';
import { runIdChip } from './receipts.js';

const DATA = new URL('../data/search/', import.meta.url);
const VENDOR = new URL('../vendor/', import.meta.url);

const TOP_N = 20; // results shown for a live query
const PARITY_QUERY_INDEX = 0; // canned query #1 is the parity probe

// Module state. Everything is loaded lazily and at most once.
const state = {
  mode: 'absent',
  queries: null, // example_queries.json
  meta: null, // embeddings_meta.json
  vendor: null, // vendor_manifest.json
  sizes: {}, // measured Content-Length per lazy asset
  codes: null, // Int8Array(rows * dim)
  scales: null, // Float32Array(rows)
  items: null, // items_meta.json
  extractor: null, // transformers.js pipeline
  parity: null, // { overlap, k, ms }
  activating: false,
};

// ---------------------------------------------------------------- asset probing

/** HEAD a lazy asset: returns its byte size, or null if it is not there. */
async function probeSize(url) {
  try {
    const res = await fetch(url, { method: 'HEAD', cache: 'no-cache' });
    if (!res.ok) return null;
    const len = res.headers.get('content-length');
    return len === null ? 0 : Number(len);
  } catch {
    return null;
  }
}

export async function probe() {
  const [queries, meta] = await Promise.all([
    loadJSON('search/example_queries.json'),
    loadJSON('search/embeddings_meta.json'),
  ]);
  state.queries = queries;
  state.meta = meta;

  const [codes, scales, items, vendorManifest] = await Promise.all([
    probeSize(new URL('embeddings_int8.bin', DATA)),
    probeSize(new URL('scales_f32.bin', DATA)),
    probeSize(new URL('items_meta.json', DATA)),
    fetch(new URL('vendor_manifest.json', VENDOR), { cache: 'no-cache' })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null),
  ]);
  state.sizes = { codes, scales, items };
  state.vendor = vendorManifest;

  const payloadReady = Boolean(meta) && codes !== null && scales !== null && items !== null;
  const modelReady = Boolean(vendorManifest && vendorManifest.transformers_js && vendorManifest.model);
  if (!queries) state.mode = 'absent';
  else if (payloadReady && modelReady) state.mode = 'live';
  else state.mode = 'fallback';
  return state.mode;
}

/** Bytes the "activate" click will actually pull, measured, not guessed. */
function activationBytes() {
  const payload = (state.sizes.codes || 0) + (state.sizes.scales || 0) + (state.sizes.items || 0);
  const model = state.vendor ? state.vendor.total_bytes || 0 : 0;
  return { payload, model, total: payload + model };
}

function mb(bytes) {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

// ------------------------------------------------------------------ the scanner

/**
 * Score every payload row against a unit-norm query vector.
 *
 * score_i = scale_i * sum_j q_j * code_ij  -- dequantizing by the row scale is
 * algebraically identical to dequantizing each component, and keeps the inner
 * loop to one multiply-add over Int8Array/Float32Array.
 *
 * Deterministic: j ascends, so the float accumulation order is fixed; the
 * caller's tie-break makes the ranking total.
 */
export function scanInt8(codes, scales, rows, dim, query, mask) {
  const out = new Float32Array(rows);
  for (let i = 0; i < rows; i += 1) {
    if (mask && mask[i] === 0) {
      out[i] = -Infinity;
      continue;
    }
    const base = i * dim;
    let acc = 0;
    for (let j = 0; j < dim; j += 1) acc += query[j] * codes[base + j];
    out[i] = acc * scales[i];
  }
  return out;
}

function scoreAll(query, mask) {
  return scanInt8(state.codes, state.scales, state.meta.rows, state.meta.dim, query, mask);
}

/**
 * Top-k row indices: score descending, ties broken by ASCENDING row index.
 *
 * The explicit secondary key is the point. A bare comparator on scores leaves
 * tied rows in an engine-defined order, which would make the same query return
 * different rankings in different browsers; this matches
 * export_search.top_k_rows (np.lexsort on (row, -score)) exactly.
 *
 * The comparator subtracts nothing: facet-masked rows carry -Infinity, and
 * `-Infinity - -Infinity` is NaN, which would make a difference-based
 * comparator inconsistent and the sort's output engine-defined.
 */
export function topK(scores, k) {
  const n = scores.length;
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i += 1) idx[i] = i;
  const sorted = idx.sort((a, b) => {
    const sa = scores[a];
    const sb = scores[b];
    if (sa > sb) return -1;
    if (sa < sb) return 1;
    return a - b;
  });
  const out = [];
  for (let r = 0; r < Math.min(k, n); r += 1) {
    const i = sorted[r];
    if (!Number.isFinite(scores[i])) break; // masked out by a facet
    out.push(i);
  }
  return out;
}

/** Facet mask over items_meta, or null when no facet is active. */
function facetMask(brand, priceMax) {
  if (!brand && !priceMax) return null;
  const items = state.items;
  const rows = state.meta.rows;
  const mask = new Uint8Array(rows);
  for (let i = 0; i < rows; i += 1) {
    if (brand && items.brand[i] !== brand) continue;
    if (priceMax) {
      const p = items.price_usd[i];
      if (p === null || p === undefined || p > priceMax) continue;
    }
    mask[i] = 1;
  }
  return mask;
}

// -------------------------------------------------------------------- rendering

function evidenceCaption() {
  return `<p class="caption">Similarity scores below are a <strong>capability demonstration, not
    evaluation evidence</strong>: they are cosine similarities over item text, with no held-out
    interactions, no full-catalog ranking protocol and no confidence interval. They are the only
    numbers on this site with no results-log record behind them, which is why none of them is
    drawn as a <span class="traced sample" tabindex="-1">clickable dotted number</span>. The
    embeddings' provenance is anchored; the scores are not a metric.</p>`;
}

function resultRow(rank, hit) {
  const price =
    typeof hit.price_usd === 'number' ? `$${hit.price_usd.toFixed(2)}` : '<span class="muted">no price</span>';
  const brand = hit.brand && hit.brand !== 'unknown' ? fmt.esc(hit.brand) : '<span class="muted">unknown brand</span>';
  return `<tr>
    <td class="rec-rank">${rank}</td>
    <td>${fmt.sig(hit.score, 4)}</td>
    <td><div class="hist-title">${fmt.esc(hit.title || hit.item_id)}</div>
        <div class="hist-meta">${brand} · ${price} · ${fmt.esc(hit.main_category || '--')}
        · <code>${fmt.esc(hit.item_id)}</code></div></td>
  </tr>`;
}

function resultsTable(hits, note) {
  if (!hits.length) {
    return `<p class="muted">No items match those facets. Widen the price ceiling or clear the brand.</p>`;
  }
  return `<div class="scroll-x"><table class="num-table">
      <thead><tr><th>#</th><th>cosine</th><th>item</th></tr></thead>
      <tbody>${hits.map((h, i) => resultRow(i + 1, h)).join('')}</tbody>
    </table></div>
    ${note ? `<p class="muted">${note}</p>` : ''}`;
}

function provenanceFooter() {
  const src = state.meta ? state.meta.source : state.queries ? state.queries.provenance : null;
  if (!src) return '';
  const q = state.queries || {};
  const rows = [
    ['model', `<code>${fmt.esc(src.model_id)}</code> @ <code>${fmt.esc(fmt.shortSha(src.model_revision, 12))}</code>`],
    ['recipe', `<code>${fmt.esc(src.recipe_id)}</code> · <code>${fmt.esc(fmt.shortHash(src.recipe_hash))}</code>`],
    [
      'item text',
      `${fmt.esc((src.recipe_fields || []).join(' + '))} from <code>gold.item_features</code> at snapshot
       <code>${fmt.esc(fmt.snapshotId(src.five_core_snapshot_id))}</code>`,
    ],
    [
      'embeddings',
      `${fmt.int(src.catalog_rows)} × ${fmt.int(state.meta ? state.meta.dim : 384)} ${fmt.esc(
        src.embedding_dtype || 'float16',
      )} · <code>${fmt.esc(fmt.shortHash(src.source_embeddings_sha256))}</code>`,
    ],
    [
      'payload',
      state.meta
        ? `top ${fmt.int(state.meta.rows)} items by <code>${fmt.esc(
            state.meta.ordering.popularity_column,
          )}</code>, ${fmt.esc(state.meta.quantization.scheme)} — min cosine vs fp16
           ${fmt.sig(state.meta.quantization.measured_min_cosine_vs_fp16, 6)}`
        : '<span class="muted">not installed (fallback mode)</span>',
    ],
    ['record', `${runIdChip(src.ann_receipt_run_id, `${src.ann_receipt_run_id} · ann_receipt`)}`],
  ];
  const browser = state.vendor
    ? `<p class="muted">In-browser runtime: <code>${fmt.esc(state.vendor.transformers_js.package)}</code>
       ${fmt.esc(state.vendor.transformers_js.version)}, model <code>${fmt.esc(
         state.vendor.model.repo,
       )}</code> @ <code>${fmt.esc(fmt.shortSha(state.vendor.model.revision, 12))}</code>
       (<code>${fmt.esc(state.vendor.model.dtype)}</code>), verified against the SHA-256s recorded in
       <code>demo/README.md</code> before install. It is a transformers.js port of the same upstream
       checkpoint the Python recipe used, so the browser adds one more quantization step on top of the
       payload's — which is what the parity receipt measures.</p>`
    : '';
  return `<div class="panel">
    <div class="rc-sect">Where these embeddings come from</div>
    <div class="kv-grid">${rows
      .map(
        ([k, v]) =>
          `<div class="kv"><div class="kv-k">${fmt.esc(k)}</div><div class="kv-v">${v}</div></div>`,
      )
      .join('')}</div>
    ${browser}
    <p class="muted">${fmt.esc(
      (state.meta && state.meta.evidence_note) || (q.evidence_note ? q.evidence_note : ''),
    )}</p>
  </div>`;
}

function parityPanel() {
  const q = state.queries;
  const exportSide = q && q.int8_parity ? q.int8_parity : null;
  const browserSide = state.parity;
  return `<div class="panel">
    <div class="rc-sect">Quantization-parity receipt</div>
    <p>Lossy twice over: the payload is int8, and the browser's ONNX model is q8 where the reference
       was fp32. The honest way to show that is to measure it rather than to claim it is small.</p>
    <div class="kv-grid">
      <div class="kv"><div class="kv-k">export side</div>
      <div class="kv-v">${
        exportSide
          ? `int8 payload vs exact fp16 reference — mean overlap@${q.top_k}
             ${fmt.sig(exportSide.mean_overlap_at_k, 3)} / ${q.top_k} over ${fmt.int(q.n_queries)} queries,
             worst ${fmt.int(exportSide.min_overlap_at_k)}`
          : '<span class="muted">--</span>'
      }</div></div>
      <div class="kv"><div class="kv-k">browser side</div>
      <div class="kv-v">${
        browserSide
          ? `canned query #${PARITY_QUERY_INDEX + 1} re-embedded here, then scanned against the int8
             payload: <strong>overlap@${browserSide.k} = ${browserSide.overlap} / ${browserSide.k}</strong>
             against the recorded Python reference top-${browserSide.k}
             <span class="muted">(embed ${browserSide.embedMs} ms, scan ${browserSide.scanMs} ms)</span>`
          : '<span class="muted">measured when you activate live search</span>'
      }</div></div>
    </div>
    <p class="caption">Overlap, not equality, is the expectation: two different quantizations of the
       same model will disagree at the tail of a top-10. A low overlap here would mean the vendored
       model is not the model the payload was built from.</p>
  </div>`;
}

// ---------------------------------------------------------------- fallback mode

function cannedList(selected) {
  const q = state.queries;
  return q.queries
    .map(
      (e, i) =>
        `<button type="button" class="shopper-chip${i === selected ? ' on' : ''}" data-canned="${i}">${fmt.esc(
          e.query,
        )}</button>`,
    )
    .join('');
}

function renderCanned(root, selected) {
  const q = state.queries;
  const entry = q.queries[selected];
  const el = root.querySelector('[data-canned-results]');
  if (!el) return;
  el.innerHTML = resultsTable(
    entry.results,
    `Precomputed at export time by the real Python model
     (<code>${fmt.esc(q.reference.model_id)}</code>, ${fmt.esc(q.reference.pooling)}), scored against the
     exact fp32 view of the pinned embeddings — not against the int8 payload, and not in this browser.`,
  );
  root.querySelectorAll('[data-canned]').forEach((b) => {
    b.classList.toggle('on', Number(b.getAttribute('data-canned')) === selected);
  });
}

// -------------------------------------------------------------------- live mode

async function fetchBinary(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${url.pathname || url}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

async function loadPayload(say) {
  say('loading int8 payload…');
  const [codesBuf, scalesBuf, items] = await Promise.all([
    fetchBinary(new URL('embeddings_int8.bin', DATA)),
    fetchBinary(new URL('scales_f32.bin', DATA)),
    loadJSON('search/items_meta.json'),
  ]);
  const { rows, dim } = state.meta;
  if (codesBuf.byteLength !== rows * dim) {
    throw new Error(`embeddings_int8.bin is ${codesBuf.byteLength} bytes, expected ${rows * dim}`);
  }
  if (scalesBuf.byteLength !== rows * 4) {
    throw new Error(`scales_f32.bin is ${scalesBuf.byteLength} bytes, expected ${rows * 4}`);
  }
  if (!items || items.rows !== rows) throw new Error('items_meta.json row count disagrees with the payload');
  state.codes = new Int8Array(codesBuf);
  state.scales = new Float32Array(scalesBuf); // little-endian: every supported target is LE
  state.items = items;
}

async function loadModel(say) {
  const vm = state.vendor;
  say('loading the in-browser runtime…');
  const modUrl = new URL(vm.transformers_js.module, VENDOR);
  const tf = await import(modUrl.href);
  const env = tf.env;
  // Offline by construction: no remote model, no remote runtime, no cache probe
  // that could reach out. If a file is missing locally this throws instead of
  // silently falling back to the Hub.
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  // No trailing slash: transformers.js joins this with the model name itself.
  env.localModelPath = new URL(vm.model.local_model_path, VENDOR).href;
  env.backends.onnx.wasm.wasmPaths = new URL(vm.transformers_js.wasm_paths, VENDOR).href;
  // Single-threaded and un-proxied: threads need SharedArrayBuffer, which needs
  // COOP/COEP headers that `python -m http.server` does not send.
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;
  say('loading MiniLM weights…');
  state.extractor = await tf.pipeline('feature-extraction', vm.model.model_name, { dtype: vm.model.dtype });
}

/** Mean-pooled, L2-normalized query embedding -- the recipe's pooling, in the browser. */
async function embed(text) {
  const out = await state.extractor(text, { pooling: 'mean', normalize: true });
  return Float32Array.from(out.data);
}

async function runParity() {
  const entry = state.queries.queries[PARITY_QUERY_INDEX];
  const t0 = performance.now();
  const qv = await embed(entry.query);
  const t1 = performance.now();
  const scores = scoreAll(qv, null);
  const hits = topK(scores, entry.top_k);
  const t2 = performance.now();
  const mine = new Set(hits.map((i) => state.items.item_id[i]));
  const ref = entry.results.map((r) => r.item_id);
  state.parity = {
    k: entry.top_k,
    overlap: ref.filter((id) => mine.has(id)).length,
    embedMs: Math.round(t1 - t0),
    scanMs: Math.round(t2 - t1),
  };
}

function brandOptions() {
  const counts = new Map();
  for (const b of state.items.brand) {
    if (!b || b === 'unknown') continue;
    counts.set(b, (counts.get(b) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .slice(0, 40)
    .map(([b, n]) => `<option value="${fmt.esc(b)}">${fmt.esc(b)} (${fmt.int(n)})</option>`)
    .join('');
}

function renderLiveUI(root) {
  const live = root.querySelector('[data-live]');
  live.innerHTML = `
    <div class="controls">
      <label class="control-label" for="search-q">Query</label>
      <input id="search-q" type="search" value="${fmt.esc(state.queries.queries[0].query)}"
             style="min-width:22rem" autocomplete="off">
      <label class="control-label" for="search-brand">Brand</label>
      <select id="search-brand"><option value="">any</option>${brandOptions()}</select>
      <label class="control-label" for="search-price">Max price</label>
      <select id="search-price">
        <option value="">any</option><option value="25">$25</option><option value="50">$50</option>
        <option value="100">$100</option><option value="250">$250</option>
      </select>
      <button type="button" class="seg-btn on" data-go>Search</button>
    </div>
    <p class="control-hint">Embedded in this tab by the vendored MiniLM, then scanned over
       ${fmt.int(state.meta.rows)} × ${fmt.int(state.meta.dim)} int8 components. Nothing is sent anywhere.</p>
    <div data-live-results></div>`;

  const run = async () => {
    const text = live.querySelector('#search-q').value.trim();
    const outEl = live.querySelector('[data-live-results]');
    if (!text) {
      outEl.innerHTML = '<p class="muted">Type a query.</p>';
      return;
    }
    outEl.innerHTML = '<p class="muted">embedding…</p>';
    const brand = live.querySelector('#search-brand').value;
    const priceMax = Number(live.querySelector('#search-price').value) || 0;
    const t0 = performance.now();
    const qv = await embed(text);
    const t1 = performance.now();
    const scores = scoreAll(qv, facetMask(brand, priceMax));
    const rows = topK(scores, TOP_N);
    const t2 = performance.now();
    const hits = rows.map((i) => ({
      item_id: state.items.item_id[i],
      title: state.items.title[i],
      brand: state.items.brand[i],
      price_usd: state.items.price_usd[i],
      main_category: state.items.main_category[i],
      score: scores[i],
    }));
    outEl.innerHTML = resultsTable(
      hits,
      `Top ${hits.length} of ${fmt.int(state.meta.rows)} items · embed ${Math.round(
        t1 - t0,
      )} ms, full scan ${Math.round(t2 - t1)} ms${brand || priceMax ? ' · facet-filtered before ranking' : ''}.`,
    );
  };
  live.querySelector('[data-go]').addEventListener('click', run);
  live.querySelector('#search-q').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') run();
  });
  run();
}

async function activate(root) {
  if (state.activating) return;
  state.activating = true;
  const box = root.querySelector('[data-activate]');
  const say = (msg) => {
    box.innerHTML = `<p class="muted">${fmt.esc(msg)}</p>`;
  };
  try {
    const t0 = performance.now();
    await loadPayload(say);
    await loadModel(say);
    say('measuring quantization parity…');
    await runParity();
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    box.innerHTML = `<p class="muted">Live search active — ${mb(
      activationBytes().total,
    )} loaded from this origin in ${secs}s. Nothing left the machine.</p>`;
    root.querySelector('[data-parity]').innerHTML = parityPanel();
    renderLiveUI(root);
  } catch (err) {
    state.activating = false;
    box.innerHTML = `<div class="warn"><p><strong>Live search could not start.</strong>
      <code>${fmt.esc(err && err.message ? err.message : String(err))}</code></p>
      <p class="muted">The canned queries below are unaffected — they are precomputed and need
      neither the model nor the payload.</p></div>`;
    console.error('[search] activation failed', err);
  }
}

// ------------------------------------------------------------------------ init

export async function init(root) {
  const mode = await probe();

  if (mode === 'absent') {
    renderPlaceholder(root, {
      file: 'search/example_queries.json',
      title: 'semantic search',
      task: 'src/batch_recsys_lab/demo/export_search.py (make demo-assets, T35)',
      note: `Search assets are ~32MB and are deliberately not committed (cut-order item #2):
             <code>make demo-assets</code> regenerates the int8 payload from the pinned embeddings
             artifact and downloads the MiniLM ONNX against the SHA-256s recorded in
             <code>demo/README.md</code>. With only the 58kB <code>example_queries.json</code> present
             this exhibit still runs in FALLBACK MODE; with nothing present you get this panel.`,
    });
    return;
  }

  const bytes = activationBytes();
  const activateBlock =
    mode === 'live'
      ? `<div class="panel" data-activate>
           <p><strong>Live search is off until you ask for it.</strong> Activating downloads
              <strong>${mb(bytes.total)}</strong> from this same loopback origin —
              ${mb(bytes.payload)} of int8 payload and item metadata, ${mb(bytes.model)} of
              MiniLM weights and WASM runtime — and then runs the model in this tab. On a laptop the
              first query lands a few seconds after the weights do.</p>
           <p><button type="button" class="seg-btn" data-activate-btn>Activate live search
              (${mb(bytes.total)})</button></p>
           <p class="muted">Everything below already works without it.</p>
         </div>`
      : `<div class="panel" data-activate>
           <p><span class="badge">fallback mode</span> Live in-browser search is <strong>not
              available</strong>: ${
                state.meta ? '' : '<code>demo/data/search/</code> payload absent. '
              }${state.vendor ? '' : '<code>demo/vendor/</code> model absent. '}Run
              <code>make demo-assets</code> to install both. What you see below is the documented
              degradation path: the ${fmt.int(state.queries.n_queries)} canned queries, computed at
              export time by the real Python model, with their exact reference results.</p>
         </div>`;

  root.innerHTML = `
    ${evidenceCaption()}
    ${activateBlock}
    <div data-live></div>
    <div data-parity>${mode === 'live' ? parityPanel() : ''}</div>
    <div class="panel">
      <div class="rc-sect">${
        mode === 'live' ? 'Canned queries (precomputed reference)' : 'Canned queries'
      }</div>
      <p class="muted">${fmt.int(state.queries.n_queries)} queries embedded at export time by
         <code>${fmt.esc(state.queries.reference.model_id)}</code> on
         <code>${fmt.esc(state.queries.reference.device)}</code>, scored against the exact fp32 view of
         the pinned embeddings. Query #${PARITY_QUERY_INDEX + 1} is the one the parity receipt replays.</p>
      <div class="shopper-chips">${cannedList(0)}</div>
      <div data-canned-results></div>
    </div>
    ${provenanceFooter()}`;

  renderCanned(root, 0);
  root.querySelectorAll('[data-canned]').forEach((b) => {
    b.addEventListener('click', () => renderCanned(root, Number(b.getAttribute('data-canned'))));
  });
  const btn = root.querySelector('[data-activate-btn]');
  if (btn) btn.addEventListener('click', () => activate(root));
  root.dataset.state = mode;
}
