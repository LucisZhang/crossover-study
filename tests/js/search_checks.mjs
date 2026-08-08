// Node harness for demo/js/search.js (Phase 6, T35). Run by tests/test_search_js.py.
//
// The browser half of the exhibit still has to be *checked*, not just eyeballed
// in T36's manual run. Three things are pinned here:
//
//   1. mode detection -- live / fallback / absent -- driven by a stubbed fetch,
//      because cut order #2 says deleting the assets must degrade rather than
//      break, and "degrades" is a behaviour, not a comment;
//   2. topK's tie-break, which must match export_search.top_k_rows exactly or
//      the same query ranks differently in different engines;
//   3. scanInt8 against the REAL payload when it is on disk, compared with a
//      reference top-20 computed in numpy -- cross-language agreement is the
//      only way to know the JS scan dequantizes the way the exporter meant.
//
// No DOM: only probe()/topK()/scanInt8() are exercised, none of which touch
// document. Rendering is T36's browser run.

import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../..');
const SEARCH_JS = path.join(REPO, 'demo/js/search.js');
const DATA = path.join(REPO, 'demo/data/search');

let failures = 0;
let ran = 0;

async function test(name, fn) {
  ran += 1;
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (err) {
    failures += 1;
    console.log(`FAIL ${name}\n     ${err && err.message ? err.message : err}`);
  }
}

/**
 * Fresh module instance with a stubbed global fetch.
 *
 * `present` is the set of paths that "exist"; everything else 404s, which is
 * exactly what a static server does for a deleted demo/data/search/.
 */
async function loadSearch(tag, present) {
  const body = new Map(Object.entries(present));
  globalThis.fetch = async (url, opts = {}) => {
    const href = String(url);
    const hit = [...body.keys()].find((k) => href.endsWith(k));
    if (hit === undefined) {
      return { ok: false, status: 404, headers: new Map(), json: async () => ({}) };
    }
    const value = body.get(hit);
    const bytes = typeof value === 'number' ? value : Buffer.byteLength(JSON.stringify(value));
    return {
      ok: true,
      status: 200,
      headers: { get: (h) => (h.toLowerCase() === 'content-length' ? String(bytes) : null) },
      json: async () => (typeof value === 'number' ? {} : value),
      arrayBuffer: async () => new ArrayBuffer(typeof value === 'number' ? value : 0),
    };
  };
  return import(`${pathToFileURL(SEARCH_JS).href}?case=${tag}`);
}

const META = { rows: 4, dim: 3, quantization: {}, ordering: {}, source: {} };
const QUERIES = {
  n_queries: 1,
  top_k: 10,
  reference: { model_id: 'm', device: 'mps', pooling: 'mean' },
  queries: [{ query: 'q', top_k: 10, results: [] }],
};
const VENDOR = { transformers_js: { module: 'transformers/x.js' }, model: { repo: 'r' }, total_bytes: 10 };

await test('mode=live when payload, item metadata and vendored model are all present', async () => {
  const m = await loadSearch('live', {
    'search/example_queries.json': QUERIES,
    'search/embeddings_meta.json': META,
    'search/embeddings_int8.bin': 12,
    'search/scales_f32.bin': 16,
    'search/items_meta.json': 99,
    'vendor/vendor_manifest.json': VENDOR,
  });
  assert.equal(await m.probe(), 'live');
});

await test('mode=fallback when the model is missing but the canned queries are there', async () => {
  const m = await loadSearch('nomodel', {
    'search/example_queries.json': QUERIES,
    'search/embeddings_meta.json': META,
    'search/embeddings_int8.bin': 12,
    'search/scales_f32.bin': 16,
    'search/items_meta.json': 99,
  });
  assert.equal(await m.probe(), 'fallback');
});

await test('mode=fallback from example_queries.json ALONE (the 58kB cut-order-#2 path)', async () => {
  const m = await loadSearch('cannedonly', { 'search/example_queries.json': QUERIES });
  assert.equal(await m.probe(), 'fallback');
});

await test('mode=fallback when the payload bins were deleted but the meta stayed', async () => {
  const m = await loadSearch('partial', {
    'search/example_queries.json': QUERIES,
    'search/embeddings_meta.json': META,
    'vendor/vendor_manifest.json': VENDOR,
  });
  assert.equal(await m.probe(), 'fallback');
});

await test('mode=absent when nothing is installed', async () => {
  const m = await loadSearch('absent', {});
  assert.equal(await m.probe(), 'absent');
});

await test('topK breaks ties by ascending row, like export_search.top_k_rows', async () => {
  const m = await loadSearch('topk', {});
  assert.deepEqual(m.topK(Float32Array.from([0.5, 0.9, 0.9, 0.1, 0.9]), 3), [1, 2, 4]);
  assert.deepEqual(m.topK(Float32Array.from([0.5, 0.9, 0.9, 0.1, 0.9]), 10), [1, 2, 4, 0, 3]);
  assert.deepEqual(m.topK(new Float32Array(20).fill(0.25), 5), [0, 1, 2, 3, 4]);
});

await test('topK drops facet-masked rows instead of ranking -Infinity', async () => {
  const m = await loadSearch('mask', {});
  const scores = Float32Array.from([0.1, -Infinity, 0.3, -Infinity]);
  assert.deepEqual(m.topK(scores, 10), [2, 0]);
});

await test('topK survives an all-masked scan (every comparison is -Inf vs -Inf)', async () => {
  // A difference-based comparator returns NaN here and the sort order becomes
  // engine-defined; the result must be an empty ranking, deterministically.
  const m = await loadSearch('allmask', {});
  assert.deepEqual(m.topK(Float32Array.from([-Infinity, -Infinity, -Infinity]), 5), []);
});

await test('scanInt8 dequantizes by the row scale', async () => {
  const m = await loadSearch('scan', {});
  const codes = Int8Array.from([127, 0, 0, 0, 127, 0]);
  const scales = Float32Array.from([2, 3]);
  const got = m.scanInt8(codes, scales, 2, 3, Float32Array.from([1, 0, 0]), null);
  assert.equal(got[0], 127 * 2);
  assert.equal(got[1], 0);
});

await test('scanInt8 honours the facet mask', async () => {
  const m = await loadSearch('scanmask', {});
  const codes = Int8Array.from([100, 100]);
  const scales = Float32Array.from([1, 1]);
  const got = m.scanInt8(codes, scales, 2, 1, Float32Array.from([1]), Uint8Array.from([0, 1]));
  assert.equal(got[0], -Infinity);
  assert.equal(got[1], 100);
});

// --- cross-language parity against the real payload (skipped if not exported) ---

const havePayload =
  existsSync(path.join(DATA, 'embeddings_int8.bin')) &&
  existsSync(path.join(DATA, 'scales_f32.bin')) &&
  existsSync(path.join(DATA, 'embeddings_meta.json')) &&
  existsSync(process.env.SEARCH_REFERENCE_JSON || '/nonexistent');

if (!havePayload) {
  console.log('skip demo/data/search payload parity (payload or numpy reference absent)');
} else {
  await test('scanInt8 + topK reproduce the numpy reference ranking on the real 50k payload', async () => {
    const m = await loadSearch('real', {});
    const meta = JSON.parse(readFileSync(path.join(DATA, 'embeddings_meta.json'), 'utf8'));
    const ref = JSON.parse(readFileSync(process.env.SEARCH_REFERENCE_JSON, 'utf8'));
    const codes = new Int8Array(readFileSync(path.join(DATA, 'embeddings_int8.bin')).buffer);
    const scalesBuf = readFileSync(path.join(DATA, 'scales_f32.bin'));
    const scales = new Float32Array(
      scalesBuf.buffer.slice(scalesBuf.byteOffset, scalesBuf.byteOffset + scalesBuf.byteLength),
    );
    assert.equal(codes.length, meta.rows * meta.dim, 'payload size');
    assert.equal(scales.length, meta.rows, 'scales length');

    const query = Float32Array.from(ref.query_vector);
    const scores = m.scanInt8(codes, scales, meta.rows, meta.dim, query, null);
    const got = m.topK(scores, ref.top_rows.length);
    assert.deepEqual(got, ref.top_rows, 'JS ranking must equal the numpy ranking');
    // Same arithmetic, different language: float32 accumulation order is fixed
    // on both sides, so the scores agree to well within display precision.
    for (let r = 0; r < got.length; r += 1) {
      const d = Math.abs(scores[got[r]] - ref.top_scores[r]);
      assert.ok(d < 1e-5, `score ${r} differs by ${d}`);
    }
  });
}

console.log(`\n${ran - failures}/${ran} checks passed`);
process.exit(failures === 0 ? 0 : 1);
