// search.js -- exhibit 3, live semantic search. STUB (T31 shell; T35 builds it).
//
// Data contract: demo/data/search/ (AGREED, NOT COMMITTED) -- int8 item payload +
// scales + metadata + ~12 canned queries with reference top-10 from the real
// Python model, which doubles as the in-UI quantization-parity receipt. The model
// itself is a local vendored asset (env.allowRemoteModels = false); it is fetched
// once by a download script that verifies the SHA-256 recorded in demo/README.md.
//
// Cut order #2: deleting demo/data/search/ and demo/vendor/ must leave every other
// exhibit green. That is why this module probes for its assets and degrades here
// instead of at boot.

import { renderPlaceholder } from './data.js';

const META = 'search/embeddings_meta.json';

async function assetsPresent() {
  try {
    const res = await fetch(new URL(`../data/${META}`, import.meta.url), { method: 'GET', cache: 'no-cache' });
    return res.ok;
  } catch {
    return false;
  }
}

export async function init(root) {
  if (!(await assetsPresent())) {
    renderPlaceholder(root, {
      file: META,
      title: 'semantic search',
      task: 'src/batch_recsys_lab/demo/export_search.py + make demo-assets (T35)',
      note: 'Search assets are ~50MB and are deliberately not committed: <code>make demo-assets</code> regenerates the int8 payload from the pinned embeddings artifact and downloads the MiniLM ONNX against the SHA-256 recorded in demo/README.md. The documented degradation path is the precomputed-queries fallback — which is what you are looking at.',
    });
    return;
  }
  root.innerHTML = `<div class="placeholder">
    <p class="placeholder-title">Search assets present; exhibit UI lands in T35</p>
    <p><code>demo/data/search/</code> is in place. The query box, lazy model load with size
       warning, and the quantization-parity receipt are built in T35.</p>
  </div>`;
}
