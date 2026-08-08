// shoppers.js -- exhibit 2, "pick a shopper". STUB (T31 shell; T33 builds the UI).
//
// Data contract: demo/data/shoppers.json (AGREED schema in docs/demo-data-schemas.md),
// written by T28's export_shoppers.py. Rankings and per-user metrics come from the
// per-user parquets the eval records name; titles/prices/timelines are declared
// descriptive. Cold users carry cold_collapse=true and must render as
// "empty by design", not as a zero.

import { doc, renderPlaceholder } from './data.js';
import * as fmt from './fmt.js';

const FILE = 'shoppers.json';

export async function init(root) {
  const d = doc(FILE);
  if (!d) {
    renderPlaceholder(root, {
      file: FILE,
      title: 'pick-a-shopper',
      task: 'src/batch_recsys_lab/demo/export_shoppers.py (T28)',
      note: '30 curated real users (IDs re-hashed) across the five history-depth segments, with their TRAIN timeline, side-by-side top-10 from each model, TEST purchases badged as hits, and per-user NDCG@10.',
    });
    return;
  }
  // Data has landed but the exhibit UI is T33; say so rather than guessing at it.
  root.innerHTML = `<div class="placeholder">
    <p class="placeholder-title">Data exported; exhibit UI lands in T33</p>
    <p><code>demo/data/${FILE}</code> is present:
       ${fmt.int((d.shopper_order || []).length)} shoppers across
       ${fmt.int((d.segments || []).length)} segments, seed <code>${fmt.esc(String(d.seed))}</code>.</p>
  </div>`;
}
