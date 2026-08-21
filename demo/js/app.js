// app.js -- boot and section init.
//
// One rule here: no exhibit may take the page down. Each section is initialised
// independently inside a try/catch, so a missing document or a bad render
// degrades to a visible panel and the rest of the page stays live.

import { initData, doc, DOCUMENTS } from './data.js';
import { initReceipts, initReceiptsSection } from './receipts.js';
import * as crossover from './crossover.js';
import * as regime from './regime.js';
import * as contrast from './contrast.js';
import * as shoppers from './shoppers.js';
import * as search from './search.js';
import * as dq from './dq.js';
import * as lineage from './lineage.js';
import * as fmt from './fmt.js';

const SECTIONS = [
  { id: 'exhibit-crossover', init: crossover.init },
  { id: 'exhibit-regime', init: regime.init },
  { id: 'exhibit-contrast', init: contrast.init },
  { id: 'exhibit-shoppers', init: shoppers.init },
  { id: 'exhibit-search', init: search.init },
  { id: 'exhibit-dq', init: dq.init },
  { id: 'exhibit-lineage', init: lineage.init },
  { id: 'exhibit-receipts', init: async (el) => initReceiptsSection(el) },
];

function renderDataStatus(el, status) {
  if (!el) return;
  const items = DOCUMENTS.map((d) => {
    const ok = status.missing.indexOf(d.name) === -1;
    return `<li class="${ok ? 'ok' : 'absent'}"><code>${fmt.esc(d.name)}</code> ${
      ok ? 'loaded' : `absent${d.required ? ' (required!)' : ''}`
    }</li>`;
  }).join('');
  const manifest = doc('trace_manifest.json');
  el.innerHTML = `
    <details class="data-status">
      <summary>Data files loaded (${status.loaded.length}/${DOCUMENTS.length})</summary>
      <ul class="status-list">${items}</ul>
      ${
        manifest
          ? `<p class="muted">Trace manifest: ${fmt.int(
              (manifest.entries || []).length,
            )} traced leaves across ${fmt.int(
              (manifest.files || []).length,
            )} documents; log hash <code>${fmt.esc(fmt.shortHash(manifest.runs_jsonl_sha256))}</code>.</p>`
          : ''
      }
    </details>`;
}

async function boot() {
  const status = await initData();

  const drawer = document.getElementById('receipts-drawer');
  if (drawer) initReceipts(drawer);

  renderDataStatus(document.getElementById('data-status'), status);

  for (const s of SECTIONS) {
    const el = document.getElementById(s.id);
    if (!el) continue;
    try {
      // eslint-disable-next-line no-await-in-loop
      await s.init(el);
    } catch (err) {
      el.innerHTML = `<div class="placeholder"><p class="placeholder-title">This exhibit failed to render</p>
        <p><code>${fmt.esc(err && err.message ? err.message : String(err))}</code></p>
        <p class="muted">The rest of the page is unaffected.</p></div>`;
      // Keep the console honest for the offline/console-clean check in T36.
      console.error(`[${s.id}]`, err);
    }
  }

  document.documentElement.dataset.booted = 'true';
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
