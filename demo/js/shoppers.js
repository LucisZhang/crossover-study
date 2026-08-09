// shoppers.js -- exhibit 2, "pick a shopper" (T33).
//
// Data contract: demo/data/shoppers.json (SHIPPED schema, docs/demo-data-schemas.md).
// Rankings and per-user metrics come from the per-user parquets the eval records
// name (traced via getTraced/tracedSpan); titles/prices/timelines are declared
// descriptive. Cold users (segment "0") carry cold_collapse=true on the als and
// item_knn arms and must render as "empty by design", not as a blank list.
//
// MANDATORY: the six shoppers per segment are a stratified draw (2 blend hits,
// 4 misses), not a random sample -- blend's real per-segment hitrate@10 is
// ~1.5-4%. The curation panel renders that disclosure prominently, with traced
// numbers, above the picker.

import { doc, renderPlaceholder } from './data.js';
import { tracedSpan, runIdChip } from './receipts.js';
import * as fmt from './fmt.js';

const FILE = 'shoppers.json';

// Column order the exhibit shows: popularity / item-kNN / ALS / content / blend.
const MODEL_ORDER = ['pop_t12m', 'item_knn', 'als', 'content', 'blend'];

// Per-arm mechanism notes: the same non-signal class, three different causes.
const COLD_NOTES = {
  als:
    'No TRAIN history: ALS scores every catalog item 0, so its stored top-list is an ' +
    'index tie-break, not a recommendation (models/als.py) -- no personalized signal, empty by design.',
  item_knn:
    'No TRAIN history: item-kNN has no seed items to walk neighbors from, so every score is 0 and ' +
    'its stored top-list is an index tie-break, not a recommendation (models/item_knn.py) -- ' +
    'no personalized signal, empty by design.',
  content:
    'No TRAIN history: the MiniLM user profile is all-zero with an empty history, so every score ' +
    'is exactly 0 and the stored top-list is an index tie-break, not a recommendation ' +
    '(models/content.py "Cold-start collapse") -- no personalized signal, empty by design.',
};

let state = null; // { root, data, segment, shopperId }

function ptr(shopperId, ...rest) {
  return `/shoppers/${shopperId}${rest.length ? '/' + rest.join('/') : ''}`;
}

function fmtDate(iso) {
  if (typeof iso !== 'string') return '--';
  const m = iso.match(/^(\d{4}-\d{2}-\d{2})/);
  return m ? m[1] : iso;
}

function historyItemHtml(it) {
  return `<li class="history-item">
    <span class="hist-date">${fmt.esc(fmtDate(it.ts))}</span>
    <span class="hist-title">${fmt.esc(it.title || it.item_id)}</span>
    <span class="hist-meta">${fmt.esc(it.brand || 'unknown')}${
      typeof it.price_usd === 'number' ? ` · $${fmt.sig(it.price_usd, 3)}` : ''
    }${typeof it.rating === 'number' ? ` · ${fmt.sig(it.rating, 2)}★` : ''}</span>
  </li>`;
}

function testPurchaseHtml(it) {
  return `<li class="history-item">
    <span class="hist-date">${fmt.esc(fmtDate(it.ts))}</span>
    <span class="hist-title">${fmt.esc(it.title || it.item_id)}</span>
    <span class="hist-meta">${fmt.esc(it.brand || 'unknown')}</span>
  </li>`;
}

function recItemHtml(shopperId, model, it) {
  const hit = it.hit === true;
  return `<li class="rec-item${hit ? ' hit' : ''}">
    <span class="rec-rank">${fmt.int(it.rank)}</span>
    <span class="rec-title">${fmt.esc(it.title || it.item_id)}
      ${hit ? '<span class="badge yes rec-hit-badge" title="in this user’s TEST purchases">hit</span>' : ''}
    </span>
    <span class="rec-meta">${fmt.esc(it.brand || 'unknown')}${
      typeof it.price_usd === 'number' ? ` · $${fmt.sig(it.price_usd, 3)}` : ''
    } · <span class="mono">${fmt.esc(it.catalog_index)}</span></span>
  </li>`;
}

function modelColHtml(data, shopperId, model) {
  const rec = data.shoppers[shopperId].recommendations[model];
  const label = (data.model_labels || {})[model] || model;
  const runId = rec ? rec.run_id : null;
  const ndcgSpan = tracedSpan(FILE, ptr(shopperId, 'recommendations', model, 'ndcg@10'), fmt.metric);
  const body =
    rec && rec.cold_collapse
      ? `<div class="cold-panel">
          <p class="cold-panel-title">no personalized signal &mdash; empty by design</p>
          <p class="cold-panel-note muted">${fmt.esc(COLD_NOTES[model] || COLD_NOTES.als)}</p>
        </div>`
      : `<ol class="rec-list">${(rec && rec.top10 ? rec.top10 : []).map((it) => recItemHtml(shopperId, model, it)).join('')}</ol>`;
  return `<div class="model-col">
    <div class="model-col-head">
      <span class="model-col-label">${fmt.esc(label)}</span>
      ${runIdChip(runId, 'run')}
    </div>
    <div class="model-col-metric">NDCG@10: ${ndcgSpan}</div>
    ${body}
  </div>`;
}

function shopperCardHtml(data, shopperId) {
  const s = data.shoppers[shopperId];
  if (!s) return '<p class="muted">Shopper not found.</p>';
  const historyHtml =
    s.history && s.history.length
      ? `<ol class="history-list">${s.history.map(historyItemHtml).join('')}</ol>`
      : '<p class="muted">No TRAIN history &mdash; this is a cold-start user (segment 0).</p>';
  const testHtml =
    s.test_purchases && s.test_purchases.length
      ? `<ol class="history-list test-list">${s.test_purchases.map(testPurchaseHtml).join('')}</ol>`
      : '<p class="muted">No TEST purchases recorded.</p>';
  const cols = MODEL_ORDER.map((m) => modelColHtml(data, shopperId, m)).join('');
  return `
    <div class="shopper-card panel">
      <div class="shopper-card-head">
        <span class="mono">${fmt.esc(shopperId)}</span>
        <span class="badge">segment ${fmt.esc(s.segment)}</span>
        <span class="muted">n_train = ${fmt.int(s.n_train)}</span>
      </div>
      <h3>TRAIN history</h3>
      ${historyHtml}
      <h3>Held-out TEST purchases</h3>
      ${testHtml}
      <h3>Top-10 by model</h3>
      <div class="scroll-x"><div class="model-cols">${cols}</div></div>
    </div>`;
}

function curationTableHtml(data) {
  const rows = (data.segments || [])
    .map((seg) => {
      const c = (data.curation || {})[seg];
      if (!c) return '';
      const base = `/curation/${seg}`;
      return `<tr>
        <th>${fmt.esc(seg)}</th>
        <td>${tracedSpan(FILE, `${base}/eval_users`, fmt.int)}</td>
        <td>${c.drawn ? fmt.int(c.drawn.from_hit_stratum) : '--'}</td>
        <td>${c.drawn ? fmt.int(c.drawn.from_miss_stratum) : '--'}</td>
        <td>${tracedSpan(FILE, `${base}/blend_hitrate@10/value`, (v) => fmt.pct(v, 2))}</td>
      </tr>`;
    })
    .join('');
  return `
    <table class="num-table curation-table">
      <thead><tr>
        <th>segment</th><th>eval users</th><th>hit-stratum draws</th><th>miss-stratum draws</th>
        <th>blend hitrate@10 (real)</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function curationPanelHtml(data) {
  const rule = data.curation_rule || {};
  return `
    <div class="warn curation-panel">
      <p><strong>These 30 shoppers are not a random sample.</strong> Each segment shows
        ${fmt.int(rule.drawn_from_hit_stratum)} users where blend's top-10 hits a TEST purchase and
        ${fmt.int(rule.drawn_from_miss_stratum)} where it misses (${fmt.int(rule.per_segment)} per
        segment) &mdash; a stratified draw, chosen so both outcomes are visible. blend's real
        per-segment hitrate@10 is roughly 1.5&ndash;4% (traced below): the exhibit over-samples
        hits by about 50&times;.</p>
      ${curationTableHtml(data)}
      <p class="muted">Curation rule <code>${fmt.esc(rule.rule_id || '--')}</code>, pre-declared in
        <code>${fmt.esc(rule.declared_in || 'EXPERIMENT_LOG.md')}</code>. Seed
        <code>${fmt.esc(String(data.seed))}</code>.</p>
    </div>`;
}

function chipHtml(data, shopperId) {
  const s = data.shoppers[shopperId];
  const on = shopperId === state.shopperId;
  return `<button type="button" class="shopper-chip${on ? ' on' : ''}" data-shopper="${fmt.esc(
    shopperId,
  )}" aria-pressed="${on}">
    <span class="mono">${fmt.esc(shopperId)}</span>
    <span class="chip-ntrain">n_train ${fmt.int(s.n_train)}</span>
  </button>`;
}

function segTabsHtml(data) {
  return `<div class="seg-switch shopper-seg-switch" role="tablist">
    ${(data.segments || [])
      .map(
        (seg) =>
          `<button type="button" class="seg-btn${seg === state.segment ? ' on' : ''}" data-segment="${fmt.esc(
            seg,
          )}" role="tab" aria-selected="${seg === state.segment}">${fmt.esc(seg)}</button>`,
      )
      .join('')}
  </div>`;
}

function chipsHtml(data) {
  const ids = (data.shopper_order || []).filter((id) => data.shoppers[id].segment === state.segment);
  return `<div class="shopper-chips">${ids.map((id) => chipHtml(data, id)).join('')}</div>`;
}

function render() {
  const { root, data } = state;
  root.innerHTML = `
    ${curationPanelHtml(data)}
    <div class="controls">
      <span class="control-label">history-depth segment</span>
      ${segTabsHtml(data)}
    </div>
    ${chipsHtml(data)}
    ${shopperCardHtml(data, state.shopperId)}
  `;
}

function onClick(ev) {
  const segBtn = ev.target.closest('[data-segment]');
  if (segBtn) {
    const seg = segBtn.getAttribute('data-segment');
    state.segment = seg;
    const firstInSeg = (state.data.shopper_order || []).find(
      (id) => state.data.shoppers[id].segment === seg,
    );
    if (firstInSeg) state.shopperId = firstInSeg;
    render();
    return;
  }
  const chip = ev.target.closest('[data-shopper]');
  if (chip) {
    state.shopperId = chip.getAttribute('data-shopper');
    render();
  }
}

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
  const firstSegment = (d.segments || [])[0];
  const firstShopper = (d.shopper_order || []).find((id) => d.shoppers[id].segment === firstSegment);
  state = { root, data: d, segment: firstSegment, shopperId: firstShopper };
  root.addEventListener('click', onClick);
  render();
}
