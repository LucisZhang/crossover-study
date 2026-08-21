// regime.js -- exhibit 1b: the measured regime map (Phase 8, T8-1 + T8-2).
//
// 2D stratification of the recorded one-shot TEST runs: user history depth x
// target-item learnability at the TRAIN cutoff (support and recency buckets),
// with per-cell GT-mass share and the paired delta vs pop-t12m. Three maps, one
// per challenger arm: item-kNN-t12m and ALS-decay hl365 (T8-2, recomposed) and
// static ALS (the T8-1 baseline). All values come from kind="regime_map"
// records via demo/data/phase8.json; the winner-with-CI status of a cell is a
// classification of traced booleans/signs, not a new number.
//
// Cell status colours reuse the DQ matrix's pass/fail treatment (site tokens),
// so a green cell means the same thing everywhere on this site: the claim holds
// with the check behind it.

import { doc, getTraced, renderPlaceholder } from './data.js';
import { tracedSpan, runIdChip } from './receipts.js';
import * as fmt from './fmt.js';

const P8 = 'phase8.json';

const state = {
  map: null, // key into regime_map.maps
  metric: 'ndcg@10',
};

let root = null;

const str = (v) => String(v);

/** Cells are stored as flat lists; index them as bucket-row x segment-col. */
function cellIndex(cells) {
  const idx = new Map();
  cells.forEach((c, i) => idx.set(`${c.bucket} ${c.segment}`, i));
  return idx;
}

/**
 * Winner-with-CI classification, from traced leaves only:
 *   'win2'  arm beats pop with the 95% CI excluding zero on BOTH metrics
 *           (the preregistered crossover criterion: NDCG@10 + Recall@20 guard)
 *   'win1'  CI excludes zero, positive, on the displayed metric only
 *   'loss'  CI excludes zero, negative, on the displayed metric
 *   'zero'  both arms score exactly 0 on both metrics (structural cell)
 *   'ns'    everything else — no CI separates the arms here
 */
function cellStatus(cell, armKey, metric) {
  const d = cell.delta;
  const both =
    d['ndcg@10'].excludes_zero === true &&
    d['ndcg@10'].delta > 0 &&
    d['recall@20'].excludes_zero === true &&
    d['recall@20'].delta > 0;
  if (both) return 'win2';
  const m = d[metric];
  if (m.excludes_zero === true && m.delta > 0) return 'win1';
  if (m.excludes_zero === true && m.delta < 0) return 'loss';
  const zero = ['ndcg@10', 'recall@20'].every(
    (mm) => cell.arms.pop_t12m[mm].value === 0 && cell.arms[armKey][mm].value === 0,
  );
  if (zero) return 'zero';
  return 'ns';
}

const STATUS_LABEL = {
  win2: 'arm wins, CI excludes 0 on both metrics',
  win1: 'arm wins on this metric only',
  loss: 'pop-t12m wins, CI excludes 0',
  zero: 'structural 0 — both arms score exactly 0',
  ns: 'no CI separates the arms',
};

function axisTable(rm, map, axis) {
  const mapBase = `/regime_map/maps/${map.key}`;
  const cells = map.cells[axis];
  const idx = cellIndex(cells);
  const buckets = rm.axes[axis].labels;
  const segs = doc(P8).segments;
  const head = [`item TRAIN ${axis}`, ...segs].map((h) => `<th>${fmt.esc(h)}</th>`).join('');
  const rows = buckets
    .map((b) => {
      const tds = segs
        .map((s) => {
          const i = idx.get(`${b} ${s}`);
          if (i === undefined) return '<td class="rm-cell rm-absent">--</td>';
          const cell = cells[i];
          const base = `${mapBase}/cells/${axis}/${i}`;
          const status = cellStatus(cell, map.arm_key, state.metric);
          const dbase = `${base}/delta/${state.metric}`;
          const lo = getTraced(P8, `${dbase}/ci_lo`);
          const hi = getTraced(P8, `${dbase}/ci_hi`);
          const title = `${map.arm_label} − pop-t12m · depth ${s} × ${axis} ${b} · ${fmt.int(
            cell.n_users,
          )} users · ${STATUS_LABEL[status]}`;
          return `<td class="rm-cell rm-${status}" title="${fmt.esc(title)}">
            <span class="rm-delta">${tracedSpan(P8, `${dbase}/delta`, fmt.delta)}</span>
            ${status === 'win2' ? '<span class="rm-star" aria-label="crossover cell">★</span>' : ''}
            <span class="cell-ci">${fmt.esc(fmt.deltaCi(lo.value, hi.value))}</span>
            <span class="rm-gt">GT ${tracedSpan(P8, `${base}/gt_share`, fmt.pct)}</span>
          </td>`;
        })
        .join('');
      const spec = rm.axes[axis].spec && rm.axes[axis].spec[b];
      return `<tr><th scope="row" ${spec ? `title="${fmt.esc(spec)}"` : ''}>${fmt.esc(b)}</th>${tds}</tr>`;
    })
    .join('');
  return `<div class="scroll-x"><table class="num-table rm-table">
    <caption>Columns: user history depth (TRAIN interactions). Rows: the TEST ground-truth item's
      TRAIN ${fmt.esc(axis)} bucket (hover a row label for the frozen definition). Each cell:
      paired Δ ${fmt.esc(fmt.metricLabel(state.metric))} (${fmt.esc(map.arm_label)} − pop-t12m),
      95% CI, and that cell's share of TEST GT interactions.</caption>
    <thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

/** The cells meeting the preregistered crossover criterion, derived from traced flags. */
function crossoverList(rm, map) {
  const rows = [];
  for (const axis of rm.axes_order) {
    map.cells[axis].forEach((cell, i) => {
      if (cellStatus(cell, map.arm_key, state.metric) !== 'win2') return;
      const base = `/regime_map/maps/${map.key}/cells/${axis}/${i}`;
      rows.push(`<tr>
        <th scope="row">${fmt.esc(axis)}, depth ${fmt.esc(cell.segment)}, ${fmt.esc(cell.bucket)}</th>
        <td>${tracedSpan(P8, `${base}/delta/ndcg@10/delta`, fmt.delta)}</td>
        <td>${tracedSpan(P8, `${base}/delta/recall@20/delta`, fmt.delta)}</td>
        <td>${tracedSpan(P8, `${base}/gt_share`, fmt.pct)}</td>
        <td>${tracedSpan(P8, `${base}/n_users`, fmt.int)}</td>
      </tr>`);
    });
  }
  if (!rows.length) {
    return `<p class="caption"><strong>No cell meets the preregistered crossover criterion for
      ${fmt.esc(map.arm_label)}</strong> — no positive paired delta with the 95% CI excluding zero
      on both NDCG@10 and the Recall@20 guard.</p>`;
  }
  return `<h3>Cells meeting the preregistered crossover criterion</h3>
    <div class="scroll-x"><table class="num-table">
    <caption>Positive paired delta with the 95% CI excluding zero on BOTH NDCG@10 and the
      Recall@20 guard — the criterion preregistered in docs/engineering-log/EXPERIMENT_LOG.md (2026-08-17) before any
      T8-2 record existed. Record ${runIdChip(map.run_id)}.</caption>
    <thead><tr><th>cell (axis, depth, bucket)</th><th>Δ NDCG@10</th><th>Δ Recall@20</th>
      <th>GT share</th><th>users</th></tr></thead>
    <tbody>${rows.join('')}</tbody></table></div>`;
}

function churnPanel(rm) {
  const g = rm.gate;
  const hd = rm.headline;
  const supportRow = (b, label) => `<tr>
    <th scope="row">${fmt.esc(label)}</th>
    <td>${tracedSpan(P8, `/regime_map/headline/gt_interactions_by_support/${b}/n`, fmt.int)}</td>
    <td>${tracedSpan(P8, `/regime_map/headline/gt_interactions_by_support/${b}/share`, fmt.pct)}</td>
    <td>${tracedSpan(P8, `/regime_map/headline/catalog_items_by_support/${b}/n`, fmt.int)}</td>
    <td>${tracedSpan(P8, `/regime_map/headline/catalog_items_by_support/${b}/share`, fmt.pct)}</td>
  </tr>`;
  return `<div class="panel">
    <h3>The churn receipt (T8-1): measured, no longer inferred</h3>
    <p>${fmt.esc(g.statistic)}: <strong>${tracedSpan(
      P8,
      '/regime_map/gate/measured_share',
      fmt.pct,
    )}</strong> — against preregistered bands of &lt;${tracedSpan(
      P8,
      '/regime_map/gate/wrong_below',
      (v) => fmt.pct(v, 0),
    )} (diagnosis wrong) and ≥${tracedSpan(
      P8,
      '/regime_map/gate/supported_at_or_above',
      (v) => fmt.pct(v, 0),
    )} (supported). Verdict on the record: <em>${fmt.esc(g.verdict)}</em> ${runIdChip(g.run_id)}.</p>
    <div class="scroll-x"><table class="num-table">
      <caption>TEST = ${tracedSpan(P8, '/regime_map/headline/gt_interactions_total', fmt.int)} GT
        interactions from ${tracedSpan(P8, '/regime_map/headline/n_users', fmt.int)} users against a
        ${tracedSpan(P8, '/regime_map/headline/catalog_size', fmt.int)}-item catalog. Items with no
        TRAIN interaction at all absorb a third of TEST purchase mass from ~5% of the catalog — the
        structural ceiling every TRAIN-frozen model shares.</caption>
      <thead><tr><th>item TRAIN support</th><th>GT interactions</th><th>GT share</th>
        <th>catalog items</th><th>catalog share</th></tr></thead>
      <tbody>
        ${supportRow('zero', 'zero (0 TRAIN interactions)')}
        ${supportRow('low', 'low (1-4, below the 5-core)')}
        ${supportRow('high', 'high (≥5)')}
      </tbody>
    </table></div>
  </div>`;
}

/** *emphasis* in log-quoted text -> <em>, everything else escaped. */
function emQuote(text) {
  return fmt
    .esc(text)
    .replace(/\*([^*]+)\*/g, '<em>$1</em>');
}

function caveatBlock(rm, map) {
  const lineage =
    map.input_equivalence && map.input_equivalence.exception_used === true
      ? `<p class="caption"><strong>Lineage disclosure.</strong> This map was recomposed on the
         machine of record after the T8-2 substrate migration, under the single-use, digest-gated
         lineage exception <code>${tracedSpan(P8, `/regime_map/maps/${map.key}/input_equivalence/exception_id`, str)}</code>
         (validation ${tracedSpan(P8, `/regime_map/maps/${map.key}/input_equivalence/status`, str)}:
         the rebuilt item-stats parquet hashes byte-identically to the one the original T8-1 record
         anchored). Declared in the record itself — ${runIdChip(map.run_id)}.</p>`
      : '';
  return `
    <blockquote class="rm-caveat">
      <p>${emQuote(rm.caveat)}</p>
      <footer class="muted">${fmt.esc(rm.caveat_source)}</footer>
    </blockquote>
    <p class="caption"><strong>${emQuote(rm.multiplicity)}</strong></p>
    ${lineage}`;
}

function legend() {
  const item = (cls, label) =>
    `<li><span class="rm-chip rm-${cls}"></span>${fmt.esc(label)}</li>`;
  return `<ul class="rm-legend">
    ${item('win2', 'crossover: arm beats pop-t12m, CI excludes 0 on both metrics ★')}
    ${item('win1', 'positive with CI excluding 0 on the shown metric only')}
    ${item('loss', 'pop-t12m wins, CI excludes 0')}
    ${item('ns', 'CI includes 0')}
    ${item('zero', 'structural 0 — both arms score exactly 0')}
  </ul>`;
}

function render() {
  const p8 = doc(P8);
  const rm = p8.regime_map;
  const map = rm.maps[state.map];

  const mapBtns = rm.map_order
    .map(
      (k) =>
        `<button type="button" class="seg-btn${k === state.map ? ' on' : ''}" data-map="${fmt.esc(
          k,
        )}">${fmt.esc(rm.maps[k].label)}</button>`,
    )
    .join('');
  const metricBtns = ['ndcg@10', 'recall@20']
    .map(
      (m) =>
        `<button type="button" class="seg-btn${m === state.metric ? ' on' : ''}" data-metric="${fmt.esc(
          m,
        )}">${fmt.esc(fmt.metricLabel(m))}</button>`,
    )
    .join('');

  root.innerHTML = `
    ${churnPanel(rm)}
    <div class="controls">
      <span class="control-label">arm − pop-t12m</span>
      <div class="seg-switch">${mapBtns}</div>
      <span class="control-label">metric</span>
      <div class="seg-switch">${metricBtns}</div>
    </div>
    <p class="caption"><strong>Derived, like every recomposition on this page.</strong> Cells
      regroup the per-user top-50 lists and ground-truth pairs already committed by the one-shot
      TEST runs (${runIdChip(map.source_run_ids.pop_t12m)} and
      ${runIdChip(map.source_run_ids[map.arm_key])}) — no re-scoring, no refitting, no new
      ground-truth consultation; bucket edges preregistered before any per-cell outcome existed.
      The recomposition itself is an appended record: ${runIdChip(map.run_id)}.</p>
    ${legend()}
    ${axisTable(rm, map, 'support')}
    ${axisTable(rm, map, 'recency')}
    ${crossoverList(rm, map)}
    ${caveatBlock(rm, map)}`;
  wire();
}

function wire() {
  root.querySelectorAll('[data-map]').forEach((b) =>
    b.addEventListener('click', () => {
      state.map = b.getAttribute('data-map');
      render();
    }),
  );
  root.querySelectorAll('[data-metric]').forEach((b) =>
    b.addEventListener('click', () => {
      state.metric = b.getAttribute('data-metric');
      render();
    }),
  );
}

export async function init(el) {
  root = el;
  const p8 = doc(P8);
  if (!p8 || !p8.regime_map) {
    renderPlaceholder(el, {
      file: P8,
      title: 'Regime map',
      task: 'the Phase 8 exporter (plan §8b line 277)',
      note: 'Written from the kind="regime_map" records; until exported this exhibit stays a panel.',
    });
    return;
  }
  state.map = p8.regime_map.map_order[0];
  render();
}
