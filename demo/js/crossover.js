// crossover.js -- exhibit 1, the front door.
//
// NDCG@10 (switchable to Recall@20) by user-history segment, one line per model
// with 95% bootstrap CI bands, plus the n* slider that shows how the routed
// policy would have scored at every grid position.
//
// Sources: demo/data/crossover.json (FROZEN schema, always present) and
// demo/data/policy_grid.json (AGREED schema, written by T27 -- absent is fine).

import { doc, getTraced, traceEntries } from './data.js';
import { tracedSpan, tracedMetricWithCi, derivedSpan, runIdChip } from './receipts.js';
import { segmentLineChart, SLOTS } from './charts.js';
import * as fmt from './fmt.js';

const CX = 'crossover.json';
const PG = 'policy_grid.json';

const state = {
  metric: 'ndcg@10',
  variant: 'B',
  gridIdx: null, // index into policy_grid.n_star_grid
};

let root = null;

/** Lowest n_train in a segment label: "0"->0, "1-4"->1, "20+"->20. Frozen edges. */
function segmentMin(label) {
  const m = String(label).match(/^(\d+)/);
  return m ? Number(m[1]) : 0;
}

function nStarValue(raw) {
  return raw === null || raw === undefined ? Infinity : Number(raw);
}

// ------------------------------------------------------------------------ chart

function buildChart(cx) {
  const order = cx.model_order;
  const series = [];
  order.forEach((key, i) => {
    const m = cx.models[key];
    if (!m || m.plot === false) return;
    const values = cx.segments.map((s) => m.segments[s][state.metric].value);
    const ci_lo = cx.segments.map((s) => m.segments[s][state.metric].ci_lo);
    const ci_hi = cx.segments.map((s) => m.segments[s][state.metric].ci_hi);
    series.push({
      key,
      label: m.label,
      values,
      ci_lo,
      ci_hi,
      color: SLOTS[i],
      highlight: !!m.highlight,
      runId: m.run_id,
    });
  });

  // Routed overlay: takes the hybrid palette slot, because at the shipped
  // position (B, n*=inf) the routed line IS the hybrid run.
  const routed = routedSeries(cx);
  if (routed) series.push(routed);

  const ref = cx.models[order.find((k) => cx.models[k] && cx.models[k].plot !== false)];
  const nUsers = cx.segments.map((s) => ref.segments[s].n_users);

  return segmentLineChart({
    segments: cx.segments,
    nUsers,
    nUsersFmt: nUsers.map(fmt.int),
    nUsersRunId: ref.run_id,
    series,
    title: cx.title,
    subtitle:
      state.metric === 'ndcg@10'
        ? cx.subtitle
        : cx.subtitle.replace('NDCG@10', fmt.metricLabel(state.metric)),
    xLabel: cx.xlabel,
    yLabel: fmt.metricLabel(state.metric),
    // The per-line run_ids go in an HTML strip under the chart rather than in
    // the figure's small print: the browser wraps real text, and the chips stay
    // clickable. In-SVG text width can only be estimated here.
    footnote: 'rendered from demo/data/crossover.json, a projection of results/runs.jsonl (append-only)',
  });
}

/** The reference figure's small-print receipts row, as flowing HTML. */
function receiptsStrip(cx) {
  const chips = cx.model_order
    .map((k) => {
      const m = cx.models[k];
      if (!m) return '';
      return `<li>${fmt.esc(m.label)}${m.plot === false ? ' <span class="muted">(not plotted)</span>' : ''}:
        ${runIdChip(m.run_id)}</li>`;
    })
    .join('');
  const pg = doc(PG);
  const grid = pg
    ? `<li>n* grid recomposition: ${runIdChip(pg.record_run_id)}</li>`
    : '';
  return `<ul class="receipts-strip">${chips}${grid}</ul>`;
}

function currentCell(pg) {
  if (!pg) return null;
  const label = pg.n_star_labels[state.gridIdx];
  const cells = pg.cells && pg.cells[state.variant];
  return cells && cells[label] ? { label, cell: cells[label] } : { label, cell: null };
}

function routedSeries(cx) {
  const pg = doc(PG);
  if (!pg) return null;
  const cur = currentCell(pg);
  if (!cur || !cur.cell) return null;
  const values = cx.segments.map((s) => {
    const seg = cur.cell.segments && cur.cell.segments[s];
    return seg && seg[state.metric] ? seg[state.metric].value : null;
  });
  if (values.some((v) => v === null)) return null;
  return {
    key: 'routed',
    label: `routed: ${pg.variants[state.variant].label}, n*=${fmt.nStarLabel(cur.label)}`,
    values,
    band: false,
    dash: '7 5',
    color: SLOTS[5],
    highlight: false,
    z: 2, // drawn above the recorded lines: at n*=inf it lies exactly on blend
    runId: pg.record_run_id,
  };
}

// ------------------------------------------------------------------ policy panel

/**
 * Share of TEST users routed to the low (blend) arm.
 *
 * policy/select.py routes `n_train < n*` to the low arm, so a whole segment
 * routes low iff its minimum n_train is below n*. The exporter writes this as a
 * traced leaf (`/cells/<v>/<label>/low_share`); we prefer that. The in-page sum
 * over the traced per-segment n_users is kept only as a fallback for a
 * policy_grid.json that predates the field, and is marked derived when used.
 */
function lowShare(pg, cell, label) {
  if (!cell) return null;
  const exported = getTraced(PG, `/cells/${state.variant}/${label}/low_share`);
  if (typeof exported.value === 'number') {
    return { share: exported.value, pointer: `/cells/${state.variant}/${label}/low_share`, derived: false };
  }
  if (!cell.segments) return null;
  const nStar = nStarValue(cell.n_star);
  let low = 0;
  let total = 0;
  for (const s of pg.segments || Object.keys(cell.segments)) {
    const n = cell.segments[s] && cell.segments[s].n_users;
    if (typeof n !== 'number') return null;
    total += n;
    if (segmentMin(s) < nStar) low += n;
  }
  return total > 0 ? { share: low / total, low, total, derived: true } : null;
}

function policyPanel() {
  const pg = doc(PG);
  if (!pg) {
    return `<div class="panel placeholder">
      <p class="placeholder-title">n* slider: grid not exported yet</p>
      <p><code>demo/data/policy_grid.json</code> is written by T27
         (<code>policy/grid_test.py</code> → <code>kind="policy_grid"</code> record → exporter).
         Until then the chart shows the five recorded model lines with no routed overlay.</p>
      <p class="muted">The shipped policy is already decided and is not waiting on this file:
         variant B (blend → pop-t12m) at n*=∞, i.e. blend everywhere. No finite n* beat
         blend-everywhere on VAL.</p>
    </div>`;
  }

  const labels = pg.n_star_labels;
  const cur = currentCell(pg);
  const shipped = pg.shipped || {};
  const isShipped = shipped.variant === state.variant && shipped.n_star_label === cur.label;
  const cell = cur.cell;
  const ls = lowShare(pg, cell, cur.label);
  const lowKey = pg.variants[state.variant].low;
  const highKey = pg.variants[state.variant].high;
  const lowLabel = labelFor(lowKey);
  const highLabel = labelFor(highKey);

  const ticks = labels
    .map(
      (l, i) =>
        `<span class="tick${i === state.gridIdx ? ' on' : ''}${
          shipped.variant === state.variant && shipped.n_star_label === l ? ' shipped' : ''
        }">${fmt.esc(fmt.nStarLabel(l))}</span>`,
    )
    .join('');

  const variantBtns = Object.entries(pg.variants)
    .map(
      ([k, v]) =>
        `<button type="button" class="seg-btn${k === state.variant ? ' on' : ''}" data-variant="${fmt.esc(
          k,
        )}">${fmt.esc(k)} · ${fmt.esc(v.label)}</button>`,
    )
    .join('');

  const base = `/cells/${state.variant}/${cur.label}/global/${state.metric}`;
  const globalHtml = cell
    ? tracedMetricWithCi(PG, base)
    : '<span class="untraced">cell absent</span>';

  const identity =
    cell && cell.identity && cell.identity.equals_run_id
      ? `<p class="muted">Identity assertion: this cell is bit-identical to run
         ${runIdChip(cell.identity.equals_run_id)} — checked in-run at export, not re-scored here.</p>`
      : '';

  const manifestGap = traceEntries(PG).length === 0
    ? `<p class="warn">policy_grid.json is present but has no <code>trace_manifest.json</code> entries yet —
       its numbers render as untraced until the manifest is re-exported. <code>make demo-verify</code> is
       the authority on this.</p>`
    : '';

  return `<div class="panel">
    <h3>Where would you put n*?</h3>
    <p>The routing rule is <code>n_train &lt; n*</code> → <strong>${fmt.esc(lowLabel)}</strong>,
       otherwise <strong>${fmt.esc(highLabel)}</strong>. The slider snaps to the recorded grid only
       — ${fmt.esc(labels.map(fmt.nStarLabel).join(', '))} — because those are the positions the
       segment edges make expressible without re-scoring.</p>
    <div class="seg-switch">${variantBtns}</div>
    <div class="slider-wrap">
      <input type="range" id="nstar" min="0" max="${labels.length - 1}" step="1" value="${state.gridIdx}"
             aria-label="n* grid position">
      <div class="ticks">${ticks}</div>
    </div>
    <div class="kv-grid">
      <div class="kv"><div class="kv-k">position</div><div class="kv-v">variant ${fmt.esc(
        state.variant,
      )}, n* = ${fmt.esc(fmt.nStarLabel(cur.label))}${
        isShipped ? ' <span class="badge shipped">shipped policy</span>' : ''
      }</div></div>
      <div class="kv"><div class="kv-k">global ${fmt.esc(fmt.metricLabel(state.metric))} [95% CI]</div>
        <div class="kv-v">${globalHtml}</div></div>
      <div class="kv"><div class="kv-k">routed to ${fmt.esc(lowLabel)}</div>
        <div class="kv-v">${
          ls === null
            ? '<span class="untraced">--</span>'
            : ls.derived
              ? `${derivedSpan(fmt.pct(ls.share), pg.record_run_id, 'sum of traced segment n_users')} of users
                 <span class="muted">(${fmt.int(ls.low)} of ${fmt.int(ls.total)})</span>`
              : `${tracedSpan(PG, ls.pointer, fmt.pct)} of users`
        }</div></div>
    </div>
    ${identity}
    ${manifestGap}
    ${coldCollapseCaption(pg)}
    <p class="caption"><strong>Derived.</strong> Every cell in this grid is recomposed from the
       recorded one-shot TEST runs — per-user outputs selected by segment, then re-aggregated. No
       re-scoring, no refitting, no additional consultation of the TEST ground truth. The
       recomposition itself is an appended record: ${runIdChip(pg.record_run_id)}.</p>
    ${
      pg.n_star_selected_on_val === null
        ? `<p class="caption">On VAL, no finite n* beat blend-everywhere, which is why the shipped
           policy is uniform blend rather than a history-depth router. The slider exists to show
           that result, not to invite tuning on TEST.</p>`
        : ''
    }
  </div>`;
}

/**
 * The one caption this exhibit must not omit: variant B's n*=0 and n*=1 cells
 * carry identical metrics, and a reader who does not know why will read it as a
 * bug. The exporter ships its own wording in notes.variant_b_cold_collapse
 * (copied from the policy_grid record); use it when present.
 */
function coldCollapseCaption(pg) {
  const note =
    pg.notes && pg.notes.variant_b_cold_collapse
      ? pg.notes.variant_b_cold_collapse
      : 'Cold users (n_train = 0) have no content history, so the blend arm collapses to pure ' +
        'pop-t12m for them — and they are the only users the n*=1 position routes differently ' +
        'from n*=0.';
  const cells = pg.cells && pg.cells.B;
  const shares =
    cells && cells['0'] && cells['1'] && typeof cells['0'].low_share === 'number'
      ? `<br><span class="muted">Only the routed share moves: ${tracedSpan(
          PG,
          '/cells/B/0/low_share',
          fmt.pct,
        )} at n*=0 versus ${tracedSpan(
          PG,
          '/cells/B/1/low_share',
          fmt.pct,
        )} at n*=1 — the same users, the same ranking, a different label on the arm that produced it.</span>`
      : '';
  return `<p class="caption"><strong>Variant B at n*=0 and n*=1 is not a bug.</strong>
    ${fmt.esc(note)}${shares}</p>`;
}

function labelFor(modelKey) {
  const cx = doc(CX);
  return cx && cx.models[modelKey] ? cx.models[modelKey].label : modelKey;
}

// ------------------------------------------------------------------ number tables

function modelTable(cx) {
  const head = ['model', 'users', `global ${fmt.metricLabel(state.metric)}`, ...cx.segments]
    .map((h) => `<th>${fmt.esc(h)}</th>`)
    .join('');
  const rows = cx.model_order
    .map((key) => {
      const m = cx.models[key];
      if (!m) return '';
      const note =
        m.plot === false
          ? `<div class="row-note">not plotted — numerically identical to ${fmt.esc(
              labelFor(m.identical_to || 'blend'),
            )}</div>`
          : '';
      const segs = cx.segments
        .map((s) => {
          const base = `/models/${key}/segments/${s}/${state.metric}`;
          const lo = getTraced(CX, `${base}/ci_lo`);
          const hi = getTraced(CX, `${base}/ci_hi`);
          return `<td>${tracedSpan(CX, `${base}/value`, fmt.metric)}
            <div class="cell-ci">${fmt.esc(fmt.ci(lo.value, hi.value))}</div></td>`;
        })
        .join('');
      return `<tr class="${m.highlight ? 'hl' : ''}">
        <th scope="row"><span class="swatch" style="background:${
          SLOTS[cx.model_order.indexOf(key)]
        }"></span>${fmt.esc(m.label)}${note}</th>
        <td>${tracedSpan(CX, `/models/${key}/n_users`, fmt.int)}</td>
        <td>${tracedMetricWithCi(CX, `/models/${key}/global/${state.metric}`)}</td>
        ${segs}
      </tr>`;
    })
    .join('');
  return `<div class="scroll-x"><table class="num-table">
    <caption>Full-catalog ranking against all ${tracedSpan(
      CX,
      '/models/blend/catalog_size',
      fmt.int,
    )} items, TRAIN-seen items excluded. Brackets are 95% user-bootstrap CIs.</caption>
    <thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function deltaTable(cx) {
  const rows = (cx.paired_delta_order || [])
    .map((key) => {
      const d = cx.paired_deltas[key];
      if (!d) return '';
      const base = `/paired_deltas/${key}/global/${state.metric}`;
      const lo = getTraced(CX, `${base}/ci_lo`);
      const hi = getTraced(CX, `${base}/ci_hi`);
      const ez = getTraced(CX, `${base}/excludes_zero`);
      return `<tr>
        <th scope="row">${fmt.esc(d.label)}</th>
        <td>${tracedSpan(CX, `${base}/delta`, fmt.delta)}</td>
        <td class="cell-ci">${fmt.esc(fmt.deltaCi(lo.value, hi.value))}</td>
        <td>${
          ez.value === true
            ? '<span class="badge yes">CI excludes 0</span>'
            : '<span class="badge no">CI includes 0</span>'
        }</td>
        <td>${tracedSpan(CX, `/paired_deltas/${key}/n_common_users`, fmt.int)}</td>
        <td>${runIdChip(d.run_id)}</td>
      </tr>`;
    })
    .join('');
  return `<div class="scroll-x"><table class="num-table">
    <caption>Paired deltas, computed per user on the common user set and bootstrapped as a paired
      statistic — the comparison, not two independent intervals eyeballed against each other.</caption>
    <thead><tr><th>comparison</th><th>Δ ${fmt.esc(
      fmt.metricLabel(state.metric),
    )}</th><th>95% CI</th><th>verdict</th><th>users</th><th>record</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

// ------------------------------------------------------------------------- render

function render() {
  const cx = doc(CX);
  const metricBtns = cx.metrics
    .map(
      (m) =>
        `<button type="button" class="seg-btn${m === state.metric ? ' on' : ''}" data-metric="${fmt.esc(
          m,
        )}">${fmt.esc(fmt.metricLabel(m))}</button>`,
    )
    .join('');

  const hybrid = cx.models.hybrid;
  const hybridDelta = cx.paired_deltas && cx.paired_deltas.hybrid_vs_blend;

  root.innerHTML = `
    <div class="controls">
      <span class="control-label">metric</span>
      <div class="seg-switch">${metricBtns}</div>
      <span class="control-hint">split: <code>${fmt.esc(cx.split)}</code> · frozen TEST, one-shot</span>
    </div>
    <div class="chart-wrap scroll-x"><div class="chart-box" id="cx-chart"></div></div>
    ${receiptsStrip(cx)}
    <p class="caption"><strong>Derived overlay.</strong> The dashed line is the routed policy at the
      slider position — recomposed from the recorded one-shot TEST runs; no re-scoring. It is drawn
      without a CI band to keep the five recorded bands readable; its interval is in the panel below.</p>
    ${
      hybrid
        ? `<p class="caption"><strong>${fmt.esc(hybrid.label)}</strong> is not drawn as a sixth line
           because it lands exactly on the blend line: the confirming hybrid run
           ${runIdChip(hybrid.run_id)} at n*=∞ routes every user to blend. The receipt is the paired
           delta against blend, which is exactly zero in every cell${
             hybridDelta
               ? ` — global Δ ${tracedSpan(
                   CX,
                   `/paired_deltas/hybrid_vs_blend/global/${state.metric}/delta`,
                   fmt.delta,
                 )}`
               : ''
           }.</p>`
        : ''
    }
    <div id="cx-policy"></div>
    <h3>Every plotted number</h3>
    ${modelTable(cx)}
    <h3>Paired comparisons</h3>
    ${deltaTable(cx)}
    <p class="caption">Full-catalog ranking: each TEST user is scored against the entire
      ${tracedSpan(CX, '/models/blend/catalog_size', fmt.int)}-item catalog with their TRAIN-seen
      items removed. No sampled negatives anywhere on this page.</p>`;

  root.querySelector('#cx-chart').innerHTML = buildChart(cx);
  root.querySelector('#cx-policy').innerHTML = policyPanel();
  wire();
}

function wire() {
  root.querySelectorAll('[data-metric]').forEach((b) =>
    b.addEventListener('click', () => {
      state.metric = b.getAttribute('data-metric');
      render();
    }),
  );
  root.querySelectorAll('[data-variant]').forEach((b) =>
    b.addEventListener('click', () => {
      state.variant = b.getAttribute('data-variant');
      render();
    }),
  );
  const slider = root.querySelector('#nstar');
  if (slider) {
    slider.addEventListener('input', () => {
      const idx = Number(slider.value);
      if (idx === state.gridIdx) return;
      state.gridIdx = idx; // integer index => the slider can only land on grid points
      render();
      const again = root.querySelector('#nstar');
      if (again) again.focus();
    });
  }
}

export async function init(el) {
  root = el;
  const cx = doc(CX);
  if (!cx) {
    el.innerHTML =
      '<p class="warn">crossover.json failed to load. This document is required; run <code>make demo-export</code>.</p>';
    return;
  }
  if (!cx.metrics.includes(state.metric)) state.metric = cx.metrics[0];

  const pg = doc(PG);
  if (pg) {
    const shipped = pg.shipped || { variant: 'B', n_star_label: 'inf' };
    state.variant = pg.variants[shipped.variant] ? shipped.variant : Object.keys(pg.variants)[0];
    const idx = pg.n_star_labels.indexOf(shipped.n_star_label);
    state.gridIdx = idx >= 0 ? idx : pg.n_star_labels.length - 1;
  }
  render();
}
