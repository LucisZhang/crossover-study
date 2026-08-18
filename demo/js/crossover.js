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
import { segmentLineChart, figureHeader, SLOTS } from './charts.js';
import * as fmt from './fmt.js';

const CX = 'crossover.json';
const PG = 'policy_grid.json';
const P8 = 'phase8.json';

// Palette slots for the Phase 8 arms: 6 and 7 are the only unused reference
// slots (0-4 are the recorded lines, 5 is the routed overlay).
const P8_SLOTS = { itemknn_t12m: SLOTS[6], alsdecay_hl365: SLOTS[7] };

const state = {
  metric: 'ndcg@10',
  variant: 'B',
  gridIdx: null, // index into policy_grid.n_star_grid
  depth: 'frozen', // 'frozen' (5 recorded segments) | 'deep' (T8-3 exploratory buckets)
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

/**
 * The chart spec, built once per render and consumed twice: figureHeader() draws
 * the title/subtitle/legend as HTML above the plot, segmentLineChart() draws the
 * plot itself. One object, so the legend can never disagree with the lines.
 */
function chartSpec(cx) {
  if (state.depth === 'deep') return deepChartSpec(cx);
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

  // Phase 8 recency-matched arms (T8-2): same frozen segments, own palette slots.
  const p8 = doc(P8);
  if (p8 && p8.arms) {
    for (const key of p8.arm_order || []) {
      const m = p8.arms[key];
      if (!m || m.plot === false) continue;
      series.push({
        key,
        label: m.label,
        values: cx.segments.map((s) => m.segments[s][state.metric].value),
        ci_lo: cx.segments.map((s) => m.segments[s][state.metric].ci_lo),
        ci_hi: cx.segments.map((s) => m.segments[s][state.metric].ci_hi),
        color: P8_SLOTS[key] || SLOTS[7],
        highlight: false,
        runId: m.run_id,
      });
    }
  }

  // Routed overlay: takes the hybrid palette slot, because at the shipped
  // position (B, n*=inf) the routed line IS the hybrid run.
  const routed = routedSeries(cx);
  if (routed) series.push(routed);

  const ref = cx.models[order.find((k) => cx.models[k] && cx.models[k].plot !== false)];
  const nUsers = cx.segments.map((s) => ref.segments[s].n_users);

  return {
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
    footnote: 'rendered from demo/data/crossover.json + phase8.json, projections of results/runs.jsonl (append-only)',
  };
}

/**
 * T8-3 exploratory depth axis: seven buckets, only the two arms the
 * deep_buckets record recomposed (pop-t12m and static ALS). The 0/1-4/5-9/10-19
 * values are asserted equal to the recorded per-segment means in-record
 * (results.self_check), so the left half of this chart is the same object as
 * the frozen chart's left half.
 */
function deepChartSpec(cx) {
  const p8 = doc(P8);
  const db = p8 && p8.deep_buckets;
  if (!db) return null;
  const labels = db.labels;
  const series = (db.arm_keys || []).map((key) => {
    const m = cx.models[key];
    return {
      key,
      label: m ? m.label : key,
      values: labels.map((b) => db.buckets[b].arms[key][state.metric].value),
      ci_lo: labels.map((b) => db.buckets[b].arms[key][state.metric].ci_lo),
      ci_hi: labels.map((b) => db.buckets[b].arms[key][state.metric].ci_hi),
      color: SLOTS[cx.model_order.indexOf(key)],
      highlight: false,
      runId: db.run_id,
    };
  });
  const nUsers = labels.map((b) => db.buckets[b].n_users);
  return {
    segments: labels,
    nUsers,
    nUsersFmt: nUsers.map(fmt.int),
    nUsersRunId: db.run_id,
    series,
    title: `${cx.title} — deep depth buckets (exploratory)`,
    subtitle:
      `${fmt.metricLabel(state.metric)} by user history depth, buckets 20-49 / 50-99 / 100+ ` +
      'recomposed from the recorded one-shot TEST runs (T8-3, exploratory/derived; ' +
      'boundaries preregistered before any per-bucket outcome). Thin right-hand buckets, wide CIs.',
    xLabel: cx.xlabel,
    yLabel: fmt.metricLabel(state.metric),
    footnote: 'rendered from demo/data/phase8.json /deep_buckets, a projection of results/runs.jsonl (append-only)',
  };
}

/** The reference figure's small-print receipts row, as flowing HTML. */
function receiptsStrip(cx) {
  const p8 = doc(P8);
  if (state.depth === 'deep') {
    const db = p8 && p8.deep_buckets;
    if (!db) return '';
    const src = (db.arm_keys || [])
      .map((k) => `<li>${fmt.esc(labelFor(k))} (source run): ${runIdChip(db.source_run_ids[k])}</li>`)
      .join('');
    return `<ul class="receipts-strip">
      <li>deep-bucket recomposition: ${runIdChip(db.run_id)}</li>${src}</ul>`;
  }
  const chips = cx.model_order
    .map((k) => {
      const m = cx.models[k];
      if (!m) return '';
      return `<li>${fmt.esc(m.label)}${m.plot === false ? ' <span class="muted">(not plotted)</span>' : ''}:
        ${runIdChip(m.run_id)}</li>`;
    })
    .join('');
  const p8chips = p8
    ? (p8.arm_order || [])
        .map((k) => {
          const m = p8.arms[k];
          return m ? `<li>${fmt.esc(m.label)}: ${runIdChip(m.run_id)}</li>` : '';
        })
        .join('')
    : '';
  const pg = doc(PG);
  const grid = pg
    ? `<li>n* grid recomposition: ${runIdChip(pg.record_run_id)}</li>`
    : '';
  return `<ul class="receipts-strip">${chips}${p8chips}${grid}</ul>`;
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
  const p8 = doc(P8);
  const p8rows = p8
    ? (p8.arm_order || [])
        .map((key) => {
          const m = p8.arms[key];
          if (!m) return '';
          const segs = cx.segments
            .map((s) => {
              const base = `/arms/${key}/segments/${s}/${state.metric}`;
              const lo = getTraced(P8, `${base}/ci_lo`);
              const hi = getTraced(P8, `${base}/ci_hi`);
              return `<td>${tracedSpan(P8, `${base}/value`, fmt.metric)}
                <div class="cell-ci">${fmt.esc(fmt.ci(lo.value, hi.value))}</div></td>`;
            })
            .join('');
          return `<tr>
            <th scope="row"><span class="swatch" style="background:${P8_SLOTS[key] || SLOTS[7]}"></span>${fmt.esc(
              m.label,
            )} <span class="badge">T8-2 · recency-matched</span></th>
            <td>${tracedSpan(P8, `/arms/${key}/n_users`, fmt.int)}</td>
            <td>${tracedMetricWithCi(P8, `/arms/${key}/global/${state.metric}`)}</td>
            ${segs}
          </tr>`;
        })
        .join('')
    : '';
  return `<div class="scroll-x"><table class="num-table">
    <caption>Full-catalog ranking against all ${tracedSpan(
      CX,
      '/models/blend/catalog_size',
      fmt.int,
    )} items, TRAIN-seen items excluded. Brackets are 95% user-bootstrap CIs. The T8-2 rows are the
    Phase 8 recency-matched arms — each is one preregistered TEST evaluation (selection on VAL),
    added after the recency-asymmetry confound was named in plan §8b.</caption>
    <thead><tr>${head}</tr></thead><tbody>${rows}${p8rows}</tbody></table></div>`;
}

/** ALS-decay 3-seed spread: three traced globals + an in-page (dashed) mean ± sd. */
function seedsCaption() {
  const p8 = doc(P8);
  const arm = p8 && p8.arms && p8.arms.alsdecay_hl365;
  if (!arm || !Array.isArray(arm.seeds) || !arm.seeds.length) return '';
  const vals = arm.seeds.map((s) => s.global[state.metric].value);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const sd =
    vals.length > 1
      ? Math.sqrt(vals.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) / (vals.length - 1))
      : 0;
  const parts = arm.seeds
    .map(
      (s, i) =>
        `${tracedSpan(P8, `/arms/alsdecay_hl365/seeds/${i}/global/${state.metric}/value`, fmt.metric)}
         <span class="muted">(seed ${fmt.esc(String(s.model_seed))}, ${runIdChip(s.run_id)})</span>`,
    )
    .join(' · ');
  return `<p class="caption"><strong>ALS-decay across the frozen 3-seed set.</strong>
    TEST global ${fmt.esc(fmt.metricLabel(state.metric))}: ${parts} —
    mean ± sd ${derivedSpan(
      `${fmt.metric(mean)} ± ${fmt.metric(sd)}`,
      arm.run_id,
      'in-page mean and sample sd of the three traced seed globals',
    )}. The plotted line is the primary seed (${runIdChip(arm.run_id)}), which carries the
    per-user artifact.</p>`;
}

function deltaRow(file, key, d, badge) {
  const base = `/paired_deltas/${key}/global/${state.metric}`;
  const lo = getTraced(file, `${base}/ci_lo`);
  const hi = getTraced(file, `${base}/ci_hi`);
  const ez = getTraced(file, `${base}/excludes_zero`);
  return `<tr>
    <th scope="row">${fmt.esc(d.label)}${badge ? ` <span class="badge">${fmt.esc(badge)}</span>` : ''}</th>
    <td>${tracedSpan(file, `${base}/delta`, fmt.delta)}</td>
    <td class="cell-ci">${fmt.esc(fmt.deltaCi(lo.value, hi.value))}</td>
    <td>${
      ez.value === true
        ? '<span class="badge yes">CI excludes 0</span>'
        : '<span class="badge no">CI includes 0</span>'
    }</td>
    <td>${tracedSpan(file, `/paired_deltas/${key}/n_common_users`, fmt.int)}</td>
    <td>${runIdChip(d.run_id)}</td>
  </tr>`;
}

function deltaTable(cx) {
  const rows = (cx.paired_delta_order || [])
    .map((key) => {
      const d = cx.paired_deltas[key];
      return d ? deltaRow(CX, key, d, null) : '';
    })
    .join('');
  const p8 = doc(P8);
  const p8rows = p8
    ? (p8.paired_delta_order || [])
        .map((key) => {
          const d = p8.paired_deltas[key];
          return d ? deltaRow(P8, key, d, 'T8-2') : '';
        })
        .join('')
    : '';
  return `<div class="scroll-x"><table class="num-table">
    <caption>Paired deltas, computed per user on the common user set and bootstrapped as a paired
      statistic — the comparison, not two independent intervals eyeballed against each other.</caption>
    <thead><tr><th>comparison</th><th>Δ ${fmt.esc(
      fmt.metricLabel(state.metric),
    )}</th><th>95% CI</th><th>verdict</th><th>users</th><th>record</th></tr></thead>
    <tbody>${rows}${p8rows}</tbody></table></div>`;
}

// ------------------------------------------------------------ deep buckets (T8-3)

function deepTable(cx) {
  const p8 = doc(P8);
  const db = p8 && p8.deep_buckets;
  if (!db) return '';
  const head = ['bucket', 'users', 'share', ...db.arm_keys.map(labelFor), `Δ ${fmt.metricLabel(state.metric)} (ALS − pop)`, 'verdict']
    .map((h) => `<th>${fmt.esc(h)}</th>`)
    .join('');
  const rows = db.labels
    .map((b, bi) => {
      const cell = db.buckets[b];
      const base = `/deep_buckets/buckets/${b}`;
      const arms = db.arm_keys
        .map((k) => {
          const mb = `${base}/arms/${k}/${state.metric}`;
          const lo = getTraced(P8, `${mb}/ci_lo`);
          const hi = getTraced(P8, `${mb}/ci_hi`);
          return `<td>${tracedSpan(P8, `${mb}/value`, fmt.metric)}
            <div class="cell-ci">${fmt.esc(fmt.ci(lo.value, hi.value))}</div></td>`;
        })
        .join('');
      const dbase = `${base}/delta/${state.metric}`;
      const lo = getTraced(P8, `${dbase}/ci_lo`);
      const hi = getTraced(P8, `${dbase}/ci_hi`);
      const ez = getTraced(P8, `${dbase}/excludes_zero`);
      const deep = bi >= 4;
      return `<tr${deep ? ' class="deep-row"' : ''}>
        <th scope="row">${fmt.esc(b)}${deep ? ' <span class="badge exploratory">exploratory</span>' : ''}</th>
        <td>${tracedSpan(P8, `${base}/n_users`, fmt.int)}</td>
        <td>${tracedSpan(P8, `${base}/user_share`, fmt.pct)}</td>
        ${arms}
        <td>${tracedSpan(P8, `${dbase}/delta`, fmt.delta)}
          <div class="cell-ci">${fmt.esc(fmt.deltaCi(lo.value, hi.value))}</div></td>
        <td>${
          ez.value === true
            ? '<span class="badge yes">CI excludes 0</span>'
            : '<span class="badge no">CI includes 0</span>'
        }</td>
      </tr>`;
    })
    .join('');
  return `<div class="scroll-x"><table class="num-table">
    <caption>T8-3 recomposition ${runIdChip(db.run_id)}: the per-user metric vectors of the recorded
      one-shot TEST runs, regrouped into seven depth buckets fixed from TRAIN-history counts before
      any per-bucket outcome was examined (${fmt.esc(db.preregistered)}). Buckets 0–10-19 reproduce
      the recorded per-segment means bit-identically (the record's self_check).</caption>
    <thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function deepCaptions() {
  return `
    <p class="caption"><strong>Exploratory, and labeled that way on purpose.</strong> The deeper
      buckets are thin — ${tracedSpan(P8, '/deep_buckets/buckets/50-99/n_users', fmt.int)} users at
      50-99 and ${tracedSpan(P8, '/deep_buckets/buckets/100+/n_users', fmt.int)} at 100+, with CIs
      several times wider than the shallow buckets'. The study's first sign flip appears at 50-99
      on NDCG@10 (${tracedSpan(P8, '/deep_buckets/buckets/50-99/delta/ndcg@10/delta', fmt.delta)})
      but its CI straddles zero, and the same bucket is significantly negative on Recall@20
      (${tracedSpan(P8, '/deep_buckets/buckets/50-99/delta/recall@20/delta', fmt.delta)}, CI
      excluding zero), so the flip does not survive a change of metric:
      <strong>not a crossover</strong>. Switch the metric toggle above to see it. 100+ reverts
      negative (CI includes zero).</p>
    <p class="caption">Only pop-t12m and static ALS appear here: the deep_buckets record recomposed
      exactly those two arms. The T8-2 recency-matched arms have per-segment records on the frozen
      five segments only, so drawing them on this axis would require numbers no committed record
      contains.</p>`;
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
  const p8 = doc(P8);
  const deepAvailable = !!(p8 && p8.deep_buckets);
  if (!deepAvailable) state.depth = 'frozen';
  const depthBtns = deepAvailable
    ? `<span class="control-label">depth axis</span>
       <div class="seg-switch">
         <button type="button" class="seg-btn${state.depth === 'frozen' ? ' on' : ''}" data-depth="frozen">frozen 5 segments</button>
         <button type="button" class="seg-btn${state.depth === 'deep' ? ' on' : ''}" data-depth="deep">20-49 / 50-99 / 100+</button>
       </div>
       ${state.depth === 'deep' ? '<span class="badge exploratory">exploratory · derived (T8-3)</span>' : ''}`
    : '';

  const hybrid = cx.models.hybrid;
  const hybridDelta = cx.paired_deltas && cx.paired_deltas.hybrid_vs_blend;
  const spec = chartSpec(cx);

  // The header sits OUTSIDE .scroll-x on purpose: the plot has a 720px floor and
  // scrolls sideways on a narrow viewport, but the subtitle is prose and must
  // wrap to the column instead of riding that scroller.
  const chartBlock = `
    <div class="controls">
      <span class="control-label">metric</span>
      <div class="seg-switch">${metricBtns}</div>
      ${depthBtns}
      <span class="control-hint">split: <code>${fmt.esc(cx.split)}</code> · frozen TEST, one-shot</span>
    </div>
    <figure class="chart-figure">
      ${figureHeader(spec)}
      <div class="chart-wrap scroll-x"><div class="chart-box" id="cx-chart"></div></div>
    </figure>
    ${receiptsStrip(cx)}`;

  if (state.depth === 'deep') {
    root.innerHTML = `${chartBlock}
      <h3>Every plotted number (deep buckets)</h3>
      ${deepTable(cx)}
      ${deepCaptions()}`;
  } else {
    root.innerHTML = `${chartBlock}
    <p class="caption"><strong>Derived overlay.</strong> The dashed line is the routed policy at the
      slider position — recomposed from the recorded one-shot TEST runs; no re-scoring. It is drawn
      without a CI band to keep the recorded bands readable; its interval is in the panel below.</p>
    ${
      p8 && p8.arms
        ? `<p class="caption"><strong>Recency-matched arms (T8-2).</strong> The two Phase 8 lines
           give every arm the same trailing-12-month freshness pop-t12m always had: item-kNN with
           co-occurrence restricted to trailing-12m TRAIN interactions, and ALS with time-decayed
           confidence (half-life ${tracedSpan(
             P8,
             '/arms/alsdecay_hl365/half_life_days',
             fmt.int,
           )} days, selected on VAL). Each is exactly one preregistered TEST evaluation. Neither
           crosses popularity at any frozen depth segment — the measured crossover appears only in
           the regime map's stale-item cells (Exhibit 1b).</p>`
        : ''
    }
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
    ${seedsCaption()}
    <h3>Paired comparisons</h3>
    ${deltaTable(cx)}
    <p class="caption">Full-catalog ranking: each TEST user is scored against the entire
      ${tracedSpan(CX, '/models/blend/catalog_size', fmt.int)}-item catalog with their TRAIN-seen
      items removed. No sampled negatives anywhere on this page.</p>`;
  }

  root.querySelector('#cx-chart').innerHTML = segmentLineChart(spec);
  const policyEl = root.querySelector('#cx-policy');
  if (policyEl) policyEl.innerHTML = policyPanel();
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
  root.querySelectorAll('[data-depth]').forEach((b) =>
    b.addEventListener('click', () => {
      if (b.getAttribute('data-depth') === state.depth) return;
      state.depth = b.getAttribute('data-depth');
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
