// lineage.js -- exhibit 5, pipeline lineage + time-travel toggle.
//
// Data contract: demo/data/lineage.json and demo/data/timetravel.json (AGREED
// schemas), written by T30's export_lineage.py. Every lineage leaf uses the
// results_artifact source kind, anchored on the kind="lineage" record's
// /artifact_sha256. Every timetravel leaf is a runs_record copy. The toggle's
// point: ops moved the Iceberg snapshots while reproduce-headline stayed
// byte_exact, twice.

import { doc, renderPlaceholder } from './data.js';
import { tracedSpan, runIdChip } from './receipts.js';
import * as fmt from './fmt.js';

const FILE = 'lineage.json';
const TT = 'timetravel.json';

const LAYER_ORDER = ['raw', 'bronze', 'silver', 'gold', 'cache', 'eval', 'ops'];
const LAYER_LABEL = {
  raw: 'raw',
  bronze: 'bronze',
  silver: 'silver',
  gold: 'gold',
  cache: 'cache',
  eval: 'eval',
  ops: 'ops',
};

const state = { toggle: 'pinned' }; // 'pinned' | 'today'

let root = null;

// ------------------------------------------------------------------- lineage

function stageRow(stage, idx) {
  const isOps = stage.layer === 'ops';
  const footnote = stage.wall_clock_source === 'runtime_not_persisted_at_build';
  const wallClock = footnote
    ? `<span class="muted">null</span>`
    : tracedSpan(FILE, `/stages/${idx}/wall_clock_s`, fmt.duration);
  const rowsIn =
    stage.rows_in === null || stage.rows_in === undefined
      ? '<span class="muted">--</span>'
      : tracedSpan(FILE, `/stages/${idx}/rows_in`, fmt.int);
  const rowsOut =
    stage.rows_out === null || stage.rows_out === undefined
      ? '<span class="muted">--</span>'
      : tracedSpan(FILE, `/stages/${idx}/rows_out`, fmt.int);
  const bytes =
    stage.bytes === null || stage.bytes === undefined
      ? '<span class="muted">--</span>'
      : tracedSpan(FILE, `/stages/${idx}/bytes`, fmt.int);
  const snap =
    stage.snapshot_id === null || stage.snapshot_id === undefined
      ? '<span class="muted">--</span>'
      : `<code class="mono" title="${fmt.esc(String(stage.snapshot_id))}">${tracedSpan(
          FILE,
          `/stages/${idx}/snapshot_id`,
          truncateSnapshot,
        )}</code>`;
  return `<tr class="${isOps ? 'lineage-row-ops' : ''}">
    <td><span class="layer-badge layer-${fmt.esc(stage.layer)}">${fmt.esc(
      LAYER_LABEL[stage.layer] || stage.layer,
    )}</span></td>
    <th scope="row">${fmt.esc(stage.stage)}</th>
    <td class="lineage-table-cell"><code>${fmt.esc(stage.table)}</code></td>
    <td>${rowsIn}</td>
    <td>${rowsOut}</td>
    <td>${bytes}</td>
    <td>${wallClock}${footnote ? ' <sup class="foot-ref">†</sup>' : ''}</td>
    <td>${snap}</td>
  </tr>`;
}

/** Iceberg snapshot ids are int64; show the first/last few digits, full value
 *  in the wrapping <code>'s title attribute. Display rounding only, per
 *  fmt.js's own rule -- this lives here (not fmt.js) because it is not a
 *  numeric rounding, it is a string truncation of an opaque id. */
function truncateSnapshot(v) {
  const s = fmt.snapshotId(v);
  return s.length > 12 ? `${s.slice(0, 6)}…${s.slice(-4)}` : s;
}

function lineageSection(d) {
  const rows = d.stages.map((s, i) => stageRow(s, i)).join('');
  const footnoteText = (d.footnotes && d.footnotes.runtime_not_persisted_at_build) || '';
  return `
    <div class="controls">
      <span class="control-label">stages</span>
      <span class="control-hint">${fmt.int(d.stages_count)} stages across
        ${LAYER_ORDER.filter((l) => d.stages.some((s) => s.layer === l)).length} layers ·
        complete = <strong>${fmt.bool(d.complete)}</strong></span>
    </div>
    <div class="scroll-x">
      <table class="num-table lineage-table">
        <caption>Bronze→silver→gold→eval→ops, one row per stage. Rows in/out, bytes and wall
          clock are traced against <code>results/lineage.json</code>
          (<code>${fmt.esc(fmt.shortHash(d.artifact_sha256))}</code>, descriptive — it anchors
          the stage leaves above, it is not itself a measured value), which anchors on the
          <code>kind="lineage"</code> record. Snapshot IDs are truncated; hover for the full value,
          click to open the record.</caption>
        <thead><tr>
          <th>layer</th><th>stage</th><th>table</th><th>rows in</th><th>rows out</th>
          <th>bytes</th><th>wall clock</th><th>snapshot</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${
      footnoteText
        ? `<p class="caption"><sup>†</sup> ${fmt.esc(footnoteText)}</p>`
        : ''
    }`;
}

// ---------------------------------------------------------------- time-travel

function snapshotList(file, base, snapshotIds) {
  return Object.entries(snapshotIds || {})
    .map(
      ([table, id]) => `<div class="rc-snap">
        <code>${fmt.esc(table)}</code>
        ${tracedSpan(file, `${base}/${escPointer(table)}`, fmt.snapshotId)}
      </div>`,
    )
    .join('');
}

function escPointer(tok) {
  return String(tok).replace(/~/g, '~0').replace(/\//g, '~1');
}

function pinnedPanel(tt) {
  const headlineMetrics = Object.keys((tt.pinned && tt.pinned.headline) || {})
    .map((metric) => {
      const base = `/pinned/headline/${escPointer(metric)}`;
      return `<div class="kv">
        <div class="kv-k">${fmt.esc(fmt.metricLabel(metric))} [95% CI]</div>
        <div class="kv-v">${tracedSpan(TT, `${base}/value`, fmt.metric)}
          <span class="ci-wrap muted">${fmt.esc(
            fmt.ci(
              (tt.pinned.headline[metric] || {}).ci_lo,
              (tt.pinned.headline[metric] || {}).ci_hi,
            ),
          )}</span></div>
      </div>`;
    })
    .join('');
  return `<div class="panel">
    <h3>Pinned snapshot</h3>
    <p class="muted">The headline run ${runIdChip(tt.pinned.run_id)} pinned these Iceberg
      snapshot IDs; every reported number on this site traces back to them.</p>
    <div class="rc-snaps">${snapshotList(TT, '/pinned/snapshot_ids', tt.pinned.snapshot_ids)}</div>
    <div class="kv-grid">${headlineMetrics}</div>
  </div>`;
}

function todayPanel(tt) {
  const opsRows = (tt.ops_chain || [])
    .map(
      (op, i) => `<tr>
        <td>${runIdChip(op.run_id, op.step)}</td>
        <td><code>${fmt.esc(op.table)}</code></td>
        <td class="cell-ci">${
          op.snapshot_id_before === null || op.snapshot_id_before === undefined
            ? '<span class="muted">--</span>'
            : `<code>${tracedSpan(TT, `/ops_chain/${i}/snapshot_id_before`, fmt.snapshotId)}</code>`
        } &rarr; <code>${tracedSpan(TT, `/ops_chain/${i}/snapshot_id_after`, fmt.snapshotId)}</code></td>
      </tr>`,
    )
    .join('');
  return `<div class="panel">
    <h3>Tables since then</h3>
    <p class="muted">${fmt.int((tt.ops_chain || []).length)} recorded ops steps moved
      <code>local.ops.interactions_monthly</code> to snapshot
      ${tracedSpan(TT, '/today/snapshot_ids/local.ops.interactions_monthly', fmt.snapshotId)},
      last written by ${runIdChip(tt.today.source_run_id)}.</p>
    <div class="scroll-x">
      <table class="num-table">
        <caption>Each step is an appended <code>kind="ops"</code> record — backfill, monthly
          appends, an upsert, two compaction shapes, and staggered snapshot expiry.</caption>
        <thead><tr><th>step</th><th>table</th><th>snapshot before &rarr; after</th></tr></thead>
        <tbody>${opsRows}</tbody>
      </table>
    </div>
    ${
      tt.notes && tt.notes.ops_table_dropped
        ? `<p class="caption">${fmt.esc(tt.notes.ops_table_dropped)}</p>`
        : ''
    }
  </div>`;
}

function reproducePunchline(tt) {
  const chips = (tt.reproduce || [])
    .map(
      (r) =>
        `<li><span class="verdict ${fmt.esc(r.verdict)}">${fmt.esc(r.verdict)}</span> ${runIdChip(
          r.run_id,
          fmt.ts(r.run_ts),
        )}</li>`,
    )
    .join('');
  return `<div class="panel">
    <h3>The catalog moved. The headline did not.</h3>
    <p>Between the headline run ${runIdChip(tt.pinned.run_id)} and today, the ops table went
      through ${fmt.int((tt.ops_chain || []).length)} recorded snapshot-moving steps — roughly
      40 snapshots across backfill, monthly appends, an upsert, compaction and expiry. The pinned
      re-run reproduced the headline byte-for-byte, twice:</p>
    <ul class="rc-list">${chips}</ul>
    <p class="caption">Reproduction is against the pinned snapshot IDs above, not against
      whatever the ops table happens to be today — that is the point of pinning.</p>
  </div>`;
}

function toggleSection(tt) {
  const btns = ['pinned', 'today']
    .map(
      (k) =>
        `<button type="button" class="seg-btn${state.toggle === k ? ' on' : ''}" data-toggle="${k}">${
          k === 'pinned' ? 'pinned snapshot' : 'tables since then'
        }</button>`,
    )
    .join('');
  return `
    <div class="controls">
      <span class="control-label">time travel</span>
      <div class="seg-switch">${btns}</div>
    </div>
    <div id="lineage-tt-panel">${state.toggle === 'pinned' ? pinnedPanel(tt) : todayPanel(tt)}</div>
    ${reproducePunchline(tt)}`;
}

// ------------------------------------------------------------------------ init

function render() {
  const d = doc(FILE);
  const tt = doc(TT);
  const parts = [];
  if (d) {
    parts.push('<h3>Pipeline lineage</h3>');
    parts.push(lineageSection(d));
  } else {
    parts.push(`<p class="warn"><code>demo/data/${FILE}</code> is not present.</p>`);
  }
  if (tt) {
    parts.push('<h3>Time travel</h3>');
    parts.push(toggleSection(tt));
  } else {
    parts.push(`<p class="warn"><code>demo/data/${TT}</code> is not present.</p>`);
  }
  root.innerHTML = parts.join('\n');
  wire();
}

function wire() {
  root.querySelectorAll('[data-toggle]').forEach((b) =>
    b.addEventListener('click', () => {
      const v = b.getAttribute('data-toggle');
      if (v === state.toggle) return;
      state.toggle = v;
      render();
    }),
  );
}

export async function init(el) {
  root = el;
  const d = doc(FILE);
  const tt = doc(TT);
  if (!d && !tt) {
    renderPlaceholder(root, {
      file: FILE,
      title: 'pipeline lineage',
      task: 'src/batch_recsys_lab/demo/export_lineage.py (T30)',
      note: 'Bronze→silver→gold DAG with per-stage rows in/out, bytes, runtime and the Iceberg snapshot chain, plus a time-travel toggle between the pinned snapshot and today’s tables.',
    });
    return;
  }
  render();
}
