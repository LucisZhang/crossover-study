// dq.js -- exhibit 4, data-quality dashboard.
//
// Data contract: demo/data/dq.json (SHIPPED, docs/demo-data-schemas.md), written
// by T29's export_dq.py and anchored by a kind="dq_export" record carrying the
// SHA-256 of both data/waterfall.json and the Spark job's dq_raw.json.
//
// Renders four panels: the raw->gold reconciliation waterfall (with an
// on-screen "rows_in - drops = rows_out" arithmetic line per edge), the 7x79
// contract pass/measured/fail matrix, the quarantine ledger + k-core funnel,
// and the measured-rates panel (each rate labeled with its own `source`).

import { doc, renderPlaceholder } from './data.js';
import { tracedSpan, derivedSpan } from './receipts.js';
import * as fmt from './fmt.js';

const FILE = 'dq.json';

let root = null;

// ------------------------------------------------------------------- helpers

/** Share formatter that does not collapse very small shares (e.g. 4.6e-8) to "0.0000%". */
function shareFmt(v) {
  if (typeof v !== 'number') return '--';
  if (v !== 0 && Math.abs(v) < 1e-4) return fmt.sig(v, 3);
  return fmt.pct(v, 4);
}

function escPointer(tok) {
  return String(tok).replace(/~/g, '~0').replace(/\//g, '~1');
}

/** rows_in - sum(non-"kept" reason rows) == rows_out, shown literally. */
function waterfallArithmetic(stage, idx) {
  const drops = stage.reasons.filter((r) => r.reason !== 'kept');
  const dropTerms = drops
    .map((r, ri) => {
      const pointer = `/waterfall/stages/${idx}/reasons/${stage.reasons.indexOf(r)}/rows`;
      return `${ri > 0 ? '&minus; ' : '&minus; '}${tracedSpan(FILE, pointer, fmt.int)} <span class="muted">(${fmt.esc(
        r.reason,
      )})</span>`;
    })
    .join(' ');
  return `<div class="wf-arith">
    ${tracedSpan(FILE, `/waterfall/stages/${idx}/rows_in`, fmt.int)}
    ${dropTerms || '<span class="muted">&minus; 0</span>'}
    = ${tracedSpan(FILE, `/waterfall/stages/${idx}/rows_out`, fmt.int)}
    <span class="wf-check ${stage.sum_ok && stage.count_ok ? 'ok' : 'bad'}">
      ${stage.sum_ok && stage.count_ok ? 'reconciles exactly' : 'MISMATCH'}
    </span>
  </div>`;
}

function waterfallBar(stage, idx, maxRows) {
  const kept = stage.reasons.find((r) => r.reason === 'kept');
  const drops = stage.reasons.filter((r) => r.reason !== 'kept' && r.rows > 0);
  const total = stage.rows_in || 1;
  const keptPct = ((kept ? kept.rows : stage.rows_out) / (maxRows || total)) * 100;
  const segs = drops
    .map((r) => {
      const w = (r.rows / (maxRows || total)) * 100;
      return `<span class="wf-seg wf-drop" style="width:${w}%" title="${fmt.esc(r.reason)}: ${fmt.esc(
        String(r.rows),
      )} rows"></span>`;
    })
    .join('');
  return `<div class="wf-bar-row">
    <div class="wf-bar-label">
      <code>${fmt.esc(stage.target_table)}</code>
      <span class="muted">(${fmt.esc(stage.stage_from)} &rarr; ${fmt.esc(stage.stage_to)})</span>
    </div>
    <div class="wf-bar-track">
      <span class="wf-seg wf-kept" style="width:${keptPct}%"></span>${segs}
    </div>
    <div class="wf-bar-count">
      ${tracedSpan(FILE, `/waterfall/stages/${idx}/rows_out`, fmt.int)} rows
    </div>
  </div>`;
}

function waterfallReasonsTable(stage, idx) {
  const rows = stage.reasons
    .map((r, ri) => {
      return `<tr>
        <td>${fmt.esc(r.reason)}</td>
        <td>${tracedSpan(FILE, `/waterfall/stages/${idx}/reasons/${ri}/rows`, fmt.int)}</td>
        <td>${tracedSpan(FILE, `/waterfall/stages/${idx}/reasons/${ri}/share_of_rows_in`, shareFmt)}</td>
      </tr>`;
    })
    .join('');
  return `<table class="num-table wf-reasons">
    <thead><tr><th>reason</th><th>rows</th><th>share of rows in</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function waterfallStage(stage, idx, maxRows) {
  return `<div class="wf-stage">
    <h4>${fmt.esc(stage.stage)}</h4>
    ${waterfallBar(stage, idx, maxRows)}
    ${waterfallArithmetic(stage, idx)}
    ${waterfallReasonsTable(stage, idx)}
  </div>`;
}

function waterfallSection(d) {
  const wf = d.waterfall;
  const reviewStages = wf.stages.filter((s) => s.dataset === 'reviews');
  const itemStages = wf.stages.filter((s) => s.dataset === 'items');
  const maxRows = Math.max(...reviewStages.map((s) => s.rows_in));
  const reviewIdx = wf.stages.map((s, i) => i).filter((i) => wf.stages[i].dataset === 'reviews');
  const itemIdx = wf.stages.map((s, i) => i).filter((i) => wf.stages[i].dataset === 'items');

  const reviewHtml = reviewStages.map((s, si) => waterfallStage(s, reviewIdx[si], maxRows)).join('');

  // Items waterfall shown compactly: one line, both stages are no-op (1,610,012 -> 1,610,012).
  const itemsCompact = itemStages
    .map((s, si) => {
      const idx = itemIdx[si];
      return `<div class="wf-compact-row">
        <code>${fmt.esc(s.stage)}</code>:
        ${tracedSpan(FILE, `/waterfall/stages/${idx}/rows_in`, fmt.int)} &rarr;
        ${tracedSpan(FILE, `/waterfall/stages/${idx}/rows_out`, fmt.int)}
        <span class="wf-check ${s.sum_ok && s.count_ok ? 'ok' : 'bad'}">
          ${s.sum_ok && s.count_ok ? 'reconciles exactly' : 'MISMATCH'}
        </span>
      </div>`;
    })
    .join('');

  return `
    <div class="panel">
      <h3>Reconciliation waterfall — reviews</h3>
      <p class="muted">Raw &rarr; bronze &rarr; silver &rarr; 5-core gold, every drop with a reason.
        Each edge's <code>sum_ok</code>/<code>count_ok</code> flags from
        <code>results/dq/waterfall.json</code> are rendered as an explicit reconciliation mark, and
        the rows_in &minus; drops = rows_out arithmetic is shown literally, not just asserted.</p>
      ${reviewHtml}
      <p class="caption">Waterfall reconciles end to end:
        ${tracedSpan(FILE, '/waterfall/reconciles', fmt.bool)}
        across ${tracedSpan(FILE, '/waterfall/ledger_rows_checked', fmt.int)} ledger rows checked
        (${derivedSpan('run ' + (wf.run_id || ''), wf.run_id, 'waterfall.run_id')}).</p>
    </div>
    <div class="panel">
      <h3>Reconciliation waterfall — items (compact)</h3>
      <p class="muted">Both item-dataset stages are lossless no-ops.</p>
      ${itemsCompact}
    </div>`;
}

// ------------------------------------------------------------------ matrix

function statusBadgeClass(status) {
  if (status === 'pass') return 'dq-status-pass';
  if (status === 'measured') return 'dq-status-measured';
  return 'dq-status-fail';
}

function contractMatrixSection(d) {
  const matrix = d.contract_matrix;
  const tables = d.contract_tables;
  const tableNames = Object.keys(matrix).sort();
  // Union of check ids across tables, stable order per table encounter.
  const allChecks = [];
  const seen = new Set();
  for (const t of tableNames) {
    for (const c of Object.keys(matrix[t])) {
      const key = c;
      if (!seen.has(key)) {
        seen.add(key);
        allChecks.push(key);
      }
    }
  }

  const headRow = `<tr><th>table</th>${allChecks
    .map((c) => `<th class="dq-matrix-check-head" title="${fmt.esc(c)}">${fmt.esc(c)}</th>`)
    .join('')}</tr>`;

  const bodyRows = tableNames
    .map((t) => {
      const meta = tables[t] || {};
      const cells = allChecks
        .map((c) => {
          const cell = matrix[t][c];
          if (!cell) return '<td class="dq-matrix-cell dq-status-absent">&middot;</td>';
          const pointer = `/contract_matrix/${escPointer(t)}/${escPointer(c)}/status`;
          const isSchemaMeasured = cell.status === 'measured' && cell.kind === 'schema_conformance';
          const label = cell.status === 'pass' ? 'P' : cell.status === 'measured' ? 'M' : 'F';
          return `<td class="dq-matrix-cell ${statusBadgeClass(cell.status)}${
            isSchemaMeasured ? ' dq-has-note' : ''
          }" title="${fmt.esc(cell.kind)}${cell.column ? ' · ' + fmt.esc(cell.column) : ''} · ${fmt.esc(
            cell.status,
          )}">${tracedSpan(FILE, pointer, () => label)}${
            isSchemaMeasured ? '<sup class="foot-ref">†</sup>' : ''
          }</td>`;
        })
        .join('');
      return `<tr>
        <th scope="row"><code>${fmt.esc(t)}</code>
          <div class="muted row-note">${fmt.esc(meta.contract_name)} v${fmt.esc(meta.contract_version)} ·
            ${tracedSpan(FILE, `/contract_tables/${escPointer(t)}/total_rows`, fmt.int)} rows</div>
        </th>
        ${cells}
      </tr>`;
    })
    .join('');

  const summary = d.contract_summary;
  return `<div class="panel">
    <h3>Contract matrix — ${tableNames.length} tables &times; ${allChecks.length} checks</h3>
    <p>${tracedSpan(FILE, '/contract_summary/tables', fmt.int)} tables,
      ${tracedSpan(FILE, '/contract_summary/checks', fmt.int)} checks:
      ${tracedSpan(FILE, '/contract_summary/pass', fmt.int)} pass,
      ${tracedSpan(FILE, '/contract_summary/measured', fmt.int)} measured,
      <strong>${tracedSpan(FILE, '/contract_summary/fail', fmt.int)} failing</strong>
      (<code>any_fail = ${fmt.esc(fmt.bool(summary.any_fail))}</code>).</p>
    <div class="scroll-x">
      <table class="num-table dq-matrix">
        <caption>P = pass, M = measured (not a failure — see note below), F = fail. &middot; =
          check does not apply to that table. Hover a cell for the check kind and column.</caption>
        <thead>${headRow}</thead>
        <tbody>${bodyRows}</tbody>
      </table>
    </div>
    <p class="caption"><sup>†</sup> ${fmt.esc(
      'status: "measured" is not a failure. 26 of the 28 measured checks are schema_conformance rows carrying the recorded T8 nullability downgrade: Spark/Iceberg createOrReplace marks every column physically nullable, but the declared-non-null column is still hard-enforced by its own not_null check.',
    )}</p>
  </div>`;
}

// --------------------------------------------------------------- quarantine

function quarantinePanel(d) {
  const q = d.quarantine;
  const reasonRows = q.by_reason
    .map(
      (r, i) => `<tr>
        <td><code>${fmt.esc(r.table)}</code></td>
        <td>${fmt.esc(r.reason)}</td>
        <td>${tracedSpan(FILE, `/quarantine/by_reason/${i}/rows`, fmt.int)}</td>
        <td>${tracedSpan(FILE, `/quarantine/by_reason/${i}/share_of_input`, shareFmt)}</td>
      </tr>`,
    )
    .join('');
  return `<div class="panel">
    <h3>Quarantine ledger</h3>
    <p>${tracedSpan(FILE, '/quarantine/total_rows', fmt.int)} rows total quarantined
      (build ${derivedSpan(q.build_run_id, q.build_run_id, 'quarantine.build_run_id')}).</p>
    <table class="num-table">
      <thead><tr><th>table</th><th>reason</th><th>rows</th><th>share of input</th></tr></thead>
      <tbody>${reasonRows}</tbody>
    </table>
  </div>`;
}

function kcoreFunnelPanel(d) {
  const rows = d.kcore_funnel
    .map(
      (f, i) => `<tr class="${f.converged ? 'hl' : ''}">
        <td>${tracedSpan(FILE, `/kcore_funnel/${i}/iteration`, fmt.int)}</td>
        <td>${tracedSpan(FILE, `/kcore_funnel/${i}/users`, fmt.int)}</td>
        <td>${tracedSpan(FILE, `/kcore_funnel/${i}/items`, fmt.int)}</td>
        <td>${tracedSpan(FILE, `/kcore_funnel/${i}/interactions`, fmt.int)}</td>
        <td>${tracedSpan(FILE, `/kcore_funnel/${i}/converged`, fmt.bool)}</td>
        <td>${tracedSpan(FILE, `/kcore_funnel/${i}/wall_clock_s`, fmt.duration)}</td>
      </tr>`,
    )
    .join('');
  const starts = d.reconciliation.checks.find((c) => c.name === 'funnel_starts_at_silver');
  const ends = d.reconciliation.checks.find((c) => c.name === 'funnel_ends_at_gold');
  return `<div class="panel">
    <h3>K-core convergence funnel — ${fmt.int(d.kcore_funnel.length)} iterations</h3>
    <p class="muted">Iteration 0 is the silver row count; the converged (highlighted) iteration is
      the published 5-core row count.</p>
    <div class="scroll-x">
      <table class="num-table">
        <caption>funnel_starts_at_silver: ${starts ? fmt.int(starts.lhs) : '--'} =
          ${starts ? fmt.int(starts.rhs) : '--'} (${starts ? fmt.bool(starts.ok) : '--'}) &middot;
          funnel_ends_at_gold: ${ends ? fmt.int(ends.lhs) : '--'} = ${ends ? fmt.int(ends.rhs) : '--'}
          (${ends ? fmt.bool(ends.ok) : '--'})</caption>
        <thead><tr><th>iter</th><th>users</th><th>items</th><th>interactions</th><th>converged</th><th>wall clock</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

// ---------------------------------------------------------------- rates

const RATE_SOURCE_LABEL = {
  contract_ledger: 'recorded contract evidence',
  dq_export_job: 'measured at export by the read-only DQ job',
};

function ratesPanel(d) {
  const rates = d.measured_rates;
  const rows = Object.entries(rates)
    .map(([key, r]) => {
      const pointer = `/measured_rates/${escPointer(key)}/rate`;
      const sourceLabel = RATE_SOURCE_LABEL[r.source] || r.source;
      return `<tr>
        <td><code>${fmt.esc(key)}</code></td>
        <td><code>${fmt.esc(r.table)}</code></td>
        <td>${tracedSpan(FILE, pointer, (v) => fmt.pct(v, 3))}</td>
        <td>${tracedSpan(FILE, `/measured_rates/${escPointer(key)}/rows`, fmt.int)} /
          ${tracedSpan(FILE, `/measured_rates/${escPointer(key)}/denominator`, fmt.int)}</td>
        <td><span class="kind-tag dq-source-${fmt.esc(r.source)}">${fmt.esc(sourceLabel)}</span></td>
      </tr>`;
    })
    .join('');
  return `<div class="panel">
    <h3>Measured rates — ${fmt.int(Object.keys(rates).length)} entries</h3>
    <p class="muted">Each rate is labeled with its own <code>source</code>:
      <span class="kind-tag dq-source-contract_ledger">${fmt.esc(
        RATE_SOURCE_LABEL.contract_ledger,
      )}</span> rows are recorded <code>dq_results</code> measures; <span class="kind-tag dq-source-dq_export_job">${fmt.esc(
        RATE_SOURCE_LABEL.dq_export_job,
      )}</span> rows are read-only counts the export job took because no contract check measures a
      null-price rate directly.</p>
    <div class="scroll-x">
      <table class="num-table">
        <caption>Unknown-brand and null-price rates headline this panel; price-unparseable,
          brand-from-manufacturer and join-loss rates are included for completeness.</caption>
        <thead><tr><th>rate</th><th>table</th><th>value</th><th>rows / denominator</th><th>source</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

// ------------------------------------------------------------------------ init

function render(d) {
  const parts = [];
  parts.push(waterfallSection(d));
  parts.push(contractMatrixSection(d));
  parts.push(quarantinePanel(d));
  parts.push(kcoreFunnelPanel(d));
  parts.push(ratesPanel(d));
  root.innerHTML = parts.join('\n');
}

export async function init(el) {
  root = el;
  const d = doc(FILE);
  if (!d) {
    renderPlaceholder(root, {
      file: FILE,
      title: 'data quality',
      task: 'src/batch_recsys_lab/demo/dq_export_job.py + export_dq.py (T29)',
      note: 'Raw→gold reconciliation waterfall with a reason for every dropped row, the contract pass/fail matrix, the quarantine ledger by reason, and the measured Unknown-brand / null-price rates.',
    });
    return;
  }
  render(d);
}
