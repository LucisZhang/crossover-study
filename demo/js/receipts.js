// receipts.js -- exhibit 6: the receipts drawer, and the traced-value affordance.
//
// Rule the whole site obeys: a number a viewer can see is rendered through
// getTraced(), which knows the results/runs.jsonl record it was copied from.
// tracedSpan() turns that into markup carrying data-run-id plus a dotted
// underline; a click anywhere on the page bubbles to the delegated handler here
// and opens the provenance card for that record.
//
// The drawer only displays. Every field it shows is a verbatim copy of the
// record, made by demo/export_receipts.py -- no arithmetic happens here.

import { getTraced, receipt, doc, entriesForRun, headlineRunId } from './data.js';
import * as fmt from './fmt.js';

let drawerEl = null;
let bodyEl = null;
let lastFocused = null;

// ------------------------------------------------------------ traced affordance

/**
 * Render one traced value.
 *   tracedSpan('crossover.json', '/models/blend/global/ndcg@10/value', fmt.metric)
 * Returns markup. When the leaf has no manifest entry the value is still shown,
 * but marked untraced -- it must never silently look like evidence.
 */
export function tracedSpan(file, pointer, format = fmt.metric, opts = {}) {
  const t = getTraced(file, pointer);
  const text = t.value === undefined ? '--' : format(t.value);
  if (!t.traced || !t.run_id) {
    return `<span class="untraced" title="${fmt.esc(
      `no trace_manifest entry for ${file}${pointer}`,
    )}">${fmt.esc(text)}</span>`;
  }
  const title = `${t.run_id} · ${t.source && t.source.source_pointer ? t.source.source_pointer : file + pointer}`;
  return `<span class="traced" data-run-id="${fmt.esc(t.run_id)}" tabindex="0" role="button" title="${fmt.esc(
    title,
  )}${opts.titleSuffix ? ` · ${opts.titleSuffix}` : ''}">${fmt.esc(text)}</span>`;
}

/** Traced value + its CI, as one clickable unit. */
export function tracedMetricWithCi(file, base) {
  const v = tracedSpan(file, `${base}/value`, fmt.metric);
  const lo = getTraced(file, `${base}/ci_lo`);
  const hi = getTraced(file, `${base}/ci_hi`);
  const runId = lo.run_id || hi.run_id;
  const ciText = fmt.ci(lo.value, hi.value);
  const ciSpan = runId
    ? `<span class="traced ci" data-run-id="${fmt.esc(runId)}" tabindex="0" role="button" title="${fmt.esc(
        `95% bootstrap CI · ${runId}`,
      )}">${fmt.esc(ciText)}</span>`
    : `<span class="untraced ci">${fmt.esc(ciText)}</span>`;
  return `${v} <span class="ci-wrap">${ciSpan}</span>`;
}

/**
 * A number the UI computed from traced components (e.g. a share of users summed
 * over segment n_users). Marked distinctly -- dashed, not dotted -- and anchored
 * to the record its components came from.
 */
export function derivedSpan(text, runId, note) {
  if (!runId) return `<span class="untraced">${fmt.esc(text)}</span>`;
  return `<span class="traced derived" data-run-id="${fmt.esc(runId)}" tabindex="0" role="button" title="${fmt.esc(
    `derived in-page from traced components · ${note || ''} · ${runId}`,
  )}">${fmt.esc(text)}</span>`;
}

/** A bare run_id rendered as a receipt link. */
export function runIdChip(runId, label) {
  if (!runId) return '<span class="untraced">--</span>';
  return `<span class="traced run-chip" data-run-id="${fmt.esc(
    runId,
  )}" tabindex="0" role="button" title="open receipt">${fmt.esc(label || runId)}</span>`;
}

// -------------------------------------------------------------------- the drawer

function row(label, valueHtml, opts = {}) {
  if (valueHtml === null || valueHtml === undefined) return '';
  return `<div class="rc-row${opts.absent ? ' absent' : ''}">
    <div class="rc-k">${fmt.esc(label)}</div>
    <div class="rc-v">${valueHtml}</div>
  </div>`;
}

function mono(v) {
  return `<code>${fmt.esc(v)}</code>`;
}

function absentRow(label) {
  return row(label, '<span class="muted">not present in this record kind</span>', { absent: true });
}

function renderCard(runId) {
  const rec = receipt(runId);
  if (!rec) {
    return `<div class="rc-card"><p class="muted">No receipts.json card for <code>${fmt.esc(
      runId,
    )}</code>. The export writes the closure of run_ids the trace manifest depends on;
    if you are seeing this, the manifest and receipts are out of step -- <code>make demo-verify</code> would fail.</p></div>`;
  }

  const absent = new Set(rec.fields_absent_in_record || []);
  const isHeadline = runId === headlineRunId();
  const cites = entriesForRun(runId).length;

  const out = [];
  out.push(`<div class="rc-card">`);
  out.push(`<div class="rc-head">
      <div class="rc-kind">${fmt.esc(rec.kind || 'record')}${isHeadline ? ' · headline' : ''}</div>
      <div class="rc-id"><code>${fmt.esc(rec.run_id)}</code></div>
    </div>`);

  out.push('<div class="rc-sect">Run</div>');
  out.push(row('recorded', fmt.ts(rec.run_ts)));
  out.push(
    row(
      'git',
      `${mono(fmt.shortSha(rec.git_sha, 12))} <span class="muted">· dirty: ${fmt.bool(rec.git_dirty)}</span>`,
    ),
  );
  out.push(row('hardware', rec.hardware ? fmt.esc(rec.hardware) : null));
  out.push(row('wall clock', typeof rec.wall_clock_s === 'number' ? `${fmt.duration(rec.wall_clock_s)} <span class="muted">(${rec.wall_clock_s}s)</span>` : null));

  out.push('<div class="rc-sect">Inputs</div>');
  out.push(absent.has('config_path') ? absentRow('config') : row('config', rec.config_path ? mono(rec.config_path) : null));
  out.push(row('config hash', rec.config_hash ? mono(fmt.shortHash(rec.config_hash)) : null));
  out.push(
    absent.has('dataset_manifest_hash')
      ? absentRow('dataset manifest')
      : row('dataset manifest', rec.dataset_manifest_hash ? mono(fmt.shortHash(rec.dataset_manifest_hash)) : null),
  );
  if (rec.splits) {
    out.push(
      row(
        'splits',
        `v${fmt.esc(rec.splits.version)} · frozen ${fmt.esc(rec.splits.frozen_at)}<br>${mono(
          fmt.shortHash(rec.splits.file_hash),
        )}`,
      ),
    );
  } else if (absent.has('splits')) {
    out.push(absentRow('splits'));
  }

  if (rec.iceberg_snapshots) {
    const rows = Object.entries(rec.iceberg_snapshots)
      .map(([t, id]) => `<div class="rc-snap"><code>${fmt.esc(t)}</code><code>${fmt.esc(fmt.snapshotId(id))}</code></div>`)
      .join('');
    out.push(row('iceberg snapshots', `<div class="rc-snaps">${rows}</div>`));
  } else if (absent.has('iceberg_snapshots')) {
    out.push(absentRow('iceberg snapshots'));
  }

  if (rec.seeds) {
    const s = Object.entries(rec.seeds)
      .map(([k, v]) => `${fmt.esc(k)}=${v === null ? '<span class="muted">n/a</span>' : fmt.esc(String(v))}`)
      .join(' · ');
    out.push(row('seeds', s));
  } else if (absent.has('seeds')) {
    out.push(absentRow('seeds'));
  }

  if (rec.model) {
    out.push('<div class="rc-sect">Model</div>');
    out.push(row('name', mono(rec.model.name)));
    if (rec.model.params && Object.keys(rec.model.params).length) {
      out.push(row('params', `<pre class="rc-json">${fmt.esc(fmt.json(rec.model.params))}</pre>`));
    }
  } else if (absent.has('model')) {
    out.push('<div class="rc-sect">Model</div>');
    out.push(absentRow('model'));
  }

  if (Array.isArray(rec.reproduce) && rec.reproduce.length) {
    out.push('<div class="rc-sect">Reproduction</div>');
    const items = rec.reproduce
      .map(
        (r) =>
          `<li><span class="verdict ${fmt.esc(r.verdict)}">${fmt.esc(r.verdict)}</span> ${runIdChip(r.run_id)}</li>`,
      )
      .join('');
    out.push(row('verdicts', `<ul class="rc-list">${items}</ul>`));
  }
  if (rec.repro_command) {
    out.push(row('reproduce with', `<pre class="rc-cmd">${fmt.esc(rec.repro_command)}</pre>`));
  }

  out.push('<div class="rc-sect">Trace</div>');
  out.push(
    row(
      'cited by',
      `${fmt.int(cites)} traced ${cites === 1 ? 'leaf' : 'leaves'} across <code>demo/data/</code>`,
    ),
  );
  out.push(
    row(
      'checked by',
      '<code>make demo-verify</code> — re-resolves every leaf against the append-only log',
    ),
  );

  out.push('</div>');
  return out.join('');
}

export function openReceipt(runId) {
  if (!drawerEl) return;
  lastFocused = document.activeElement;
  bodyEl.innerHTML = renderCard(runId);
  drawerEl.hidden = false;
  drawerEl.classList.add('open');
  document.body.classList.add('drawer-open');
  const close = drawerEl.querySelector('.rc-close');
  if (close) close.focus();
}

export function closeReceipt() {
  if (!drawerEl) return;
  drawerEl.classList.remove('open');
  drawerEl.hidden = true;
  document.body.classList.remove('drawer-open');
  if (lastFocused && typeof lastFocused.focus === 'function') lastFocused.focus();
}

/** Install the drawer and the delegated open handler. Call once at boot. */
export function initReceipts(root) {
  drawerEl = root;
  const receipts = doc('receipts.json');
  root.innerHTML = `
    <div class="rc-bar">
      <span class="rc-title">Receipt</span>
      <button type="button" class="rc-close" aria-label="Close receipt">close</button>
    </div>
    <div class="rc-body"></div>
    <div class="rc-foot">${
      receipts && receipts.note ? fmt.esc(receipts.note) : 'Fields are verbatim copies of results/runs.jsonl records.'
    }</div>`;
  bodyEl = root.querySelector('.rc-body');
  root.hidden = true;

  root.querySelector('.rc-close').addEventListener('click', closeReceipt);

  // Delegated: works for HTML and for SVG elements (Element.closest is defined
  // on SVGElement too), so chart marks open receipts as well.
  document.addEventListener('click', (ev) => {
    const target = ev.target instanceof Element ? ev.target.closest('[data-run-id]') : null;
    if (!target) return;
    ev.preventDefault();
    openReceipt(target.getAttribute('data-run-id'));
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      closeReceipt();
      return;
    }
    if (ev.key !== 'Enter' && ev.key !== ' ') return;
    const target = ev.target instanceof Element ? ev.target.closest('[data-run-id]') : null;
    if (!target) return;
    ev.preventDefault();
    openReceipt(target.getAttribute('data-run-id'));
  });
}

/** Exhibit 6's own section: the index of every record the site can show. */
export function initReceiptsSection(root) {
  const receipts = doc('receipts.json');
  if (!receipts) {
    root.innerHTML = '<p class="muted">receipts.json not loaded.</p>';
    return;
  }
  const order = receipts.run_order || Object.keys(receipts.runs || {});
  const rows = order
    .map((rid) => {
      const r = receipts.runs[rid] || {};
      return `<tr>
        <td>${runIdChip(rid)}</td>
        <td><span class="kind-tag">${fmt.esc(r.kind || '--')}</span></td>
        <td>${fmt.esc(r.model && r.model.name ? r.model.name : '--')}</td>
        <td>${fmt.esc(fmt.ts(r.run_ts))}</td>
        <td><code>${fmt.esc(fmt.shortSha(r.git_sha))}</code></td>
      </tr>`;
    })
    .join('');
  root.innerHTML = `
    <p>Every number on this page is a copy of a value in
       <code>results/runs.jsonl</code>, the append-only results log. Click any
       <span class="traced sample" tabindex="-1">dotted number</span> to open the record it came
       from: config path and hash, git SHA, dataset manifest hash, frozen-splits hash, Iceberg
       snapshot IDs, seeds, model params, wall clock, hardware. Dashed numbers are aggregates this
       page computed from traced components; the receipt names the record the components came from.</p>
    <p class="muted">Headline run ${runIdChip(receipts.headline_run_id)} carries its reproduction
       verdicts and the command that regenerates it.</p>
    <div class="scroll-x">
      <table class="rec-table">
        <thead><tr><th>run_id</th><th>kind</th><th>model</th><th>recorded</th><th>git</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}
