// contrast.js -- exhibit 1c: Phase 9 regime contrast (Amazon Electronics vs ML-32M).
//
// demo/data/contrast.json is exporter-generated like every other document
// (src/batch_recsys_lab/demo/export_contrast.py, run by `make demo-export`) and
// every numeric leaf it holds has a trace_manifest.json entry, so
// `make demo-verify` re-resolves this panel's numbers independently. Two source
// kinds back it:
//   - results/runs.jsonl records (global arm NDCG@10, both churn shares), one of
//     them cited through a record_selector because its run_id is shared by two
//     records in the append-only log;
//   - results/confirmatory_ml32m_test.json, a committed derived T9-3c analysis
//     that by design never appends to runs.jsonl -- anchored by SHA-256 plus the
//     per-user parquets its source records published ("derived_artifact").
// This module reads the document's own leaves rather than getTraced(): the
// values are identical (the manifest records the same leaf), and each block
// carries the run_id it came from so runIdChip() opens the matching receipt.

import { doc, renderPlaceholder } from './data.js';
import { runIdChip } from './receipts.js';
import * as fmt from './fmt.js';

const C = 'contrast.json';

function churnRow(c) {
  const cc = c.churn_contrast;
  return `
    <table class="num-table">
      <caption>${fmt.esc(cc.statistic)} -- the churn gate (§8b/§8c), Amazon Electronics vs ML-32M.</caption>
      <thead><tr><th>dataset</th><th>measured share</th><th>band</th><th>verdict</th><th>record</th></tr></thead>
      <tbody>
        <tr>
          <th scope="row">Amazon Electronics</th>
          <td>${fmt.pct(cc.amazon_electronics.value, 2)}</td>
          <td>${fmt.esc(cc.amazon_electronics.band)}</td>
          <td>${fmt.esc(cc.amazon_electronics.verdict)}</td>
          <td>${runIdChip(cc.amazon_electronics.run_id)}</td>
        </tr>
        <tr>
          <th scope="row">ML-32M</th>
          <td>${fmt.pct(cc.ml32m.value, 2)}</td>
          <td>${fmt.esc(cc.ml32m.band)}</td>
          <td>${fmt.esc(cc.ml32m.verdict)}</td>
          <td>${runIdChip(cc.ml32m.run_id)}</td>
        </tr>
      </tbody>
    </table>
    <p class="caption">Difference: <strong>${fmt.pct(cc.difference, 2)}</strong> (ML-32M minus Amazon
      Electronics) -- the low-churn regime the crossover contrast below runs in.</p>`;
}

function globalArmTable(c) {
  const g = c.ml32m_global_ndcg10;
  const order = ['pop_t12m', 'itemknn_t12m', 'als', 'blend_alpha0_1'];
  const label = {
    pop_t12m: 'pop-t12m (P*)',
    itemknn_t12m: 'item-kNN-t12m (M*)',
    als: 'ALS',
    blend_alpha0_1: 'content-pop blend (α=0.1)',
  };
  const rows = order
    .map((k) => {
      const a = g[k];
      // Seeded arms name their seed: the quoted ALS value is the canonical
      // primary-seed record (§6 admits only that seed's per-user artifact for
      // paired tests), not a 3-seed mean.
      const seed =
        a.model_seed === null || a.model_seed === undefined
          ? '<span class="muted">deterministic</span>'
          : `seed ${fmt.esc(String(a.model_seed))}`;
      return `<tr><th scope="row">${fmt.esc(label[k])}</th>
        <td>${fmt.metric(a.value)}</td>
        <td><code>${fmt.esc(a.model_name)}</code></td>
        <td>${seed}</td>
        <td>${runIdChip(a.run_id)}</td></tr>`;
    })
    .join('');
  return `
    <table class="num-table">
      <caption>ML-32M global TEST NDCG@10 per arm (§8c one-shot TEST runs). Seeded arms quote the
        canonical primary-seed record, not a multi-seed mean.</caption>
      <thead><tr><th>arm</th><th>NDCG@10</th><th>model</th><th>seed</th><th>record</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function verdictPanel(c) {
  const v = c.verdict;
  return `
    <div class="panel">
      <h3>Confirmatory verdict (T9-3c, preregistered T9-3b)</h3>
      <p><strong>${fmt.esc(v.verdict)}</strong> -- ${fmt.esc(v.headline)}. Policy: n* =
        ${fmt.esc(String(v.n_star))}, crossover bucket ${fmt.esc(v.crossover_bucket)}.
        ${runIdChip(c.source_run_ids.m_star, 'M* record')}
        ${runIdChip(c.source_run_ids.p_star, 'P* record')}</p>
      <p class="caption">Derived analysis file: <code>${fmt.esc(c.confirmatory_source.path)}</code>
        (git ${fmt.shortSha(c.confirmatory_source.git_sha)}, generated
        ${fmt.esc(fmt.ts(c.confirmatory_source.generated_ts))}) -- appends nothing to
        <code>results/runs.jsonl</code>; regroups the per-user vectors already committed by the
        ten one-shot ML-32M TEST runs named in its <code>source_run_ids</code>.</p>
    </div>`;
}

function bucketTable(c) {
  const rows = c.winning_deep_buckets
    .map(
      (b) => `<tr>
        <th scope="row">${fmt.esc(b.label)}</th>
        <td>${fmt.int(b.n_users)}</td>
        <td>${fmt.delta(b.delta)}</td>
        <td>${fmt.deltaCi(b.ci_lo, b.ci_hi)}</td>
        <td>${b.p_value_uncorrected < 0.002 ? '2/1001 (floor)' : b.p_value_uncorrected.toFixed(6)}</td>
        <td>${b.q_value.toFixed(6)}</td>
        <td>${fmt.bool(b.bh_significant)}</td>
        <td>${fmt.pct(b.user_share, 2)}</td>
      </tr>`,
    )
    .join('');
  const d0 = c.d4_depth0;
  return `
    <div class="scroll-x"><table class="num-table">
      <caption>The three BH-significant, positive M* − P* NDCG@10 depth buckets (§7 condition i) --
        the coherent region behind the D1 verdict above.</caption>
      <thead><tr><th>depth bucket</th><th>users</th><th>Δ NDCG@10</th><th>95% CI</th><th>p (uncorrected)</th>
        <th>q (BH)</th><th>BH-sig</th><th>GT share</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="caption"><strong>D4 (depth 0, cold users):</strong> Δ NDCG@10 =
      ${fmt.delta(d0.delta)} ${fmt.deltaCi(d0.ci_lo, d0.ci_hi)}, BH-significant =
      ${fmt.bool(d0.bh_significant)} -- flagged <code>${fmt.esc(c.verdict.d4_token)}</code>: the
      coherent-region win does not erase a significant cold-user loss.</p>`;
}

function figures(c) {
  return `
    <div class="fig-pair">
      <figure>
        <img src="${fmt.esc(c.figures.regime_map)}" alt="ML-32M crossover chart: NDCG@10 by user history depth, M* vs P*." loading="lazy">
        <figcaption class="caption">Regime map (ML-32M TEST) -- ${runIdChip(c.source_run_ids.m_star, 'M*')}
          vs ${runIdChip(c.source_run_ids.p_star, 'P*')}.</figcaption>
      </figure>
      <figure>
        <img src="${fmt.esc(c.figures.deep_buckets)}" alt="ML-32M deep-bucket crossover chart, extended history-depth axis." loading="lazy">
        <figcaption class="caption">Deep-bucket extension (20-49 / 50-99 / 100+).</figcaption>
      </figure>
    </div>`;
}

function caveats(c) {
  const items = c.caveats.map((cv) => `<li>${fmt.esc(cv.text)}</li>`).join('');
  return `
    <blockquote class="rm-caveat">
      <p><strong>Caveats.</strong></p>
      <ul>${items}</ul>
      <footer class="muted">Recall@20 guard: ${fmt.esc(c.recall_guard.definition)} --
        agreeing labels: ${c.recall_guard.agreeing_labels.length ? fmt.esc(c.recall_guard.agreeing_labels.join(', ')) : 'none'}
        (metric_robust = ${fmt.bool(c.recall_guard.metric_robust)}).</footer>
    </blockquote>`;
}

export async function init(el) {
  const c = doc(C);
  if (!c) {
    renderPlaceholder(el, {
      file: C,
      title: 'Regime contrast',
      task: 'T9-4 (demo/data/contrast.json)',
      note: 'Amazon Electronics vs ML-32M churn contrast and the T9-3c confirmatory verdict.',
    });
    return;
  }
  el.innerHTML = `
    ${churnRow(c)}
    ${globalArmTable(c)}
    ${verdictPanel(c)}
    ${bucketTable(c)}
    ${figures(c)}
    ${caveats(c)}`;
}
