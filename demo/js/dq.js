// dq.js -- exhibit 4, data-quality dashboard. STUB (T31 shell; T34 builds the UI).
//
// Data contract: demo/data/dq.json (AGREED schema), written by T29's export_dq.py
// and anchored by a kind="dq_export" record carrying the SHA-256 of both
// data/waterfall.json and the Spark job's dq_raw.json. The on-screen
// reconciliation waterfall must sum exactly.

import { doc, renderPlaceholder } from './data.js';
import * as fmt from './fmt.js';

const FILE = 'dq.json';

export async function init(root) {
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
  const stages = (d.waterfall && d.waterfall.stages) || [];
  root.innerHTML = `<div class="placeholder">
    <p class="placeholder-title">Data exported; exhibit UI lands in T34</p>
    <p><code>demo/data/${FILE}</code> is present: ${fmt.int(stages.length)} waterfall stages,
       reconciles = <strong>${fmt.bool(d.waterfall && d.waterfall.reconciles)}</strong>.</p>
  </div>`;
}
