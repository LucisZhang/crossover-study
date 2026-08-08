// lineage.js -- exhibit 5, pipeline lineage + time-travel toggle.
// STUB (T31 shell; T34 builds the UI).
//
// Data contract: demo/data/lineage.json and demo/data/timetravel.json (AGREED
// schemas), written by T30's export_lineage.py. Every lineage leaf uses the
// results_artifact source kind, anchored on the kind="lineage" record's
// /artifact_sha256. The toggle's point: ops moved the Iceberg snapshots while
// reproduce-headline stayed byte_exact, twice.

import { doc, renderPlaceholder } from './data.js';
import * as fmt from './fmt.js';

const FILE = 'lineage.json';
const TT = 'timetravel.json';

export async function init(root) {
  const d = doc(FILE);
  const tt = doc(TT);
  if (!d) {
    renderPlaceholder(root, {
      file: FILE,
      title: 'pipeline lineage',
      task: 'src/batch_recsys_lab/demo/export_lineage.py (T30)',
      note: 'Bronze→silver→gold DAG with per-stage rows in/out, bytes, runtime and the Iceberg snapshot chain, plus a time-travel toggle between the pinned snapshot and today’s tables.',
    });
    return;
  }
  root.innerHTML = `<div class="placeholder">
    <p class="placeholder-title">Data exported; exhibit UI lands in T34</p>
    <p><code>demo/data/${FILE}</code> is present: ${fmt.int(d.stages_count)} stages,
       complete = <strong>${fmt.bool(d.complete)}</strong>.
       ${tt ? `Time-travel document present with ${fmt.int((tt.reproduce || []).length)} reproduce verdicts.` : `<code>demo/data/${TT}</code> is not present yet.`}</p>
  </div>`;
}
