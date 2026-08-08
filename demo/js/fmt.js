// fmt.js -- ALL display rounding lives here.
//
// demo/data/*.json carry full precision (schema rule 2). Nothing outside this
// module may call toFixed / toPrecision / Math.round on an evidence value. If
// you find yourself rounding elsewhere, add a formatter here instead.

const SIG_DEFAULT = 4;

/** Significant-digit format that never falls back to exponential notation. */
export function sig(v, n = SIG_DEFAULT) {
  if (v === null || v === undefined || Number.isNaN(v)) return '--';
  if (typeof v !== 'number') return String(v);
  if (v === 0) return '0';
  const mag = Math.floor(Math.log10(Math.abs(v)));
  const decimals = Math.min(20, Math.max(0, n - 1 - mag));
  return v.toFixed(decimals);
}

/** A metric point estimate (ndcg@10, recall@20, ...): 4 significant digits. */
export function metric(v) {
  return sig(v, SIG_DEFAULT);
}

/** A 95% bootstrap CI, bracketed. */
export function ci(lo, hi) {
  if (lo === null || lo === undefined || hi === null || hi === undefined) return '';
  return `[${sig(lo, SIG_DEFAULT)}, ${sig(hi, SIG_DEFAULT)}]`;
}

/** A paired delta, explicitly signed (sign is the finding, never drop it). */
export function delta(v) {
  if (typeof v !== 'number') return '--';
  if (v === 0) return '0';
  return (v > 0 ? '+' : '−') + sig(Math.abs(v), SIG_DEFAULT);
}

/** Signed CI for a delta. */
export function deltaCi(lo, hi) {
  if (typeof lo !== 'number' || typeof hi !== 'number') return '';
  return `[${delta(lo)}, ${delta(hi)}]`;
}

/** Thousands-separated integer (user counts, catalog size, row counts). */
export function int(n) {
  if (typeof n !== 'number') return '--';
  const neg = n < 0;
  const s = Math.abs(Math.round(n)).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return (neg ? '−' : '') + s;
}

/** Fraction in [0,1] -> percent string. */
export function pct(x, dp = 1) {
  if (typeof x !== 'number') return '--';
  return `${(x * 100).toFixed(dp)}%`;
}

/** Axis tick label: decimals derived from the tick step so ticks render exactly. */
export function axisTick(v, step) {
  if (!Number.isFinite(step) || step <= 0) return String(v);
  const mag = Math.floor(Math.log10(step));
  let decimals = Math.max(0, -mag);
  // 2.5e-3-style steps need one extra place.
  const mant = step / Math.pow(10, mag);
  if (Math.abs(mant - Math.round(mant)) > 1e-9) decimals += 1;
  return v.toFixed(Math.min(20, decimals));
}

/** Metric key -> display label ("ndcg@10" -> "NDCG@10"). */
export function metricLabel(key) {
  return String(key).toUpperCase();
}

/** n* grid label -> display ("inf" -> the infinity glyph). */
export function nStarLabel(label) {
  return label === 'inf' ? '∞' : String(label);
}

/** Short git SHA. */
export function shortSha(sha, n = 7) {
  return typeof sha === 'string' ? sha.slice(0, n) : '--';
}

/** "sha256:abcdef.." -> "sha256:abcdef1234...9876" (middle elided). */
export function shortHash(h, head = 12, tail = 6) {
  if (typeof h !== 'string') return '--';
  const [prefix, body] = h.includes(':') ? [h.slice(0, h.indexOf(':') + 1), h.slice(h.indexOf(':') + 1)] : ['', h];
  if (body.length <= head + tail + 1) return h;
  return `${prefix}${body.slice(0, head)}…${body.slice(-tail)}`;
}

/** Wall clock seconds -> "20m 36s" / "1h 02m" / "12.3s". */
export function duration(s) {
  if (typeof s !== 'number') return '--';
  if (s < 60) return `${s.toFixed(1)}s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.round(s % 60);
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  return `${m}m ${String(sec).padStart(2, '0')}s`;
}

/** ISO timestamp -> "2026-08-07 06:14:09 UTC" (no locale dependence). */
export function ts(iso) {
  if (typeof iso !== 'string') return '--';
  const m = iso.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})/);
  return m ? `${m[1]} ${m[2]} UTC` : iso;
}

/** Iceberg snapshot ids are int64; render as plain digits, never grouped. */
export function snapshotId(v) {
  return v === null || v === undefined ? '--' : String(v);
}

/** Boolean -> yes/no. */
export function bool(v) {
  return v === true ? 'yes' : v === false ? 'no' : '--';
}

/** HTML-escape (used by the string-built SVG in charts.js and by card render). */
export function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Pretty JSON for the receipts drawer's model-params block. */
export function json(v) {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}
