// charts.js -- hand-rolled SVG, no chart library.
//
// The segment line chart is a direct port of the geometry, palette and
// CI-band treatment of src/batch_recsys_lab/eval/crossover_chart.py, so the
// on-site chart reads as the same object as the committed figure
// results/figures/crossover_test.svg:
//
//   * figure box 10.0 x 6.2 in  -> viewBox 1000 x 620 (1 pt = 1000/720 units)
//   * axes rect from subplots_adjust(left .085, right .775, top .84, bottom .22)
//   * y always from 0, top = max(ci_hi) * 1.22 ; x from -0.3 to nSeg-0.7
//   * CI bands as fills at alpha 0.13 under 2 pt lines (3 pt when highlighted)
//   * markers wear a 1 pt surface-coloured ring
//   * direct labels at the line ends, de-overlapped, drawn in ink (never in the
//     series colour -- the relief rule for the sub-3:1 palette slots)
//   * segment sizes as a muted n= row under the ticks
//
// SVG is built as markup and handed to innerHTML rather than through
// createElementNS: the HTML parser assigns the SVG namespace for us, so the SVG
// namespace URI -- the one URL a hand-built chart would otherwise have to
// contain -- never appears in this tree at all. The offline scanner finds zero
// URLs under demo/js/, and that is not a coincidence.

import { axisTick, esc } from './fmt.js';

// --- dataviz reference palette (light mode), verbatim from crossover_chart.py ---
export const COLORS = {
  SURFACE: '#fcfcfb',
  INK: '#0b0b0b',
  INK2: '#52514e',
  MUTED: '#898781',
  GRID: '#e1e0d9',
  BASELINE: '#c3c2b7',
};

/** Categorical slots in fixed order; assigned by model_order index, never re-ranked. */
export const SLOTS = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948', '#e87ba4', '#eb6834'];

/** points -> viewBox units (figure is 10 in = 720 pt wide, drawn at 1000 units). */
export const PT = 1000 / 720;

/** Same system stack as css/site.css; set on the svg root so the chart is self-contained. */
const SANS =
  '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, Noto Sans, sans-serif';

export const FIG = {
  w: 1000,
  h: 620,
  // matplotlib figure fractions; y is flipped when converted to SVG coordinates.
  left: 0.085,
  right: 0.775,
  top: 0.84,
  bottom: 0.22,
};

const PLOT = {
  x0: FIG.left * FIG.w,
  x1: FIG.right * FIG.w,
  y0: (1 - FIG.top) * FIG.h, // top edge in SVG coords
  y1: (1 - FIG.bottom) * FIG.h, // bottom edge in SVG coords
};

// ---------------------------------------------------------------- generic helpers

/** Linear scale factory. */
export function makeScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const fn = (v) => r0 + ((v - d0) / span) * (r1 - r0);
  fn.domain = domain;
  fn.range = range;
  return fn;
}

/**
 * MaxNLocator-compatible ticks (steps 1/2/2.5/5/10, at most `nbins` intervals),
 * so the y axis lands on the same 0.000 / 0.002 / ... grid as the matplotlib figure.
 */
export function niceTicks(max, nbins = 10, min = 0) {
  const span = max - min;
  if (!(span > 0)) return { ticks: [min], step: 1 };
  const mantissas = [1, 2, 2.5, 5, 10];
  let mag = Math.floor(Math.log10(span)) - 2;
  for (let guard = 0; guard < 40; guard += 1, mag += 1) {
    for (const m of mantissas) {
      const step = m * Math.pow(10, mag);
      if (Math.ceil(span / step) <= nbins) {
        const ticks = [];
        for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) {
          ticks.push(Number((Math.round(t / step) * step).toPrecision(15)));
        }
        return { ticks, step };
      }
    }
  }
  return { ticks: [min, max], step: span };
}

/** De-overlap direct-label y positions: preserve order, enforce min_gap. Port of _spread_labels. */
export function spreadLabels(ys, minGap) {
  const order = [...ys.keys()].sort((a, b) => ys[b] - ys[a]);
  const placed = [];
  const out = new Array(ys.length).fill(0);
  for (const i of order) {
    let y = ys[i];
    if (placed.length && y > placed[placed.length - 1] - minGap) {
      y = placed[placed.length - 1] - minGap;
    }
    placed.push(y);
    out[i] = y;
  }
  return out;
}

function attrs(o) {
  return Object.entries(o)
    .filter(([, v]) => v !== null && v !== undefined && v !== false)
    .map(([k, v]) => `${k}="${esc(v)}"`)
    .join(' ');
}

/**
 * Text node. `anchor` is start|middle|end; `baseline` is alphabetic|central|top.
 * "top" is emulated with a dy shift rather than dominant-baseline:hanging, which
 * is the more portable of the two.
 */
export function svgText(x, y, text, opts = {}) {
  const size = opts.size ?? 12;
  const yy = opts.baseline === 'top' ? y + size * 0.8 : y;
  return `<text ${attrs({
    x,
    y: yy,
    'text-anchor': opts.anchor || 'start',
    'dominant-baseline': opts.baseline === 'central' ? 'central' : null,
    'font-size': size,
    'font-weight': opts.weight || null,
    fill: opts.fill || COLORS.INK2,
    transform: opts.transform || null,
    class: opts.cls || null,
    'data-run-id': opts.runId || null,
    tabindex: opts.runId ? '0' : null,
    role: opts.runId ? 'button' : null,
  })}>${esc(text)}</text>`;
}

export function svgLine(x1, y1, x2, y2, o = {}) {
  return `<line ${attrs({
    x1,
    y1,
    x2,
    y2,
    stroke: o.stroke || COLORS.GRID,
    'stroke-width': o.width ?? 0.8 * PT,
    'stroke-dasharray': o.dash || null,
  })} />`;
}

export function svgRect(x, y, w, h, o = {}) {
  return `<rect ${attrs({
    x,
    y,
    width: w,
    height: h,
    rx: o.rx ?? 0,
    fill: o.fill || 'none',
    stroke: o.stroke || null,
    'stroke-width': o.width ?? null,
    'fill-opacity': o.fillOpacity ?? null,
  })} />`;
}

/** Rough advance width for the system sans stack; only used to size the legend box. */
function estWidth(text, size) {
  return String(text).length * size * 0.53;
}

// ------------------------------------------------------- the segment line chart

/**
 * spec = {
 *   segments: ["0","1-4",...],
 *   nUsers: [12866, ...] | null,
 *   nUsersRunId: "<run_id>" | null,
 *   series: [{ key, label, values[], ci_lo[], ci_hi[], color, highlight, dash,
 *              band=true, runId, directLabel=true }],
 *   yLabel, xLabel, title, subtitle,
 *   footnote: "…"           // small print under the axes (run ids etc.)
 *   footnoteRuns: [{ label, run_id }]
 * }
 * Returns SVG markup. Assign with element.innerHTML.
 */
export function segmentLineChart(spec) {
  const segments = spec.segments;
  const n = segments.length;
  const series = spec.series.filter((s) => Array.isArray(s.values) && s.values.length === n);
  if (!series.length) return '<p class="muted">No series to plot.</p>';

  const x = makeScale([-0.3, n - 0.7], [PLOT.x0, PLOT.x1]);
  const rawMax = Math.max(...series.map((s) => Math.max(...(s.band === false ? s.values : s.ci_hi || s.values))));
  const yTop = rawMax * 1.22;
  const y = makeScale([0, yTop], [PLOT.y1, PLOT.y0]);
  const { ticks, step } = niceTicks(yTop);

  const parts = [];

  // Surface + title block (figure-level, left-aligned with the axes).
  parts.push(svgRect(0, 0, FIG.w, FIG.h, { fill: COLORS.SURFACE }));
  if (spec.title) {
    parts.push(
      svgText(PLOT.x0, (1 - 0.955) * FIG.h, spec.title, {
        size: 15 * PT,
        weight: 'bold',
        fill: COLORS.INK,
        baseline: 'top',
      }),
    );
  }
  if (spec.subtitle) {
    parts.push(
      svgText(PLOT.x0, (1 - 0.905) * FIG.h, spec.subtitle, { size: 9.5 * PT, fill: COLORS.INK2, baseline: 'top' }),
    );
  }

  // Chrome: recessive solid hairline grid, baseline-weight left/bottom spines only.
  for (const t of ticks) {
    parts.push(svgLine(PLOT.x0, y(t), PLOT.x1, y(t), { stroke: COLORS.GRID, width: 0.8 * PT }));
  }
  parts.push(svgLine(PLOT.x0, PLOT.y0, PLOT.x0, PLOT.y1, { stroke: COLORS.BASELINE, width: 0.8 * PT }));
  parts.push(svgLine(PLOT.x0, PLOT.y1, PLOT.x1, PLOT.y1, { stroke: COLORS.BASELINE, width: 0.8 * PT }));

  // Marks: CI bands under the lines; highlighted series above the rest, and
  // anything with z=2 (the routed overlay) above that -- at the shipped n* the
  // routed line coincides exactly with blend, and a dashed line drawn on top is
  // how the viewer sees that rather than seeing nothing.
  const zOf = (s) => (s.z !== undefined ? s.z : s.highlight ? 1 : 0);
  const ordered = [...series].sort((a, b) => zOf(a) - zOf(b));
  for (const s of ordered) {
    if (s.band !== false && s.ci_lo && s.ci_hi) {
      const up = s.ci_hi.map((v, i) => `${x(i)},${y(v)}`);
      const dn = s.ci_lo.map((v, i) => `${x(i)},${y(v)}`).reverse();
      parts.push(
        `<polygon ${attrs({
          points: [...up, ...dn].join(' '),
          fill: s.color,
          'fill-opacity': 0.13,
          stroke: 'none',
        })} />`,
      );
    }
  }
  for (const s of ordered) {
    const pts = s.values.map((v, i) => `${x(i)},${y(v)}`).join(' ');
    parts.push(
      `<polyline ${attrs({
        points: pts,
        fill: 'none',
        stroke: s.color,
        'stroke-width': (s.highlight ? 3.0 : 2.0) * PT,
        'stroke-linejoin': 'round',
        'stroke-dasharray': s.dash || null,
        class: 'series-line',
        'data-run-id': s.runId || null,
      })} />`,
    );
    const r = ((s.highlight ? 6.5 : 5.0) / 2) * PT;
    for (let i = 0; i < s.values.length; i += 1) {
      parts.push(
        `<circle ${attrs({
          cx: x(i),
          cy: y(s.values[i]),
          r,
          fill: s.color,
          stroke: COLORS.SURFACE,
          'stroke-width': 1.0 * PT,
          class: 'series-dot',
          'data-run-id': s.runId || null,
        })}><title>${esc(`${s.label} · ${segments[i]}`)}</title></circle>`,
      );
    }
  }

  // Y ticks (labels only; matplotlib's 3 pt ticks + 3.5 pt pad).
  for (const t of ticks) {
    parts.push(
      svgText(PLOT.x0 - 6.5 * PT, y(t), axisTick(t, step), {
        size: 9 * PT,
        anchor: 'end',
        baseline: 'central',
        fill: COLORS.INK2,
      }),
    );
  }
  parts.push(
    svgText(0, 0, spec.yLabel || '', {
      size: 10 * PT,
      anchor: 'middle',
      fill: COLORS.INK2,
      transform: `translate(${PLOT.x0 - 46 * PT}, ${(PLOT.y0 + PLOT.y1) / 2}) rotate(-90)`,
    }),
  );

  // X ticks + the muted segment-size row + axis label.
  for (let i = 0; i < n; i += 1) {
    parts.push(
      svgText(x(i), PLOT.y1 + 6.5 * PT, segments[i], {
        size: 11 * PT,
        anchor: 'middle',
        baseline: 'top',
        fill: COLORS.INK2,
      }),
    );
  }
  if (spec.nUsers) {
    const yN = PLOT.y1 + 0.085 * (PLOT.y1 - PLOT.y0);
    for (let i = 0; i < n; i += 1) {
      if (spec.nUsers[i] === null || spec.nUsers[i] === undefined) continue;
      parts.push(
        svgText(x(i), yN, `n=${spec.nUsersFmt ? spec.nUsersFmt[i] : spec.nUsers[i]}`, {
          size: 8 * PT,
          anchor: 'middle',
          baseline: 'top',
          fill: COLORS.MUTED,
          cls: 'traced-svg',
          runId: spec.nUsersRunId || null,
        }),
      );
    }
  }
  if (spec.xLabel) {
    parts.push(
      svgText(x((n - 1) / 2), PLOT.y1 + 0.155 * (PLOT.y1 - PLOT.y0), spec.xLabel, {
        size: 10 * PT,
        anchor: 'middle',
        baseline: 'top',
        fill: COLORS.INK2,
      }),
    );
  }

  // Direct labels at the line ends, in ink; the coloured endpoint carries identity.
  const labelled = series.filter((s) => s.directLabel !== false);
  const ends = spreadLabels(
    labelled.map((s) => s.values[n - 1]),
    0.048 * yTop,
  );
  labelled.forEach((s, i) => {
    parts.push(
      svgText(x(n - 1 + 0.12), y(ends[i]), s.label, {
        size: 9 * PT,
        anchor: 'start',
        baseline: 'central',
        fill: s.highlight ? COLORS.INK : COLORS.INK2,
        weight: s.highlight ? 'bold' : null,
        cls: 'traced-svg',
        runId: s.runId || null,
      }),
    );
  });

  // Legend, center-left inside the axes, surface frame so lines stay legible under it.
  const lf = 8.5 * PT;
  const rowH = lf * 1.5;
  const handle = 1.6 * lf;
  const pad = 0.4 * lf;
  const boxW = pad * 2 + handle + 0.8 * lf + Math.max(...series.map((s) => estWidth(s.label, lf)));
  const boxH = pad * 2 + rowH * series.length;
  const bx = PLOT.x0 + 0.4 * lf;
  const by = (PLOT.y0 + PLOT.y1) / 2 - boxH / 2;
  parts.push(
    svgRect(bx, by, boxW, boxH, {
      fill: COLORS.SURFACE,
      stroke: COLORS.GRID,
      width: 0.8 * PT,
      rx: 3,
      fillOpacity: 0.95,
    }),
  );
  series.forEach((s, i) => {
    const ly = by + pad + rowH * (i + 0.5);
    parts.push(
      svgLine(bx + pad, ly, bx + pad + handle, ly, {
        stroke: s.color,
        width: (s.highlight ? 3.0 : 2.0) * PT,
        dash: s.dash || null,
      }),
    );
    parts.push(
      `<circle ${attrs({
        cx: bx + pad + handle / 2,
        cy: ly,
        r: ((s.highlight ? 6.5 : 5.0) / 2) * PT,
        fill: s.color,
        stroke: COLORS.SURFACE,
        'stroke-width': 1.0 * PT,
      })} />`,
    );
    parts.push(
      svgText(bx + pad + handle + 0.8 * lf, ly, s.label, {
        size: lf,
        baseline: 'central',
        fill: COLORS.INK2,
        cls: 'traced-svg',
        runId: s.runId || null,
      }),
    );
  });

  // Receipts small print (every number traces to the log).
  const foot = [];
  if (spec.footnote) foot.push({ text: spec.footnote, runId: null });
  for (const f of spec.footnoteRuns || []) foot.push({ text: `${f.label}: ${f.run_id}`, runId: f.run_id });
  let fy = (1 - 0.062) * FIG.h;
  let fx = PLOT.x0;
  // Width is estimated, not measured, so this deliberately over-estimates:
  // gaps between entries are acceptable, overlapping text is not. Callers with
  // many entries should render them as HTML below the chart instead.
  for (const f of foot) {
    const w = estWidth(f.text, 6.5 * PT) * 1.08;
    if (fx > PLOT.x0 && fx + w > FIG.w - 20) {
      fx = PLOT.x0;
      fy += 12.4;
    }
    parts.push(
      svgText(fx, fy, f.text, {
        size: 6.5 * PT,
        fill: COLORS.MUTED,
        cls: f.runId ? 'traced-svg' : null,
        runId: f.runId,
      }),
    );
    if (f.runId) {
      fx += w + 14;
    } else {
      fx = PLOT.x0;
      fy += 12.4;
    }
  }

  // font-family is set as an attribute, not left to the stylesheet: the chart
  // must render identically if the SVG is ever pulled out of the page.
  return `<svg class="chart" viewBox="0 0 ${FIG.w} ${
    FIG.h
  }" preserveAspectRatio="xMidYMid meet" font-family="${SANS}" role="img" aria-label="${esc(
    spec.title || 'segment chart',
  )}">${parts.join('')}</svg>`;
}
