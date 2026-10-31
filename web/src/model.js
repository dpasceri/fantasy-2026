// Port of guillotine/model.py: weekly score distributions fit from projected-vs-actual history, in this
// league's scoring. Marginals are empirical quantiles of actual / projected by position and projection
// bucket; dependence is a one-factor Gaussian copula by NFL team and game.
import { POS } from "./league.js";
import { Rng } from "./rng.js";
import { median, mean, ndtri, quantileSorted, searchsorted, sortNum, pyRound } from "./stats.js";

export const Q = 256;
export const LEVELS = Float64Array.from({ length: Q }, (_, i) => i / (Q - 1));
export const MIN_PROJ = 0.5;
const CORR_MIN_PROJ = 3.0; // only fantasy-relevant players enter the correlation fit

/** [pos, proj, actual, team key, game key] for every calibration player-week (last season + this one so far). */
export function calibrationRows(lg) {
  const rows = [];
  const sets = [[lg.raw.proj.calib, lg.raw.stats.calib, lg.meta.calib_season, 18], [lg.raw.proj.season, lg.raw.stats.season, lg.season, lg.leg - 1]];
  for (const [P, S, season, last] of sets) {
    for (let w = 1; w <= last; w++) {
      const pw = P[w] || {}, sw = S[w] || {};
      for (const [pid, x] of Object.entries(pw)) {
        const p = lg.pts(x.stats);
        if (p < MIN_PROJ || !x.game) continue;
        const s = sw[pid];
        rows.push([x.pos, p, s ? lg.pts(s.stats) : 0.0, `${season}-${w}-${x.team}`, `${season}-${w}-${x.game}`]);
      }
    }
  }
  return rows;
}

export function fit(lg, seed = 7) {
  const rng = new Rng(seed);
  const rows = calibrationRows(lg);
  const n = rows.length;
  const z = new Float64Array(n);
  const out = { centers: {}, tables: {} }, report = {};
  for (const P of POS) {
    const m = [];
    for (let i = 0; i < n; i++) if (rows[i][0] === P) m.push(i);
    if (!m.length) throw new Error(`No ${P} calibration data`);
    const nb = Math.min(14, Math.max(5, Math.floor(m.length / 300)));
    const projSorted = sortNum(m.map((i) => rows[i][1]));
    const edges = Array.from({ length: nb + 1 }, (_, k) => quantileSorted(projSorted, k / nb));
    const buckets = Array.from({ length: nb }, () => []);
    for (const i of m) buckets[Math.min(nb - 1, Math.max(0, searchsorted(edges, rows[i][1], "right") - 1))].push(i);
    const centers = [], tables = [], rep = [];
    for (const idx of buckets) {
      const r = sortNum(idx.map((i) => rows[i][2] / rows[i][1]));
      centers.push(median(idx.map((i) => rows[i][1])));
      tables.push(Float64Array.from(LEVELS, (q) => quantileSorted(r, q)));
      // randomized probability integral transform (handles the point mass at 0)
      for (const i of idx) {
        const x = rows[i][2] / rows[i][1];
        const lo = searchsorted(r, x, "left"), hi = searchsorted(r, x, "right");
        const u = (lo + rng.random() * (hi - lo)) / r.length;
        z[i] = ndtri(Math.min(Math.max(u, 1e-6), 1 - 1e-6));
      }
      const a = idx.map((i) => rows[i][2]);
      const as = sortNum(a);
      const sd = Math.sqrt(mean(a.map((v) => v * v)) - mean(a) ** 2);
      rep.push({ proj: pyRound(median(idx.map((i) => rows[i][1])), 1), n: idx.length, mean_actual: pyRound(mean(a), 2),
        sd: pyRound(sd, 2), p_under_half: pyRound(idx.filter((i) => rows[i][2] < 0.5 * rows[i][1]).length / idx.length, 3),
        p10: pyRound(quantileSorted(as, 0.1), 1), p90: pyRound(quantileSorted(as, 0.9), 1) });
    }
    out.centers[P] = Float64Array.from(centers);
    out.tables[P] = tables;
    report[P] = rep;
  }

  // residual correlations: same NFL team, and opposing teams in the same game
  const byTeam = new Map(), byGame = new Map();
  for (let i = 0; i < n; i++) {
    if (rows[i][1] < CORR_MIN_PROJ) continue;
    push(byTeam, rows[i][3], i);
    push(byGame, rows[i][4], i);
  }
  const same = new Map(), cross = new Map();
  const add = (M, i, j) => {
    const key = [rows[i][0], rows[j][0]].sort().join("-");
    const s = M.get(key) || [0, 0];
    s[0] += z[i] * z[j];
    s[1] += 1;
    M.set(key, s);
  };
  for (const g of byTeam.values()) for (let x = 0; x < g.length; x++) for (let y = x + 1; y < g.length; y++) add(same, g[x], g[y]);
  for (const g of byGame.values())
    for (let x = 0; x < g.length; x++) for (let y = x + 1; y < g.length; y++) if (rows[g[x]][3] !== rows[g[y]][3]) add(cross, g[x], g[y]);
  const pi = Object.fromEntries(POS.map((p, k) => [p, k]));
  const pairs = [...same].filter(([k, v]) => v[1] >= 50 && k !== "QB-QB").map(([k, v]) => [k.split("-"), v[0] / v[1], v[1]]);
  const a = fitLoadings(pairs.map(([k, c, w]) => [pi[k[0]], pi[k[1]], c, w]));
  let num = 0, den = 0;
  for (const [k, [sc, m]] of cross) {
    if (m < 50) continue;
    const [p, q] = k.split("-"), ab = a[pi[p]] * a[pi[q]];
    num += m * (sc / m) * ab;
    den += m * ab * ab;
  }
  const g = den > 0 ? Math.min(1, Math.max(0, num / den)) : 0;
  out.loadings = Object.fromEntries(POS.map((p) => [p, a[pi[p]]]));
  out.game_share = g;
  report.copula = {
    loadings: Object.fromEntries(POS.map((p) => [p, pyRound(a[pi[p]], 3)])),
    game_share: pyRound(g, 3),
    same_team_corr: Object.fromEntries(pairs.map(([k, c]) => [k.join("-"), pyRound(c, 3)]).sort((x, y) => (x[0] < y[0] ? -1 : 1))),
    opp_team_corr: Object.fromEntries([...cross].filter(([, v]) => v[1] >= 50).sort((x, y) => (x[0] < y[0] ? -1 : 1))
      .map(([k, v]) => [k, pyRound(v[0] / v[1], 3)])),
    n_obs: n,
  };
  return { model: out, report };
}

function push(map, key, v) {
  const a = map.get(key);
  if (a) a.push(v);
  else map.set(key, [v]);
}

/**
 * Loadings a (4, bounded to [0, 0.95]) minimizing sum w * (c - a_i * a_j)^2 over same-team pairs
 * (scipy's least_squares with bounds in model.py). Projected Gauss-Newton with Levenberg damping.
 */
export function fitLoadings(pairs) {
  const clip = (v) => Math.min(0.95, Math.max(0, v));
  const loss = (a) => pairs.reduce((s, [i, j, c, w]) => s + w * (c - a[i] * a[j]) ** 2, 0);
  let a = [0.3, 0.3, 0.3, 0.3], f = loss(a), lam = 1e-3;
  for (let it = 0; it < 200; it++) {
    const JtJ = Array.from({ length: 4 }, () => new Float64Array(4)), Jtr = new Float64Array(4);
    for (const [i, j, c, w] of pairs) {
      const r = Math.sqrt(w) * (c - a[i] * a[j]);
      const J = new Float64Array(4);
      J[i] -= Math.sqrt(w) * a[j];
      J[j] -= Math.sqrt(w) * a[i];
      for (let p = 0; p < 4; p++) {
        Jtr[p] += J[p] * r;
        for (let q = 0; q < 4; q++) JtJ[p][q] += J[p] * J[q];
      }
    }
    // a loading pinned at a bound with the gradient pushing outward stays put; solve for the rest
    const fixed = a.map((v, p) => (v >= 0.95 && Jtr[p] < 0) || (v <= 0 && Jtr[p] > 0));
    let improved = false;
    while (lam < 1e10) {
      const A = JtJ.map((row, p) => row.map((v, q) => (fixed[p] || fixed[q] ? +(p === q) : v + (p === q ? lam * (v + 1e-9) : 0))));
      const step = solve4(A, Jtr.map((v, p) => (fixed[p] ? 0 : v)));
      const b = a.map((v, p) => clip(v - step[p]));
      const fb = loss(b);
      if (fb < f) {
        const done = f - fb < 1e-15 * (1 + f);
        a = b; f = fb; lam = Math.max(lam / 10, 1e-12); improved = true;
        if (done) return a;
        break;
      }
      lam *= 10;
    }
    if (!improved) break;
  }
  return a;
}

// Gaussian elimination with partial pivoting for the damped 4x4 normal equations.
function solve4(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-300) continue;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = Math.abs(M[r][r]) < 1e-300 ? 0 : s / M[r][r];
  }
  return x;
}

export class PlayerModel {
  constructor(model) {
    this.centers = model.centers;
    this.logCenters = Object.fromEntries(POS.map((p) => [p, model.centers[p].map(Math.log)]));
    this.tables = model.tables;
    this.loading = model.loadings;
    this.game_share = model.game_share;
  }

  /** Score quantiles at LEVELS for a player with this projection, written into out (length Q). */
  quantiles(pos, proj, out = new Float64Array(Q)) {
    if (proj <= 0 || !(pos in this.centers)) return out.fill(0);
    const lc = this.logCenters[pos], t = this.tables[pos], x = Math.log(proj);
    if (x <= lc[0]) for (let q = 0; q < Q; q++) out[q] = proj * t[0][q];
    else if (x >= lc[lc.length - 1]) for (let q = 0; q < Q; q++) out[q] = proj * t[t.length - 1][q];
    else {
      const k = searchsorted(lc, x) - 1, wgt = (x - lc[k]) / (lc[k + 1] - lc[k]);
      for (let q = 0; q < Q; q++) out[q] = proj * ((1 - wgt) * t[k][q] + wgt * t[k + 1][q]);
    }
    return out;
  }
}
