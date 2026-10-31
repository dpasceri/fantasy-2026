// Port of guillotine/sim.py: the Monte Carlo engine.
//
// One sim draws every relevant player's week from the fitted distributions (copula-correlated by NFL team and
// game), plays the lineup each manager would set from projections (bench subs cover inactive starters), chops
// the lowest alive score, and repeats through the final week. Every variant of my roster or lineup is scored
// on the same draws (common random numbers). Python vectorizes over sims; here each sim is a loop, which is
// what JS is fast at. The Universe is built on the main thread; workers get its plain-data form (`U.data()`).
import { bestLineup, eligible, P_OUT, POS } from "./league.js";
import { Q } from "./model.js";
import { Rng } from "./rng.js";
import { ndtr } from "./stats.js";

export const LEVEL_STEP = 0.10; // weekly random-walk sd (log scale) of a player's true level vs today's projection
// Absences after this week (sim.py has the derivation): weekly hazard, share season-long, mean length otherwise
const INJ_HAZARD = { QB: 0.086, other: 0.064 };
const INJ_LONG = { QB: 0.20, other: 0.126 };
const INJ_MEAN_WEEKS = { QB: 1.45, other: 2.07 };
const INJ_LONG_WEEKS = 99;
export const CHUNK = 8000;

// ndtr(z) * (Q - 1) by table lookup on [-8.5, 8.5] in steps of 1/256 (linear interpolation error < 5e-7 in
// probability, far finer than the Q-level quantile tables it indexes)
const NT_LO = -8.5, NT_STEP = 256, NT = Float64Array.from({ length: 17 * NT_STEP + 2 }, (_, i) => ndtr(NT_LO + i / NT_STEP) * (Q - 1));
const NT_MAX = NT.length - 2;
function uOf(z) {
  const x = (z - NT_LO) * NT_STEP;
  if (x <= 0) return 0;
  if (x >= NT_MAX) return Q - 1;
  const i = x | 0;
  return NT[i] + (x - i) * (NT[i + 1] - NT[i]);
}

/** Every player the sims touch, with per-week projections, quantile tables and NFL team/game ids. */
export class Universe {
  constructor(lg, pm, pids, weeks) {
    this.lg = lg;
    this.pm = pm;
    this.pids = [...new Set(pids)];
    this.ix = new Map(this.pids.map((p, i) => [p, i]));
    this.weeks = [...weeks];
    const n = (this.n = this.pids.length), W = this.weeks.length;
    this.pos = this.pids.map((p) => lg.pos(p));
    this.proj = Array.from({ length: W }, () => new Float64Array(n));
    this.qflat = Array.from({ length: W }, () => new Float32Array(n * Q));
    this.team = Array.from({ length: W }, () => new Int32Array(n));
    this.game = Array.from({ length: W }, () => new Int32Array(n));
    this.n_team = [];
    this.n_game = [];
    this.load = Float32Array.from(this.pos, (p) => pm.loading[p] || 0);
    this.cload = Float32Array.from(this.load, (a) => Math.sqrt(1 - a * a));
    this.p_out = Float32Array.from(this.pids, (p) => {
      const t = (lg.info[p] || {}).inj || "";
      return t === "Questionable" || t === "Doubtful" ? P_OUT[t] : 0;
    });
    // week 0 under way: players whose game has kicked off are locked (in or out of the lineup) and carry their
    // points so far; their score is that plus the unplayed share of a full-game draw
    this.locked = new Set();
    this.starters0 = new Map();
    this.started = new Uint8Array(n);
    this.frac_left = new Float32Array(n).fill(1);
    this.pts0 = new Float32Array(n);
    if (lg.live && W && this.weeks[0] === lg.week) {
      this.pids.forEach((p, i) => {
        const [s, f, a] = lg.gameState(p);
        this.started[i] = s !== "pre" ? 1 : 0;
        this.frac_left[i] = f;
        this.pts0[i] = a;
        if (s !== "pre") this.locked.add(p);
      });
      for (const rid of lg.rosters.keys()) this.starters0.set(rid, lg.starters(rid, lg.week));
    }
    const grp = this.pos.map((p) => (p === "QB" ? "QB" : "other"));
    this.haz = Float32Array.from(grp, (g) => INJ_HAZARD[g]);
    this.p_long = Float64Array.from(grp, (g) => INJ_LONG[g]);
    this.p_geo = Float64Array.from(grp, (g) => 1 / INJ_MEAN_WEEKS[g]);
    const qbuf = new Float64Array(Q);
    this.weeks.forEach((w, wi) => {
      const teams = new Map(), games = new Map();
      this.pids.forEach((p, i) => {
        const x = lg.proj[w][p] || {};
        const pr = lg.projPts(p, w);
        this.proj[wi][i] = pr;
        this.qflat[wi].set(pm.quantiles(this.pos[i], pr, qbuf), i * Q);
        const tk = x.team || `_${i}`, gk = x.game || `_${i}`;
        if (!teams.has(tk)) teams.set(tk, teams.size);
        if (!games.has(gk)) games.set(gk, games.size);
        this.team[wi][i] = teams.get(tk);
        this.game[wi][i] = games.get(gk);
      });
      this.n_team.push(teams.size);
      this.n_game.push(games.size);
    });
  }

  /** What a worker needs to draw paths: typed arrays and numbers only. */
  data() {
    const { n, weeks, qflat, team, game, n_team, n_game, load, cload, p_out, started, frac_left, pts0, haz, p_long, p_geo, proj } = this;
    return { n, W: weeks.length, game_share: this.pm.game_share, anyLocked: this.locked.size > 0,
      qflat, team, game, n_team, n_game, load, cload, p_out, started, frac_left, pts0, haz, p_long, p_geo, proj };
  }

  /**
   * Lineup index rows for a team: Int32Array(3 * S) = starters, first and second bench fallback by slot (n = empty).
   * With `starters` (the team's Sleeper lineup) in a week under way, locked starters keep their slot, locked bench
   * players stay out, and the open slots and their fallbacks go to the best unlocked players.
   */
  lineupIdx(wi, roster, starters = null) {
    const slots = this.lg.slots, S = slots.length;
    const lock = wi === 0 && starters != null ? this.locked : new Set();
    const st = starters || [];
    const fixed = Array.from({ length: S }, (_, k) => (st[k] && lock.has(st[k]) && this.ix.has(st[k]) ? st[k] : null));
    const free = fixed.map((p, k) => (p === null ? k : -1)).filter((k) => k >= 0);
    const cands = roster.filter((p) => this.ix.has(p) && !lock.has(p)).map((p) => [this.proj[wi][this.ix.get(p)], p, this.pos[this.ix.get(p)]]);
    const [lu, subs] = bestLineup(free.map((k) => slots[k]), cands, true);
    const out = new Int32Array(3 * S).fill(this.n);
    fixed.forEach((p, k) => { if (p) out[k] = this.ix.get(p); });
    lu.forEach(([s, p], j) => {
      const k = free.splice(free.findIndex((k) => slots[k] === s), 1)[0];
      if (p) out[k] = this.ix.get(p);
      if (subs[j].length > 0) out[S + k] = this.ix.get(subs[j][0]);
      if (subs[j].length > 1) out[2 * S + k] = this.ix.get(subs[j][1]);
    });
    return out;
  }

  /** Index rows for a hand-picked starting lineup (subs = best eligible bench). */
  fixedLineupIdx(wi, starters, roster) {
    const slots = this.lg.slots, S = slots.length;
    const val = (p) => this.proj[wi][this.ix.get(p)];
    const posOf = (p) => this.pos[this.ix.get(p)];
    const left = [...starters].sort((a, b) => val(b) - val(a));
    const assign = new Array(S).fill(null);
    const take = (k, pred) => {
      const j = left.findIndex(pred);
      if (j >= 0) assign[k] = left.splice(j, 1)[0];
    };
    slots.forEach((s, k) => { if (POS.includes(s)) take(k, (p) => posOf(p) === s); }); // fixed positions first
    slots.forEach((s, k) => { if (assign[k] === null) take(k, (p) => eligible(s, posOf(p))); });
    const lock = wi === 0 ? this.locked : new Set(); // a bench player whose game has started can't come in
    const bench = roster.filter((p) => this.ix.has(p) && !starters.includes(p) && !lock.has(p) && val(p) > 0).sort((a, b) => val(b) - val(a));
    const out = new Int32Array(3 * S).fill(this.n);
    slots.forEach((s, k) => {
      if (assign[k]) out[k] = this.ix.get(assign[k]);
      const b = bench.filter((p) => eligible(s, posOf(p)));
      if (b.length > 0) out[S + k] = this.ix.get(b[0]);
      if (b.length > 1) out[2 * S + k] = this.ix.get(b[1]);
    });
    return out;
  }
}

/** Per-sim path state across weeks: drift of each player's level and weeks left on an absence. */
export function pathState(n) {
  return { loglev: new Float32Array(n), outLeft: new Int16Array(n) };
}

/** Draw week wi of one sim into score/avail (length n + 1; the last entry is an empty slot). */
export function drawWeek(U, wi, rng, st, score, avail, G, T) {
  const n = U.n, g = U.game_share, sg = Math.sqrt(g), st1 = Math.sqrt(1 - g);
  const ng = U.n_game[wi], nt = U.n_team[wi], game = U.game[wi], team = U.team[wi], q = U.qflat[wi];
  const load = U.load, cload = U.cload, loglev = st.loglev, outLeft = st.outLeft;
  for (let j = 0; j < ng; j++) G[j] = sg * rng.normal();
  for (let j = 0; j < nt; j++) T[j] = st1 * rng.normal();
  const drift = wi > 0 ? -0.5 * LEVEL_STEP * LEVEL_STEP * wi : 0;
  for (let i = 0; i < n; i++) {
    const u = uOf(load[i] * (G[game[i]] + T[team[i]]) + cload[i] * rng.normal());
    const i0 = u >= Q - 2 ? Q - 2 : u | 0, fr = u - i0, b = i * Q + i0;
    let s = q[b] + fr * (q[b + 1] - q[b]);
    let a;
    if (wi === 0) {
      a = rng.random() >= U.p_out[i];
      if (U.anyLocked && U.started[i]) {
        s = U.pts0[i] + U.frac_left[i] * s;
        a = true;
      }
    } else {
      loglev[i] += rng.normal() * LEVEL_STEP;
      s *= Math.exp(loglev[i] + drift);
      if (outLeft[i] === 0 && rng.random() < U.haz[i])
        outLeft[i] = rng.random() < U.p_long[i] ? INJ_LONG_WEEKS : rng.geometric(U.p_geo[i]);
      a = outLeft[i] === 0;
      if (outLeft[i] > 0) outLeft[i]--;
    }
    score[i] = s;
    avail[i] = a ? 1 : 0;
  }
  score[n] = 0;
  avail[n] = 0;
}

/** Team total for one lineup: idx is Int32Array(3 * S) at offset off. */
export function teamScore(score, avail, idx, off, S) {
  let tot = 0;
  for (let k = 0; k < S; k++) {
    const a = idx[off + k], b1 = idx[off + S + k], b2 = idx[off + 2 * S + k];
    tot += avail[a] ? score[a] : avail[b1] ? score[b1] : avail[b2] ? score[b2] : 0;
  }
  return tot;
}

function scratch(U) {
  const ng = Math.max(1, ...U.n_game), nt = Math.max(1, ...U.n_team);
  return { score: new Float32Array(U.n + 1), avail: new Uint8Array(U.n + 1), G: new Float64Array(ng), T: new Float64Array(nt) };
}

/**
 * Season / elimination sims for one chunk. plan: {S, T, V, me_t, wi0, alive0: Uint8Array(V*T), mineOf: {wi: variant -> row},
 * opp: {wi: Int32Array(T*3*S)}, mine: {wi: Int32Array(rows*3*S)}}. Returns sums to be added across chunks.
 */
export function seasonChunk(U, plan, ci, C, seed) {
  const { S, T, V, me_t, wi0 } = plan, W = U.W, L = 3 * S;
  const rng = new Rng(seed, ci);
  const chops = new Float64Array(V * T * W), wins = new Float64Array(V * T);
  const lineHist = new Float64Array(W * 400), myHist = new Float64Array(W * 400);
  const { score, avail, G, T: TT } = scratch(U);
  const alive = new Uint8Array(V * T), opp = new Float64Array(T), mine = new Float64Array(V), uniq = new Float64Array(V);
  for (let c = 0; c < C; c++) {
    alive.set(plan.alive0);
    const st = pathState(U.n);
    for (let wi = wi0; wi < W; wi++) {
      let nAlive = 0;
      for (let t = 0; t < T; t++) nAlive += alive[t];
      if (nAlive <= 1) break;
      drawWeek(U, wi, rng, st, score, avail, G, TT);
      const oi = plan.opp[wi], mi = plan.mine[wi], of = plan.mineOf[wi], U1 = mi.length / L;
      for (let t = 0; t < T; t++) opp[t] = teamScore(score, avail, oi, t * L, S);
      for (let u = 0; u < U1; u++) uniq[u] = teamScore(score, avail, mi, u * L, S);
      for (let v = 0; v < V; v++) mine[v] = uniq[of[v]];
      const wasAlive = alive[me_t] === 1; // variant 0, going into this week
      for (let v = 0; v < V; v++) {
        let k = -1, lo = Infinity;
        for (let t = 0; t < T; t++) {
          if (!alive[v * T + t]) continue;
          const x = t === me_t ? mine[v] : opp[t];
          if (x < lo || k < 0) { lo = x; k = t; }
        }
        alive[v * T + k] = 0;
        chops[(v * T + k) * W + wi]++;
        if (v === 0 && wasAlive) {
          let line = Infinity;
          for (let t = 0; t < T; t++) if (t !== me_t && alive[t] | (t === k)) line = Math.min(line, opp[t]);
          lineHist[wi * 400 + bin(line)]++;
          myHist[wi * 400 + bin(mine[0])]++;
        }
      }
    }
    for (let v = 0; v < V; v++) {
      let w = 0;
      while (w < T - 1 && !alive[v * T + w]) w++;
      wins[v * T + w]++;
    }
  }
  return { chops, wins, lineHist, myHist };
}

const bin = (x) => Math.trunc(Math.min(Math.max(x * 2, 0), 399));

/** One-week lineup sims: P(my lineup scores below the lowest rival), mean and sum of squares per lineup. */
export function weekChunk(U, job, ci, C, seed) {
  const { S, T, V, me_t, opp: oi, mine: mi } = job, L = 3 * S;
  const rng = new Rng(seed, ci);
  const chopped = new Float64Array(V), s1 = new Float64Array(V), s2 = new Float64Array(V);
  const { score, avail, G, T: TT } = scratch(U);
  const st = pathState(U.n);
  for (let c = 0; c < C; c++) {
    drawWeek(U, 0, rng, st, score, avail, G, TT);
    let line = Infinity;
    for (let t = 0; t < T; t++) if (t !== me_t) line = Math.min(line, teamScore(score, avail, oi, t * L, S));
    for (let v = 0; v < V; v++) {
      const m = teamScore(score, avail, mi, v * L, S);
      if (m < line) chopped[v]++;
      s1[v] += m;
      s2[v] += m * m;
    }
  }
  return { chopped, s1, s2 };
}

/**
 * Every legal starting lineup from players projected > 0 (by position counts). In a week under way, `starters`
 * (my Sleeper lineup) pins it: every locked starter is in, no locked bench player is.
 */
export function enumerateLineups(U, wi, roster, slots, starters = null) {
  const lock = wi === 0 && starters != null ? U.locked : new Set();
  const must = new Set((starters || []).filter((p) => p && lock.has(p)));
  const pool = roster.filter((p) => U.ix.has(p) && (must.has(p) || (U.proj[wi][U.ix.get(p)] > 0 && !lock.has(p))));
  const need = Object.fromEntries(POS.map((p) => [p, slots.filter((s) => s === p).length]));
  const nFlex = slots.filter((s) => !POS.includes(s)).length;
  const rest = pool.filter((p) => !must.has(p));
  const mustList = pool.filter((p) => must.has(p));
  const out = [];
  for (const combo of combinations(rest, slots.length - mustList.length)) {
    const lu = [...mustList, ...combo];
    const c = { QB: 0, RB: 0, WR: 0, TE: 0 };
    for (const p of lu) c[U.pos[U.ix.get(p)]]++;
    if (c.QB !== need.QB || ["RB", "WR", "TE"].some((p) => c[p] < need[p])) continue;
    if (["RB", "WR", "TE"].reduce((s, p) => s + c[p] - need[p], 0) !== nFlex) continue;
    if (!flexFits(lu.map((p) => U.pos[U.ix.get(p)]), slots)) continue;
    out.push(lu);
  }
  return out;
}

// Python checks only position counts, which is exact for FLEX-only leagues; with REC_FLEX or WRRB_FLEX the
// extras must also fit the flex slots they'd go in.
function flexFits(posList, slots) {
  const lu = bestLineup(slots, posList.map((p, i) => [1, i, p]));
  return lu.every(([, p]) => p !== null);
}

function* combinations(arr, k) {
  const n = arr.length;
  if (k < 0 || k > n) return;
  const idx = Array.from({ length: k }, (_, i) => i);
  while (true) {
    yield idx.map((i) => arr[i]);
    let i = k - 1;
    while (i >= 0 && idx[i] === n - k + i) i--;
    if (i < 0) return;
    idx[i]++;
    for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
  }
}
