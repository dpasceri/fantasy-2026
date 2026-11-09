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
import { ndtr, ndtri0 } from "./stats.js";

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

/**
 * Draw week wi of one sim into score/avail (length n + 1; the last entry is an empty slot). The draws happen in
 * one fixed order: game factors, team factors, then per player. The per-player loops live in their own functions
 * so each stays within V8's inlining budget and the random-number methods are inlined at every call site.
 */
export function drawWeek(U, wi, rng, st, score, avail, G, T) {
  const n = U.n, g = U.game_share, sg = Math.sqrt(g), st1 = Math.sqrt(1 - g), ng = U.n_game[wi], nt = U.n_team[wi];
  for (let j = 0; j < ng + nt; j++) {
    const z = rng.normal();
    if (j < ng) G[j] = sg * z;
    else T[j - ng] = st1 * z;
  }
  if (wi === 0) drawFirst(U, rng, score, avail, G, T);
  else drawLater(U, wi, rng, st, score, avail, G, T);
  score[n] = 0;
  avail[n] = 0;
}

// This week: known injury tags decide who sits; a game under way carries its points so far.
function drawFirst(U, rng, score, avail, G, T) {
  const n = U.n, game = U.game[0], team = U.team[0], q = U.qflat[0], load = U.load, cload = U.cload;
  const pOut = U.p_out, locked = U.anyLocked, started = U.started, pts0 = U.pts0, fracLeft = U.frac_left;
  for (let i = 0; i < n; i++) {
    const u = uOf(load[i] * (G[game[i]] + T[team[i]]) + cload[i] * rng.normal());
    const i0 = u >= Q - 2 ? Q - 2 : u | 0, fr = u - i0, b = i * Q + i0;
    let s = q[b] + fr * (q[b + 1] - q[b]);
    let a = rng.random() >= pOut[i];
    if (locked && started[i]) {
      s = pts0[i] + fracLeft[i] * s;
      a = true;
    }
    score[i] = s;
    avail[i] = a ? 1 : 0;
  }
}

// Weeks out for a new absence: season-long, or geometric. Out of the hot loop, so it isn't inlined there.
function absence(rng, pLong, pGeo) {
  return rng.random() < pLong ? INJ_LONG_WEEKS : rng.geometric(pGeo);
}

// Later weeks: the level drifts, and new absences start and run down.
function drawLater(U, wi, rng, st, score, avail, G, T) {
  const n = U.n, game = U.game[wi], team = U.team[wi], q = U.qflat[wi], load = U.load, cload = U.cload;
  const loglev = st.loglev, outLeft = st.outLeft, haz = U.haz, pLong = U.p_long, pGeo = U.p_geo;
  const drift = -0.5 * LEVEL_STEP * LEVEL_STEP * wi;
  for (let i = 0; i < n; i++) {
    const u = uOf(load[i] * (G[game[i]] + T[team[i]]) + cload[i] * rng.normal());
    const i0 = u >= Q - 2 ? Q - 2 : u | 0, fr = u - i0, b = i * Q + i0;
    const s = q[b] + fr * (q[b + 1] - q[b]);
    const lv = Math.fround(loglev[i] + rng.normal() * LEVEL_STEP); // rounds as the Float32Array store does
    loglev[i] = lv;
    let o = outLeft[i];
    if (o === 0 && rng.random() < haz[i]) o = absence(rng, pLong[i], pGeo[i]);
    score[i] = s * Math.exp(lv + drift);
    avail[i] = o === 0 ? 1 : 0;
    outLeft[i] = o > 0 ? o - 1 : o;
  }
}

/**
 * Totals for a set of lineup rows that share slot triples (see run.js's slotTriples): each distinct
 * (starter, sub, sub) is scored once into tv, then every row adds its S slots in slot order, as teamScore does.
 */
function rowScores(score, avail, trip, of, S, tv, out, R) {
  const nTrip = trip.length / 3;
  for (let j = 0; j < nTrip; j++) {
    const a = trip[3 * j], b1 = trip[3 * j + 1], b2 = trip[3 * j + 2];
    tv[j] = avail[a] ? score[a] : avail[b1] ? score[b1] : avail[b2] ? score[b2] : 0;
  }
  for (let u = 0; u < R; u++) {
    let tot = 0;
    for (let k = 0; k < S; k++) tot += tv[of[u * S + k]];
    out[u] = tot;
  }
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

// ---- Season sims: stratified groups of paths, recombined
//
// Given a season's game and NFL-team factors, every player's path (scores, level drift, absences) is independent
// of every other player's, and so every fantasy team's season is independent of every other team's. The season
// sims lean on that twice. Paths are drawn in groups of GROUP that share one draw of the factors, and
//   1. the group's team seasons are recombined: each team takes its season from any path of the group (ROUNDS
//      rounds of random per-team shifts), which is GROUP * ROUNDS seasons with the exact joint law for the price
//      of GROUP draws;
//   2. each variant of my roster on each path is played against the rivals of every path of the group, so a
//      pickup's value is averaged over GROUP pairings per path, not one.
// Both are product-form estimators (Kuntz, Crucinio & Johansen 2022) applied given the factors, and unbiased for
// any group size. On top, the draws are stratified, which leaves every single path's law exact as well: Latin
// hypercube samples of the factors over the groups of a job, and of each player's weekly score noise, injury
// tag and new absences over the paths of a group. Together a season path is worth about three plain ones for
// chop and win odds and four or more for the value of a pickup (test/season.test.js checks the engine against
// plain Monte Carlo, and its precision).
// This week's own numbers (who is chopped now, the score to beat) don't come from here: run.js takes them from plain
// sims of the week alone (week0Chunk), as it always has. With few games left in a week under way, what is still
// random is mostly those games' factors, which a group shares, so grouping gains nothing there and can lose; for the
// same reason the season then runs in smaller groups (GROUP_LIVE). run.js also measures, on every run, what the
// paths were worth, and draws more if that falls short (seasonPrecision).
export const GROUP = 16; // paths per group; a power of two, at most 32
export const GROUP_LIVE = 8; // ... in a week under way
export const ROUNDS = 32; // recombined seasons per path, for every team's odds
export const GROUPS = 250; // groups per job
const N_PERM = 1024; // stratum orders are drawn from this many random permutations, each with its own random shift

/**
 * Worker-side state for seasonGroups over plans that share their teams: per week, the distinct slot triples of
 * every lineup row (the T teams' rows, then each plan's rows for my variants) and the players they touch.
 * A plan is {S, T, V, me_t, wi0, alive0: Uint8Array(V*T), opp: {wi: Int32Array(T*3*S)}, mine: {wi: rows},
 * mineOf: {wi: variant -> row}}; variants with the same alive0 are one case (the same rivals).
 */
export function prepSeason(U, plans, K = GROUP) {
  const { S, T, me_t } = plans[0], W = U.W, L = 3 * S, n = U.n, n1 = n + 1;
  const wLo = Math.min(...plans.map((p) => p.wi0));
  const E = { U, plans, K, S, T, W, L, me_t, wLo, trip: [], of: [], nRow: [], rowOff: [], tb: [], used: [], live: [], delta: [] };
  let nTot = 0; // lineup totals live in one array: week w's row u on path k is tot[tb[w] + u * K + k]
  for (let w = 0; w < W; w++) {
    const rows = w >= wLo ? [plans[0].opp[w]] : [new Int32Array(0)], off = [];
    let nRow = rows[0].length / L;
    for (const p of plans) {
      off.push(nRow);
      if (w >= p.wi0) {
        rows.push(p.mine[w]);
        nRow += p.mine[w].length / L;
      }
    }
    const seen = new Map(), trip = [], of = new Int32Array(nRow * S), used = new Uint8Array(n1);
    let u = 0;
    for (const r of rows)
      for (let o = 0; o < r.length; o += L, u++)
        for (let k = 0; k < S; k++) {
          const a = r[o + k], b1 = r[o + S + k], b2 = r[o + 2 * S + k], key = (a * n1 + b1) * n1 + b2;
          let j = seen.get(key);
          if (j === undefined) {
            seen.set(key, (j = trip.length / 3));
            trip.push(a, b1, b2);
            used[a] = used[b1] = used[b2] = 1;
          }
          of[u * S + k] = j;
        }
    E.trip.push(Int32Array.from(trip));
    E.of.push(of);
    E.nRow.push(nRow);
    // a variant's row differs from its plan's first row (my roster as it is) in a few slots, so its total is
    // that row's plus the changed slots' differences: delta = [row, first row, n, (old triple, new triple) * n, ...]
    const delta = [];
    plans.forEach((p, pi) => {
      if (w < p.wi0) return;
      for (let r = 1; r < p.mine[w].length / L; r++) {
        const u = off[pi] + r, b = off[pi], ch = [];
        for (let k = 0; k < S; k++) if (of[u * S + k] !== of[b * S + k]) ch.push(of[b * S + k], of[u * S + k]);
        delta.push(u, b, ch.length / 2, ...ch);
      }
    });
    E.delta.push(Int32Array.from(delta));
    E.rowOff.push(off);
    E.tb.push(nTot);
    nTot += nRow * K;
    E.used.push(used);
  }
  E.tot = new Float64Array(nTot);
  // rowAt[plan][v * W + w]: where variant v's totals for week w start in tot
  E.rowAt = plans.map((p, pi) => {
    const at = new Int32Array(p.V * W).fill(-1);
    for (let v = 0; v < p.V; v++) for (let w = p.wi0; w < W; w++) at[v * W + w] = E.tb[w] + (E.rowOff[w][pi] + p.mineOf[w][v]) * K;
    return at;
  });
  // live[w][i]: player i is in some lineup row in week w or later, so his level and absences still matter
  for (let w = W - 1; w >= 0; w--) E.live[w] = E.used[w].map((x, i) => (x || (w + 1 < W && E.live[w + 1][i]) ? 1 : 0));
  // cases per plan: variants that face the same rivals
  E.cases = plans.map((p) => {
    const ids = new Map(), caseOf = new Int32Array(p.V), list = [];
    for (let v = 0; v < p.V; v++) {
      const mask = p.alive0.subarray(v * T, (v + 1) * T), key = mask.join("");
      if (!ids.has(key)) {
        ids.set(key, list.length);
        list.push({ mask: Uint8Array.from(mask), vars: [] });
      }
      caseOf[v] = ids.get(key);
      list[caseOf[v]].vars.push(v);
    }
    return list;
  });
  // per case and week, its variants' distinct rows (rows[w]: offsets into tot) and which one each variant has (rowOf)
  E.cases.forEach((list, pi) => {
    for (const cs of list) {
      cs.rows = Array.from({ length: W }, () => new Int32Array(0));
      cs.rowOf = new Int32Array(cs.vars.length * W);
      for (let w = plans[pi].wi0; w < W; w++) {
        const ids = new Map();
        cs.vars.forEach((v, a) => {
          const at = E.rowAt[pi][v * W + w];
          if (!ids.has(at)) ids.set(at, ids.size);
          cs.rowOf[a * W + w] = ids.get(at);
        });
        cs.rows[w] = Int32Array.from(ids.keys());
      }
      cs.maxRows = Math.max(1, ...cs.rows.map((r) => r.length));
    }
  });
  // stratum orders for the Latin hypercube draws
  const rng = new Rng(0x5eed, K);
  E.perm = new Uint8Array(N_PERM * K);
  for (let r = 0; r < N_PERM; r++) {
    for (let k = 0; k < K; k++) E.perm[r * K + k] = k;
    for (let k = K - 1; k > 0; k--) {
      const j = Math.floor(rng.random() * (k + 1)), t = E.perm[r * K + k];
      E.perm[r * K + k] = E.perm[r * K + j];
      E.perm[r * K + j] = t;
    }
  }
  E.nF = U.n_game.map((g, w) => g + U.n_team[w]);
  E.score = new Float32Array(n1 * K);
  E.avail = new Uint8Array(n1 * K);
  E.plays = new Uint8Array(n1 * K); // this week's score is counted in some lineup
  E.anyPlays = new Uint8Array(n1);
  E.loglev = new Float64Array(n * K);
  E.levWeek = new Uint8Array(n * K); // the week loglev stands at
  E.outLeft = new Int16Array(n * K);
  E.stepSd = Float64Array.from({ length: W + 1 }, (_, d) => LEVEL_STEP * Math.sqrt(d));
  E.tv = new Float64Array(Math.max(1, ...E.trip.map((t) => t.length / 3)) * K);
  return E;
}

/** Factors for G groups: a Latin hypercube sample over the groups in every coordinate (game factors, then team factors). */
function drawFactors(E, G, rng) {
  const { U, W, nF } = E, sg = Math.sqrt(U.game_share), st = Math.sqrt(1 - U.game_share), order = new Int32Array(G);
  const F = [];
  for (let w = 0; w < W; w++) {
    const f = new Float64Array(G * nF[w]);
    if (w >= E.wLo)
      for (let j = 0; j < nF[w]; j++) {
        const sc = j < U.n_game[w] ? sg : st;
        for (let g = 0; g < G; g++) order[g] = g;
        for (let g = G - 1; g > 0; g--) {
          const r = Math.floor(rng.random() * (g + 1)), t = order[g];
          order[g] = order[r];
          order[r] = t;
        }
        for (let g = 0; g < G; g++) f[g * nF[w] + j] = sc * ndtri0((order[g] + rng.random()) / G);
      }
    F.push(f);
  }
  return F;
}

/**
 * Week w of one group, all K paths at once, into the lineup totals (E.tot). Who is available comes first: a new
 * absence (or sitting out on an injury tag) goes to the paths in the lowest strata of a uniform, so absences are
 * spread evenly over the group's paths. Then only the players a lineup actually counts are scored: each one's score
 * noise is one draw per stratum of the normal, and his level catches up on the weeks since it was last needed (a
 * random walk's steps add up to one normal draw).
 */
function drawGroupWeek(E, w, Fw, fb, rng) {
  const { U, K, perm, score, avail, plays, anyPlays, loglev, levWeek, outLeft, stepSd, tv, tot } = E, n = U.n, invK = 1 / K;
  const game = U.game[w], team = U.team[w], q = U.qflat[w], load = U.load, cload = U.cload, ng = U.n_game[w];
  const live = E.live[w], haz = U.haz, pLong = U.p_long, pGeo = U.p_geo, pOut = U.p_out;
  const locked = U.anyLocked && w === 0, started = U.started, pts0 = U.pts0, fracLeft = U.frac_left;
  const drift = -0.5 * LEVEL_STEP * LEVEL_STEP * w;
  for (let i = 0; i < n; i++) {
    if (!live[i]) continue;
    const b0 = i * K;
    if (w > 0) { // new absences start and run down
      const hz = haz[i] * K, sure = hz | 0, edge = hz - sure;
      const h = rng.nextI32() >>> 0, r = (h & (N_PERM - 1)) * K, sh = (h >>> 10) & (K - 1);
      for (let k = 0; k < K; k++) {
        let o = outLeft[b0 + k];
        if (o === 0) {
          const s = (perm[r + k] + sh) & (K - 1);
          if (s < sure || (s === sure && rng.random() < edge)) o = absence(rng, pLong[i], pGeo[i]);
        }
        avail[b0 + k] = o === 0 ? 1 : 0;
        outLeft[b0 + k] = o > 0 ? o - 1 : o;
      }
    } else if (pOut[i] > 0 && !(locked && started[i])) { // this week: an injury tag decides who sits
      const hz = pOut[i] * K, sure = hz | 0, edge = hz - sure;
      const h = rng.nextI32() >>> 0, r = (h & (N_PERM - 1)) * K, sh = (h >>> 10) & (K - 1);
      for (let k = 0; k < K; k++) {
        const s = (perm[r + k] + sh) & (K - 1);
        avail[b0 + k] = s < sure || (s === sure && rng.random() < edge) ? 0 : 1;
      }
    } else for (let k = 0; k < K; k++) avail[b0 + k] = 1;
  }
  // who plays: in each slot the starter, else the first available fallback
  const trip = E.trip[w], nTrip = trip.length / 3;
  plays.fill(0);
  anyPlays.fill(0);
  for (let j = 0; j < nTrip; j++) {
    const a = trip[3 * j], b1 = trip[3 * j + 1], b2 = trip[3 * j + 2], ak = a * K, b1k = b1 * K, b2k = b2 * K;
    for (let k = 0; k < K; k++) {
      if (avail[ak + k]) { plays[ak + k] = 1; anyPlays[a] = 1; }
      else if (avail[b1k + k]) { plays[b1k + k] = 1; anyPlays[b1] = 1; }
      else if (avail[b2k + k]) { plays[b2k + k] = 1; anyPlays[b2] = 1; }
    }
  }
  for (let i = 0; i < n; i++) {
    if (!anyPlays[i]) continue;
    const b0 = i * K, f = load[i] * (Fw[fb + game[i]] + Fw[fb + ng + team[i]]), c = cload[i], qb = i * Q;
    const h = rng.nextI32() >>> 0, r = (h & (N_PERM - 1)) * K, sh = (h >>> 10) & (K - 1);
    for (let k = 0; k < K; k++) {
      if (!plays[b0 + k]) continue;
      const s = (perm[r + k] + sh) & (K - 1);
      const u = uOf(f + c * ndtri0((s + rng.random()) * invK));
      const i0 = u >= Q - 2 ? Q - 2 : u | 0, fr = u - i0, b = qb + i0;
      let x = q[b] + fr * (q[b + 1] - q[b]);
      if (w > 0) {
        const lv = loglev[b0 + k] + rng.normal() * stepSd[w - levWeek[b0 + k]];
        loglev[b0 + k] = lv;
        levWeek[b0 + k] = w;
        x *= Math.exp(lv + drift);
      } else if (locked && started[i]) x = pts0[i] + fracLeft[i] * x; // a game under way carries its points so far
      score[b0 + k] = x;
    }
  }
  // lineup totals: each distinct slot triple once, then the teams' rows slot by slot (as teamScore adds them) and
  // the variants' rows as their plan's first row plus the slots that differ
  for (let j = 0; j < nTrip; j++) {
    const a = trip[3 * j] * K, b1 = trip[3 * j + 1] * K, b2 = trip[3 * j + 2] * K, o = j * K;
    for (let k = 0; k < K; k++) tv[o + k] = avail[a + k] ? score[a + k] : avail[b1 + k] ? score[b1 + k] : avail[b2 + k] ? score[b2 + k] : 0;
  }
  const { S, T } = E, of = E.of[w], tb = E.tb[w], off = E.rowOff[w], plans = E.plans;
  const sumRow = (u) => {
    const o = tb + u * K;
    for (let k = 0; k < K; k++) tot[o + k] = 0;
    for (let s = 0; s < S; s++) {
      const t = of[u * S + s] * K;
      for (let k = 0; k < K; k++) tot[o + k] += tv[t + k];
    }
  };
  for (let t = 0; t < T; t++) sumRow(t);
  for (let pi = 0; pi < plans.length; pi++) if (w >= plans[pi].wi0) sumRow(off[pi]);
  const dl = E.delta[w];
  for (let j = 0; j < dl.length;) {
    const o = tb + dl[j] * K, ob = tb + dl[j + 1] * K, nc = dl[j + 2];
    j += 3;
    for (let k = 0; k < K; k++) tot[o + k] = tot[ob + k];
    for (let e = 0; e < nc; e++, j += 2) {
      const t0 = dl[j] * K, t1 = dl[j + 1] * K;
      for (let k = 0; k < K; k++) tot[o + k] += tv[t1 + k] - tv[t0 + k];
    }
  }
}

/**
 * Season / elimination sims for one job of G groups. job: {ci, G, seed, rounds}. Returns sums to be added across
 * jobs: chops (T * W) and wins (T) over `seasons` recombined seasons, for every team with my roster as it is; and
 * per plan and roster variant, wins over `pairs` pairings of my paths with the rivals' paths, and differs: the
 * paths (of `paths`) on which, as drawn, the variant's season ends differently (won or not) from that of its case's
 * first variant. That is what plain paired sims would count, so it gives the variance they would have had.
 * Round 0 is the paths as drawn, so with rounds = 1 the teams' odds are plain Monte Carlo on stratified draws.
 */
export function seasonGroups(E, job) {
  const { U, plans, K, T, W, me_t, wLo, tot, tb } = E, { ci, G, seed } = job, R = job.rounds ?? ROUNDS;
  const rng = new Rng(seed, ci), F = drawFactors(E, G, rng);
  const chops = new Float64Array(T * W), wins = new Float64Array(T);
  const out = plans.map((p) => ({ wins: new Float64Array(p.V), differs: new Float64Array(p.V) }));
  const pathOf = new Int32Array(R * T), alive = new Int32Array(T), at = new Int32Array(T), own = new Uint8Array(K);
  // A case's chop lines, one per path: the lowest alive rival each week if I survive. By week, the lines from lowest
  // up (sLo; sMe: I lose a tie with it) and, for every count c, the set of the c lowest lines as bits (pre). A score
  // clears a prefix of that order, so the lines one of my paths survives all season are the AND of one prefix per
  // week. NP pads the sorted lines to a power of two with lines nothing clears, for a search without branches.
  const NP = 2 * K, lineLo = new Float64Array(W * K), lineMe = new Uint8Array(W * K), sIdx = new Int32Array(K);
  const sLo = new Float64Array(W * NP).fill(Infinity), sMe = new Uint8Array(W * NP).fill(1), pre = new Int32Array(W * (K + 1));
  const rank = new Int32Array(W * Math.max(...E.cases.flat().map((c) => c.maxRows)) * K);
  for (let g = 0; g < G; g++) {
    E.loglev.fill(0);
    E.levWeek.fill(0);
    E.outLeft.fill(0);
    for (let w = wLo; w < W; w++) drawGroupWeek(E, w, F[w], g * E.nF[w], rng);
    for (let j = 0; j < R * T; j++) pathOf[j] = j < T ? 0 : (rng.nextI32() >>> 0) & (K - 1); // round 0: as drawn
    if (job.onGroup) job.onGroup(E, pathOf, R); // tests: check this group's sums against brute force
    // 1. every team's odds with my roster as it is: recombined seasons
    if (wLo === 0)
      for (let r = 0; r < R; r++)
        for (let j = 0; j < K; j++) {
          for (let t = 0; t < T; t++) {
            at[t] = t * K + ((j + pathOf[r * T + t]) & (K - 1));
            alive[t] = t;
          }
          for (let w = 0, nA = T; w < W && nA > 1; w++, nA--) { // the lowest alive score goes, the earliest team on a tie
            const b = tb[w];
            let lo = Infinity, la = -1;
            for (let a = 0; a < nA; a++) {
              const x = tot[b + at[alive[a]]];
              if (x < lo) { lo = x; la = a; }
            }
            chops[alive[la] * W + w]++;
            for (let a = la; a < nA - 1; a++) alive[a] = alive[a + 1]; // keeps team order
          }
          wins[alive[0]]++;
        }
    // 2. my roster variants: every path of mine against the rivals of every path, case by case
    for (let pi = 0; pi < plans.length; pi++) {
      const p = plans[pi], w0 = p.wi0, res = out[pi];
      for (const cs of E.cases[pi]) {
        for (let l = 0; l < K; l++) { // path l's rivals: their own elimination
          let nA = 0;
          for (let t = 0; t < T; t++) if (t !== me_t && cs.mask[t]) alive[nA++] = t;
          for (let w = w0; w < W; w++, nA--) {
            const b = tb[w] + l;
            let lo = Infinity, la = -1;
            for (let a = 0; a < nA; a++) {
              const x = tot[b + alive[a] * K];
              if (x < lo) { lo = x; la = a; }
            }
            lineLo[w * K + l] = lo;
            lineMe[w * K + l] = la >= 0 && me_t < alive[la] ? 1 : 0;
            for (let a = la; a >= 0 && a < nA - 1; a++) alive[a] = alive[a + 1];
            if (la < 0) nA++; // nobody left: the line stays at infinity
          }
        }
        for (let w = w0; w < W; w++) {
          const sb = w * NP, pb = w * (K + 1);
          for (let l = 0; l < K; l++) { // insertion sort by (line, I lose a tie): a score clears the lines below it and the ties it wins
            const lo = lineLo[w * K + l], me = lineMe[w * K + l];
            let a = l;
            for (; a > 0 && (sLo[sb + a - 1] > lo || (sLo[sb + a - 1] === lo && sMe[sb + a - 1] > me)); a--) {
              sLo[sb + a] = sLo[sb + a - 1];
              sMe[sb + a] = sMe[sb + a - 1];
              sIdx[a] = sIdx[a - 1];
            }
            sLo[sb + a] = lo;
            sMe[sb + a] = me;
            sIdx[a] = l;
          }
          pre[pb] = 0;
          for (let c = 0; c < K; c++) pre[pb + c + 1] = pre[pb + c] | (1 << sIdx[c]);
        }
        // how many lines each distinct row clears on each of my paths, week by week
        const mr = cs.maxRows * K;
        for (let w = w0; w < W; w++) {
          const rows = cs.rows[w], sb = w * NP, ro = w * mr;
          for (let u = 0; u < rows.length; u++)
            for (let m = 0; m < K; m++) {
              const x = tot[rows[u] + m];
              if (u > 0 && x === tot[rows[0] + m]) { // most variants score what the case's first one does on most paths
                rank[ro + u * K + m] = rank[ro + m];
                continue;
              }
              let c = 0;
              for (let st = K; st > 0; st >>= 1) { // (a line below my score, or a tie I win) * st: no branch to mispredict
                const e = sb + c + st - 1;
                c += ((sLo[e] < x) | ((sLo[e] === x) & (sMe[e] ^ 1))) * st;
              }
              rank[ro + u * K + m] = c;
            }
        }
        const rowOf = cs.rowOf;
        for (let a = 0; a < cs.vars.length; a++) {
          let nw = 0, nd = 0;
          for (let m = 0; m < K; m++) {
            let b = -1; // the lines my path m survives all season, as bits
            for (let w = w0; w < W; w++) b &= pre[w * (K + 1) + rank[w * mr + rowOf[a * W + w] * K + m]];
            nw += popcount(b);
            const won = (b >>> m) & 1; // against its own path's rivals: the season as drawn
            if (a === 0) own[m] = won;
            else nd += won ^ own[m];
          }
          res.wins[cs.vars[a]] += nw;
          res.differs[cs.vars[a]] += nd;
        }
      }
    }
  }
  return { chops, wins, plans: out, seasons: wLo === 0 ? G * K * R : 0, pairs: G * K * K, paths: G * K };
}

const bin = (x) => Math.trunc(Math.min(Math.max(x * 2, 0), 399));

function popcount(y) {
  y -= (y >>> 1) & 0x55555555;
  y = (y & 0x33333333) + ((y >>> 2) & 0x33333333);
  return Math.imul((y + (y >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
}

/**
 * This week alone, plain Monte Carlo, on a season plan whose first week it is: sums over C sims of every team's chop
 * with my roster as it is (chops), my chop per roster variant (firstOut) and the histograms of the lowest rival
 * score and mine.
 */
export function week0Chunk(U, plan, ci, C, seed) {
  const { S, T, V, me_t } = plan, L = 3 * S, rows = plan.mine[0], of = plan.mineOf[0], nRows = rows.length / L;
  const rng = new Rng(seed, ci), { score, avail, G, T: TT } = scratch(U), st = pathState(U.n);
  const chops = new Float64Array(T), firstOut = new Float64Array(V), lineHist = new Float64Array(400), myHist = new Float64Array(400);
  const mine = new Float64Array(nRows);
  for (let c = 0; c < C; c++) {
    drawWeek(U, 0, rng, st, score, avail, G, TT);
    let lo = Infinity, lt = -1, line = Infinity, t1 = -1; // lowest of all (the earliest team on a tie), lowest rival
    for (let t = 0; t < T; t++) {
      const x = teamScore(score, avail, plan.opp[0], t * L, S);
      if (x < lo) { lo = x; lt = t; }
      if (t !== me_t && x < line) { line = x; t1 = t; }
    }
    chops[lt]++;
    for (let u = 0; u < nRows; u++) mine[u] = teamScore(score, avail, rows, u * L, S);
    for (let v = 0; v < V; v++) {
      const m = mine[of[v]];
      if (m < line || (m === line && me_t < t1)) firstOut[v]++;
    }
    lineHist[bin(line)]++;
    myHist[bin(mine[of[0]])]++;
  }
  return { chops, firstOut, lineHist, myHist };
}

/** One-week lineup sims: P(my lineup scores below the lowest rival), mean and sum of squares per lineup. */
export function weekChunk(U, job, ci, C, seed) {
  const { S, T, V, me_t, opp: oi, trip, mineTrip } = job, L = 3 * S;
  const rng = new Rng(seed, ci);
  const chopped = new Float64Array(V), s1 = new Float64Array(V), s2 = new Float64Array(V);
  const { score, avail, G, T: TT } = scratch(U);
  const st = pathState(U.n), mine = new Float64Array(V), tv = new Float64Array(Math.max(1, trip.length / 3));
  for (let c = 0; c < C; c++) {
    drawWeek(U, 0, rng, st, score, avail, G, TT);
    let line = Infinity;
    for (let t = 0; t < T; t++) if (t !== me_t) line = Math.min(line, teamScore(score, avail, oi, t * L, S));
    rowScores(score, avail, trip, mineTrip, S, tv, mine, V);
    for (let v = 0; v < V; v++) {
      const m = mine[v];
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
