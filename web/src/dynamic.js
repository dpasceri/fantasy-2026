// Port of guillotine/dynamic.py: season-with-auctions sims (what is a FAAB dollar worth, how hard to bid).
//
// Each path plays the rest of the season week by week: lineups from projections (known injuries benched),
// lowest alive score chopped, then the chopped roster goes to a sealed-bid FAAB auction. Rivals bid from the
// fitted bid model; I bid m x the league's typical bid for my gain. Highest bid wins and pays its bid, claims
// are processed from the highest bid down, budgets and the roster cap are enforced.
//
// This file runs inside workers. prepDyn puts the plain data from run.js in index form: players are universe
// indices, teams are 0..T-1 in base.teams order, positions are POS indices. Each distinct roster (players in
// order, which also breaks ties between equal projections) gets a number, and the worker caches by that number:
// the roster, its projection order and weakest starters per week, and the roster an auction win turns it into.
// Every cache returns exactly what recomputing would.
import { Bids } from "./auction.js";
import { eligible, POS, slotOrder } from "./league.js";
import { Rng } from "./rng.js";
import { drawWeek, pathState, teamScore } from "./sim.js";

export const HORIZON = 4;
export const MIN_GAIN = 2.0; // I don't bid on players worth less than this many lineup points a week
// Rosters a worker keeps cached results for. The caches are exact, so this only trades speed for memory: at 60k a
// 3-worker phone run peaks near 2 GB (3 GB at 300k) and runs no slower.
const MAX_ROSTERS = 60_000;
const NP = POS.length + 1; // position index stride; the extra column is for positions without a slot

/** Worker-side state for pathsChunk, from the Universe data and run.js's {ctx, base}. */
export function prepDyn(U, { ctx, base }) {
  const { ix } = ctx, n = U.n, W = U.W, S = ctx.slots.length, teams = base.teams, T = teams.length;
  // elig[k * NP + p]: can position p fill the k-th slot in bestLineup's fill order
  const elig = new Uint8Array(S * NP);
  slotOrder(ctx.slots).forEach((s, k) => POS.forEach((p, pi) => { elig[k * NP + pi] = eligible(s, p) ? 1 : 0; }));
  const idxOf = (pids) => pids.map((p) => ix.get(p));
  const baseRosters = teams.map((t) => idxOf(base.rosters[t]));
  const pendingPool = base.pending_pool ? idxOf(base.pending_pool) : null;
  const futOf = Array.from({ length: W }, (_, wi) => { // the weeks an auction before week wi + 1 looks at
    const f = [];
    for (let w2 = wi + 1; w2 < Math.min(wi + 1 + HORIZON, W); w2++) f.push(w2);
    return f;
  });
  const bm = new Bids(ctx.bm);
  const maxPool = Math.max(ctx.maxRoster, ...baseRosters.map((r) => r.length), pendingPool ? pendingPool.length : 0) + 1;
  return {
    U, n, W, S, T, teams, meT: teams.indexOf(ctx.me), bm, uMe: bm.um.get(ctx.me) || 0, ix, pos: ctx.posIdx, elig,
    proj: ctx.proj, rosVal: ctx.rosVal, maxRoster: ctx.maxRoster, weeks: base.weeks, week0Idx: base.week0Idx,
    baseRosters, budget0: Float64Array.from(teams, (t) => base.budget[t]), pendingPool, fut0: [0, 1, 2, 3].slice(0, W), futOf,
    // caches by roster number
    ids: new Map(), rosterOf: [], ordOf: [], weakOf: [], trans: new Map(),
    // scratch
    used: new Uint8Array(n + 1), picked: new Int32Array(S), taken: new Uint8Array(n + 1), el: new Array(HORIZON),
    cB: new Float64Array(T * maxPool), cU: new Float64Array(T * maxPool), cG: new Float64Array(T * maxPool),
    cT: new Int32Array(T * maxPool), cP: new Int32Array(T * maxPool), cOrd: new Int32Array(T * maxPool),
  };
}

/** Roster number for an array of player indices (never mutated after this). */
function intern(env, r) {
  const key = r.join(",");
  let id = env.ids.get(key);
  if (id === undefined) {
    env.ids.set(key, (id = env.ids.size));
    env.rosterOf[id] = r;
  }
  return id;
}

/** Roster `id`'s players with a projection in week wi, best first; ties keep roster order (bestLineup's stable sort). */
function projOrder(env, id, wi) {
  const byW = (env.ordOf[id] ??= new Array(env.W));
  let o = byW[wi];
  if (!o) {
    const pw = env.proj[wi], r = env.rosterOf[id];
    byW[wi] = o = [];
    for (let k = 0; k < r.length; k++) { // stable insertion: equal projections stay in roster order
      const i = r[k];
      if (!(pw[i] > 0)) continue;
      let j = o.length;
      o.push(i);
      for (; j > 0 && pw[o[j - 1]] < pw[i]; j--) o[j] = o[j - 1];
      o[j] = i;
    }
  }
  return o;
}

/** bestLineup's greedy fill over an ordered candidate list into env.picked (-1 = empty slot); avail null = everyone plays. */
function fill(env, o, avail, off) {
  const { elig, pos, used, picked, S } = env;
  for (let k = 0; k < S; k++) {
    picked[k] = -1;
    for (let j = 0; j < o.length; j++) {
      const i = o[j];
      if (!used[i] && (avail === null || avail[off + i]) && elig[k * NP + pos[i]]) {
        used[i] = 1;
        picked[k] = i;
        break;
      }
    }
  }
  for (let k = 0; k < S; k++) if (picked[k] >= 0) used[picked[k]] = 0;
  return picked;
}

/** Weakest starter by position for roster `id` in week w2 (depends only on the roster and the week). */
function weakest(env, id, w2) {
  const byW = (env.weakOf[id] ??= new Array(env.W));
  let m = byW[w2];
  if (!m) {
    const lu = fill(env, projOrder(env, id, w2), null, 0), pw = env.proj[w2], { elig, S } = env;
    byW[w2] = m = new Float64Array(NP).fill(NaN);
    for (let p = 0; p < POS.length; p++) {
      let lo = Infinity;
      for (let k = 0; k < S; k++) if (elig[k * NP + p]) lo = Math.min(lo, lu[k] >= 0 ? pw[lu[k]] : 0);
      m[p] = lo;
    }
  }
  return m;
}

/** Roster `id` after winning player i before week w1: appended, and over the cap the lowest rest-of-season projection goes. */
function moveTo(env, id, i, w1) {
  const key = (id * env.n + i) * env.W + w1;
  let nid = env.trans.get(key);
  if (nid === undefined) {
    const r = [...env.rosterOf[id], i];
    if (r.length > env.maxRoster) {
      const rosNext = env.rosVal[w1];
      let drop = -1;
      for (let j = 0; j < r.length; j++) if (r[j] !== i && (drop < 0 || rosNext[r[j]] < rosNext[r[drop]])) drop = j;
      r.splice(drop, 1);
    }
    env.trans.set(key, (nid = intern(env, r)));
  }
  return nid;
}

/**
 * One waiver run, in place: rid (roster number per team) and budget change. My bids follow setting st in week wk,
 * forced bids included when useForce. out = [my spend, my studs, league spend, top price].
 */
function auction(env, pool, alive, rid, budget, fut, leg, pr, boost, fittedMe, st, wk, useForce, out) {
  const { meT, teams, bm, proj, pos, el, cB, cU, cG, cT, cP, cOrd, taken } = env;
  const F = fut.length, w1 = fut[0], nPool = pool.length;
  const mMult = wk < st.switch_week ? st.m_early : st.m_late, force = useForce ? st.forceIdx : null;
  let nc = 0, cMe = null;
  for (let a = 0; a < alive.length; a++) {
    const t = alive[a], bt = budget[t], bids = bt > 0;
    if (bids) for (let k = 0; k < F; k++) el[k] = weakest(env, rid[t], fut[k]);
    const isMe = t === meT && !fittedMe;
    const c = bids && !isMe ? bm.terms(teams[t], bt, leg) : null;
    const q0 = c ? Bids.q(c, 0) : 0; // most players add nothing to most lineups
    if (bids && isMe) cMe = bm.terms(teams[t], bt, wk);
    for (let j = 0; j < nPool; j++) {
      const uPart = pr.random(), z = pr.normal(), uTie = pr.random(); // always drawn: keeps paths aligned
      if (!bids) continue;
      const i = pool[j], P = pos[i];
      let g = 0;
      for (let k = 0; k < F; k++) g += Math.max(0, proj[fut[k]][i] - el[k][P]);
      g /= F;
      let b;
      if (isMe) {
        if (force !== null && force[i] >= 0) b = force[i];
        else if (g < MIN_GAIN) b = 0;
        else b = Math.round(mMult * Math.exp(Bids.m(cMe, g) - env.uMe));
      } else {
        let q = g === 0 ? q0 : Bids.q(c, g);
        if (boost[0]) q = 1 / (1 + Math.exp(-(Math.log(q / (1 - q)) + boost[0])));
        b = uPart < q ? Math.round(Math.exp(Bids.m(c, g) + boost[1] + bm.sigma * z)) : 0;
      }
      b = Math.min(b, bt);
      if (b >= 1) {
        cB[nc] = b; cU[nc] = uTie; cT[nc] = t; cP[nc] = i; cG[nc] = g;
        nc++;
      }
    }
  }
  // highest bid first, then the tie-break draw; equal claims keep their order (a stable sort, like Array#sort)
  for (let a = 0; a < nc; a++) {
    let j = a;
    while (j > 0 && (cB[cOrd[j - 1]] < cB[a] || (cB[cOrd[j - 1]] === cB[a] && cU[cOrd[j - 1]] < cU[a]))) {
      cOrd[j] = cOrd[j - 1];
      j--;
    }
    cOrd[j] = a;
  }
  let spent = 0, studs = 0, total = 0, top = 0;
  for (let a = 0; a < nc; a++) {
    const q = cOrd[a], b = cB[q], t = cT[q], i = cP[q];
    if (taken[i] || budget[t] < b) continue;
    taken[i] = 1;
    budget[t] -= b;
    total += b;
    top = Math.max(top, b);
    rid[t] = moveTo(env, rid[t], i, w1);
    if (t === meT) {
      spent += b;
      studs += cG[q] >= 8 ? 1 : 0;
    }
  }
  for (let a = 0; a < nc; a++) taken[cP[a]] = 0;
  out[0] = spent; out[1] = studs; out[2] = total; out[3] = top;
}

/** All weeks of C paths: score/avail as Float32Array/Uint8Array (W * C * (n + 1)). */
function drawPaths(U, C, rng) {
  const n1 = U.n + 1, W = U.W;
  const score = new Float32Array(W * C * n1), avail = new Uint8Array(W * C * n1);
  const G = new Float64Array(Math.max(1, ...U.n_game)), T = new Float64Array(Math.max(1, ...U.n_team));
  const s = new Float32Array(n1), a = new Uint8Array(n1), st = pathState(U.n);
  for (let c = 0; c < C; c++) {
    st.loglev.fill(0);
    st.outLeft.fill(0);
    for (let wi = 0; wi < W; wi++) {
      drawWeek(U, wi, rng, st, s, a, G, T);
      const off = (wi * C + c) * n1;
      score.set(s, off);
      avail.set(a, off);
    }
  }
  return { score, avail };
}

/** job: {ci, C, seed, settings}. Returns one accumulator per setting. */
export function pathsChunk(env, job) {
  const { U, W, T, S, meT, n, weeks } = env, { ci, C, seed, settings } = job, n1 = n + 1;
  const rng = new Rng(seed, ci);
  const { score, avail } = drawPaths(U, C, rng);
  if (env.ids.size > MAX_ROSTERS) { // between jobs only: no path holds an old roster number now
    env.ids.clear();
    env.trans.clear();
    env.rosterOf.length = env.ordOf.length = env.weakOf.length = 0;
  }
  const baseIds = Int32Array.from(env.baseRosters, (r) => intern(env, r));
  // week 0 under way: lineups are partly locked, so score every team's lock-aware lineup on all paths at once
  let week0 = null;
  if (env.week0Idx && !env.pendingPool) {
    week0 = new Float64Array(C * T);
    for (let c = 0; c < C; c++)
      for (let t = 0; t < T; t++) week0[c * T + t] = teamScore(score.subarray(c * n1), avail.subarray(c * n1), env.week0Idx, t * 3 * S, S);
  }
  const sets = settings.map((st) => ({
    st, boost: st.rival_boost || [0, 0], track: !!st.track_all, fitted: !!st.fitted_me, // track: keep playing after I'm chopped
    chopped0: st.chopped0 != null ? env.teams.indexOf(st.chopped0) : -1, // condition on this team going out in week 0
    tot: { wins: 0, weeks: 0, spent: 0, studs: 0, n: C, win_by: new Array(T).fill(0), weeks_by: new Array(T).fill(0),
      league_spend: new Array(W + 1).fill(0), top_price: new Array(W + 1).fill(0), burned: new Array(W + 1).fill(0) },
  }));
  for (const s of sets) { // my forced bids by player index (-1 = none)
    s.st = { ...s.st, forceIdx: null };
    if (s.st.force) {
      s.st.forceIdx = new Float64Array(n).fill(-1);
      for (const [p, b] of Object.entries(s.st.force)) if (env.ix.has(p)) s.st.forceIdx[env.ix.get(p)] = b;
    }
  }
  // a team's lineup score this week on a path depends only on its roster: the settings share it until they
  // diverge, so each path is played under every setting before the next path
  const memo = new Map(); // roster number * W + week -> score on the current path
  const rid = new Int32Array(T), budget = new Float64Array(T), r4 = [0, 0, 0, 0];
  for (let c = 0; c < C; c++) {
    memo.clear();
    for (const s of sets) {
      const { st, tot, boost, track, fitted } = s;
      const pr = new Rng(seed, ci, c, 1);
      rid.set(baseIds);
      budget.set(env.budget0);
      budget[meT] = st.budget;
      const alive = [];
      for (let t = 0; t < T; t++) alive.push(t);
      if (env.pendingPool) {
        const wk = weeks[0] - 1;
        auction(env, env.pendingPool, alive, rid, budget, env.fut0, wk, pr, boost, fitted, st, wk, true, r4);
        tot.spent += r4[0];
        tot.studs += r4[1];
        tot.league_spend[0] += r4[2];
        tot.top_price[0] += r4[3];
      }
      let meOut = null;
      for (let wi = 0; wi < W; wi++) {
        const wk = weeks[wi];
        let lowT = -1, useForce = false;
        if (wi === 0 && s.chopped0 >= 0) {
          lowT = s.chopped0;
          useForce = true;
        } else {
          const off = (wi * C + c) * n1;
          let low = Infinity;
          for (let a = 0; a < alive.length; a++) {
            const t = alive[a];
            let tt;
            if (wi === 0 && week0) tt = week0[c * T + t];
            else {
              const mk = rid[t] * W + wi;
              tt = memo.get(mk);
              if (tt === undefined) {
                const lu = fill(env, projOrder(env, rid[t], wi), avail, off);
                tt = 0;
                for (let k = 0; k < S; k++) if (lu[k] >= 0) tt += score[off + lu[k]];
                memo.set(mk, tt);
              }
            }
            if (tt < low || lowT < 0) {
              low = tt;
              lowT = t;
            }
          }
        }
        alive.splice(alive.indexOf(lowT), 1);
        tot.weeks_by[lowT] += wi;
        tot.burned[wi + 1] += budget[lowT];
        if (lowT === meT) {
          meOut = wi;
          if (!track) break;
        }
        if (alive.length === 1 || wi === W - 1) break;
        const rv = env.rosVal[wi + 1], pool = env.rosterOf[rid[lowT]].filter((i) => rv[i] > 0);
        auction(env, pool, alive, rid, budget, env.futOf[wi], wk, pr, boost, fitted, st, wk, useForce, r4);
        tot.spent += r4[0];
        tot.studs += r4[1];
        tot.league_spend[wi + 1] += r4[2];
        tot.top_price[wi + 1] += r4[3];
      }
      if (alive.length === 1) {
        tot.win_by[alive[0]] += 1;
        tot.weeks_by[alive[0]] += W;
      }
      if (meOut === null) {
        tot.wins += alive.length === 1 && alive[0] === meT ? 1 : 0;
        tot.weeks += W;
      } else tot.weeks += meOut;
    }
  }
  return sets.map((s) => s.tot);
}

/** Add one chunk's accumulators into acc (same settings order). */
export function addTotals(acc, part) {
  if (!acc) return part.map((x) => ({ ...x, win_by: [...x.win_by], weeks_by: [...x.weeks_by],
    league_spend: [...x.league_spend], top_price: [...x.top_price], burned: [...x.burned] }));
  acc.forEach((a, i) => {
    for (const [k, v] of Object.entries(part[i])) {
      if (Array.isArray(v)) v.forEach((x, j) => { a[k][j] += x; });
      else a[k] += v;
    }
  });
  return acc;
}

/** Per-setting summary rows (dynamic.run's output). */
export function summarize(settings, acc) {
  return settings.map((st, i) => {
    const tot = acc[i], n = tot.n, p = tot.wins / n;
    const row = { ...st, p_win: p, se: Math.sqrt((p * (1 - p)) / n), e_weeks: tot.weeks / n, avg_spent: tot.spent / n,
      studs_bought: tot.studs / n, n };
    if (st.track_all) for (const k of ["win_by", "weeks_by", "league_spend", "top_price", "burned"]) row[k] = tot[k].map((v) => v / n);
    return row;
  });
}
