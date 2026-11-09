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
// Two more shortcuts work within a path, across the settings a job plays on it: settings that reach an auction
// with the same rivals and the same random stream share the rivals' claims, and settings that leave their forced
// first auction in the same state share the rest of the season. Every cache returns exactly what recomputing would.
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
const CLAIM_SLOTS = 6; // sets of rival claims a path keeps per auction

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
  const projF = new Float64Array(W * n); // proj[w][i] at w * n + i
  ctx.proj.forEach((pw, w) => projF.set(pw, w * n));
  return {
    U, n, W, S, T, teams, meT: teams.indexOf(ctx.me), bm, uMe: bm.um.get(ctx.me) || 0, ix, pos: ctx.posIdx, elig,
    proj: ctx.proj, rosVal: ctx.rosVal, maxRoster: ctx.maxRoster, weeks: base.weeks, week0Idx: base.week0Idx,
    baseRosters, budget0: Float64Array.from(teams, (t) => base.budget[t]), pendingPool, fut0: [0, 1, 2, 3].slice(0, W), futOf,
    // caches by roster number
    rosterOf: [], ordOf: [], weakOf: [],
    // two open-addressing tables (-1 = free): roster numbers by content, and (roster, player, week) -> next roster
    idTab: new Int32Array(1 << 12).fill(-1), trN: 0,
    trId: new Int32Array(1 << 12).fill(-1), trAdd: new Int32Array(1 << 12), trVal: new Int32Array(1 << 12),
    // lineup scores on the current path by roster number and week (valid where the stamp is the path's)
    memoV: new Float64Array(1024 * W), memoAt: new Int32Array(1024 * W),
    // scratch
    used: new Uint8Array(n + 1), picked: new Int32Array(S), taken: new Uint8Array(n + 1), projF,
    el: new Float64Array(HORIZON * NP), fo: new Int32Array(HORIZON), // a team's weakest starters and projection offsets for the weeks an auction looks at
    cB: new Float64Array(T * maxPool), cU: new Float64Array(T * maxPool), cG: new Float64Array(T * maxPool),
    cT: new Int32Array(T * maxPool), cP: new Int32Array(T * maxPool), cOrd: new Int32Array(T * maxPool),
    // per path: the rivals' claims by auction (CLAIM_SLOTS sets each), valid while their path stamp is current
    poolOf: [], path: 0, claimNext: new Int32Array(W + 1), scratchSet: claimSet(T, maxPool),
    claimSets: Array.from({ length: (W + 1) * CLAIM_SLOTS }, () => claimSet(T, maxPool)),
  };
}

/** The rivals' claims in one auction, with what they depend on: the random stream, the pool and every rival's roster and budget. */
function claimSet(T, maxPool) {
  const cap = T * maxPool;
  return { path: -1, a: 0, b: 0, c: 0, d: 0, a2: 0, b2: 0, c2: 0, d2: 0, pool: null, fut: null, leg: 0, b0: 0, b1: 0, nAlive: 0,
    alive: new Int32Array(T), rid: new Int32Array(T), budget: new Float64Array(T), n: 0, meAt: -1, uMe: new Float64Array(maxPool),
    B: new Float64Array(cap), U: new Float64Array(cap), G: new Float64Array(cap), T: new Int32Array(cap), P: new Int32Array(cap) };
}

function hashRoster(r) {
  let h = 0x811c9dc5;
  for (let k = 0; k < r.length; k++) h = Math.imul(h ^ r[k], 0x01000193);
  return h ^ (h >>> 15);
}

/** Roster number for an array of player indices (never mutated after this). */
function intern(env, r) {
  let tab = env.idTab, mask = tab.length - 1, at = hashRoster(r) & mask;
  seek: for (let id; (id = tab[at]) >= 0; at = (at + 1) & mask) {
    const x = env.rosterOf[id];
    if (x.length !== r.length) continue;
    for (let k = 0; k < r.length; k++) if (x[k] !== r[k]) continue seek;
    return id;
  }
  const id = env.rosterOf.length;
  env.rosterOf.push(r);
  tab[at] = id;
  if (2 * (id + 1) > tab.length) { // keep the table under half full
    tab = env.idTab = new Int32Array(2 * tab.length).fill(-1);
    mask = tab.length - 1;
    env.rosterOf.forEach((x, k) => {
      let a = hashRoster(x) & mask;
      while (tab[a] >= 0) a = (a + 1) & mask;
      tab[a] = k;
    });
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
  const add = i * env.W + w1, hash = (x, y) => Math.imul(Math.imul(x, 0x9e3779b1) ^ y, 0x85ebca6b) >>> 8;
  let ids = env.trId, mask = ids.length - 1, at = hash(id, add) & mask;
  for (; ids[at] >= 0; at = (at + 1) & mask) if (ids[at] === id && env.trAdd[at] === add) return env.trVal[at];
  const r = [...env.rosterOf[id], i];
  if (r.length > env.maxRoster) {
    const rosNext = env.rosVal[w1];
    let drop = -1;
    for (let j = 0; j < r.length; j++) if (r[j] !== i && (drop < 0 || rosNext[r[j]] < rosNext[r[drop]])) drop = j;
    r.splice(drop, 1);
  }
  const nid = intern(env, r);
  ids[at] = id;
  env.trAdd[at] = add;
  env.trVal[at] = nid;
  if (2 * ++env.trN > ids.length) { // keep the table under half full
    const old = [ids, env.trAdd, env.trVal], size = 2 * ids.length;
    ids = env.trId = new Int32Array(size).fill(-1);
    env.trAdd = new Int32Array(size);
    env.trVal = new Int32Array(size);
    mask = size - 1;
    for (let a = 0; a < old[0].length; a++) {
      if (old[0][a] < 0) continue;
      let b = hash(old[0][a], old[1][a]) & mask;
      while (ids[b] >= 0) b = (b + 1) & mask;
      ids[b] = old[0][a];
      env.trAdd[b] = old[1][a];
      env.trVal[b] = old[2][a];
    }
  }
  return nid;
}

/** Roster `id`'s weakest starters in the weeks `fut`, and those weeks' offsets into projF, into env.el / env.fo. */
function lookAhead(env, id, fut) {
  const { el, fo, n } = env;
  for (let k = 0; k < fut.length; k++) {
    const m = weakest(env, id, fut[k]);
    for (let p = 0; p < NP; p++) el[k * NP + p] = m[p];
    fo[k] = fut[k] * n;
  }
}

/** The players of roster `id` that go to auction after week wi (those with a projection left); one array per roster and week. */
function poolFor(env, id, wi) {
  const byW = (env.poolOf[id] ??= new Array(env.W));
  let p = byW[wi];
  if (!p) {
    const rv = env.rosVal[wi + 1];
    byW[wi] = p = env.rosterOf[id].filter((i) => rv[i] > 0);
  }
  return p;
}

/**
 * Every rival's claims in one waiver run, drawn from pr into e in the order the draws are made (teams in alive
 * order, players in pool order). Three draws per team and player are always taken, mine too, so paths stay aligned.
 * My own bids depend on the setting, so unless I bid like a fitted manager only my tie-break draws are kept (e.uMe)
 * and where my claims go among the rivals' (e.meAt).
 */
function rivalClaims(env, e, pool, alive, nAlive, rid, budget, fut, leg, pr, boost, fittedMe) {
  const { meT, teams, bm, projF, pos, el, fo } = env, F = fut.length, nPool = pool.length;
  e.path = env.path; e.a = pr.a; e.b = pr.b; e.c = pr.c; e.d = pr.d;
  e.pool = pool; e.fut = fut; e.leg = leg; e.b0 = boost[0]; e.b1 = boost[1]; e.nAlive = nAlive;
  e.meAt = -1;
  let n = 0;
  for (let a = 0; a < nAlive; a++) {
    const t = alive[a], bt = budget[t], bids = bt > 0;
    e.alive[a] = t; e.rid[a] = rid[t]; e.budget[a] = bt;
    if (t === meT && !fittedMe) {
      e.meAt = n;
      for (let j = 0; j < nPool; j++) {
        pr.random();
        pr.normal();
        e.uMe[j] = pr.random();
      }
      continue;
    }
    if (bids) lookAhead(env, rid[t], fut);
    const c = bids ? bm.terms(teams[t], bt, leg) : null;
    const q0 = c ? Bids.q(c, 0) : 0; // most players add nothing to most lineups
    for (let j = 0; j < nPool; j++) {
      const uPart = pr.random(), z = pr.normal(), uTie = pr.random(); // always drawn: keeps paths aligned
      if (!bids) continue;
      const i = pool[j], P = pos[i];
      let g = 0;
      for (let k = 0; k < F; k++) g += Math.max(0, projF[fo[k] + i] - el[k * NP + P]);
      g /= F;
      let q = g === 0 ? q0 : Bids.q(c, g);
      if (boost[0]) q = 1 / (1 + Math.exp(-(Math.log(q / (1 - q)) + boost[0])));
      let b = uPart < q ? Math.round(Math.exp(Bids.m(c, g) + boost[1] + bm.sigma * z)) : 0;
      b = Math.min(b, bt);
      if (b >= 1) {
        e.B[n] = b; e.U[n] = uTie; e.T[n] = t; e.P[n] = i; e.G[n] = g;
        n++;
      }
    }
  }
  e.n = n; e.a2 = pr.a; e.b2 = pr.b; e.c2 = pr.c; e.d2 = pr.d;
}

/** Would the rivals' claims in e be drawn again here: same random stream, pool, weeks, and every rival's roster and budget? */
function sameRivals(e, env, pool, alive, nAlive, rid, budget, fut, leg, pr, boost) {
  if (e.path !== env.path || e.a !== pr.a || e.b !== pr.b || e.c !== pr.c || e.d !== pr.d || e.pool !== pool || e.fut !== fut
    || e.leg !== leg || e.b0 !== boost[0] || e.b1 !== boost[1] || e.nAlive !== nAlive) return false;
  for (let a = 0; a < nAlive; a++) {
    const t = alive[a];
    if (e.alive[a] !== t || (t !== env.meT && (e.rid[a] !== rid[t] || e.budget[a] !== budget[t]))) return false;
  }
  return true;
}

/**
 * One waiver run (the ai-th auction of the path), in place: rid (roster number per team) and budget change. My bids
 * follow setting st in week wk, forced bids included when useForce. out = [my spend, my studs, league spend, top price].
 */
function auction(env, ai, pool, alive, nAlive, rid, budget, fut, leg, pr, boost, fittedMe, st, wk, useForce, out) {
  const { meT, teams, bm, projF, pos, el, fo, cB, cU, cG, cT, cP, cOrd, taken } = env;
  const F = fut.length, w1 = fut[0], nPool = pool.length;
  const mMult = wk < st.switch_week ? st.m_early : st.m_late, force = useForce ? st.forceIdx : null;
  // the rivals' claims: drawn once per path for every setting that gets here with the same rivals
  let e = null;
  if (fittedMe) rivalClaims(env, (e = env.scratchSet), pool, alive, nAlive, rid, budget, fut, leg, pr, boost, true);
  else {
    const sets = env.claimSets, s0 = ai * CLAIM_SLOTS;
    for (let k = 0; k < CLAIM_SLOTS && e === null; k++) if (sameRivals(sets[s0 + k], env, pool, alive, nAlive, rid, budget, fut, leg, pr, boost)) e = sets[s0 + k];
    if (e !== null) { // skip the stream to where drawing them left it
      pr.a = e.a2; pr.b = e.b2; pr.c = e.c2; pr.d = e.d2;
    } else {
      e = sets[s0 + env.claimNext[ai]];
      env.claimNext[ai] = (env.claimNext[ai] + 1) % CLAIM_SLOTS;
      rivalClaims(env, e, pool, alive, nAlive, rid, budget, fut, leg, pr, boost, false);
    }
  }
  // all claims in the order they were drawn: the rivals before me, mine, the rivals after me
  const meAt = e.meAt < 0 ? e.n : e.meAt, bt = budget[meT];
  let nc = 0;
  for (let k = 0; k < meAt; k++, nc++) {
    cB[nc] = e.B[k]; cU[nc] = e.U[k]; cT[nc] = e.T[k]; cP[nc] = e.P[k]; cG[nc] = e.G[k];
  }
  if (e.meAt >= 0 && bt > 0) {
    lookAhead(env, rid[meT], fut);
    const cMe = bm.terms(teams[meT], bt, wk);
    for (let j = 0; j < nPool; j++) {
      const i = pool[j], P = pos[i];
      let g = 0;
      for (let k = 0; k < F; k++) g += Math.max(0, projF[fo[k] + i] - el[k * NP + P]);
      g /= F;
      let b;
      if (force !== null && force[i] >= 0) b = force[i];
      else if (g < MIN_GAIN) b = 0;
      else b = Math.round(mMult * Math.exp(Bids.m(cMe, g) - env.uMe));
      b = Math.min(b, bt);
      if (b >= 1) {
        cB[nc] = b; cU[nc] = e.uMe[j]; cT[nc] = meT; cP[nc] = i; cG[nc] = g;
        nc++;
      }
    }
  }
  for (let k = meAt; k < e.n; k++, nc++) {
    cB[nc] = e.B[k]; cU[nc] = e.U[k]; cT[nc] = e.T[k]; cP[nc] = e.P[k]; cG[nc] = e.G[k];
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

function growMemo(env, mk) {
  let cap = env.memoAt.length;
  while (cap <= mk) cap *= 2;
  const v = new Float64Array(cap), at = new Int32Array(cap);
  v.set(env.memoV);
  at.set(env.memoAt);
  env.memoV = v;
  env.memoAt = at;
}

/**
 * Seasons already played on this path from just after a forced first auction. Entry k of cont holds a state (class of
 * setting, random stream, next week, who is alive, their rosters and budgets) and what the rest of that season added
 * to wins, weeks, spent and studs. Returns the entry in this state, or -1 after recording the state as entry `n`.
 */
function seasonFrom(cont, n, cls, pr, wi, alive, nAlive, rid, budget) {
  const { T, KI, KF, I, F } = cont, where = wi * (T + 1) + nAlive;
  seek: for (let k = 0; k < n; k++) {
    const o = k * KI, f = k * KF;
    if (I[o] !== cls || I[o + 1] !== pr.a || I[o + 2] !== pr.b || I[o + 3] !== pr.c || I[o + 4] !== pr.d || I[o + 5] !== where) continue;
    for (let a = 0; a < nAlive; a++) {
      const t = alive[a];
      if (I[o + 6 + a] !== t || I[o + 6 + T + a] !== rid[t] || F[f + a] !== budget[t]) continue seek;
    }
    return k;
  }
  const o = n * KI, f = n * KF;
  I[o] = cls; I[o + 1] = pr.a; I[o + 2] = pr.b; I[o + 3] = pr.c; I[o + 4] = pr.d; I[o + 5] = where;
  for (let a = 0; a < nAlive; a++) {
    const t = alive[a];
    I[o + 6 + a] = t; I[o + 6 + T + a] = rid[t]; F[f + a] = budget[t];
  }
  return -1;
}

/** job: {ci, C, seed, settings}. Returns one accumulator per setting. */
export function pathsChunk(env, job) {
  const { U, W, T, S, meT, n, weeks } = env, { ci, C, seed, settings } = job, n1 = n + 1;
  const rng = new Rng(seed, ci);
  const { score, avail } = drawPaths(U, C, rng);
  if (env.rosterOf.length > MAX_ROSTERS) { // between jobs only: no path holds an old roster number now
    env.idTab.fill(-1);
    env.trId.fill(-1);
    env.trN = 0;
    env.rosterOf.length = env.ordOf.length = env.weakOf.length = env.poolOf.length = 0;
  }
  const baseIds = Int32Array.from(env.baseRosters, (r) => intern(env, r));
  // week 0 under way: lineups are partly locked, so score every team's lock-aware lineup on all paths at once
  let week0 = null;
  if (env.week0Idx && !env.pendingPool) {
    week0 = new Float64Array(C * T);
    for (let c = 0; c < C; c++)
      for (let t = 0; t < T; t++) week0[c * T + t] = teamScore(score.subarray(c * n1), avail.subarray(c * n1), env.week0Idx, t * 3 * S, S);
  }
  const classes = new Map(); // settings that play the rest of a season alike once they are in the same state
  const sets = settings.map((st) => {
    const key = JSON.stringify([st.m_early, st.m_late, st.switch_week, st.rival_boost ?? null, !!st.track_all, !!st.fitted_me, st.chopped0 ?? null]);
    if (!classes.has(key)) classes.set(key, classes.size);
    return {
      st, boost: st.rival_boost || [0, 0], track: !!st.track_all, fitted: !!st.fitted_me, // track: keep playing after I'm chopped
      chopped0: st.chopped0 != null ? env.teams.indexOf(st.chopped0) : -1, // condition on this team going out in week 0
      cls: classes.get(key),
      tot: { wins: 0, weeks: 0, spent: 0, studs: 0, n: C, win_by: new Array(T).fill(0), weeks_by: new Array(T).fill(0),
        league_spend: new Array(W + 1).fill(0), top_price: new Array(W + 1).fill(0), burned: new Array(W + 1).fill(0) },
    };
  });
  for (const s of sets) { // my forced bids by player index (-1 = none)
    s.st = { ...s.st, forceIdx: null };
    if (s.st.force) {
      s.st.forceIdx = new Float64Array(n).fill(-1);
      for (const [p, b] of Object.entries(s.st.force)) if (env.ix.has(p)) s.st.forceIdx[env.ix.get(p)] = b;
    }
  }
  // a team's lineup score this week on a path depends only on its roster: the settings share it until they
  // diverge, so each path is played under every setting before the next path (env.memoV, by roster number and week)
  const rid = new Int32Array(T), budget = new Float64Array(T), alive = new Int32Array(T), r4 = [0, 0, 0, 0], pr = new Rng(0);
  // the seasons already played on this path from just after a forced first auction: state (class, random stream,
  // who is alive, every roster and budget) -> what the rest of the season added to wins, weeks, spent and studs
  const cont = { T, KI: 6 + 2 * T, KF: T + 4, I: new Int32Array(sets.length * (6 + 2 * T)), F: new Float64Array(sets.length * (T + 4)) };
  for (let c = 0; c < C; c++) {
    env.path++;
    env.claimNext.fill(0);
    const p0 = new Rng(seed, ci, c, 1);
    let nCont = 0;
    for (const s of sets) {
      const { st, tot, boost, track, fitted } = s;
      pr.a = p0.a; pr.b = p0.b; pr.c = p0.c; pr.d = p0.d;
      rid.set(baseIds);
      budget.set(env.budget0);
      budget[meT] = st.budget;
      let nAlive = T, open = -1, wins0 = 0, weeks0 = 0, spent0 = 0, studs0 = 0, hit = -1;
      for (let t = 0; t < T; t++) alive[t] = t;
      if (env.pendingPool) {
        const wk = weeks[0] - 1;
        auction(env, 0, env.pendingPool, alive, nAlive, rid, budget, env.fut0, wk, pr, boost, fitted, st, wk, true, r4);
        tot.spent += r4[0];
        tot.studs += r4[1];
        if (track) {
          tot.league_spend[0] += r4[2];
          tot.top_price[0] += r4[3];
        } else if ((hit = seasonFrom(cont, nCont, s.cls, pr, 0, alive, nAlive, rid, budget)) < 0) {
          open = nCont++; // the first setting here in this state: its season from here on is kept
          wins0 = tot.wins; weeks0 = tot.weeks; spent0 = tot.spent; studs0 = tot.studs;
        }
      }
      let meOut = null;
      for (let wi = 0; wi < W && hit < 0; wi++) {
        const wk = weeks[wi];
        let lowT = -1, useForce = false;
        if (wi === 0 && s.chopped0 >= 0) {
          lowT = s.chopped0;
          useForce = true;
        } else {
          const off = (wi * C + c) * n1;
          let low = Infinity;
          for (let a = 0; a < nAlive; a++) {
            const t = alive[a];
            let tt;
            if (wi === 0 && week0) tt = week0[c * T + t];
            else {
              const mk = rid[t] * W + wi;
              if (mk >= env.memoAt.length) growMemo(env, mk);
              if (env.memoAt[mk] === env.path) tt = env.memoV[mk];
              else {
                const lu = fill(env, projOrder(env, rid[t], wi), avail, off);
                tt = 0;
                for (let k = 0; k < S; k++) if (lu[k] >= 0) tt += score[off + lu[k]];
                env.memoV[mk] = tt;
                env.memoAt[mk] = env.path;
              }
            }
            if (tt < low || lowT < 0) {
              low = tt;
              lowT = t;
            }
          }
        }
        let at = 0;
        while (alive[at] !== lowT) at++;
        for (nAlive--; at < nAlive; at++) alive[at] = alive[at + 1];
        if (track) { // the league-wide tallies are only reported for the power-ranking settings
          tot.weeks_by[lowT] += wi;
          tot.burned[wi + 1] += budget[lowT];
        }
        if (lowT === meT) {
          meOut = wi;
          if (!track) break;
        }
        if (nAlive === 1 || wi === W - 1) break;
        auction(env, wi + 1, poolFor(env, rid[lowT], wi), alive, nAlive, rid, budget, env.futOf[wi], wk, pr, boost, fitted, st, wk, useForce, r4);
        tot.spent += r4[0];
        tot.studs += r4[1];
        if (track) {
          tot.league_spend[wi + 1] += r4[2];
          tot.top_price[wi + 1] += r4[3];
        } else if (useForce && open < 0) {
          if ((hit = seasonFrom(cont, nCont, s.cls, pr, wi + 1, alive, nAlive, rid, budget)) >= 0) break;
          open = nCont++;
          wins0 = tot.wins; weeks0 = tot.weeks; spent0 = tot.spent; studs0 = tot.studs;
        }
      }
      if (hit >= 0) { // a setting of this class has been here on this path, in this state: the rest of its season is this one's
        const f = hit * cont.KF + T;
        tot.wins += cont.F[f]; tot.weeks += cont.F[f + 1]; tot.spent += cont.F[f + 2]; tot.studs += cont.F[f + 3];
        continue;
      }
      if (track && nAlive === 1) {
        tot.win_by[alive[0]] += 1;
        tot.weeks_by[alive[0]] += W;
      }
      if (meOut === null) {
        tot.wins += nAlive === 1 && alive[0] === meT ? 1 : 0;
        tot.weeks += W;
      } else tot.weeks += meOut;
      if (open >= 0) {
        const f = open * cont.KF + T;
        cont.F[f] = tot.wins - wins0; cont.F[f + 1] = tot.weeks - weeks0; cont.F[f + 2] = tot.spent - spent0; cont.F[f + 3] = tot.studs - studs0;
      }
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
