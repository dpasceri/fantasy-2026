// Port of guillotine/dynamic.py: season-with-auctions sims (what is a FAAB dollar worth, how hard to bid).
//
// Each path plays the rest of the season week by week: lineups from projections (known injuries benched),
// lowest alive score chopped, then the chopped roster goes to a sealed-bid FAAB auction. Rivals bid from the
// fitted bid model; I bid m x the league's typical bid for my gain. Highest bid wins and pays its bid, claims
// are processed from the highest bid down, budgets and the roster cap are enforced.
//
// This file runs inside workers: env holds plain data only (see run.js), plus the worker's caches: `ids` numbers
// each distinct roster (players in order, which also breaks ties between equal projections) and `mcache` holds
// weakest-starter lookups by roster number and week. Every cache returns exactly what recomputing would.
import { Bids } from "./auction.js";
import { bestLineup, eligible, POS } from "./league.js";
import { Rng } from "./rng.js";
import { drawWeek, pathState, teamScore } from "./sim.js";

export const HORIZON = 4;
export const MIN_GAIN = 2.0; // I don't bid on players worth less than this many lineup points a week

function minelig(slots, roster, projw, posOf, ix) {
  const lu = bestLineup(slots, roster.map((p) => [projw[ix.get(p)], p, posOf.get(p)]));
  const out = {};
  for (const P of POS) {
    let m = Infinity;
    for (const [s, p] of lu) if (eligible(s, P)) m = Math.min(m, p ? projw[ix.get(p)] : 0);
    out[P] = m;
  }
  return out;
}

function intern(ids, key) {
  let id = ids.get(key);
  if (id === undefined) ids.set(key, (id = ids.size));
  return id;
}

/** Weakest starter by position for team t's roster in week w2 (depends only on the roster and the week). */
function weakest(env, R, t, w2) {
  const { ctx } = env, id = R.ids.get(t);
  let byWeek = env.mcache.get(id);
  if (!byWeek) env.mcache.set(id, (byWeek = new Array(env.U.W)));
  return (byWeek[w2] ??= minelig(ctx.slots, R.rosters.get(t), ctx.proj[w2], ctx.posOf, ctx.ix));
}

/**
 * One waiver run, in place. R = {rosters, ids, owned} for this path: rosters start as the shared base arrays and
 * are copied the first time an auction changes them. Returns [my spend, my studs, league spend, top price].
 */
function auction(env, pool, alive, R, budget, myBid, fut, leg, pr, boost, fittedMe) {
  const { me, proj, posOf, ix, bm } = env.ctx;
  const rosNext = env.ctx.rosVal[fut[0]], F = fut.length;
  const pI = pool.map((p) => ix.get(p)), pP = pool.map((p) => posOf.get(p));
  const claims = [];
  for (const t of alive) {
    const bt = budget.get(t);
    const el = bt > 0 ? fut.map((w2) => weakest(env, R, t, w2)) : null;
    const c = el && (t !== me || fittedMe) ? bm.terms(t, bt, leg) : null;
    const q0 = c ? Bids.q(c, 0) : 0; // most players add nothing to most lineups
    for (let j = 0; j < pool.length; j++) {
      const uPart = pr.random(), z = pr.normal(), uTie = pr.random(); // always drawn: keeps paths aligned
      if (!el) continue;
      const P = pP[j], i = pI[j];
      let g = 0;
      for (let k = 0; k < F; k++) g += Math.max(0, proj[fut[k]][i] - el[k][P]);
      g /= F;
      let b;
      if (t === me && !fittedMe) b = myBid(pool[j], g);
      else {
        let q = g === 0 ? q0 : Bids.q(c, g);
        if (boost[0]) q = 1 / (1 + Math.exp(-(Math.log(q / (1 - q)) + boost[0])));
        b = uPart < q ? Math.round(Math.exp(Bids.m(c, g) + boost[1] + bm.sigma * z)) : 0;
      }
      b = Math.min(b, bt);
      if (b >= 1) claims.push([b, uTie, t, pool[j], g]);
    }
  }
  claims.sort((x, y) => y[0] - x[0] || y[1] - x[1]);
  const taken = new Set();
  let spent = 0, studs = 0, total = 0, top = 0;
  for (const [b, , t, p, g] of claims) {
    if (taken.has(p) || budget.get(t) < b) continue;
    taken.add(p);
    budget.set(t, budget.get(t) - b);
    total += b;
    top = Math.max(top, b);
    let r = R.rosters.get(t);
    if (!R.owned.has(t)) {
      R.rosters.set(t, (r = [...r]));
      R.owned.add(t);
    }
    r.push(p);
    if (r.length > env.ctx.maxRoster) {
      let drop = -1;
      for (let j = 0; j < r.length; j++) if (r[j] !== p && (drop < 0 || rosNext[ix.get(r[j])] < rosNext[ix.get(r[drop])])) drop = j;
      r.splice(drop, 1);
    }
    R.ids.set(t, intern(env.ids, r.join(",")));
    if (t === me) {
      spent += b;
      studs += g >= 8 ? 1 : 0;
    }
  }
  return [spent, studs, total, top];
}

/** All weeks of C paths: score/avail as Float32Array/Uint8Array (W * C * (n + 1)). */
function drawPaths(U, C, rng) {
  const n1 = U.n + 1, W = U.W;
  const score = new Float32Array(W * C * n1), avail = new Uint8Array(W * C * n1);
  const G = new Float64Array(Math.max(1, ...U.n_game)), T = new Float64Array(Math.max(1, ...U.n_team));
  const s = new Float32Array(n1), a = new Uint8Array(n1);
  for (let c = 0; c < C; c++) {
    const st = pathState(U.n);
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
  const { U, ctx, base } = env, { ci, C, seed, settings } = job;
  const W = U.W, n1 = U.n + 1, weeks = base.weeks, me = ctx.me, bm = ctx.bm;
  const rng = new Rng(seed, ci);
  const { score, avail } = drawPaths(U, C, rng);
  const T = base.teams.length, tix = new Map(base.teams.map((t, i) => [t, i]));
  if (env.ids.size > 300_000) { // between jobs only: no path holds an old roster number now
    env.ids.clear();
    env.mcache.clear();
  }
  const baseIds = new Map(base.teams.map((t) => [t, intern(env.ids, base.rosters[t].join(","))]));
  // a team's lineup score this week on path c depends only on its roster: settings share it until they diverge
  const lcache = Array.from({ length: W }, () => new Map());
  // week 0 under way: lineups are partly locked, so score every team's lock-aware lineup on all paths at once
  let week0 = null;
  if (base.week0Idx && !base.pending_pool) {
    week0 = new Float64Array(C * T);
    for (let c = 0; c < C; c++)
      for (let t = 0; t < T; t++) week0[c * T + t] = teamScore(score.subarray(c * n1), avail.subarray(c * n1), base.week0Idx, t * 3 * base.S, base.S);
  }
  const out = [];
  for (const st of settings) {
    const track = !!st.track_all; // keep playing after I'm chopped, to see who wins
    const fitted = !!st.fitted_me; // I bid like my own history (for a fair power ranking)
    const tot = { wins: 0, weeks: 0, spent: 0, studs: 0, n: C, win_by: new Array(T).fill(0), weeks_by: new Array(T).fill(0),
      league_spend: new Array(W + 1).fill(0), top_price: new Array(W + 1).fill(0), burned: new Array(W + 1).fill(0) };
    const boost = st.rival_boost || [0, 0];
    const force = st.force || {};
    const uMe = bm.um.get(me) || 0;
    for (let c = 0; c < C; c++) {
      const pr = new Rng(seed, ci, c, 1);
      const R = { rosters: new Map(base.teams.map((t) => [t, base.rosters[t]])), ids: new Map(baseIds), owned: new Set() };
      const budget = new Map(base.teams.map((t) => [t, base.budget[t]]));
      budget.set(me, st.budget);
      const alive = [...base.teams];
      const policy = (p, g, wk) => {
        if (g < MIN_GAIN) return 0;
        const m = wk < st.switch_week ? st.m_early : st.m_late;
        return Math.round(m * Math.exp(bm.mu(me, g, budget.get(me), wk) - uMe));
      };
      const forced = (wk) => (p, g) => (p in force ? force[p] : policy(p, g, wk));
      if (base.pending_pool) {
        const wk = weeks[0] - 1;
        const r = auction(env, base.pending_pool, alive, R, budget, forced(wk), [0, 1, 2, 3].slice(0, W), wk, pr, boost, fitted);
        tot.spent += r[0];
        tot.studs += r[1];
        tot.league_spend[0] += r[2];
        tot.top_price[0] += r[3];
      }
      let meOut = null;
      for (let wi = 0; wi < W; wi++) {
        const wk = weeks[wi];
        let lowT, myBid;
        if (wi === 0 && st.chopped0 != null) {
          lowT = st.chopped0; // condition on this team going out this week
          myBid = forced(wk);
        } else {
          const off = (wi * C + c) * n1, pw = ctx.proj[wi];
          let low = Infinity;
          for (const t of alive) {
            let tt;
            if (wi === 0 && week0) tt = week0[c * T + tix.get(t)];
            else {
              const rk = R.ids.get(t);
              let memo = lcache[wi].get(rk);
              if (!memo) lcache[wi].set(rk, (memo = new Float64Array(C).fill(NaN)));
              tt = memo[c];
              if (Number.isNaN(tt)) {
                const cands = [];
                for (const p of R.rosters.get(t)) {
                  const i = ctx.ix.get(p);
                  if (avail[off + i]) cands.push([pw[i], p, ctx.posOf.get(p)]);
                }
                tt = 0;
                for (const [, p] of bestLineup(ctx.slots, cands)) if (p) tt += score[off + ctx.ix.get(p)];
                memo[c] = tt;
              }
            }
            if (tt < low || lowT === undefined) {
              low = tt;
              lowT = t;
            }
          }
          myBid = (p, g) => policy(p, g, wk);
        }
        alive.splice(alive.indexOf(lowT), 1);
        tot.weeks_by[tix.get(lowT)] += wi;
        tot.burned[wi + 1] += budget.get(lowT);
        if (lowT === me) {
          meOut = wi;
          if (!track) break;
        }
        if (alive.length === 1 || wi === W - 1) break;
        const pool = R.rosters.get(lowT).filter((p) => ctx.rosVal[wi + 1][ctx.ix.get(p)] > 0);
        const fut = [];
        for (let w2 = wi + 1; w2 < Math.min(wi + 1 + HORIZON, W); w2++) fut.push(w2);
        const r = auction(env, pool, alive, R, budget, myBid, fut, wk, pr, boost, fitted);
        tot.spent += r[0];
        tot.studs += r[1];
        tot.league_spend[wi + 1] += r[2];
        tot.top_price[wi + 1] += r[3];
      }
      if (alive.length === 1) {
        tot.win_by[tix.get(alive[0])] += 1;
        tot.weeks_by[tix.get(alive[0])] += W;
      }
      if (meOut === null) {
        tot.wins += alive.length === 1 && alive[0] === me ? 1 : 0;
        tot.weeks += W;
      } else tot.weeks += meOut;
    }
    out.push(tot);
  }
  return out;
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
