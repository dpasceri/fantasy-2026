// Port of guillotine/dynamic.py: season-with-auctions sims (what is a FAAB dollar worth, how hard to bid).
//
// Each path plays the rest of the season week by week: lineups from projections (known injuries benched),
// lowest alive score chopped, then the chopped roster goes to a sealed-bid FAAB auction. Rivals bid from the
// fitted bid model; I bid m x the league's typical bid for my gain. Highest bid wins and pays its bid, claims
// are processed from the highest bid down, budgets and the roster cap are enforced.
//
// This file runs inside workers: env holds plain data only (see run.js), plus `mcache`, the worker's cache of
// weakest-starter lookups. Both caches here return exactly what recomputing would: they're keyed on everything
// the value depends on, including roster order (it breaks ties between equal projections).
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

/** minelig for a roster (by its key) in week w2: depends only on the roster and that week's projections. */
function mineligCached(ctx, mcache, roster, rkey, w2) {
  const key = `${w2}|${rkey}`;
  let v = mcache.get(key);
  if (v === undefined) {
    if (mcache.size > 400_000) mcache.clear();
    v = minelig(ctx.slots, roster, ctx.proj[w2], ctx.posOf, ctx.ix);
    mcache.set(key, v);
  }
  return v;
}

/** One waiver run, in place (rosters and their keys). Returns [my spend, my studs, league spend, top price]. */
function auction(ctx, mcache, pool, alive, rosters, keys, budget, myBid, fut, leg, pr, boost, fittedMe) {
  const { me, proj, posOf, ix, bm } = ctx;
  const rosNext = ctx.rosVal[fut[0]];
  const claims = [];
  for (const t of alive) {
    const el = budget.get(t) > 0 ? fut.map((w2) => mineligCached(ctx, mcache, rosters.get(t), keys.get(t), w2)) : null;
    for (const p of pool) {
      const uPart = pr.random(), z = pr.normal(), uTie = pr.random(); // always drawn: keeps paths aligned
      if (!el) continue;
      const P = posOf.get(p), i = ix.get(p);
      let g = 0;
      fut.forEach((w2, k) => { g += Math.max(0, proj[w2][i] - el[k][P]); });
      g /= fut.length;
      let b;
      if (t === me && !fittedMe) b = myBid(p, g);
      else {
        let q = bm.p_bid(t, g, budget.get(t));
        if (boost[0]) q = 1 / (1 + Math.exp(-(Math.log(q / (1 - q)) + boost[0])));
        b = uPart < q ? Math.round(Math.exp(bm.mu(t, g, budget.get(t), leg) + boost[1] + bm.sigma * z)) : 0;
      }
      b = Math.min(b, budget.get(t));
      if (b >= 1) claims.push([b, uTie, t, p, g]);
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
    const r = rosters.get(t);
    r.push(p);
    if (r.length > ctx.maxRoster) {
      let drop = -1;
      for (let j = 0; j < r.length; j++) if (r[j] !== p && (drop < 0 || rosNext[ix.get(r[j])] < rosNext[ix.get(r[drop])])) drop = j;
      r.splice(drop, 1);
    }
    keys.set(t, r.join(","));
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
  const mcache = env.mcache, baseKeys = new Map(base.teams.map((t) => [t, base.rosters[t].join(",")]));
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
      const rosters = new Map(base.teams.map((t) => [t, [...base.rosters[t]]]));
      const keys = new Map(baseKeys);
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
        const r = auction(ctx, mcache, base.pending_pool, alive, rosters, keys, budget, forced(wk), [0, 1, 2, 3].slice(0, W), wk, pr, boost, fitted);
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
              const rk = keys.get(t);
              let memo = lcache[wi].get(rk);
              if (!memo) lcache[wi].set(rk, (memo = new Float64Array(C).fill(NaN)));
              tt = memo[c];
              if (Number.isNaN(tt)) {
                const cands = [];
                for (const p of rosters.get(t)) {
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
        const pool = rosters.get(lowT).filter((p) => ctx.rosVal[wi + 1][ctx.ix.get(p)] > 0);
        const fut = [];
        for (let w2 = wi + 1; w2 < Math.min(wi + 1 + HORIZON, W); w2++) fut.push(w2);
        const r = auction(ctx, mcache, pool, alive, rosters, keys, budget, myBid, fut, wk, pr, boost, fitted);
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
