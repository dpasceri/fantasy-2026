// Port of guillotine/auction.py: the rival FAAB bid model, fit from every bid this league has placed.
//
//   P(bid > 0)          = logistic(a0 + a1 * log(1 + gain) + a2 * log(budget) + v_j)
//   log(bid) | bid > 0  = b0 + b1 * gain + b2 * log(budget) + b3 * leg + u_j + e,  e ~ N(0, s^2)
// v_j and u_j are manager effects shrunk toward 0 (v_j by a N(0, 1) prior, u_j by empirical Bayes).
// A league with fewer than MIN_BIDS bids gets PRIOR instead, with budgets in units of its starting FAAB.
import { bestLineup, eligible, POS } from "./league.js";
import { Rng } from "./rng.js";
import { lstsq, mean, quantileSorted, solveSPD, sortNum, variance } from "./stats.js";

export const HORIZON = 4;
export const MIN_BIDS = 20;
export const PRIOR = {
  a: [-3.1959181701605246, 1.7018781859274152, -0.27955391252144474],
  beta: [-0.09015987071757711, 0.1765344719074891, 0.2882372787789881, 0.0852055796510904],
  sigma: 0.9287977672850494, budget: 200,
};

/** The fitted parameters and the draws that use them; plain data, so it can go to a worker. */
export class Bids {
  constructor(p) {
    Object.assign(this, p);
    this.vm = new Map(Object.entries(p.v).map(([k, x]) => [+k, x]));
    this.um = new Map(Object.entries(p.u).map(([k, x]) => [+k, x]));
  }

  params() {
    const { a, beta, sigma, v, u, ref, shift } = this;
    return { a, beta, sigma, v, u, ref, shift };
  }

  p_bid(rid, g, budget) {
    const eta = this.a[0] + this.a[1] * Math.log1p(g) + this.a[2] * Math.log(Math.max(budget * this.ref, 1)) + (this.vm.get(rid) || 0);
    return 1 / (1 + Math.exp(-eta));
  }

  mu(rid, g, budget, leg) {
    return this.beta[0] + this.shift + this.beta[1] * g + this.beta[2] * Math.log(Math.max(budget * this.ref, 1)) +
      this.beta[3] * leg + (this.um.get(rid) || 0);
  }

  median_bid(rid, g, budget, leg) {
    return Math.min(Math.exp(this.mu(rid, g, budget, leg)), budget);
  }

  /** One rival bid in whole dollars (0 = no bid). */
  sample1(rid, g, budget, leg, rng, q = this.p_bid(rid, g, budget), m = this.mu(rid, g, budget, leg)) {
    if (budget <= 0) return 0;
    const bid = rng.random() < q;
    const amt = Math.min(Math.round(Math.exp(m + this.sigma * rng.normal())), budget);
    return bid ? amt : 0;
  }
}

export class BidModel extends Bids {
  constructor(lg) {
    const rows = bidHistory(lg).filter((r) => r.budget > 0);
    const nPos = rows.filter((r) => (r.bid || 0) > 0).length;
    super(nPos < MIN_BIDS ? priorFit(lg, rows) : fitBids(rows));
    this.lg = lg;
    this.rows = rows;
  }

  summary() {
    const r3 = (x) => Math.round(x * 1000) / 1000;
    return {
      participation: Object.fromEntries(["a0", "a_log1p_gain", "a_log_budget"].map((k, i) => [k, r3(this.a[i])])),
      size: { ...Object.fromEntries(["b0", "b_gain", "b_log_budget", "b_leg"].map((k, i) => [k, r3(this.beta[i])])),
        sigma: r3(this.sigma), tau: r3(this.tau), r2: this.prior ? NaN : r3(this.r2) },
      n_rows: this.rows.length, n_bids: this.rows.filter((r) => (r.bid || 0) > 0).length, prior: this.prior,
    };
  }
}

function counts(rows) {
  const n_bids = {}, n_offered = {};
  for (const r of rows) {
    n_offered[r.rid] = (n_offered[r.rid] || 0) + 1;
    n_bids[r.rid] = (n_bids[r.rid] || 0) + ((r.bid || 0) > 0 ? 1 : 0);
  }
  return { n_bids, n_offered };
}

function priorFit(lg, rows) {
  const ref = PRIOR.budget / lg.budget;
  return { prior: true, ref, shift: -Math.log(ref), a: [...PRIOR.a], beta: [...PRIOR.beta], sigma: PRIOR.sigma,
    v: {}, u: {}, tau: 0, r2: NaN, ...counts(rows) };
}

function fitBids(rows) {
  const rids = [...new Set(rows.map((r) => r.rid))].sort((a, b) => a - b);
  const ri = new Map(rids.map((r, k) => [r, k]));
  const p = 3 + rids.length;
  const x1 = rows.map((r) => Math.log1p(r.gain)), x2 = rows.map((r) => Math.log(r.budget));
  const y = rows.map((r) => ((r.bid || 0) > 0 ? 1 : 0)), R = rows.map((r) => ri.get(r.rid));

  // participation: penalized logistic regression (convex), by Newton's method with step halving
  const nll = (th) => {
    let s = 0;
    for (let i = 0; i < rows.length; i++) {
      const eta = th[0] + th[1] * x1[i] + th[2] * x2[i] + th[3 + R[i]];
      s += (eta > 0 ? eta + Math.log1p(Math.exp(-eta)) : Math.log1p(Math.exp(eta))) - y[i] * eta;
    }
    for (let k = 3; k < p; k++) s += 0.5 * th[k] * th[k];
    return s;
  };
  let th = new Float64Array(p);
  th.set([-6.0, 1.5, 0.3]);
  let f = nll(th);
  for (let it = 0; it < 100; it++) {
    const g = new Float64Array(p), H = Array.from({ length: p }, () => new Float64Array(p));
    for (let i = 0; i < rows.length; i++) {
      const eta = th[0] + th[1] * x1[i] + th[2] * x2[i] + th[3 + R[i]];
      const mu = 1 / (1 + Math.exp(-eta)), w = mu * (1 - mu), d = mu - y[i];
      const x = [1, x1[i], x2[i]], j = 3 + R[i];
      for (let a = 0; a < 3; a++) {
        g[a] += d * x[a];
        for (let b = 0; b < 3; b++) H[a][b] += w * x[a] * x[b];
        H[a][j] += w * x[a];
        H[j][a] += w * x[a];
      }
      g[j] += d;
      H[j][j] += w;
    }
    for (let k = 3; k < p; k++) {
      g[k] += th[k];
      H[k][k] += 1;
    }
    const step = solveSPD(H, g);
    let t = 1, next, fn;
    do {
      next = th.map((v, k) => v - t * step[k]);
      fn = nll(next);
      t /= 2;
    } while (fn > f && t > 1e-10);
    const done = f - fn < 1e-12 * (1 + Math.abs(f));
    th = next;
    f = fn;
    if (done) break;
  }
  const a = [th[0], th[1], th[2]];
  const v = Object.fromEntries(rids.map((r) => [r, th[3 + ri.get(r)]]));

  // bid size: least squares, then each manager's mean residual shrunk by empirical Bayes
  const pos = rows.filter((r) => (r.bid || 0) > 0);
  const X = pos.map((r) => [1, r.gain, Math.log(r.budget), r.leg]);
  const Y = pos.map((r) => Math.log(Math.min(r.bid, r.budget)));
  const beta = [...lstsq(X, Y)];
  const res = Y.map((yy, i) => yy - X[i].reduce((s, x, k) => s + x * beta[k], 0));
  const s2 = variance(res);
  const by = new Map();
  pos.forEach((r, i) => {
    const e = by.get(r.rid) || [0, 0];
    e[0] += res[i];
    e[1] += 1;
    by.set(r.rid, e);
  });
  const means = [...by].map(([rid, [s, n]]) => [rid, s / n, n]);
  const tau2 = Math.max(variance(means.map((m) => m[1])) - s2 / mean(means.map((m) => m[2])), 0);
  const u = Object.fromEntries(means.map(([rid, m, n]) => [rid, tau2 > 0 ? (m * n) / (n + s2 / tau2) : 0]));
  return { prior: false, ref: 1, shift: 0, a, beta, sigma: Math.sqrt(s2), v, u, tau: Math.sqrt(tau2),
    r2: 1 - s2 / variance(Y), ...counts(rows) };
}

/** Weakest starter a player of each position would displace. */
function minelig(lg, roster, w) {
  const lu = bestLineup(lg.slots, roster.map((p) => [lg.projPts(p, w), p, lg.pos(p)]));
  const out = {};
  for (const P of POS) {
    const v = lu.filter(([s]) => eligible(s, P)).map(([, p]) => (p ? lg.projPts(p, w) : 0));
    out[P] = v.length ? Math.min(...v) : Infinity;
  }
  return out;
}

/** Avg weekly lineup gain from adding pid to roster over weeks w0..w1. cache: a Map, reused per roster object. */
export function gain(lg, roster, pid, w0, w1, cache = null) {
  const last = Math.min(w1, lg.final_week);
  if (last < w0) return 0;
  let tot = 0;
  for (let w = w0; w <= last; w++) {
    let me;
    if (cache) {
      let byW = cache.get(roster);
      if (!byW) cache.set(roster, (byW = new Map()));
      me = byW.get(w);
      if (!me) byW.set(w, (me = minelig(lg, roster, w)));
    } else me = minelig(lg, roster, w);
    tot += Math.max(0, lg.projPts(pid, w) - me[lg.pos(pid)]);
  }
  return tot / (last - w0 + 1);
}

export function bidHistory(lg) {
  const rows = [];
  for (let leg = 1; leg <= lg.leg; leg++) {
    const tx = lg.tx[leg];
    const chop = tx.find((t) => t.type === "chopped");
    const chopTime = chop ? chop.status_updated : null;
    const runs = [...new Set(tx.filter((t) => t.type === "waiver").map((t) => t.status_updated))].sort((a, b) => a - b);
    const ms = new Map(lg.matchups[leg].map((m) => [m.roster_id, m]));
    for (const run of runs) {
      const claims = tx.filter((t) => t.type === "waiver" && t.status_updated === run);
      const postChop = chopTime !== null && run > chopTime;
      const w0 = postChop ? leg + 1 : leg;
      let offered = new Set(claims.map((t) => Object.keys(t.adds)[0]));
      if (postChop && chop.drops) for (const p of Object.keys(chop.drops)) offered.add(p);
      offered = [...offered].filter((p) => POS.includes(lg.pos(p)));
      // budgets just before this run
      const spent = new Map();
      const addS = (r, x) => spent.set(r, (spent.get(r) || 0) + x);
      for (let l2 = 1; l2 <= leg; l2++) {
        for (const t of lg.tx[l2]) {
          if (t.status !== "complete" || t.status_updated >= run) continue;
          if (t.type === "waiver") addS(t.roster_ids[0], t.settings.waiver_bid);
          else if (t.type === "trade")
            for (const wb of t.waiver_budget || []) {
              addS(wb.sender, wb.amount);
              addS(wb.receiver, -wb.amount);
            }
        }
      }
      const bids = new Map(), winners = new Set();
      for (const t of claims) {
        const key = `${t.roster_ids[0]}|${Object.keys(t.adds)[0]}`;
        bids.set(key, Math.max(bids.get(key) || 0, t.settings.waiver_bid));
        if (t.status === "complete") winners.add(key);
      }
      const choppedRid = chop && postChop ? chop.roster_ids[0] : null;
      const cache = new Map();
      for (const [rid, m] of ms) {
        const roster = m.players || [];
        if (!roster.length || rid === choppedRid) continue;
        const budget = lg.budget - (spent.get(rid) || 0);
        for (const pid of offered) {
          if (roster.includes(pid)) continue;
          const key = `${rid}|${pid}`;
          rows.push({ leg, run, rid, pid, budget, gain: gain(lg, roster, pid, w0, w0 + HORIZON - 1, cache),
            bid: bids.has(key) ? bids.get(key) : null, won: winners.has(key) });
        }
      }
    }
  }
  return rows;
}

/**
 * League FAAB by completed week: spent on claims, burned (chopped team's unspent budget), left with alive teams.
 * linear_left is the pace where every team spends its budget evenly over the season's total_rosters - 2 auctions.
 */
export function faabHistory(lg) {
  const N = lg.league.total_rosters, B = lg.budget, nAuctions = N - 2;
  const spent = new Map(), dead = new Set(), rows = [];
  const addS = (r, x) => spent.set(r, (spent.get(r) || 0) + x);
  for (let leg = 1; leg <= lg.leg; leg++) {
    const tx = [...lg.tx[leg]].sort((a, b) => a.status_updated - b.status_updated);
    if (!tx.some((t) => t.type === "chopped")) continue; // week still in progress
    let legSpend = 0, burned = 0, top = 0;
    const buyers = new Set();
    for (const t of tx) {
      const r = t.roster_ids && t.roster_ids.length ? t.roster_ids[0] : null;
      if (t.type === "waiver" && t.status === "complete") {
        const b = t.settings.waiver_bid;
        addS(r, b);
        legSpend += b;
        top = Math.max(top, b);
        if (b > 0) buyers.add(r);
      } else if (t.type === "trade" && t.status === "complete") {
        for (const wb of t.waiver_budget || []) {
          addS(wb.sender, wb.amount);
          addS(wb.receiver, -wb.amount);
        }
      } else if (t.type === "chopped") {
        burned += B - (spent.get(r) || 0);
        dead.add(r);
      }
    }
    const k = dead.size;
    let left = 0;
    for (const r of lg.rosters.keys()) if (!dead.has(r)) left += B - (spent.get(r) || 0);
    rows.push({ leg, spent: legSpend, burned, top_price: top, buyers: buyers.size, alive: N - k, left,
      linear_left: Math.round((N - k) * B * Math.max(0, 1 - k / nAuctions) * 10) / 10 });
  }
  return rows;
}

/** Draws of the highest rival bid. rivals: Map rid -> budget, gains: Map rid -> gain. */
export function topRivalBids(bm, rivals, gains, leg, rng, n = 200000) {
  const top = new Float64Array(n);
  for (const [rid, budget] of rivals) {
    if (budget <= 0) continue;
    const g = gains.get(rid), q = bm.p_bid(rid, g, budget), m = bm.mu(rid, g, budget, leg);
    for (let i = 0; i < n; i++) {
      const b = bm.sample1(rid, g, budget, leg, rng, q, m);
      if (b > top[i]) top[i] = b;
    }
  }
  return top;
}

/** P(my bid b beats the top rival bid) for b = 0..maxBid. Ties count half (waiver order). */
export function pWinCurve(top, maxBid) {
  const n = top.length;
  let hi = maxBid + 2;
  for (const x of top) hi = Math.max(hi, Math.trunc(x) + 1);
  const cnt = new Float64Array(hi);
  for (const x of top) cnt[Math.trunc(x)]++;
  const pw = new Float64Array(maxBid + 1);
  let below = 0;
  for (let b = 0; b <= maxBid; b++) {
    pw[b] = below / n + (0.5 * cnt[b]) / n;
    below += cnt[b];
  }
  pw[0] = 0; // a $0 claim never beats a real bid, and only wins uncontested players
  return pw;
}

/** Model's top-rival-bid distribution vs the price that actually cleared, per contested player. */
export function priceBacktest(lg, bm, seed = 5) {
  const rng = new Rng(seed);
  const byRun = new Map();
  for (const r of bm.rows) {
    const key = `${r.run}|${r.pid}`;
    if (!byRun.has(key)) byRun.set(key, []);
    byRun.get(key).push(r);
  }
  const out = [];
  for (const rs of byRun.values()) {
    const bids = rs.filter((r) => (r.bid || 0) > 0);
    if (!bids.length || Math.max(...bids.map((r) => r.bid)) < (15 * lg.budget) / 200) continue; // contested: $15+ in a $200 league
    const top = topRivalBids(bm, new Map(rs.map((r) => [r.rid, r.budget])), new Map(rs.map((r) => [r.rid, r.gain])), rs[0].leg, rng, 40000);
    const actual = Math.max(...bids.map((r) => r.bid));
    const s = sortNum(top);
    let lt = 0, eq = 0;
    for (const x of top) {
      if (x < actual) lt++;
      else if (x === actual) eq++;
    }
    out.push({ leg: rs[0].leg, player: lg.label(rs[0].pid), actual, p10: quantileSorted(s, 0.1), median: quantileSorted(s, 0.5),
      p90: quantileSorted(s, 0.9), pct_rank: (lt + 0.5 * eq) / top.length });
  }
  return out.sort((a, b) => a.leg - b.leg || b.actual - a.actual);
}
