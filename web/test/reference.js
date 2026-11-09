// Plain Monte Carlo season sims, written for clarity: one path at a time, every team and every variant of my roster
// played through the elimination directly. This is what sim.js's seasonGroups has to agree with (test/season.test.js),
// and it is the engine the season sims used before they were stratified and recombined: same draws, same numbers.
import { Rng } from "../src/rng.js";
import { drawWeek, pathState, teamScore } from "../src/sim.js";

const bin = (x) => Math.trunc(Math.min(Math.max(x * 2, 0), 399));

/** One chunk of C plain paths on a plan (see sim.js's prepSeason). Returns sums: chops[(v*T+t)*W+w], wins[v*T+t]. */
export function seasonPlain(U, plan, ci, C, seed) {
  const { S, T, V, me_t, wi0 } = plan, W = U.W, L = 3 * S, n = U.n;
  const rng = new Rng(seed, ci), st = pathState(n);
  const score = new Float32Array(n + 1), avail = new Uint8Array(n + 1);
  const G = new Float64Array(Math.max(1, ...U.n_game)), F = new Float64Array(Math.max(1, ...U.n_team));
  const chops = new Float64Array(V * T * W), wins = new Float64Array(V * T), lineHist = new Float64Array(400), myHist = new Float64Array(400);
  const alive = new Uint8Array(V * T), tot = new Float64Array(T);
  for (let c = 0; c < C; c++) {
    alive.set(plan.alive0);
    st.loglev.fill(0);
    st.outLeft.fill(0);
    for (let wi = wi0; wi < W; wi++) {
      drawWeek(U, wi, rng, st, score, avail, G, F);
      for (let t = 0; t < T; t++) tot[t] = teamScore(score, avail, plan.opp[wi], t * L, S);
      for (let v = 0; v < V; v++) {
        const mine = teamScore(score, avail, plan.mine[wi], plan.mineOf[wi][v] * L, S);
        let lo = Infinity, lt = -1, line = Infinity; // lowest alive score, earliest team on a tie
        for (let t = 0; t < T; t++) {
          if (!alive[v * T + t]) continue;
          const x = t === me_t ? mine : tot[t];
          if (x < lo) { lo = x; lt = t; }
          if (t !== me_t && x < line) line = x;
        }
        if (lt < 0) continue;
        if (v === 0 && wi === 0) {
          lineHist[bin(line)]++;
          myHist[bin(mine)]++;
        }
        alive[v * T + lt] = 0;
        chops[(v * T + lt) * W + wi]++;
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

/**
 * What seasonGroups must add up to for one group, by brute force from the group's lineup totals (E.tot) and the
 * rounds' path choices: every recombined season and every pairing played out one at a time. (A case's variants sit
 * next to each other in a plan, which `first` relies on.)
 */
export function groupBrute(E, pathOf, R) {
  const { plans, K, T, W, me_t } = E;
  const chops = new Float64Array(T * W), wins = new Float64Array(T);
  const tot = (w, row, k) => E.tot[E.tb[w] + row * K + k];
  const pick = (r, j, t) => (j + pathOf[r * T + t]) % K;
  for (let r = 0; r < R; r++)
    for (let j = 0; j < K; j++) {
      const alive = new Array(T).fill(true);
      for (let w = 0; w < W; w++) {
        let lt = -1;
        for (let t = 0; t < T; t++) if (alive[t] && (lt < 0 || tot(w, t, pick(r, j, t)) < tot(w, lt, pick(r, j, lt)))) lt = t;
        alive[lt] = false;
        chops[lt * W + w]++;
      }
      wins[alive.indexOf(true)]++;
    }
  const out = plans.map((p, pi) => {
    const res = { wins: new Float64Array(p.V), differs: new Float64Array(p.V) };
    const mask = (v) => p.alive0.subarray(v * T, (v + 1) * T).join("");
    const won = (v, m, j) => { // my path m against the rivals of path j: do I win?
      const alive = Array.from({ length: T }, (_, t) => !!p.alive0[v * T + t]);
      for (let w = p.wi0; w < W; w++) {
        let lt = -1, lx = Infinity;
        for (let t = 0; t < T; t++) {
          if (!alive[t]) continue;
          const x = t === me_t ? tot(w, E.rowOff[w][pi] + p.mineOf[w][v], m) : tot(w, t, j);
          if (x < lx) { lx = x; lt = t; }
        }
        if (lt < 0) break;
        alive[lt] = false;
        if (lt === me_t) return false;
      }
      return true;
    };
    for (let v = 0; v < p.V; v++) {
      let first = v; // the first variant with the same rivals
      while (first > 0 && mask(first - 1) === mask(v)) first--;
      for (let m = 0; m < K; m++) {
        for (let j = 0; j < K; j++) if (won(v, m, j)) res.wins[v]++;
        if (won(v, m, m) !== won(first, m, m)) res.differs[v]++; // the path as drawn
      }
    }
    return res;
  });
  return { chops, wins, plans: out };
}

/**
 * Draws of the highest rival bid, as auction.py's top_rival_bids makes them (the web app used to as well; it now
 * works the distribution out exactly, see auction.js's topRivalCdf). rivals: Map rid -> budget, gains: Map rid -> gain.
 */
export function topRivalBidsMC(bm, rivals, gains, leg, rng, n) {
  const top = new Float64Array(n);
  for (const [rid, budget] of rivals) {
    if (budget <= 0) continue;
    const g = gains.get(rid), q = bm.p_bid(rid, g, budget), m = bm.mu(rid, g, budget, leg), sigma = bm.sigma;
    for (let i = 0; i < n; i++) {
      const bids = rng.random() < q, z = rng.normal();
      if (bids) top[i] = Math.max(top[i], Math.min(Math.round(Math.exp(m + sigma * z)), budget));
    }
  }
  return top;
}
