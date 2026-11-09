// The season engine (sim.js's seasonGroups: stratified draws in groups, recombined) against plain Monte Carlo written
// the obvious way (reference.js). Three things are checked on a synthetic league, with an auction pending and with a
// week under way: its bookkeeping is exact (every recombined season and pairing, replayed one at a time), it is
// unbiased (same numbers as plain sims, within noise), and it is worth it (a path beats a plain one by a margin).
// Seeds are fixed, so each run of these tests sees the same draws.
import { test } from "node:test";
import assert from "node:assert/strict";
import { InlinePool } from "../src/pool.js";
import { runSeason, seasonPrecision } from "../src/run.js";
import { GROUP, GROUP_LIVE, prepSeason, seasonGroups, week0Chunk } from "../src/sim.js";
import { groupBrute, seasonPlain } from "./reference.js";
import { liveRaw, syntheticRaw } from "./synthetic.js";
import { tapRun } from "./tap.js";

const cache = {};
const league = (name) => (cache[name] ??= tapRun(name === "live" ? liveRaw() : syntheticRaw(), 5));
const sum = (parts, get) => parts.reduce((a, p) => { get(p).forEach((x, i) => { a[i] += x; }); return a; }, new Float64Array(get(parts[0]).length));

for (const name of ["pending", "live"]) {
  test(`season sums equal a brute-force replay of every recombined season and pairing (${name})`, async () => {
    const { sent } = await league(name), plans = [sent.plan, sent.plan2];
    for (const [K, rounds] of [[4, 1], [8, 5], [16, 3], [32, 2]]) {
      const E = prepSeason(sent.U, plans, K), want = { chops: new Float64Array(E.T * E.W), wins: new Float64Array(E.T), plans: plans.map((p) => [new Float64Array(p.V), new Float64Array(p.V)]) };
      let worst = 0;
      const onGroup = (E, pathOf, R) => {
        const b = groupBrute(E, pathOf, R);
        b.chops.forEach((x, i) => { want.chops[i] += x; });
        b.wins.forEach((x, i) => { want.wins[i] += x; });
        b.plans.forEach((p, pi) => { p.wins.forEach((x, v) => { want.plans[pi][0][v] += x; }); p.differs.forEach((x, v) => { want.plans[pi][1][v] += x; }); });
        // the last week's lineup totals against a slot-by-slot sum of that week's scores (still in E.score / E.avail)
        const w = E.W - 1, rows = [plans[0].opp[w], ...plans.map((p) => p.mine[w])];
        let u = 0;
        for (const r of rows)
          for (let o = 0; o < r.length; o += E.L, u++)
            for (let k = 0; k < K; k++) {
              let t = 0;
              for (let s = 0; s < E.S; s++) {
                const a = r[o + s] * K + k, b1 = r[o + E.S + s] * K + k, b2 = r[o + 2 * E.S + s] * K + k;
                t += E.avail[a] ? E.score[a] : E.avail[b1] ? E.score[b1] : E.avail[b2] ? E.score[b2] : 0;
              }
              worst = Math.max(worst, Math.abs(E.tot[E.tb[w] + u * K + k] - t));
            }
      };
      const got = seasonGroups(E, { ci: 3, G: 3, seed: 99, rounds, onGroup });
      assert.deepEqual(got.chops, want.chops, `chops, K=${K}`);
      assert.deepEqual(got.wins, want.wins, `wins, K=${K}`);
      got.plans.forEach((p, pi) => {
        assert.deepEqual(p.wins, want.plans[pi][0], `plan ${pi} wins, K=${K}`);
        assert.deepEqual(p.differs, want.plans[pi][1], `plan ${pi}: paths that end differently from the first variant's, K=${K}`);
        assert.ok(p.differs.every((x) => x <= got.paths));
        assert.ok(p.wins.every((x) => x >= 0 && x <= got.pairs));
      });
      assert.equal(got.seasons, 3 * K * rounds);
      assert.equal(got.pairs, 3 * K * K);
      assert.equal(got.wins.reduce((a, b) => a + b), got.seasons, "one winner per season");
      assert.ok(worst < 1e-9, `lineup totals match a direct sum (off by ${worst})`);
    }
  });

  test(`season sims agree with plain Monte Carlo, and a path is worth more than a plain one (${name})`, async () => {
    const { sent } = await league(name), U = sent.U, plans = [sent.plan, sent.plan2], [P1, P2] = plans, { T, me_t: me } = P1, W = U.W;
    const K = name === "live" ? GROUP_LIVE : GROUP, C = 8000, nPlain = 8, nJobs = 16, G = 2000 / K;
    // plain: every team's chop week and win, my win per roster variant of both plans
    const pa = [], pb = [];
    for (let ci = 0; ci < nPlain; ci++) {
      pa.push(seasonPlain(U, P1, ci, C, 11));
      pb.push(seasonPlain(U, P2, ci, C, 12));
    }
    const N = nPlain * C, ca = sum(pa, (p) => p.chops), wa = sum(pa, (p) => p.wins), wb = sum(pb, (p) => p.wins);
    const plain = [...ca.subarray(0, T * W), ...wa.subarray(0, T), ...Array.from({ length: P1.V }, (_, v) => wa[v * T + me]),
      ...Array.from({ length: P2.V }, (_, v) => wb[v * T + me])].map((x) => x / N);
    const E = prepSeason(U, plans, K), jobs = [];
    for (let ci = 0; ci < nJobs; ci++) {
      const r = seasonGroups(E, { ci, G, seed: 21 });
      jobs.push([...[...r.chops].map((x) => x / r.seasons), ...[...r.wins].map((x) => x / r.seasons), ...[...r.plans[0].wins].map((x) => x / r.pairs),
        ...[...r.plans[1].wins].map((x) => x / r.pairs)]);
    }
    const z = [], worth = [];
    plain.forEach((p, i) => {
      const xs = jobs.map((j) => j[i]), m = xs.reduce((a, b) => a + b) / nJobs, v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / (nJobs - 1);
      const se = Math.sqrt(v / nJobs + (p * (1 - p)) / N);
      if (se > 0) z.push((m - p) / se);
      if (i < T * W + T && p > 0.03 && p < 0.97 && v > 0) worth.push((p * (1 - p)) / (G * K * v)); // plain paths per path, for the teams' odds
    });
    const rms = Math.sqrt(z.reduce((a, b) => a + b * b, 0) / z.length), max = Math.max(...z.map(Math.abs));
    assert.ok(z.length > 60);
    assert.ok(rms < 1.3 && max < 4, `z-scores against plain sims look like noise (rms ${rms.toFixed(2)}, max ${max.toFixed(2)} of ${z.length})`);
    worth.sort((a, b) => a - b);
    const median = worth[worth.length >> 1];
    assert.ok(median > 2, `a path is worth ${median.toFixed(1)} plain ones at the median`);
  });
}

test("a run measures what its season paths were worth and draws more when that falls short", async () => {
  const { sent } = await league("live"), K = GROUP_LIVE, plans = [sent.plan, sent.plan2];
  // what each variant is valued against: the first variant with the same rivals (reference.js's groupBrute does the same)
  const bases = plans.map((p) => {
    const mask = (v) => p.alive0.subarray(v * p.T, (v + 1) * p.T).join("");
    return Int32Array.from({ length: p.V }, (_, v) => { let b = v; while (b > 0 && mask(b - 1) === mask(v)) b--; return b; });
  });
  const E = prepSeason(sent.U, plans, K), parts = [];
  for (let ci = 0; ci < 16; ci++) parts.push(seasonGroups(E, { ci, G: 1000 / K, seed: 3 }));
  const paths = 16 * 1000, easy = seasonPrecision(parts, bases, [paths, paths]), hard = seasonPrecision(parts, bases, [20 * paths, 20 * paths]);
  assert.ok(easy.worth > 1.5 * paths, `the teams' odds are worth ${Math.round(easy.worth)} plain paths, from ${paths}`);
  assert.ok(easy.short < 1, `enough for as many plain paths (${easy.short.toFixed(2)})`);
  assert.ok(Math.abs(hard.short / easy.short - 20) < 1e-9 && hard.short > 1, "20 times the target takes 20 times the paths");
  assert.ok(Math.abs(hard.worth - easy.worth) < 1e-6, "what the paths are worth doesn't depend on the target");
  // the runner: same jobs, and it tops up only when short (to at most 2.5 times the paths)
  const pool = new InlinePool();
  for (const k of ["U", "plan", "plan2"]) await pool.set(k, sent[k]);
  const ok = await runSeason(pool, ["plan", "plan2"], 64_000, 3, K, bases, [64_000, 32_000]);
  const more = await runSeason(pool, ["plan", "plan2"], 64_000, 3, K, bases, [1_280_000, 640_000]);
  assert.equal(ok.N, 64_000);
  assert.ok(ok.short <= 1 && ok.worth > 64_000);
  assert.equal(more.N, 160_000, "short of 20 times the target: topped up to the cap");
  assert.ok(more.worth > ok.worth, "and more paths are worth more");
  const few = await runSeason(pool, ["plan", "plan2"], 8_000, 3, K, bases, [640_000, 320_000]);
  assert.deepEqual([few.N, few.worth], [8_000, null], "a quick run has too few jobs to measure, and is left alone");
});

test("a week under way: this week's numbers from plain sims of the week alone agree with the season sims' first week", async () => {
  const { sent } = await league("live"), U = sent.U, P = sent.plan, { T, V, me_t: me } = P, W = U.W, C = 8000, n = 6;
  assert.ok(U.anyLocked, "some players are locked");
  const a = [], b = [];
  for (let ci = 0; ci < n; ci++) {
    a.push(week0Chunk(U, P, ci, C, 5));
    b.push(seasonPlain(U, P, ci, C, 6));
  }
  const N = n * C, chopA = sum(a, (p) => p.chops), outA = sum(a, (p) => p.firstOut), all = sum(b, (p) => p.chops);
  assert.equal(chopA.reduce((x, y) => x + y), N, "one team is chopped per sim");
  const z = [];
  const cmp = (x, y) => { const p = (x + y) / (2 * N), se = Math.sqrt((2 * p * (1 - p)) / N); if (se > 0) z.push((x - y) / N / se); };
  for (let t = 0; t < T; t++) cmp(chopA[t], all[t * W]); // variant 0, team t, week 0
  for (let v = 0; v < V; v++) cmp(outA[v], all[(v * T + me) * W]); // my chop this week per roster variant
  const max = Math.max(...z.map(Math.abs));
  assert.ok(z.length > 5 && max < 4, `max z ${max.toFixed(2)} of ${z.length}`);
  // the two histograms hold every sim, and my score histogram is the same law in both
  const hist = (parts, k) => sum(parts, (p) => p[k]);
  assert.equal(hist(a, "lineHist").reduce((x, y) => x + y), N);
  const med = (h) => { let c = 0, i = 0; while ((c += h[i]) < N / 2) i++; return i / 2; };
  assert.ok(Math.abs(med(hist(a, "myHist")) - med(hist(b, "myHist"))) <= 1 && Math.abs(med(hist(a, "lineHist")) - med(hist(b, "lineHist"))) <= 1);
});
