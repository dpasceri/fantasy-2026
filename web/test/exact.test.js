// Shortcuts that must not change a number: settings sharing work within a path of the season-with-auctions sims,
// and the rival bid distribution worked out exactly instead of drawn. Plus one more end-to-end run.
import { test } from "node:test";
import assert from "node:assert/strict";
import { BidModel, pWinCurve, topRivalCdf, topRivalQuantile } from "../src/auction.js";
import { pathsChunk, prepDyn } from "../src/dynamic.js";
import { League } from "../src/league.js";
import { Rng } from "../src/rng.js";
import { topRivalBidsMC } from "./reference.js";
import { liveRaw, syntheticRaw } from "./synthetic.js";
import { tapRun } from "./tap.js";

const cache = {};
const league = (name, me = 5) => (cache[name + me] ??= tapRun({ pending: () => syntheticRaw(), waived: () => syntheticRaw({ waived: true }), live: liveRaw }[name](), me));

// Played alone, a setting has nobody to share rival claims or the rest of a season with; in a job with the other
// settings it shares both. The two must give the same sums.
for (const name of ["pending", "waived", "live"]) {
  test(`season-with-auctions sims: a setting plays the same alone as in a job with others (${name})`, async () => {
    const { sent, jobs } = await league(name);
    const stages = jobs.filter(([kind]) => kind === "paths").map(([, js]) => js.filter((j) => j.ci === 0).flatMap((j) => j.settings)); // every setting of a stage
    assert.equal(stages.length, 4, "sweep, budgets, power rankings, bid grids");
    const env = () => prepDyn(sent.U, structuredClone(sent.dyn));
    let shared = 0;
    for (const settings of stages) {
      const job = { ci: 1, C: 25, seed: 7, settings }, together = pathsChunk(env(), job), solo = env();
      settings.forEach((st, i) => {
        const [alone] = pathsChunk(solo, { ...job, settings: [st] });
        const pick = (t) => (st.track_all ? t : { wins: t.wins, weeks: t.weeks, spent: t.spent, studs: t.studs, n: t.n });
        assert.deepEqual(pick(together[i]), pick(alone), `setting ${i}: ${JSON.stringify(st).slice(0, 120)}`);
      });
      shared += settings.length - 1;
      assert.ok(together.some((t) => t.weeks > 0));
    }
    assert.ok(shared > 30, "the jobs had settings to share with");
  });
}

test("a team with many legal lineups runs end to end: every lineup gets every sim, the best ten are listed in order", async () => {
  const { r, sent } = await league("waived", 6); // team 6 has the positions for 1,000 lineups (team 5 has no running back)
  assert.ok(r.n_lineups > 500 && sent.week.V === r.n_lineups, `${r.n_lineups} lineups`);
  assert.ok(r.lineups.length >= 10);
  const top = r.lineups.slice(0, 10).map((l) => l.p_chop);
  assert.deepEqual(top, [...top].sort((a, b) => a - b));
  assert.ok(r.lineups.every((l) => l.players.length === 8 && l.p_chop >= 0 && l.p_chop <= 1 && l.sd > 0));
  assert.equal(r.lineups.filter((l) => l.is_proj_opt).length, 1, "the max-projection lineup is listed");
  assert.equal(r.sims.season, r.sims.lineups > 0 ? 8000 : 0, "this week's numbers: plain sims of the week alone");
  const sum = (xs) => xs.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum(r.standings.map((s) => s.p_chop_now)) - 1) < 1e-9 && Math.abs(sum(r.standings.map((s) => s.p_win)) - 1) < 1e-9);
  assert.equal(r.scenarios.length, 5, "an auction for each rival who could be chopped");
  assert.ok(r.scenarios.some((sc) => sc.players.some((p) => p.bid)), "with bid grids");
});

test("the top rival bid's exact distribution matches draws of it", () => {
  const lg = new League(syntheticRaw({ budget: 200 }), 5), bm = new BidModel(lg);
  const rivals = new Map([[3, 200], [4, 120], [6, 35], [7, 0], [8, 4]]), gains = new Map([[3, 6.5], [4, 0], [6, 11], [7, 9], [8, 3]]);
  const cdf = topRivalCdf(bm, rivals, gains, 3), n = 400_000, top = topRivalBidsMC(bm, rivals, gains, 3, new Rng(1), n);
  assert.equal(cdf.length, 201);
  assert.ok(Math.abs(cdf[200] - 1) < 1e-12 && cdf.every((x, b) => b === 0 || x >= cdf[b - 1]), "a distribution function");
  const below = new Float64Array(202);
  for (const x of top) below[x + 1]++;
  let c = 0, worst = 0;
  for (let b = 0; b <= 200; b++) {
    c += below[b + 1];
    const p = cdf[b], se = Math.sqrt(Math.max(p * (1 - p), 1e-9) / n);
    worst = Math.max(worst, Math.abs(c / n - p) / se);
    assert.ok(Math.abs(c / n - p) < 5 * se + 1e-6, `P(top <= ${b}): ${c / n} drawn, ${p} exact`);
  }
  assert.ok(worst > 0.5, "and the draws are not the formula again");
  // quantiles and the chance a bid wins (ties count half; $0 never beats a bid)
  const sorted = Float64Array.from(top).sort();
  for (const q of [0.5, 0.8, 0.9, 0.95]) assert.ok(Math.abs(topRivalQuantile(cdf, q) - sorted[Math.floor((n - 1) * q)]) <= 1);
  const pw = pWinCurve(cdf, 250);
  assert.equal(pw[0], 0);
  assert.ok(Math.abs(pw[1] - (cdf[0] + 0.5 * (cdf[1] - cdf[0]))) < 1e-15 && pw[201] === 1 && pw[250] === 1);
  for (let b = 2; b <= 250; b++) assert.ok(pw[b] >= pw[b - 1]);
  // nobody can bid: the top rival bid is always $0
  assert.deepEqual([...topRivalCdf(bm, new Map([[3, 0]]), gains, 3)], [1]);
});
