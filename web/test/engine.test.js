// The simulator: a synthetic league through the whole pipeline (runs on CI), plus exact comparisons with the
// Python version on guillotine/data when it's there (it holds a real league, so those skip on CI).
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { BidModel, gain } from "../src/auction.js";
import { bestLineup, League } from "../src/league.js";
import { fit } from "../src/model.js";
import { InlinePool } from "../src/pool.js";
import { render } from "../src/report.js";
import { runAll } from "../src/run.js";
import { Rng } from "../src/rng.js";
import { erfc, ndtr, ndtri, pyRound, quantile } from "../src/stats.js";
import { GDIR, HAVE_DATA, loadLocal, PY } from "./local-data.js";
import { syntheticRaw } from "./synthetic.js";

test("normal CDF, its inverse and quantiles", () => {
  assert.ok(Math.abs(ndtr(1.959963984540054) - 0.975) < 2e-7);
  assert.ok(Math.abs(erfc(0) - 1) < 2e-7);
  for (const p of [1e-6, 0.01, 0.3, 0.5, 0.9, 0.999999]) assert.ok(Math.abs(ndtr(ndtri(p)) - p) < 1e-7 * Math.max(1, p * 10));
  assert.equal(quantile([3, 1, 2, 4], 0.5), 2.5);
  assert.equal(pyRound(2.5), 2);
  assert.equal(pyRound(3.5), 4);
});

test("random numbers: same seeds, same stream; normals look normal", () => {
  const a = new Rng(1, 2), b = new Rng(1, 2), c = new Rng(1, 3);
  assert.equal(a.random(), b.random());
  assert.notEqual(new Rng(1, 2).random(), c.random());
  let s = 0, s2 = 0;
  const n = 200000;
  for (let i = 0; i < n; i++) {
    const z = a.normal();
    s += z;
    s2 += z * z;
  }
  assert.ok(Math.abs(s / n) < 0.01 && Math.abs(s2 / n - 1) < 0.01);
});

test("a synthetic league runs end to end and the sheet renders", async () => {
  const raw = syntheticRaw();
  const lg = new League(raw, 5);
  assert.deepEqual(lg.alive(), [3, 4, 5, 6, 7, 8]);
  assert.ok(lg.pending && lg.pending.rid === 2);
  const r = await runAll(lg, new InlinePool(), { k: 160 });
  const sum = (xs) => xs.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum(r.standings.map((s) => s.p_win)) - 1) < 1e-9, "someone wins every sim");
  assert.ok(Math.abs(sum(r.standings.map((s) => s.p_chop_now)) - 1) < 1e-9, "someone is chopped this week");
  assert.ok(r.bid_model.prior, "6 bids is too few to fit, so the prior is used");
  assert.ok(r.lineups.length > 0 && r.free_agents.length > 0 && r.scenarios.length === 1);
  for (const d of r.dynamic.budgets) assert.ok(d.p_win >= 0 && d.p_win <= 1);
  const html = render(r);
  assert.match(html, /manager5 chop sheet/i);
  assert.doesNotMatch(html, /undefined|NaN%|\[object Object\]/);
});

test("the bid prior covers leagues with under 20 bids and scales with the budget", () => {
  const lg = new League(syntheticRaw({ bids: 13 }), 5);
  assert.equal(new BidModel(lg).prior, true); // 13 < 20
  // the prior scales with the league's budget: a $1000 league bids 10x a $100 one
  const a = new BidModel(new League(syntheticRaw({ budget: 100 }), 5)), b = new BidModel(new League(syntheticRaw({ budget: 1000 }), 5));
  assert.ok(Math.abs(b.median_bid(3, 8, 1000, 3) / a.median_bid(3, 8, 100, 3) - 10) < 1e-9);
});

// ---- against the Python version on the real league snapshot (local only)
const PY_REF = `
import json, sys
sys.path.insert(0, sys.argv[1])
from common import *
import auction, model
lg = League()
rows = model.calibration_rows(lg)
bm = auction.BidModel(lg)
teams = lg.alive()
out = {
  "me": lg.me, "week": lg.week, "alive": teams, "n_calib": len(rows),
  "lineups": {str(t): [p for _, p in best_lineup(lg.slots, [(lg.proj_pts(p, lg.week), p, lg.pos(p)) for p in lg.players(t)])] for t in teams},
  "beta": bm.beta.tolist(), "sigma": bm.sigma, "a": bm.a.tolist(), "u": {str(k): v for k, v in bm.u.items()},
  "gains": [round(auction.gain(lg, lg.players(t), p, lg.week, lg.week + 3), 9) for t in teams for p in lg.players(teams[0])[:3]],
  "faab": {str(t): lg.faab(t) for t in teams},
}
print(json.dumps(out))
`;

test("matches Python on the local league snapshot", { skip: !HAVE_DATA && "no guillotine/data" }, () => {
  const py = JSON.parse(execFileSync(PY, ["-c", PY_REF, GDIR], { cwd: GDIR, env: { ...process.env, GSIM_THREADS: "1" } }).toString());
  const lg = new League(loadLocal(), py.me);
  assert.equal(lg.week, py.week);
  assert.deepEqual(lg.alive(), py.alive);
  for (const t of py.alive) {
    const lu = bestLineup(lg.slots, lg.players(t).map((p) => [lg.projPts(p, lg.week), p, lg.pos(p)])).map(([, p]) => p);
    assert.deepEqual(lu, py.lineups[t], `team ${t} lineup`);
    assert.equal(lg.faab(t), py.faab[t]);
  }
  const { report } = fit(lg);
  assert.equal(report.copula.n_obs, py.n_calib);
  const bm = new BidModel(lg);
  bm.beta.forEach((b, i) => assert.ok(Math.abs(b - py.beta[i]) < 1e-9, "bid size coefficients"));
  assert.ok(Math.abs(bm.sigma - py.sigma) < 1e-9);
  bm.a.forEach((a, i) => assert.ok(Math.abs(a - py.a[i]) < 5e-3, "bid participation coefficients (optimizer tolerance)"));
  for (const [rid, u] of Object.entries(py.u)) assert.ok(Math.abs(bm.um.get(+rid) - u) < 1e-9);
  const gains = py.alive.flatMap((t) => lg.players(py.alive[0]).slice(0, 3).map((p) => pyRound(gain(lg, lg.players(t), p, lg.week, lg.week + 3), 9)));
  assert.deepEqual(gains, py.gains);
});
