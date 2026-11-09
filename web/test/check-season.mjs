// The season engine against plain Monte Carlo, at a size you choose: the long version of test/season.test.js, for
// after a change to sim.js's season sims. Not part of the test run (it takes minutes).
//
//   node test/check-season.mjs [plain chunks = 40] [engine jobs = 80] [team]
//
// It uses the league in guillotine/data when there is one (team: a Sleeper name or roster id; default the first team
// alive) and the synthetic league otherwise. Plain chunks are 8,000 seasons for each of the two season plans; engine
// jobs are 4,000 paths. Read the output like this: the z-scores of engine minus plain should look like noise (rms
// near 1, few beyond 3, whatever the size); a bias shows up as an rms that grows as you raise the counts.
import { GROUP, GROUPS, prepSeason, seasonGroups } from "../src/sim.js";
import { seasonPrecision, SEASON_PATHS, SEASON_PLAIN } from "../src/run.js";
import { HAVE_DATA, loadLocal } from "./local-data.js";
import { seasonPlain } from "./reference.js";
import { syntheticRaw } from "./synthetic.js";
import { tapRun } from "./tap.js";

const [nPlain = 40, nJobs = 80] = process.argv.slice(2, 4).map(Number), C = 8000;
const raw = HAVE_DATA ? loadLocal() : syntheticRaw({ waived: true });
const who = process.argv[4] ?? raw.rosters.find((r) => (r.players || []).length).roster_id;
const { sent, lg } = await tapRun(raw, /^\d+$/.test(String(who)) ? +who : who);
const U = sent.U, plans = [sent.plan, sent.plan2], [P1, P2] = plans, { T, me_t: me } = P1, W = U.W;
console.log(`${HAVE_DATA ? "local league" : "synthetic league"}: week ${lg.week}, ${T} teams, ${W} weeks, ${U.n} players, ${P1.V} + ${P2.V} roster variants`);

const sum = (parts, get) => parts.reduce((a, p) => { get(p).forEach((x, i) => { a[i] += x; }); return a; }, new Float64Array(get(parts[0]).length));
let t0 = performance.now();
const pa = [], pb = [];
for (let ci = 0; ci < nPlain; ci++) {
  pa.push(seasonPlain(U, P1, ci, C, 101));
  pb.push(seasonPlain(U, P2, ci, C, 102));
}
const N = nPlain * C, ca = sum(pa, (p) => p.chops), wa = sum(pa, (p) => p.wins), wb = sum(pb, (p) => p.wins);
console.log(`plain: ${N.toLocaleString()} seasons per plan, ${((performance.now() - t0) / 1000).toFixed(0)} s`);

t0 = performance.now();
const E = prepSeason(U, plans, GROUP), jobs = [], parts = [];
for (let ci = 0; ci < nJobs; ci++) {
  const r = seasonGroups(E, { ci, G: GROUPS, seed: 103 });
  parts.push(r);
  jobs.push([[...r.chops].map((x) => x / r.seasons), [...r.wins].map((x) => x / r.seasons), [...r.plans[0].wins].map((x) => x / r.pairs), [...r.plans[1].wins].map((x) => x / r.pairs)]);
}
const paths = nJobs * GROUPS * GROUP;
console.log(`engine: ${paths.toLocaleString()} paths, ${((performance.now() - t0) / 1000).toFixed(0)} s`);

// what each variant is valued against: the first variant with the same rivals
const bases = plans.map((p) => {
  const mask = (v) => p.alive0.subarray(v * p.T, (v + 1) * p.T).join("");
  return Int32Array.from({ length: p.V }, (_, v) => { let b = v; while (b > 0 && mask(b - 1) === mask(v)) b--; return b; });
});
// A plain season's variance: p(1 - p) for odds. For a value (a difference of two win odds on the same seasons) it is
// the share of seasons that end differently with the player than without, less the value squared; the engine counts
// those seasons on its paths as drawn (differs), which pins it far better than a few plain chunks' spread would.
const paired = (pi, v, d) => parts.reduce((a, r) => a + r.plans[pi].differs[v], 0) / paths - d * d;
const auction = [...bases[1].keys()].filter((v) => bases[1][v] !== v);
const groups = [ // name, plain values, engine values per job, a plain season's variance
  ["a team's chop in a week", [...ca.subarray(0, T * W)].map((x) => x / N), (j) => j[0], (i, p) => p * (1 - p)],
  ["a team's win", [...wa.subarray(0, T)].map((x) => x / N), (j) => j[1], (i, p) => p * (1 - p)],
  ["value of a free agent", Array.from({ length: P1.V - 1 }, (_, i) => (wa[(i + 1) * T + me] - wa[me]) / N), (j) => j[2].slice(1).map((x) => x - j[2][0]), (i, d) => paired(0, i + 1, d)],
  ["value of an auction player", auction.map((v) => (wb[v * T + me] - wb[bases[1][v] * T + me]) / N), (j) => auction.map((v) => j[3][v] - j[3][bases[1][v]]), (i, d) => paired(1, auction[i], d)],
];
const all = [];
console.log("\n" + "number".padEnd(28) + "compared   rms z   max |z|   beyond 3   plain seasons a path is worth (min, median)");
for (const [name, plain, get, how] of groups) {
  const z = [], worth = [];
  plain.forEach((p, i) => {
    const xs = jobs.map((j) => get(j)[i]), m = xs.reduce((a, b) => a + b) / nJobs, v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / (nJobs - 1);
    const plainVar = Math.max(how(i, p), 0), se = Math.sqrt(v / nJobs + plainVar / N);
    if (se > 0) z.push((m - p) / se);
    if (v > 0 && plainVar > 1e-4) worth.push(plainVar / (v * GROUPS * GROUP));
  });
  all.push(...z);
  worth.sort((a, b) => a - b);
  const rms = Math.sqrt(z.reduce((a, b) => a + b * b, 0) / z.length);
  console.log(name.padEnd(28) + String(z.length).padStart(8) + rms.toFixed(2).padStart(8) + Math.max(...z.map(Math.abs)).toFixed(2).padStart(10) + String(z.filter((x) => Math.abs(x) > 3).length).padStart(11)
    + (worth.length ? `   ${worth[0].toFixed(1)}, ${worth[worth.length >> 1].toFixed(1)}` : "   (too small to tell)"));
}
const rms = Math.sqrt(all.reduce((a, b) => a + b * b, 0) / all.length);
console.log(`\nall ${all.length} comparisons: rms z ${rms.toFixed(3)} (1 if unbiased), ${all.filter((x) => Math.abs(x) > 3).length} beyond 3 (expect ${(all.length * 0.0027).toFixed(1)})`);
const scale = paths / SEASON_PATHS, check = seasonPrecision(parts, bases, SEASON_PLAIN.map((n) => n * scale));
console.log(`the run's own check, scaled to these ${paths.toLocaleString()} paths: needs ${check.short.toFixed(2)}x the paths (1 or less is enough); `
  + `the teams' odds are worth ${Math.round(check.worth).toLocaleString()} plain seasons`);
