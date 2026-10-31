// Port of guillotine/run.py: run every sim and build the results object (same schema as out/results.json),
// which report.js renders as the cheat sheet.
import { BidModel, faabHistory, gain, priceBacktest, pWinCurve, sortDollars, topRivalBids } from "./auction.js";
import { addTotals, MIN_GAIN, summarize } from "./dynamic.js";
import { bestLineup, POS } from "./league.js";
import { fit, PlayerModel } from "./model.js";
import { Rng } from "./rng.js";
import { CHUNK, enumerateLineups, Universe } from "./sim.js";
import { pyRound, quantileSorted, searchsorted } from "./stats.js";

export const BID_GRID = [0, 10, 20, 30, 45, 60, 80, 100, 125, 150, 165]; // in a $200 league; scaled to the league's FAAB
export const STRESS = [0.7, 0.3]; // rivals bid ~2x as often (logit +0.7) and ~35% higher (log +0.3)
// my bid multiplier, fixed: the sweep's cells tie within noise; 0.75x all season held up best (bench/RESULTS.md)
export const POLICY = { m_early: 0.75, m_late: 0.75, switch_week: 12 };
// ...unless another style beats it by this many standard errors in this league (unpaired, so conservative; it also
// guards against picking the luckiest of 16 cells). Tuned leagues tie within noise and keep 0.75x.
export const SWITCH_Z = 3.0;
const PATH_CHUNK = 200;

export function faCandidates(lg, exclude, perPos = { QB: 3, RB: 10, WR: 10, TE: 6 }) {
  const rostered = lg.rostered();
  for (const p of exclude) rostered.add(p);
  const pool = {};
  for (const [pid, inf] of Object.entries(lg.info)) {
    if (rostered.has(pid) || !POS.includes(inf.pos) || !inf.team) continue;
    let ros = 0;
    for (let w = lg.week; w <= lg.final_week; w++) ros += lg.projPts(pid, w);
    (pool[inf.pos] ||= []).push([ros, pid]);
  }
  // Python sorts (ros, pid) tuples descending: ties on ros go to the larger pid string
  return Object.entries(perPos).flatMap(([P, k]) =>
    (pool[P] || []).sort((a, b) => b[0] - a[0] || (a[1] < b[1] ? 1 : a[1] > b[1] ? -1 : 0)).slice(0, k).map((x) => x[1]));
}

/** Dropped since the last waiver run (still on waivers), plus a pending chopped roster. */
export function onWaivers(lg) {
  const all = Object.values(lg.tx).flat();
  const runs = all.filter((t) => t.type === "waiver").map((t) => t.status_updated);
  const last = runs.length ? Math.max(...runs) : 0;
  const out = new Set();
  for (const t of all) if (t.status_updated > last) for (const p of Object.keys(t.drops || {})) out.add(p);
  for (const p of (lg.pending || {}).pool || []) out.add(p);
  return out;
}

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
function gameDays(lg) {
  const sched = new Map(lg.raw.schedule.map((g) => [g.game_id, g]));
  const out = {};
  for (const [pid, x] of Object.entries(lg.proj[lg.week])) {
    const g = sched.get(x.game);
    if (g && g.date) out[pid] = DAYS[new Date(`${g.date.slice(0, 10)}T12:00:00Z`).getUTCDay()];
  }
  return out;
}

function quantilesFromHist(h, qs) {
  let tot = 0;
  for (const x of h) tot += x;
  const c = new Float64Array(h.length);
  let s = 0;
  h.forEach((x, i) => { c[i] = (s += x) / Math.max(tot, 1); });
  return Object.fromEntries(qs.map((q) => [String(q), searchsorted(c, q) / 2]));
}

/** pts: [[bid, p_win, se]] sorted by bid -> best bid, walk-away bid, gain vs not bidding. */
export function summarizeGrid(pts) {
  const pw = pts.map((p) => p[1]), ext = [pw[0], ...pw, pw[pw.length - 1]];
  const sm = pw.map((_, i) => 0.25 * ext[i] + 0.5 * ext[i + 1] + 0.25 * ext[i + 2]);
  let j = 0;
  sm.forEach((v, i) => { if (v > sm[j]) j = i; });
  const se = pts.reduce((s, p) => s + p[2], 0) / pts.length;
  const ok = (lo) => pts.filter((_, i) => sm[i] >= lo).map((p) => p[0]);
  const good = ok(sm[j] - se);
  return { grid: pts.map(([b, p, s]) => ({ bid: b, p_win: p, se: s })), best_bid: pts[j][0], best_p_win: sm[j], p_win_if_zero: sm[0],
    walk_away: Math.max(...ok(sm[0])), good_range: [Math.min(...good), Math.max(...good)] };
}

const key = (lu) => [...lu].sort().join(",");

/** Season plan for a set of my roster variants (sim.py's SeasonPlan): per-week lineup index rows, plain data for the workers. */
function seasonPlan(U, teams, rosters, me, variants, { alive0 = null, wi0 = 0 } = {}) {
  const S = U.lg.slots.length, L = 3 * S, T = teams.length, V = variants.length, W = U.weeks.length;
  // mine[wi] holds each distinct lineup once and mineOf[wi] maps variants to it: adding a bench-level player
  // usually leaves the starters and subs unchanged, and an identical lineup scores identically
  const plan = { S, T, V, me_t: teams.indexOf(me), wi0, opp: {}, mine: {}, mineOf: {}, trip: {}, mineTrip: {}, alive0: new Uint8Array(V * T).fill(1) };
  if (alive0) alive0.forEach((row, v) => row.forEach((a, t) => { plan.alive0[v * T + t] = a ? 1 : 0; }));
  for (let wi = wi0; wi < W; wi++) {
    plan.opp[wi] = new Int32Array(T * L);
    teams.forEach((t, j) => plan.opp[wi].set(U.lineupIdx(wi, rosters.get(t), U.starters0.get(t)), j * L));
    const uniq = new Map(), rows = [];
    plan.mineOf[wi] = Int32Array.from(variants, (r) => {
      const idx = U.lineupIdx(wi, r, U.starters0.get(me)), k = idx.join();
      if (!uniq.has(k)) {
        uniq.set(k, rows.length);
        rows.push(idx);
      }
      return uniq.get(k);
    });
    plan.mine[wi] = new Int32Array(rows.length * L);
    rows.forEach((idx, u) => plan.mine[wi].set(idx, u * L));
    [plan.trip[wi], plan.mineTrip[wi]] = slotTriples(plan.mine[wi], S, U.n + 1);
  }
  return plan;
}

/**
 * The distinct (starter, sub, sub) slot triples in a set of lineup rows and, per row and slot, which one it is.
 * Variants of my roster share nearly all their slots, so a chunk scores each triple once per week.
 */
function slotTriples(rows, S, n1) {
  const L = 3 * S, R = rows.length / L, seen = new Map(), trip = [], of = new Int32Array(R * S);
  for (let u = 0; u < R; u++)
    for (let k = 0; k < S; k++) {
      const a = rows[u * L + k], b1 = rows[u * L + S + k], b2 = rows[u * L + 2 * S + k], key = (a * n1 + b1) * n1 + b2;
      let j = seen.get(key);
      if (j === undefined) {
        seen.set(key, (j = trip.length / 3));
        trip.push(a, b1, b2);
      }
      of[u * S + k] = j;
    }
  return [Int32Array.from(trip), of];
}

const nChunksOf = (nSims) => Math.max(1, Math.floor(nSims / CHUNK));

/** Season sims (run_season) on the plan the pool holds under `key`. Every variant's result is its own: it doesn't depend on the others. */
async function runSeason(pool, key, plan, W, nSims, seed, onDone) {
  const { V, T } = plan, nChunks = nChunksOf(nSims);
  const parts = await pool.map("season", Array.from({ length: nChunks }, (_, ci) => ({ plan: key, ci, C: CHUNK, seed })), onDone);
  const N = nChunks * CHUNK, sumUp = (k) => parts.reduce((a, p) => { p[k].forEach((x, i) => { a[i] += x; }); return a; }, new Float64Array(parts[0][k].length));
  const chops = sumUp("chops"), wins = sumUp("wins");
  return { N, V, T, W, chop: (v, t, w) => chops[(v * T + t) * W + w] / N, win: (v, t) => wins[v * T + t] / N,
    lineHist: sumUp("lineHist").subarray(0, 400), myHist: sumUp("myHist").subarray(0, 400) };
}

async function runWeek(pool, key, V, nSims, seed, onDone) {
  const nChunks = nChunksOf(nSims);
  const parts = await pool.map("week", Array.from({ length: nChunks }, (_, ci) => ({ plan: key, ci, C: CHUNK, seed })), onDone);
  const N = nChunks * CHUNK, p = new Float64Array(V), m = new Float64Array(V), q = new Float64Array(V);
  for (const x of parts) for (let v = 0; v < V; v++) { p[v] += x.chopped[v]; m[v] += x.s1[v]; q[v] += x.s2[v]; }
  return { N, p_chop: [...p].map((x) => x / N), mean: [...m].map((x) => x / N), sd: [...q].map((x, v) => Math.sqrt(Math.max(x / N - (m[v] / N) ** 2, 0))) };
}

/** How the settings of a dynamic stage are split into jobs: [chunks, groups of settings]. */
function pathJobs(nSettings, nPaths, size) {
  const nChunks = Math.max(1, Math.floor(nPaths / PATH_CHUNK));
  return [nChunks, Math.max(1, Math.min(nSettings, Math.ceil((3 * size) / nChunks)))];
}

/** dynamic.run: P(win) etc. per setting. Same chunk index = same draws, so settings share random numbers. */
async function runDynamic(pool, settings, nPaths, seed, onDone) {
  const [nChunks, nGroups] = pathJobs(settings.length, nPaths, pool.size);
  const groups = Array.from({ length: nGroups }, (_, g) => settings.map((_, i) => i).filter((i) => i % nGroups === g));
  const jobs = [];
  for (let ci = 0; ci < nChunks; ci++) for (const grp of groups) jobs.push({ ci, C: PATH_CHUNK, seed, settings: grp.map((i) => settings[i]), grp });
  const parts = await pool.map("paths", jobs.map(({ grp, ...j }) => j), onDone);
  const acc = new Array(settings.length);
  parts.forEach((r, k) => jobs[k].grp.forEach((i, j) => { acc[i] = addTotals(acc[i] && [acc[i]], [r[j]])[0]; }));
  return summarize(settings, acc);
}

/**
 * Run everything at full precision. opts: {onStage(name, i, n), onProgress(done, total)}; tests pass k to divide
 * every sim count by k (k = 8 is run.py's --quick).
 * Stages overlap: every stage is queued as soon as its inputs exist, so the workers run straight from one into the
 * next and the main thread does its own work (the rival bid curves) while they are busy. onStage names what the
 * main thread is waiting on; onProgress counts work done over the whole run. Each timing is when that stage's
 * result was in, in seconds from the start.
 * Returns the results object.
 */
export async function runAll(lg, pool, { k = 1, onStage = () => {}, onProgress = () => {} } = {}) {
  const t0 = Date.now(), timings = {};
  let stageNo = 0;
  const STAGES = 5;
  const stage = (name) => onStage(name, stageNo++, STAGES);
  const tick = (name) => { timings[name] = pyRound((Date.now() - t0) / 1000, 1); };
  // progress is by rough job cost: a season chunk-week is 1, a paths job W * (2 * settings + 4), a bid curve 0.16
  let planned = 0, done = 0;
  const prog = (cost) => () => { done += cost; onProgress(done, planned); };
  const pathCost = (nSet, nPaths) => { const [nc, ng] = pathJobs(nSet, nPaths, pool.size); return [nc * ng, (PATH_CHUNK / CHUNK) * (2 * Math.ceil(nSet / ng) + 4)]; };

  stage("Fitting the player and bid models");
  const grid = [...new Set(BID_GRID.map((b) => pyRound((b * lg.budget) / 200)))].sort((a, b) => a - b);
  const { model, report: modelReport } = fit(lg);
  const pm = new PlayerModel(model);
  const bm = new BidModel(lg);
  tick("fit models");

  const teams = lg.alive(), me = lg.me, meT = teams.indexOf(me);
  if (meT < 0) throw new Error("Your team has been chopped, so there's nothing left to simulate.");
  const ros = new Map(teams.map((t) => [t, lg.players(t)]));
  const pending = lg.pending;
  const fas = faCandidates(lg, onWaivers(lg));
  const weeks = [];
  for (let w = lg.week; w <= lg.final_week; w++) weeks.push(w);
  const W = weeks.length;
  const rosSum = (p, ws) => ws.reduce((s, w) => s + lg.projPts(p, w), 0);
  const pendPool = ((pending || {}).pool || []).filter((p) => rosSum(p, weeks) > 0);
  const U = new Universe(lg, pm, [...teams.flatMap((t) => ros.get(t)), ...fas, ...pendPool], weeks);
  const days = gameDays(lg);
  const L = 3 * lg.slots.length;

  // 1. static season: base + every free agent added to my roster
  const variants = [ros.get(me), ...fas.map((f) => [...ros.get(me), f])];
  // 3. the next auction's cases: the pending one, or each rival chopped this week (ordered by P(chop) once S is in)
  let cases, win0, wLo, bidLeg;
  if (pending) {
    cases = [[pending.rid, pendPool]];
    [win0, wLo, bidLeg] = [0, lg.week, lg.week - 1];
  } else {
    cases = teams.filter((t) => t !== me).map((t) => [t, ros.get(t).filter((i) => rosSum(i, weeks.slice(1)) > 0)]);
    [win0, wLo, bidLeg] = [1, lg.week + 1, lg.week];
  }
  const var2 = [], alive2 = [], lab2 = [];
  for (const [kk, plist] of cases) {
    const mask = teams.map((t) => t !== kk);
    var2.push(ros.get(me)); alive2.push(mask); lab2.push([kk, null]);
    for (const i of plist) { var2.push([...ros.get(me), i]); alive2.push(mask); lab2.push([kk, i]); }
  }
  const vOf = new Map(cases.map(([kk]) => [kk, new Map()])); // case, player -> variant
  lab2.forEach(([kk, i], v) => vOf.get(kk).set(i, v));
  // 4. season with auctions: shared inputs
  const B = lg.faab(me);
  const base = { me, teams, rosters: Object.fromEntries([...ros]), budget: Object.fromEntries(teams.map((t) => [t, lg.faab(t)])),
    pending_pool: pending && pendPool.length ? pendPool : null, weeks, S: lg.slots.length, week0Idx: null };
  if (U.locked.size && !base.pending_pool) {
    base.week0Idx = new Int32Array(teams.length * L);
    teams.forEach((t, j) => base.week0Idx.set(U.lineupIdx(0, ros.get(t), U.starters0.get(t)), j * L));
  }
  const ix = U.ix, posIdx = Int8Array.from(U.pos, (p) => (POS.includes(p) ? POS.indexOf(p) : POS.length)); // POS.length: no slot takes it
  const rosVal = U.proj.map((_, wi) => { const s = new Float64Array(U.n); for (let w = wi; w < U.proj.length; w++) U.proj[w].forEach((x, i) => { s[i] += x; }); return s; });
  rosVal.push(new Float64Array(U.n));
  const ctx = { me, bm: bm.params(), slots: lg.slots, ix, posIdx, proj: U.proj, rosVal,
    maxRoster: lg.league.roster_positions.length + (lg.league.settings.reserve_slots || 0) };
  const sweep = [];
  for (const a of [0.5, 0.75, 1.0, 1.5]) for (const b of [0.75, 1.0, 1.5, 2.0]) sweep.push({ m_early: a, m_late: b, switch_week: 12, budget: B });

  // Everything that needs only the league and the models is queued now, in this order: the season sims, the
  // aggression sweep, then the static auction sims (a variant's result doesn't depend on its place in the plan)
  const plan1 = seasonPlan(U, teams, ros, me, variants), plan2 = seasonPlan(U, teams, ros, me, var2, { alive0: alive2, wi0: win0 });
  await pool.set("U", U.data());
  await pool.set("plan", plan1);
  await pool.set("dyn", { ctx, base });
  await pool.set("plan2", plan2);
  const nS = nChunksOf(640_000 / k), nS2 = nChunksOf(320_000 / k), nW8 = nChunksOf(480_000 / k), nCurves = lab2.length - cases.length;
  const [nD1, cD1] = pathCost(sweep.length, 10_000 / k), [nD2, cD2] = pathCost(12, 10_000 / k), [nP, cP] = pathCost(2, 20_000 / k);
  const [nD3, cD3] = pathCost(pending ? 6 * grid.length : 12 * grid.length + grid.length, (pending ? 8_000 : 4_000) / k);
  planned = nS * W + nS2 * (W - win0) + nW8 * 0.3 + nCurves * 0.16 + nD1 * cD1 + nD2 * cD2 + nP * cP + nD3 * cD3;
  stage("Season sims");
  const S_p = runSeason(pool, "plan", plan1, W, 640_000 / k, 1, prog(W));
  const D1_p = runDynamic(pool, sweep, 10_000 / k, 11, prog(cD1));
  const S2_p = runSeason(pool, "plan2", plan2, W, 320_000 / k, 3, prog(W - win0));

  const S = await S_p;
  tick("season sim (static, base + FA adds)");
  const standings = teams.map((t, j) => {
    const lu = bestLineup(lg.slots, ros.get(t).map((p) => [lg.projPts(p, lg.week), p, lg.pos(p)]));
    const st = lg.rosters.get(t).settings;
    let c = 0;
    return { rid: t, owner: lg.owner(t), faab: lg.faab(t), me: t === me, p_chop_now: S.chop(0, j, 0), p_win: S.win(0, j),
      survival: weeks.map((_, w) => 1 - (c += S.chop(0, j, w))),
      proj_now: pyRound(lu.reduce((s, [, p]) => s + (p ? lg.projPts(p, lg.week) : 0), 0), 1),
      season_pts: pyRound((st.fpts || 0) + (st.fpts_decimal || 0) / 100, 1) };
  });
  const reserve = new Set(lg.rosters.get(me).reserve || []);
  const nActive = ros.get(me).filter((p) => !reserve.has(p)).length;
  const openSpots = Math.max(0, lg.league.roster_positions.filter((s) => s !== "IR").length - nActive);
  const faRows = fas.map((f, i) => ({
    pid: f, player: lg.label(f), pos: lg.pos(f), day: days[f] ?? null, proj_now: pyRound(lg.projPts(f, lg.week), 1),
    ros_avg: pyRound(rosSum(f, weeks) / weeks.length, 1), d_win: S.win(i + 1, meT) - S.win(0, meT),
    d_survive_now: S.chop(0, meT, 0) - S.chop(i + 1, meT, 0), inj: lg.info[f].inj,
  })).sort((a, b) => b.d_win - a.d_win);

  // 2. this week's lineup: every legal starting lineup, scored on the same draws (small jobs, so they go last and
  // fill the tail of the bid grids)
  const bestFa = [...faRows].sort((a, b) => b.d_survive_now - a.d_survive_now).slice(0, 1).filter((r) => r.d_survive_now > 0).map((r) => r.pid);
  const poolMe = [...ros.get(me), ...bestFa];
  const lineups = enumerateLineups(U, 0, poolMe, lg.slots, U.starters0.get(me) ?? null);
  let current = null;
  if (lg.week in lg.matchups) {
    const m = lg.matchups[lg.week].find((m) => m.roster_id === me);
    if (m) {
      current = m.starters.filter((p) => U.ix.has(p));
      if (!lineups.some((lu) => key(lu) === key(current))) lineups.push(current);
    }
  }
  const oppIdx = new Int32Array(teams.length * L), myIdx = new Int32Array(lineups.length * L);
  teams.forEach((t, j) => oppIdx.set(U.lineupIdx(0, ros.get(t), U.starters0.get(t)), j * L));
  // the free agent is only on my roster (and so only a bench fallback) in the lineups that add him
  lineups.forEach((lu, v) => myIdx.set(U.fixedLineupIdx(0, lu, lu.some((p) => bestFa.includes(p)) ? poolMe : ros.get(me)), v * L));
  const [trip, mineTrip] = slotTriples(myIdx, lg.slots.length, U.n + 1);
  const weekSet = pool.set("week", { S: lg.slots.length, T: teams.length, V: lineups.length, me_t: meT, opp: oppIdx, mine: myIdx, trip, mineTrip });

  // 3. rival bid curves, on the main thread while the workers are busy. Cases go most to least likely chopped;
  // Rng(17) is consumed in that order, one curve at a time, as before.
  stage("Auction values");
  if (!pending) cases.sort((a, b) => S.chop(0, teams.indexOf(b[0]), 0) - S.chop(0, teams.indexOf(a[0]), 0));
  const rng = new Rng(17);
  const curveOf = new Map(); // case -> player -> the row's bid-curve fields
  const gcache = new Map(); // weakest starters per roster and week, shared by every player's gain
  for (const [kk, plist] of cases) {
    const others = new Map(teams.filter((t) => t !== me && t !== kk).map((t) => [t, lg.faab(t)]));
    const rows = new Map();
    for (const i of plist) {
      await new Promise((ok) => setTimeout(ok, 0)); // let worker results through and keep the page responsive
      const gains = new Map([...others.keys()].map((t) => [t, gain(lg, ros.get(t), i, wLo, wLo + 3, gcache)]));
      const top = topRivalBids(bm, others, gains, bidLeg, rng, 60_000);
      const curve = pWinCurve(top, lg.faab(me));
      const s = sortDollars(top);
      rows.set(i, {
        gain_me: pyRound(gain(lg, ros.get(me), i, wLo, wLo + 3, gcache), 2),
        rival_top: Object.fromEntries([0.5, 0.8, 0.9, 0.95].map((q) => [String(q), quantileSorted(s, q)])),
        p_no_rival_bid: top.filter((x) => x === 0).length / top.length,
        p_beat_rivals: Object.fromEntries(grid.filter((b) => b <= lg.faab(me)).map((b) => [String(b), curve[b]])),
        rival_gains: [...gains].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([t, g]) => ({ owner: lg.owner(t), gain: pyRound(g, 2), faab: lg.faab(t) })),
      });
      prog(0.16)();
    }
    curveOf.set(kk, rows);
  }

  // 4. season with auctions: aggression sweep, then value of a dollar and the stress test
  stage("Bidding styles");
  const D1 = await D1_p;
  tick("dynamic: aggression sweep");
  const best = D1.reduce((x, y) => (y.p_win > x.p_win ? y : x));
  let pol = { ...POLICY };
  let used = D1.find((d) => d.m_early === pol.m_early && d.m_late === pol.m_late);
  if (best.p_win - used.p_win > SWITCH_Z * Math.hypot(best.se, used.se)) {
    pol = Object.fromEntries(Object.keys(POLICY).map((k) => [k, best[k]]));
    used = best;
  }
  const budgets = [...new Set([B, 0, ...budgetSteps(lg.budget).map((d) => Math.max(B - d, 0))])].sort((a, b) => b - a);
  const budSet = [...budgets.map((b) => ({ ...pol, budget: b })), ...budgets.map((b) => ({ ...pol, budget: b, rival_boost: STRESS }))];
  const D2_p = runDynamic(pool, budSet, 10_000 / k, 12, prog(cD2)).then((d) => { tick("dynamic: budget curve (+ stress test)"); return d; });
  // 4b. power rankings: play every path to the end and record who wins
  const P_p = runDynamic(pool, [{ ...pol, budget: B, track_all: true, fitted_me: true }, { ...pol, budget: B, track_all: true }], 20_000 / k, 21, prog(cP))
    .then((d) => { tick("dynamic: power rankings"); return d; });

  // 3, continued: the static value of each player in the next auction
  const S2 = await S2_p;
  const baseWin = new Map(cases.map(([kk]) => [kk, S2.win(vOf.get(kk).get(null), meT)]));
  const scen = cases.map(([kk, plist]) => {
    const rows = plist.map((i) => {
      const c = curveOf.get(kk).get(i);
      return { pid: i, player: lg.label(i), pos: lg.pos(i), inj: (lg.info[i] || {}).inj ?? null, gain_me: c.gain_me,
        d_win_static: S2.win(vOf.get(kk).get(i), meT) - baseWin.get(kk), rival_top: c.rival_top, p_no_rival_bid: c.p_no_rival_bid,
        p_beat_rivals: c.p_beat_rivals, rival_gains: c.rival_gains };
    });
    rows.sort((a, b) => b.d_win_static - a.d_win_static);
    return { rid: kk, owner: lg.owner(kk), known: !!pending, p_chop: pending ? 1.0 : S.chop(0, teams.indexOf(kk), 0),
      base_win: baseWin.get(kk), players: rows };
  });
  tick(`auction values, static (${var2.length} variants)`);

  // 5. forced-bid grids: P(win | my bid) for the players that matter in the next auction
  stage("Bid grids");
  const jobs = [];
  for (const sc of scen.slice(0, pending ? 1 : 4))
    for (const r of sc.players.filter((r) => r.gain_me >= MIN_GAIN).slice(0, pending ? 6 : 3))
      for (const b of grid)
        if (b <= B) jobs.push({ ...pol, budget: B, force: { [r.pid]: b }, _pid: r.pid, _bid: b, _k: sc.rid, ...(pending ? {} : { chopped0: sc.rid }) });
  const stressJobs = jobs.length ? jobs.filter((j) => j._pid === jobs[0]._pid && j._k === jobs[0]._k).map((j) => ({ ...j, rival_boost: STRESS })) : [];
  const D3_p = (jobs.length ? runDynamic(pool, [...jobs, ...stressJobs], (pending ? 8_000 : 4_000) / k, 13, prog(cD3)) : Promise.resolve([]))
    .then((d) => { tick(`dynamic: forced-bid grid (${jobs.length + stressJobs.length} settings)`); return d; });
  await weekSet;
  const W8_p = runWeek(pool, "week", lineups.length, 480_000 / k, 2, prog(0.3)).then((d) => { tick(`lineup sim (${lineups.length} lineups)`); return d; });
  const [D2, [P_fit, P_pol], D3, W8] = await Promise.all([D2_p, P_p, D3_p, W8_p]);

  const projOpt = key(bestLineup(lg.slots, ros.get(me).map((p) => [U.proj[0][U.ix.get(p)], p, U.pos[U.ix.get(p)]])).map(([, p]) => p).filter(Boolean));
  const lineupRows = lineups.map((lu, li) => ({
    players: [...lu].sort((a, b) => POS.indexOf(lg.pos(a)) - POS.indexOf(lg.pos(b)) || lg.projPts(b, lg.week) - lg.projPts(a, lg.week)),
    p_chop: W8.p_chop[li], mean: W8.mean[li], sd: W8.sd[li], is_current: current !== null && key(lu) === key(current),
    is_proj_opt: key(lu) === projOpt, uses_fa: lu.filter((p) => bestFa.includes(p)),
  })).sort((a, b) => a.p_chop - b.p_chop);
  const power = teams.map((t, j) => {
    const lineupPts = weeks.map((w) => bestLineup(lg.slots, ros.get(t).map((p) => [lg.projPts(p, w), p, lg.pos(p)])).reduce((s, [, p]) => s + (p ? lg.projPts(p, w) : 0), 0));
    const s = standings.find((x) => x.rid === t);
    let wi = 0;
    lineupPts.forEach((v, i) => { if (v < lineupPts[wi]) wi = i; });
    return { owner: lg.owner(t), me: t === me, faab: lg.faab(t), p_win: P_fit.win_by[j], p_win_policy: P_pol.win_by[j], e_weeks: P_fit.weeks_by[j],
      p_win_frozen: s.p_win, p_chop_now: s.p_chop_now, ros_proj: pyRound(lineupPts.reduce((a, b) => a + b, 0) / lineupPts.length, 1),
      worst_week: weeks[wi], worst_proj: pyRound(Math.min(...lineupPts), 1), survival: s.survival, season_pts: s.season_pts };
  }).sort((a, b) => b.p_win - a.p_win);
  const pace = { history: faabHistory(lg), start_total: lg.league.total_rosters * lg.budget, n_auctions: lg.league.total_rosters - 2,
    left_now: teams.reduce((s, t) => s + lg.faab(t), 0), trade_net_me: tradeFaab(lg, me),
    forecast: Object.fromEntries(["league_spend", "burned", "top_price"].map((x) => [x, P_fit[x]])) };
  for (const sc of scen)
    for (const r of sc.players)
      for (const [tag, boosted] of [["", false], ["stress_", true]]) {
        const pts = D3.filter((d) => d._k === sc.rid && d._pid === r.pid && ("rival_boost" in d) === boosted)
          .map((d) => [d._bid, d.p_win, d.se]).sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
        if (pts.length) r[`${tag}bid`] = summarizeGrid(pts);
      }

  let live = null;
  if (lg.live) { // the week is under way: where each team stands so far
    const G = Object.values(lg.live.games);
    const rows = teams.map((t) => {
      const st = (lg.starters(t, lg.week) || []).filter(Boolean).map((p) => lg.gameState(p));
      return { owner: lg.owner(t), me: t === me, pts: pyRound(st.reduce((s, x) => s + x[2], 0), 1),
        left: st.filter((x) => x[0] !== "post").length, p_chop: S.chop(0, teams.indexOf(t), 0) };
    });
    live = { as_of: lg.live.fetched_at, n: Object.fromEntries(["pre", "in", "post"].map((s) => [s, G.filter((g) => g.state === s).length])),
      in_progress: G.filter((g) => g.state === "in").map((g) => `${g.away} @ ${g.home} ${g.clock || ""}`.trim()),
      teams: rows.sort((a, b) => b.pts - a.pts), locked: [...U.locked].filter((p) => ros.get(me).includes(p)).sort() };
  }
  const labels = {};
  for (const p of new Set([...poolMe, ...lineups.flat()])) labels[p] = lg.label(p);
  timings.total = pyRound((Date.now() - t0) / 1000, 1);
  return {
    generated_at: Math.floor(Date.now() / 1000), fetched_at: lg.meta.fetched_at, league: lg.league.name, league_id: lg.meta.league,
    season: lg.season, week: lg.week, final_week: lg.final_week, me: lg.owner(me), faab_me: B, open_spots: openSpots, weeks,
    threads: pool.size, pending: pending ? { owner: lg.owner(pending.rid), n: pendPool.length } : null, live, timings,
    sims: { season: S.N, lineups: W8.N, auction_static: S2.N, dynamic_paths: D1.length ? D1[0].n : 0, grid_paths: D3.length ? D3[0].n : 0 },
    standings: [...standings].sort((a, b) => b.p_win - a.p_win), power, faab_pace: pace,
    chop_line: quantilesFromHist(S.lineHist, [0.1, 0.25, 0.5, 0.75, 0.9]),
    my_score: quantilesFromHist(S.myHist, [0.1, 0.25, 0.5, 0.75, 0.9]),
    roster: ros.get(me).map((p) => ({ pid: p, player: lg.label(p), pos: lg.pos(p), inj: lg.info[p].inj, day: days[p] ?? null,
      ir: reserve.has(p), proj: weeks.map((w) => pyRound(lg.projPts(p, w), 1)) })),
    labels,
    lineups: [...lineupRows.slice(0, 10), ...lineupRows.slice(10).filter((r) => r.is_current || r.is_proj_opt)],
    n_lineups: lineupRows.length, free_agents: faRows, scenarios: scen,
    dynamic: { sweep: D1, budgets: D2.filter((d) => !("rival_boost" in d)), budgets_stress: D2.filter((d) => "rival_boost" in d), best, used, stress: STRESS,
      default: POLICY },
    bid_model: { ...bm.summary(), managers: teams.map((t) => ({ owner: lg.owner(t), faab: lg.faab(t), bids_often: pyRound(bm.vm.get(t) || 0, 2),
      bids_hard: pyRound(bm.um.get(t) || 0, 2), n_bids: bm.n_bids[t] || 0, n_chances: bm.n_offered[t] || 0 })) },
    price_backtest: priceBacktest(lg, bm), model_report: modelReport, backtest: null,
  };
}

/** Budget cuts for the value-of-a-dollar curve: $20, 40, 80, 120 in a $200 league, scaled to the league's FAAB. */
export const budgetSteps = (budget) => [20, 40, 80, 120].map((d) => pyRound((d * budget) / 200));

/** Net FAAB this team has received in completed trades (negative = sent). */
export function tradeFaab(lg, rid) {
  let net = 0;
  for (const t of Object.values(lg.tx).flat())
    if (t.type === "trade" && t.status === "complete")
      for (const wb of t.waiver_budget || []) net += wb.amount * ((wb.receiver === rid) - (wb.sender === rid));
  return net;
}
