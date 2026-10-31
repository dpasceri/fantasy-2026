// Port of guillotine/run.py: run every sim and build the results object (same schema as out/results.json),
// which report.js renders as the cheat sheet.
import { BidModel, faabHistory, gain, priceBacktest, pWinCurve, topRivalBids } from "./auction.js";
import { addTotals, MIN_GAIN, summarize } from "./dynamic.js";
import { bestLineup, POS } from "./league.js";
import { fit, PlayerModel } from "./model.js";
import { Rng } from "./rng.js";
import { CHUNK, enumerateLineups, Universe } from "./sim.js";
import { pyRound, quantileSorted, searchsorted, sortNum } from "./stats.js";

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

/** Season sims for a set of my roster variants (sim.py's SeasonPlan + run_season). */
async function runSeason(pool, U, teams, rosters, me, variants, nSims, { seed = 1, alive0 = null, wi0 = 0, onDone } = {}) {
  const S = U.lg.slots.length, L = 3 * S, T = teams.length, V = variants.length, W = U.weeks.length;
  const plan = { S, T, V, me_t: teams.indexOf(me), wi0, opp: {}, mine: {}, alive0: new Uint8Array(V * T).fill(1) };
  if (alive0) alive0.forEach((row, v) => row.forEach((a, t) => { plan.alive0[v * T + t] = a ? 1 : 0; }));
  for (let wi = wi0; wi < W; wi++) {
    plan.opp[wi] = new Int32Array(T * L);
    teams.forEach((t, j) => plan.opp[wi].set(U.lineupIdx(wi, rosters.get(t), U.starters0.get(t)), j * L));
    plan.mine[wi] = new Int32Array(V * L);
    variants.forEach((r, v) => plan.mine[wi].set(U.lineupIdx(wi, r, U.starters0.get(me)), v * L));
  }
  await pool.set("plan", plan);
  const nChunks = Math.max(1, Math.floor(nSims / CHUNK));
  const parts = await pool.map("season", Array.from({ length: nChunks }, (_, ci) => ({ plan: "plan", ci, C: CHUNK, seed })), onDone);
  const N = nChunks * CHUNK, sumUp = (k) => parts.reduce((a, p) => { p[k].forEach((x, i) => { a[i] += x; }); return a; }, new Float64Array(parts[0][k].length));
  const chops = sumUp("chops"), wins = sumUp("wins");
  return { N, V, T, W, chop: (v, t, w) => chops[(v * T + t) * W + w] / N, win: (v, t) => wins[v * T + t] / N,
    lineHist: sumUp("lineHist").subarray(0, 400), myHist: sumUp("myHist").subarray(0, 400) };
}

async function runWeek(pool, U, oppIdx, myIdx, meT, T, V, nSims, seed, onDone) {
  await pool.set("week", { S: U.lg.slots.length, T, V, me_t: meT, opp: oppIdx, mine: myIdx });
  const nChunks = Math.max(1, Math.floor(nSims / CHUNK));
  const parts = await pool.map("week", Array.from({ length: nChunks }, (_, ci) => ({ plan: "week", ci, C: CHUNK, seed })), onDone);
  const N = nChunks * CHUNK, p = new Float64Array(V), m = new Float64Array(V), q = new Float64Array(V);
  for (const x of parts) for (let v = 0; v < V; v++) { p[v] += x.chopped[v]; m[v] += x.s1[v]; q[v] += x.s2[v]; }
  return { N, p_chop: [...p].map((x) => x / N), mean: [...m].map((x) => x / N), sd: [...q].map((x, v) => Math.sqrt(Math.max(x / N - (m[v] / N) ** 2, 0))) };
}

/** dynamic.run: P(win) etc. per setting. Same chunk index = same draws, so settings share random numbers. */
async function runDynamic(pool, settings, nPaths, seed, onDone) {
  const nChunks = Math.max(1, Math.floor(nPaths / PATH_CHUNK));
  const nGroups = Math.max(1, Math.min(settings.length, Math.ceil((3 * pool.size) / nChunks)));
  const groups = Array.from({ length: nGroups }, (_, g) => settings.map((_, i) => i).filter((i) => i % nGroups === g));
  const jobs = [];
  for (let ci = 0; ci < nChunks; ci++) for (const grp of groups) jobs.push({ ci, C: PATH_CHUNK, seed, settings: grp.map((i) => settings[i]), grp });
  const parts = await pool.map("paths", jobs.map(({ grp, ...j }) => j), onDone);
  const acc = new Array(settings.length);
  parts.forEach((r, k) => jobs[k].grp.forEach((i, j) => { acc[i] = addTotals(acc[i] && [acc[i]], [r[j]])[0]; }));
  return summarize(settings, acc);
}

/**
 * Run everything. opts: {quick (1/8 of the sims, default true), onStage(name, i, n), onProgress(done, total)}.
 * Returns the results object.
 */
export async function runAll(lg, pool, { quick = true, k = quick ? 8 : 1, onStage = () => {}, onProgress = () => {} } = {}) {
  const t0 = Date.now(), timings = {};
  let tStage = Date.now(), stageNo = 0;
  const STAGES = 8;
  const stage = (name) => onStage(name, stageNo++, STAGES);
  const tick = (name) => {
    timings[name] = pyRound((Date.now() - tStage) / 1000, 1);
    tStage = Date.now();
  };

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
  const rosSum = (p, ws) => ws.reduce((s, w) => s + lg.projPts(p, w), 0);
  const pendPool = ((pending || {}).pool || []).filter((p) => rosSum(p, weeks) > 0);
  const U = new Universe(lg, pm, [...teams.flatMap((t) => ros.get(t)), ...fas, ...pendPool], weeks);
  const days = gameDays(lg);
  await pool.set("U", U.data());

  // 1. static season: base + every free agent added to my roster
  stage("Season sims");
  const variants = [ros.get(me), ...fas.map((f) => [...ros.get(me), f])];
  const S = await runSeason(pool, U, teams, ros, me, variants, 640_000 / k, { onDone: onProgress });
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

  // 2. this week's lineup: every legal starting lineup, scored on the same draws
  stage("Lineup sims");
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
  const L = 3 * lg.slots.length;
  const oppIdx = new Int32Array(teams.length * L), myIdx = new Int32Array(lineups.length * L);
  teams.forEach((t, j) => oppIdx.set(U.lineupIdx(0, ros.get(t), U.starters0.get(t)), j * L));
  // the free agent is only on my roster (and so only a bench fallback) in the lineups that add him
  lineups.forEach((lu, v) => myIdx.set(U.fixedLineupIdx(0, lu, lu.some((p) => bestFa.includes(p)) ? poolMe : ros.get(me)), v * L));
  const W8 = await runWeek(pool, U, oppIdx, myIdx, meT, teams.length, lineups.length, 480_000 / k, 2, onProgress);
  tick(`lineup sim (${lineups.length} lineups)`);
  const projOpt = key(bestLineup(lg.slots, ros.get(me).map((p) => [U.proj[0][U.ix.get(p)], p, U.pos[U.ix.get(p)]])).map(([, p]) => p).filter(Boolean));
  const lineupRows = lineups.map((lu, li) => ({
    players: [...lu].sort((a, b) => POS.indexOf(lg.pos(a)) - POS.indexOf(lg.pos(b)) || lg.projPts(b, lg.week) - lg.projPts(a, lg.week)),
    p_chop: W8.p_chop[li], mean: W8.mean[li], sd: W8.sd[li], is_current: current !== null && key(lu) === key(current),
    is_proj_opt: key(lu) === projOpt, uses_fa: lu.filter((p) => bestFa.includes(p)),
  })).sort((a, b) => a.p_chop - b.p_chop);

  // 3. the next auction: static value of each player, rival bid curves
  stage("Auction values");
  let cases, win0, wLo, bidLeg;
  if (pending) {
    cases = [[pending.rid, pendPool]];
    [win0, wLo, bidLeg] = [0, lg.week, lg.week - 1];
  } else {
    cases = teams.filter((t) => t !== me).sort((a, b) => S.chop(0, teams.indexOf(b), 0) - S.chop(0, teams.indexOf(a), 0))
      .map((t) => [t, ros.get(t).filter((i) => rosSum(i, weeks.slice(1)) > 0)]);
    [win0, wLo, bidLeg] = [1, lg.week + 1, lg.week];
  }
  const var2 = [], alive2 = [], lab2 = [];
  for (const [kk, plist] of cases) {
    const mask = teams.map((t) => t !== kk);
    var2.push(ros.get(me)); alive2.push(mask); lab2.push([kk, null]);
    for (const i of plist) { var2.push([...ros.get(me), i]); alive2.push(mask); lab2.push([kk, i]); }
  }
  const S2 = await runSeason(pool, U, teams, ros, me, var2, 320_000 / k, { seed: 3, alive0: alive2, wi0: win0, onDone: onProgress });
  const baseWin = new Map(lab2.map(([kk, i], v) => [kk, i === null ? S2.win(v, meT) : undefined]).filter(([, x]) => x !== undefined));
  const rng = new Rng(17);
  const scen = [];
  const nCurves = lab2.filter(([, i]) => i !== null).length;
  let curves = 0;
  for (const [kk, plist] of cases) {
    const others = new Map(teams.filter((t) => t !== me && t !== kk).map((t) => [t, lg.faab(t)]));
    const rows = [];
    for (const [v, [k2, i]] of lab2.entries()) {
      if (k2 !== kk || i === null) continue;
      onProgress(++curves, nCurves);
      if (curves % 8 === 0) await new Promise((ok) => setTimeout(ok, 0)); // keep the page responsive
      const gains = new Map([...others.keys()].map((t) => [t, gain(lg, ros.get(t), i, wLo, wLo + 3)]));
      const top = topRivalBids(bm, others, gains, bidLeg, rng, 60_000);
      const curve = pWinCurve(top, lg.faab(me));
      const s = sortNum(top);
      rows.push({
        pid: i, player: lg.label(i), pos: lg.pos(i), inj: (lg.info[i] || {}).inj ?? null,
        gain_me: pyRound(gain(lg, ros.get(me), i, wLo, wLo + 3), 2), d_win_static: S2.win(v, meT) - baseWin.get(kk),
        rival_top: Object.fromEntries([0.5, 0.8, 0.9, 0.95].map((q) => [String(q), quantileSorted(s, q)])),
        p_no_rival_bid: top.filter((x) => x === 0).length / top.length,
        p_beat_rivals: Object.fromEntries(grid.filter((b) => b <= lg.faab(me)).map((b) => [String(b), curve[b]])),
        rival_gains: [...gains].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([t, g]) => ({ owner: lg.owner(t), gain: pyRound(g, 2), faab: lg.faab(t) })),
      });
    }
    rows.sort((a, b) => b.d_win_static - a.d_win_static);
    scen.push({ rid: kk, owner: lg.owner(kk), known: !!pending, p_chop: pending ? 1.0 : S.chop(0, teams.indexOf(kk), 0),
      base_win: baseWin.get(kk), players: rows });
  }
  tick(`auction values, static (${var2.length} variants)`);

  // 4. season with auctions: aggression sweep, value of a dollar, stress test
  const B = lg.faab(me);
  const base = { me, teams, rosters: Object.fromEntries([...ros]), budget: Object.fromEntries(teams.map((t) => [t, lg.faab(t)])),
    pending_pool: pending && pendPool.length ? pendPool : null, weeks, S: lg.slots.length, week0Idx: null };
  if (U.locked.size && !base.pending_pool) {
    base.week0Idx = new Int32Array(teams.length * L);
    teams.forEach((t, j) => base.week0Idx.set(U.lineupIdx(0, ros.get(t), U.starters0.get(t)), j * L));
  }
  const ix = U.ix, posOf = new Map([...ix.keys()].map((p) => [p, U.pos[ix.get(p)]]));
  const rosVal = U.proj.map((_, wi) => { const s = new Float64Array(U.n); for (let w = wi; w < U.proj.length; w++) U.proj[w].forEach((x, i) => { s[i] += x; }); return s; });
  rosVal.push(new Float64Array(U.n));
  const ctx = { me, bm: bm.params(), slots: lg.slots, ix, posOf, proj: U.proj, rosVal,
    maxRoster: lg.league.roster_positions.length + (lg.league.settings.reserve_slots || 0) };
  await pool.set("dyn", { ctx, base });

  stage("Bidding styles");
  const sweep = [];
  for (const a of [0.5, 0.75, 1.0, 1.5]) for (const b of [0.75, 1.0, 1.5, 2.0]) sweep.push({ m_early: a, m_late: b, switch_week: 12, budget: B });
  const D1 = await runDynamic(pool, sweep, 10_000 / k, 11, onProgress);
  const best = D1.reduce((x, y) => (y.p_win > x.p_win ? y : x));
  let pol = { ...POLICY };
  let used = D1.find((d) => d.m_early === pol.m_early && d.m_late === pol.m_late);
  if (best.p_win - used.p_win > SWITCH_Z * Math.hypot(best.se, used.se)) {
    pol = Object.fromEntries(Object.keys(POLICY).map((k) => [k, best[k]]));
    used = best;
  }
  tick("dynamic: aggression sweep");
  stage("Value of a FAAB dollar");
  const budgets = [...new Set([B, 0, ...budgetSteps(lg.budget).map((d) => Math.max(B - d, 0))])].sort((a, b) => b - a);
  const budSet = [...budgets.map((b) => ({ ...pol, budget: b })), ...budgets.map((b) => ({ ...pol, budget: b, rival_boost: STRESS }))];
  const D2 = await runDynamic(pool, budSet, 10_000 / k, 12, onProgress);
  tick("dynamic: budget curve (+ stress test)");

  // 4b. power rankings: play every path to the end and record who wins
  stage("Power rankings");
  const [P_fit, P_pol] = await runDynamic(pool, [{ ...pol, budget: B, track_all: true, fitted_me: true }, { ...pol, budget: B, track_all: true }], 20_000 / k, 21, onProgress);
  tick("dynamic: power rankings");
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

  // 5. forced-bid grids: P(win | my bid) for the players that matter in the next auction
  stage("Bid grids");
  const jobs = [];
  for (const sc of scen.slice(0, pending ? 1 : 4))
    for (const r of sc.players.filter((r) => r.gain_me >= MIN_GAIN).slice(0, pending ? 6 : 3))
      for (const b of grid)
        if (b <= B) jobs.push({ ...pol, budget: B, force: { [r.pid]: b }, _pid: r.pid, _bid: b, _k: sc.rid, ...(pending ? {} : { chopped0: sc.rid }) });
  const stressJobs = jobs.length ? jobs.filter((j) => j._pid === jobs[0]._pid && j._k === jobs[0]._k).map((j) => ({ ...j, rival_boost: STRESS })) : [];
  const D3 = jobs.length ? await runDynamic(pool, [...jobs, ...stressJobs], (pending ? 8_000 : 4_000) / k, 13, onProgress) : [];
  tick(`dynamic: forced-bid grid (${jobs.length + stressJobs.length} settings)`);
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
