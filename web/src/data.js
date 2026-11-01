// Port of guillotine/fetch.py: pull everything the simulator needs from Sleeper (and ESPN for live game
// clocks) into the `raw` shape League expects. Last season's weekly files never change, so they're kept in
// IndexedDB and fetched once per browser.
import { API, API2, fresh, getJSON } from "./sleeper.js";
import { POS } from "./league.js";

const ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard";
const ESPN_TEAM = { WSH: "WAS" }; // ESPN abbreviations that differ from Sleeper's
const POS_Q = POS.map((p) => `position[]=${p}`).join("&");

/** Weekly projections/stats slimmed to the fields the model uses (fetch.py's slim_weekly). */
export function slimWeekly(rows) {
  const out = {};
  for (const r of rows || []) {
    const p = r.player || {};
    if (!POS.includes(p.position)) continue;
    const stats = {};
    for (const [k, v] of Object.entries(r.stats || {})) if (v) stats[k] = v;
    out[r.player_id] = { team: r.team, opp: r.opponent, game: r.game_id, pos: p.position,
      name: `${p.first_name || ""} ${p.last_name || ""}`.trim(), inj: p.injury_status ?? null, stats };
  }
  return out;
}

// --- a tiny IndexedDB cache; every failure just means "not cached"
function db() {
  return new Promise((ok) => {
    try {
      const req = indexedDB.open("chop-sheet", 1);
      req.onupgradeneeded = () => req.result.createObjectStore("weekly");
      req.onsuccess = () => ok(req.result);
      req.onerror = () => ok(null);
    } catch {
      ok(null);
    }
  });
}
async function cacheGet(d, key) {
  if (!d) return undefined;
  return new Promise((ok) => {
    try {
      const r = d.transaction("weekly").objectStore("weekly").get(key);
      r.onsuccess = () => ok(r.result);
      r.onerror = () => ok(undefined);
    } catch {
      ok(undefined);
    }
  });
}
function cachePut(d, key, value) {
  try {
    if (d) d.transaction("weekly", "readwrite").objectStore("weekly").put(value, key);
  } catch { /* quota or private mode */ }
}

/** Run async tasks with at most `limit` in flight. */
async function limited(tasks, limit, onDone) {
  let i = 0, done = 0;
  const run = async () => {
    while (i < tasks.length) {
      const t = tasks[i++];
      await t();
      onDone(++done, tasks.length);
    }
  };
  await Promise.all(Array.from({ length: limit }, run));
}

/** The week's games: state pre / in / post and the share of the game still to play (fetch.py's live_games). */
async function liveGames(season, leg, sched) {
  const games = {};
  for (const g of sched) {
    if (g.week !== leg) continue;
    const st = { in_game: "in", complete: "post" }[g.status] || "pre";
    games[g.game_id] = { away: g.away, home: g.home, state: st, frac_left: { pre: 1.0, in: 0.5, post: 0.0 }[st], clock: null, kickoff: g.date ?? null };
  }
  let events = [];
  try {
    events = (await getJSON(fresh(`${ESPN}?seasontype=2&week=${leg}&dates=${season}`), 1)).events || [];
  } catch { /* Sleeper's status alone is fine */ }
  const byTeams = new Map(Object.entries(games).map(([gid, g]) => [[g.away, g.home].sort().join("|"), gid]));
  for (const ev of events) {
    const c = ev.competitions[0];
    const abbr = c.competitors.map((t) => ESPN_TEAM[t.team.abbreviation] || t.team.abbreviation);
    const gid = byTeams.get(abbr.sort().join("|"));
    if (!gid) continue;
    const s = c.status, g = games[gid];
    g.state = s.type.state;
    g.kickoff = ev.date;
    if (g.state === "in") {
      const per = parseInt(s.period || 1, 10), clk = parseFloat(s.clock || 0);
      g.frac_left = per <= 4 ? Math.max((4 - per) * 900 + clk, 0) / 3600 : 0.02;
      g.clock = s.type.shortDetail || `Q${per} ${s.displayClock || ""}`;
    } else g.frac_left = g.state === "pre" ? 1.0 : 0.0;
  }
  return games;
}

/** Everything for one league. onProgress(done, total) as files arrive. */
export async function loadAll(L, onProgress = () => {}) {
  const [state, league] = await Promise.all([getJSON(`${API}/state/nfl`), getJSON(`${API}/league/${L}`)]);
  const season = parseInt(league.season, 10), leg = parseInt(state.leg, 10);
  const finalWeek = (league.settings.start_week || 1) + league.total_rosters - 2; // one chop a week until one team is left
  const calib = season - 1;
  const raw = { league, matchups: {}, tx: {}, proj: { season: {}, calib: {} }, stats: { season: {}, calib: {} } };
  const d = await db();
  const tasks = [
    async () => { raw.users = await getJSON(`${API}/league/${L}/users`); },
    async () => { raw.rosters = await getJSON(`${API}/league/${L}/rosters`); },
    async () => { raw.schedule = await getJSON(fresh(`${API2}/schedule/nfl/regular/${season}`)); },
  ];
  for (let w = 1; w <= leg; w++) {
    const url = `${API}/league/${L}/matchups/${w}`;
    tasks.push(async () => { raw.matchups[w] = (await getJSON(w === leg ? fresh(url) : url)) || []; });
    tasks.push(async () => { raw.tx[w] = (await getJSON(`${API}/league/${L}/transactions/${w}`)) || []; });
  }
  const weekly = (target, kind, s, w, cache) => async () => {
    const key = `${s}/${kind}_${w}`;
    let x = cache ? await cacheGet(d, key) : undefined;
    if (x === undefined) {
      x = slimWeekly(await getJSON(`${API2}/${kind === "proj" ? "projections" : "stats"}/nfl/${s}/${w}?season_type=regular&${POS_Q}`));
      if (cache) cachePut(d, key, x);
    }
    target[w] = x;
  };
  for (let w = 1; w <= Math.max(finalWeek, 17); w++) tasks.push(weekly(raw.proj.season, "proj", season, w, false));
  for (let w = 1; w < leg; w++) tasks.push(weekly(raw.stats.season, "stats", season, w, false));
  for (let w = 1; w <= 18; w++) {
    tasks.push(weekly(raw.proj.calib, "proj", calib, w, true));
    tasks.push(weekly(raw.stats.calib, "stats", calib, w, true));
  }
  let liveStats = {};
  tasks.push(async () => { liveStats = slimWeekly(await getJSON(fresh(`${API2}/stats/nfl/${season}/${leg}?season_type=regular&${POS_Q}`))); });
  await limited(tasks, 8, onProgress);
  const fetchedAt = Math.floor(Date.now() / 1000);
  raw.live = { week: leg, fetched_at: fetchedAt, games: await liveGames(season, leg, raw.schedule), stats: liveStats };
  raw.meta = { league: L, season, leg, final_week: finalWeek, calib_season: calib, fetched_at: fetchedAt };
  return raw;
}
