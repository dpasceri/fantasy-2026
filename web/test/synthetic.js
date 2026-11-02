// A made-up guillotine league in the `raw` shape, so the whole pipeline can run on CI without real data.
// 8 teams, one chopped per week from week 1; it's week 3 with two teams gone and one auction run so far
// (both, with `waived`).
import { Rng } from "../src/rng.js";

const POSN = { QB: 24, RB: 50, WR: 60, TE: 26 };
const BASE = { QB: 17, RB: 10, WR: 10, TE: 7 };

export function syntheticRaw({ bids = 6, budget = 100, waived = false } = {}) {
  const rng = new Rng(2024);
  const players = [];
  let id = 1000;
  for (const [pos, n] of Object.entries(POSN))
    for (let i = 0; i < n; i++) players.push({ pid: String(id++), pos, team: `T${i % 32}`, level: BASE[pos] * Math.exp(-i / (n / 2.5)) + 0.5 });
  const game = (team, w) => `g${w}-${(Number(team.slice(1)) + w) % 16}`;
  const week = (w, actual) => {
    const out = {};
    for (const p of players) {
      const proj = p.level * (0.9 + 0.2 * rng.random());
      const v = actual ? Math.max(0, proj * Math.exp(0.6 * rng.normal() - 0.18)) : proj;
      out[p.pid] = { team: p.team, opp: null, game: game(p.team, w), pos: p.pos, name: `Player ${p.pid}`, inj: null, stats: { fpts: v } };
    }
    return out;
  };
  const calibProj = {}, calibStats = {}, proj = {}, stats = {};
  for (let w = 1; w <= 18; w++) {
    calibProj[w] = week(w, false);
    calibStats[w] = week(w, true);
  }
  for (let w = 1; w <= 17; w++) proj[w] = week(w, false);
  for (let w = 1; w <= 2; w++) stats[w] = week(w, true);

  // rosters: 8 teams x 13 players, snake-drafted by level; teams 1 and 2 were chopped in weeks 1 and 2
  const order = [...players].sort((a, b) => b.level - a.level);
  const rosters = Array.from({ length: 8 }, (_, t) => ({ roster_id: t + 1, owner_id: `u${t + 1}`, players: [], reserve: [],
    settings: { waiver_budget_used: 0, fpts: 200 + 10 * t, fpts_decimal: 0 } }));
  for (let r = 0; r < 13; r++) for (let j = 0; j < 8; j++) rosters[r % 2 ? 7 - j : j].players.push(order[r * 8 + j].pid);
  const chopped = [1, 2];
  const tx = { 1: [], 2: [], 3: [] };
  chopped.forEach((rid, k) => {
    tx[k + 1].push({ type: "chopped", status: "complete", status_updated: 1000 * (k + 1), roster_ids: [rid],
      drops: Object.fromEntries(rosters[rid - 1].players.map((p) => [p, rid])), adds: null });
  });
  // the week-1 chop's players went through a waiver run; the week-2 chop's roster is still on waivers (pending)
  const pool = rosters[0].players.slice(0, bids);
  pool.forEach((p, i) => {
    const winner = 3 + (i % 6);
    tx[1].push({ type: "waiver", status: "complete", status_updated: 1500, roster_ids: [winner], adds: { [p]: winner }, drops: null,
      settings: { waiver_bid: 5 + 3 * i } });
    rosters[winner - 1].players.push(p);
    rosters[winner - 1].settings.waiver_budget_used += 5 + 3 * i;
  });
  // waived: the week-2 chop's roster has been through a waiver run too, so no auction is pending (mid-week)
  if (waived) {
    const p = rosters[1].players[0];
    tx[3].push({ type: "waiver", status: "complete", status_updated: 2500, roster_ids: [4], adds: { [p]: 4 }, drops: null,
      settings: { waiver_bid: 7 } });
    rosters[3].players.push(p);
    rosters[3].settings.waiver_budget_used += 7;
  }
  rosters[0].players = [];
  rosters[1].players = [];
  const matchups = {};
  for (let w = 1; w <= 3; w++)
    matchups[w] = rosters.filter((r) => r.players.length || w <= r.roster_id).map((r) => ({ roster_id: r.roster_id, players: r.players,
      starters: r.players.slice(0, 8) }));
  const schedule = [];
  for (let w = 1; w <= 17; w++)
    for (let g = 0; g < 16; g++) schedule.push({ week: w, game_id: `g${w}-${g}`, away: `T${g}`, home: `T${g + 16}`, date: "2026-09-13", status: w < 3 ? "complete" : "pre_game" });
  return {
    meta: { league: "123", season: 2026, leg: 3, final_week: 7, calib_season: 2025, fetched_at: 1790000000 },
    league: { name: "Synthetic League", season: "2026", total_rosters: 8, status: "in_season",
      roster_positions: ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "BN", "BN", "BN", "BN", "BN"],
      scoring_settings: { fpts: 1 }, settings: { type: 3, waiver_type: 2, waiver_budget: budget, reserve_slots: 2, start_week: 1 } },
    users: rosters.map((r) => ({ user_id: r.owner_id, display_name: `manager${r.roster_id}`, metadata: {} })),
    rosters, schedule, matchups, tx, live: null,
    proj: { season: proj, calib: calibProj }, stats: { season: stats, calib: calibStats },
  };
}
