// Port of guillotine/common.py: league state, scoring and lineup logic.
//
// `raw` is everything fetched from Sleeper, in the shape of guillotine/data/:
//   { meta: {league, season, leg, final_week, calib_season, fetched_at}, league, users, rosters, schedule,
//     proj: {season: {week: {pid: row}}, calib: {...}}, stats: {season: {...}, calib: {...}},
//     matchups: {week: [...]}, tx: {week: [...]}, live }
// where a projection/stats row is {team, opp, game, pos, name, inj, stats: {key: value}} (fetch.py's slim_weekly).

export const POS = ["QB", "RB", "WR", "TE"];
export const FLEX_OK = { FLEX: ["RB", "WR", "TE"], SUPER_FLEX: ["QB", "RB", "WR", "TE"], REC_FLEX: ["WR", "TE"], WRRB_FLEX: ["RB", "WR"] };
// chance a player with this week's tag sits; Out/IR are already zeroed in projections (see common.py)
export const P_OUT = { Questionable: 0.25, Doubtful: 0.99, Out: 1.0, IR: 1.0, PUP: 1.0, Sus: 1.0, NA: 1.0 };

export const eligible = (slot, pos) => pos === slot || (FLEX_OK[slot] || []).includes(pos);

export class League {
  /** me: my roster id (number) or Sleeper display name. */
  constructor(raw, me) {
    this.raw = raw;
    this.meta = raw.meta;
    this.league = raw.league;
    this.season = this.meta.season;
    this.leg = this.meta.leg;
    this.final_week = this.meta.final_week;
    this.scoring = this.league.scoring_settings;
    this.slots = this.league.roster_positions.filter((s) => !["BN", "IR", "TAXI"].includes(s));
    this.budget = this.league.settings.waiver_budget;
    this.users = new Map(raw.users.map((u) => [u.user_id, u]));
    this.rosters = new Map(raw.rosters.map((r) => [r.roster_id, r]));
    this.proj = {};
    for (let w = 1; w <= this.final_week; w++) this.proj[w] = raw.proj.season[w] || {};
    this.matchups = {};
    this.tx = {};
    for (let w = 1; w <= this.leg; w++) {
      this.matchups[w] = raw.matchups[w] || [];
      this.tx[w] = raw.tx[w] || [];
    }
    this.info = {};
    for (let w = 1; w <= this.final_week; w++)
      for (const [pid, x] of Object.entries(this.proj[w])) this.info[pid] = { name: x.name, pos: x.pos, team: x.team, inj: x.inj };
    this._pp = new Map();
    this.me = typeof me === "number" ? me : this.ridOf(me);
    // the week to simulate from: the current leg, or the next one once all its games are final
    const sched = raw.schedule.filter((g) => g.week === this.leg);
    const done = sched.length > 0 && sched.every((g) => g.status === "complete");
    this.week = done && this.leg < this.final_week ? this.leg + 1 : this.leg;
    this.pending = this._pendingAuction();
    const live = raw.live;
    const on = live && live.week === this.week && Object.values(live.games).some((g) => g.state !== "pre");
    this.live = on ? live : null;
  }

  /** [state, share of the game left, points so far] for this player's game this week; state is pre / in / post. */
  gameState(pid) {
    if (!this.live) return ["pre", 1.0, 0.0];
    const s = this.live.stats[pid];
    const g = this.live.games[(s && s.game) || (this.proj[this.week][pid] || {}).game];
    if (!g) return ["pre", 1.0, 0.0];
    return [g.state, g.frac_left, s ? this.pts(s.stats) : 0.0];
  }

  /** The lineup this team has set in Sleeper for week w, by slot (null = empty slot). */
  starters(rid, w) {
    const m = (this.matchups[w] || []).find((m) => m.roster_id === rid);
    return m ? m.starters.map((p) => (p && p !== "0" ? p : null)) : null;
  }

  /** The latest chop whose roster hasn't been through a waiver run yet, if any. */
  _pendingAuction() {
    const all = Object.values(this.tx).flat();
    const chops = all.filter((t) => t.type === "chopped");
    if (!chops.length) return null;
    const t = chops.reduce((a, b) => (b.status_updated > a.status_updated ? b : a));
    if (all.some((x) => x.type === "waiver" && x.status_updated > t.status_updated)) return null;
    return { rid: t.roster_ids[0], pool: Object.keys(t.drops || {}).filter((p) => POS.includes(this.pos(p))), time: t.status_updated };
  }

  /** Fantasy points. Neumaier-compensated like Python 3.12+'s sum(), so a 0.5-point projection stays 0.5. */
  pts(stats) {
    let s = 0, c = 0;
    for (const k in stats) {
      const x = (this.scoring[k] || 0) * stats[k], t = s + x;
      c += Math.abs(s) >= Math.abs(x) ? s - t + x : x - t + s;
      s = t;
    }
    return s + c;
  }

  projPts(pid, w) {
    const key = `${pid}|${w}`;
    let v = this._pp.get(key);
    if (v === undefined) {
      const x = (this.proj[w] || {})[pid];
      v = x ? this.pts(x.stats) : 0.0;
      this._pp.set(key, v);
    }
    return v;
  }

  pos(pid) {
    return (this.info[pid] || {}).pos;
  }

  label(pid) {
    const i = this.info[pid];
    return i ? `${i.name} (${i.pos}, ${i.team || "FA"})` : pid;
  }

  owner(rid) {
    return (this.users.get(this.rosters.get(rid).owner_id) || {}).display_name || `roster ${rid}`;
  }

  ridOf(name) {
    for (const rid of this.rosters.keys()) if (this.owner(rid).toLowerCase() === String(name).toLowerCase()) return rid;
    throw new Error(`No team owned by ${name}`);
  }

  /** Teams still in: rosters with players that haven't been chopped (some leagues leave a chopped roster's players). */
  alive() {
    const chopped = new Set(Object.values(this.tx).flat().filter((t) => t.type === "chopped").map((t) => t.roster_ids[0]));
    return [...this.rosters.entries()].filter(([r, x]) => (x.players || []).length && !chopped.has(r)).map(([r]) => r).sort((a, b) => a - b);
  }

  faab(rid) {
    return this.budget - (this.rosters.get(rid).settings.waiver_budget_used || 0);
  }

  players(rid) {
    return [...(this.rosters.get(rid).players || [])];
  }

  rostered() {
    const s = new Set();
    for (const r of this.rosters.values()) for (const p of r.players || []) s.add(p);
    return s;
  }
}

/**
 * Max-projection lineup. cands: [[value, pid, pos], ...].
 * Fixed slots first, then flex slots from most to least restrictive; greedy is optimal for nested flex sets.
 * With withSubs, each slot also gets up to two bench fallbacks (best eligible unused players).
 */
export function bestLineup(slots, cands, withSubs = false) {
  const order = cands.filter((c) => c[0] > 0).sort((a, b) => b[0] - a[0]);
  const used = new Set(), lineup = [];
  const fixed = slots.filter((s) => POS.includes(s));
  const flex = slots.filter((s) => !POS.includes(s)).map((s, i) => [s, i])
    .sort((a, b) => (FLEX_OK[a[0]] || POS).length - (FLEX_OK[b[0]] || POS).length || a[1] - b[1]).map((x) => x[0]);
  for (const s of [...fixed, ...flex]) {
    const pick = order.find((c) => !used.has(c[1]) && eligible(s, c[2]));
    if (pick) used.add(pick[1]);
    lineup.push([s, pick ? pick[1] : null]);
  }
  if (!withSubs) return lineup;
  const bench = order.filter((c) => !used.has(c[1]));
  const subs = lineup.map(([s]) => bench.filter((c) => eligible(s, c[2])).slice(0, 2).map((c) => c[1]));
  return [lineup, subs];
}
