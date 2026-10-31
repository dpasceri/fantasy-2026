// Which Sleeper leagues the simulator can handle. Mirrors guillotine/common.py: POS and FLEX_OK.
export const POS = ["QB", "RB", "WR", "TE"];
export const FLEX_OK = {
  FLEX: ["RB", "WR", "TE"],
  REC_FLEX: ["WR", "TE"],
  WRRB_FLEX: ["RB", "WR"],
};
const NON_STARTER = new Set(["BN", "IR", "TAXI"]);
const IDP = new Set(["DL", "LB", "DB", "IDP_FLEX"]);

/** Reasons this league can't be simulated (empty list = supported). */
export function problems(league) {
  const out = [];
  const s = league.settings || {};
  const slots = league.roster_positions || [];
  if (s.type !== 3) out.push("This isn't a guillotine league. Only guillotine leagues are supported.");
  if (s.best_ball) out.push("Best ball leagues aren't supported.");
  if (s.waiver_type !== 2) out.push("Only FAAB waivers are supported.");
  if (slots.includes("K") || slots.includes("DEF")) out.push("Leagues with kicker or defense slots (K, DEF) aren't supported.");
  if (slots.some((x) => IDP.has(x))) out.push("Leagues with IDP slots aren't supported.");
  if (slots.includes("SUPER_FLEX") || slots.filter((x) => x === "QB").length > 1)
    out.push("Superflex and 2-QB leagues aren't supported.");
  const known = new Set([...POS, ...Object.keys(FLEX_OK), ...NON_STARTER, "K", "DEF", "SUPER_FLEX", ...IDP]);
  for (const x of new Set(slots)) if (!known.has(x)) out.push(`Roster slot ${x} isn't supported.`);
  if (league.status === "pre_draft" || league.status === "drafting") out.push("This league hasn't drafted yet.");
  if (league.status === "complete") out.push("This league's season is over.");
  return out;
}

/** A Sleeper league id is a long number; accept a pasted league URL too. */
export function parseLeagueId(text) {
  const m = String(text).trim().match(/(\d{15,20})/);
  return m ? m[1] : null;
}
