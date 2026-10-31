import { test } from "node:test";
import assert from "node:assert/strict";
import { parseLeagueId, problems } from "../src/checks.js";
import { teams } from "../src/sleeper.js";

const guillotine = (slots, settings = {}) => ({
  status: "in_season",
  roster_positions: slots,
  settings: { type: 3, waiver_type: 2, best_ball: 0, ...settings },
});
const BASE = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "BN", "BN", "BN", "BN", "BN"];

test("a standard guillotine league is supported", () => {
  assert.deepEqual(problems(guillotine(BASE)), []);
  assert.deepEqual(problems(guillotine([...BASE, "REC_FLEX", "WRRB_FLEX", "IR"])), []);
});

test("unsupported formats each get a reason", () => {
  assert.match(problems(guillotine(BASE, { type: 0 }))[0], /guillotine/);
  assert.match(problems(guillotine([...BASE, "K"]))[0], /kicker or defense/);
  assert.match(problems(guillotine([...BASE, "DEF"]))[0], /kicker or defense/);
  assert.match(problems(guillotine([...BASE, "SUPER_FLEX"]))[0], /Superflex and 2-QB/);
  assert.match(problems(guillotine(["QB", ...BASE]))[0], /Superflex and 2-QB/);
  assert.match(problems(guillotine([...BASE, "LB"]))[0], /IDP/);
  assert.match(problems(guillotine(BASE, { waiver_type: 0 }))[0], /FAAB/);
  assert.match(problems(guillotine(BASE, { best_ball: 1 }))[0], /Best ball/);
  assert.match(problems({ ...guillotine(BASE), status: "pre_draft" })[0], /drafted/);
  assert.match(problems(guillotine([...BASE, "XYZ"]))[0], /slot XYZ/);
});

test("league ids parse from a bare id or a pasted URL", () => {
  assert.equal(parseLeagueId(" 1234567890123456789 "), "1234567890123456789");
  assert.equal(parseLeagueId("https://sleeper.com/leagues/1234567890123456789/team"), "1234567890123456789");
  assert.equal(parseLeagueId("my league"), null);
});

test("team list keeps only alive rosters and labels them", () => {
  const t = teams({
    users: [{ user_id: "u1", display_name: "zed", metadata: { team_name: "Zed FC" } }, { user_id: "u2", display_name: "amy" }],
    rosters: [{ roster_id: 1, owner_id: "u1", players: ["a"] }, { roster_id: 2, owner_id: "u2", players: ["b"] },
              { roster_id: 3, owner_id: "u3", players: [] }],
  });
  assert.deepEqual(t.map((x) => x.label), ["amy", "zed (Zed FC)"]);
});
