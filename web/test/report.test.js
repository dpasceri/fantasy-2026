// Renders the last real results.json with report.py and with report.js and checks the HTML is identical.
// guillotine/out/ holds a real league and is gitignored, so this skips itself when it's missing (as on CI).
// Nothing from it is written anywhere; the HTML only lives in memory here.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { render } from "../src/report.js";

const GDIR = fileURLToPath(new URL("../../guillotine/", import.meta.url));
const RESULTS = GDIR + "out/results.json";
const PY = GDIR + ".venv/bin/python";
const READY = existsSync(RESULTS) && existsSync(PY);

// Variants are made on the Python side so float-ness survives (JSON.parse turns 445.0 into 445) and
// report.py and report.js see exactly the same object.
const VARIANTS = `
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
import report

def variant(name):
    r = json.load(open(sys.argv[2]))
    if name == "prior":
        r["bid_model"]["prior"] = True
        r["bid_model"]["size"]["r2"] = float("nan")
    elif name == "pending":  # flip whichever way the file has it
        r["pending"] = None if r.get("pending") else {"owner": r["scenarios"][0]["owner"]}
    elif name == "peek":  # a peek at next week: the likeliest chop is assumed, so its auction is the only one
        sc = r["scenarios"][0]
        sc["known"], sc["assumed"], sc["p_chop"] = False, True, 0.973
        r["scenarios"] = [sc]
        r["pending"] = {"owner": sc["owner"], "n": len(sc["players"]), "assumed": True, "p_chop": 0.973}
    elif name == "no_live":
        r["live"] = None
    elif name == "live_in_progress" and r.get("live"):
        r["live"]["in_progress"] = ["A @ B", "C & D"]
    elif name == "no_used":
        r["dynamic"].pop("used", None)
    elif name == "me_first":
        r["power"].sort(key=lambda x: not x["me"])
    elif name == "me_last":
        r["power"].sort(key=lambda x: bool(x["me"]))
    elif name in ("no_faab", "priced_out"):  # no bid in any grid beats skipping
        for sc in r["scenarios"]:
            for pl in sc["players"]:
                if pl.get("bid"):
                    pl["bid"]["best_bid"] = 0
        if name == "no_faab":
            r["faab_me"] = 1
    elif name == "budget_1000":  # a $1000 league: dollar thresholds scale
        r["faab_pace"]["start_total"] *= 5
        for x in r["power"]:
            x["faab"] *= 5
    elif name == "switched_policy":  # this league's sims clearly favor another bidding style
        D = r["dynamic"]
        D["default"] = {"m_early": 0.75, "m_late": 0.75, "switch_week": 12}
        D["used"] = D["best"] = next(d for d in D["sweep"] if (d["m_early"], d["m_late"]) == (0.5, 0.75))
    elif name == "no_history":
        r["faab_pace"]["history"] = []
    with tempfile.TemporaryDirectory() as d:
        with open(report.render(r, os.path.join(d, "sheet.html"))) as f:
            return {"r": r, "html": f.read()}

names = ["base", "prior", "pending", "peek", "no_live", "live_in_progress", "no_used", "me_first", "me_last", "no_history",
         "no_faab", "priced_out", "budget_1000", "switched_policy"]
print(json.dumps({"css": report.CSS, "variants": {n: variant(n) for n in names}}))
`;

// Python's json writes NaN/Infinity bare; JSON.parse needs them quoted, then a reviver turns them back.
function parsePyJson(text) {
  const quoted = text.replace(/([:,[]\s*)(NaN|-?Infinity)(?=\s*[,\]}])/g, '$1"__$2__"');
  const special = { __NaN__: NaN, __Infinity__: Infinity, "__-Infinity__": -Infinity };
  return JSON.parse(quoted, (k, v) => (typeof v === "string" && v in special ? special[v] : v));
}

let cached;
const pyRenders = () => (cached ??= parsePyJson(execFileSync(PY, ["-c", VARIANTS, GDIR, RESULTS], { maxBuffer: 256 << 20 }).toString()));

/** Compare line by line so a mismatch reports one line, not two whole pages. */
function assertSameHtml(js, py, name) {
  const a = js.split("\n"), b = py.split("\n");
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if (a[i] !== b[i]) {
      let c = 0;
      while (a[i]?.[c] === b[i]?.[c]) c++;
      const at = (s) => (s ?? "<missing>").slice(Math.max(0, c - 80), c + 80);
      assert.fail(`${name}: line ${i + 1} differs at column ${c + 1}\n  js: ${at(a[i])}\n  py: ${at(b[i])}`);
    }
  }
}

test("report.js renders the same HTML as report.py", (t) => {
  if (!READY) return t.skip("needs guillotine/out/results.json and guillotine/.venv");
  const out = pyRenders();
  for (const [name, { r, html }] of Object.entries(out.variants)) {
    assertSameHtml(render(r, { css: out.css }), html, name);
  }
});

test("prior and pending variants actually exercise their branches", (t) => {
  if (!READY) return t.skip("needs guillotine/out/results.json and guillotine/.venv");
  const v = pyRenders().variants;
  assert.ok(Number.isNaN(v.prior.r.bid_model.size.r2));
  assert.match(v.prior.html, /too few to fit/);
  assert.doesNotMatch(v.prior.html, /R² =/);
  assert.notEqual(v.pending.html.includes("auction pending for"), v.base.html.includes("auction pending for"));
  assert.match(v.peek.html, /peek: assumes .+ is chopped in week \d+/);
  assert.match(v.peek.html, /Auction: if .+ is chopped <small>97\.3% likely in week \d+, assumed chopped/);
  assert.match(v.peek.html, /If .+ is chopped \(97%\)/);
});

test("without css the <style> element is left out (the page uses sheet.css)", (t) => {
  if (!READY) return t.skip("needs guillotine/out/results.json and guillotine/.venv");
  const html = render(parsePyJson(readFileSync(RESULTS, "utf8")));
  assert.doesNotMatch(html, /<style>/);
  assert.match(html, /<div class="wrap">/);
});
