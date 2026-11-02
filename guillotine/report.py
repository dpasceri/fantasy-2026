"""Render out/results.json as the cheat sheet (out/cheatsheet.html). Called by run.py.

    python report.py     # re-render from the last results.json without re-running sims
"""
import datetime as dt
import html
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")

CSS = """
/* Layout: a laminated play-call sheet. Dense bordered call boxes with condensed caps tabs;
   the calls up top, the evidence below, the math on the back. You = blue, chop line = orange. */
:root {
  --bg: #e7ece9; --sheet: #f9fbfa; --ink: #101618; --ink-2: #434d52; --muted: #6a757a;
  --rule: #cdd6d2; --grid: #e2e8e5; --tab: #12324a; --tab-ink: #f4f8f6;
  --you: #2a78d6; --rival: #eb6834; --you-wash: rgba(42,120,214,.12);
  --good: #0ca30c; --warn: #fab219; --crit: #d03b3b; --s3: #1baf7a; --s4: #6250d6;
  --h1: #0d366b; --h2: #1c5cab; --h3: #3987e5; --h4: #86b6ef; --h5: #cde2fb;
  --f-display: "Barlow Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --f-body: "Barlow", "Segoe UI", system-ui, sans-serif;
  --f-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1315; --sheet: #161b1e; --ink: #e5ebe8; --ink-2: #b2bcb8; --muted: #86918d;
  --rule: #2b3438; --grid: #222a2e; --tab: #24465f; --tab-ink: #eef4f1;
  --you: #3987e5; --rival: #d95926; --you-wash: rgba(57,135,229,.16); --s3: #199e70; --s4: #9085e9;
  --h1: #cde2fb; --h2: #86b6ef; --h3: #3987e5; --h4: #1c5cab; --h5: #104281; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f1315; --sheet: #161b1e; --ink: #e5ebe8; --ink-2: #b2bcb8; --muted: #86918d;
  --rule: #2b3438; --grid: #222a2e; --tab: #24465f; --tab-ink: #eef4f1;
  --you: #3987e5; --rival: #d95926; --you-wash: rgba(57,135,229,.16); --s3: #199e70; --s4: #9085e9;
  --h1: #cde2fb; --h2: #86b6ef; --h3: #3987e5; --h4: #1c5cab; --h5: #104281; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.45 var(--f-body); margin: 0; }
.wrap { margin: 0 auto; padding-block: 20px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 14px; }
header.top { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: end; justify-content: space-between;
  border-bottom: 3px solid var(--ink); padding-bottom: 10px; }
header.top h1 { font: 700 clamp(30px, 5vw, 46px)/0.95 var(--f-display); letter-spacing: .01em; margin: 0; text-transform: uppercase; text-wrap: balance; }
header.top .sub { font: 500 13px/1.3 var(--f-display); letter-spacing: .08em; text-transform: uppercase; color: var(--ink-2); }
header.top .meta { font: 12px/1.5 var(--f-data); color: var(--muted); text-align: right; }
/* Widescreen: the boxes flow down newspaper columns, as many ~700px columns as the screen fits, balanced in height. */
.board { columns: 700px; column-gap: 14px; }
.board > .box { break-inside: avoid; margin: 0 0 14px; }
.box { background: var(--sheet); border: 1.5px solid var(--ink); min-width: 0; }
.box > h2 { margin: 0; background: var(--tab); color: var(--tab-ink); font: 700 15px/1 var(--f-display); letter-spacing: .1em;
  text-transform: uppercase; padding: 7px 10px 6px; display: flex; justify-content: space-between; gap: 12px; }
.box > h2 small { font: 500 11px/1.2 var(--f-data); letter-spacing: 0; text-transform: none; opacity: .85; align-self: center; }
.inner { padding: 12px 12px 14px; display: grid; gap: 10px; min-width: 0; }
.note { color: var(--ink-2); font-size: 13.5px; max-width: 70ch; margin: 0; }
.note b { color: var(--ink); }
small.sub { font: 11.5px/1.3 var(--f-data); color: var(--ink-2); }
.calls { counter-reset: call; display: grid; gap: 0; margin: 0; padding: 0; list-style: none; }
.calls li { display: grid; grid-template-columns: 34px 1fr; gap: 10px; padding: 9px 2px; border-top: 1px solid var(--rule); align-items: start; }
.calls li:first-child { border-top: 0; }
.calls li::before { counter-increment: call; content: counter(call); font: 700 22px/1 var(--f-display); color: var(--tab-ink);
  background: var(--tab); width: 30px; height: 30px; display: grid; place-items: center; }
.calls .k { font: 700 13px/1.2 var(--f-display); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); display: block; }
.calls .v { font-size: 15.5px; }
.calls .v b { font-weight: 650; }
.num { font-family: var(--f-data); font-variant-numeric: tabular-nums; }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th { font: 600 11.5px/1.2 var(--f-display); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); text-align: left;
  padding: 6px 7px; border-bottom: 1.5px solid var(--ink); vertical-align: bottom; }
td { padding: 6px 7px; border-bottom: 1px solid var(--grid); vertical-align: middle; }
td.r, th.r { text-align: right; }
td.num, td .num { font-family: var(--f-data); font-variant-numeric: tabular-nums; font-size: 12.5px; white-space: nowrap; }
tr.me td { background: var(--you-wash); }
tr.me td:first-child { box-shadow: inset 3px 0 0 var(--you); }
.chip { display: inline-flex; align-items: center; gap: 5px; font: 600 11.5px/1 var(--f-data); padding: 3px 7px 3px 5px;
  border: 1px solid var(--rule); border-radius: 3px; white-space: nowrap; color: var(--ink); }
.chip i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.chip.good i { background: var(--good); } .chip.warn i { background: var(--warn); } .chip.crit i { background: var(--crit); }
.tag { font: 600 10.5px/1 var(--f-display); letter-spacing: .08em; text-transform: uppercase; border: 1px solid var(--ink-2);
  color: var(--ink-2); padding: 3px 5px 2px; border-radius: 2px; white-space: nowrap; }
.tag.you { border-color: var(--you); color: var(--ink); background: var(--you-wash); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12.5px; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 3px; display: inline-block; border-radius: 2px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--muted); font: 11px var(--f-data); }
svg .lbl { fill: var(--ink-2); }
svg .lbl-strong { fill: var(--ink); font-weight: 600; }
.scen { border-top: 1.5px solid var(--ink); padding-top: 10px; display: grid; gap: 10px; }
.scen:first-child { border-top: 0; padding-top: 0; }
.scen h3 { margin: 0; font: 700 19px/1.1 var(--f-display); text-transform: uppercase; letter-spacing: .03em; display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: baseline; }
.scen h3 .num { font-size: 12px; color: var(--muted); font-weight: 400; letter-spacing: 0; text-transform: none; }
.minis { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)); gap: 10px; }
.mini { border: 1px solid var(--rule); padding: 8px 8px 4px; min-width: 0; }
.mini h4 { margin: 0 0 2px; font: 600 14px/1.2 var(--f-body); }
.mini .verdict { font: 12px/1.35 var(--f-data); color: var(--ink-2); margin: 0 0 4px; }
.mini .verdict b { color: var(--ink); }
.heat td { text-align: center; font-family: var(--f-data); font-size: 12.5px; padding: 7px 4px; border: 2px solid var(--sheet); }
.math { display: grid; gap: 12px; font-size: 14px; }
.math h3 { margin: 6px 0 0; font: 700 14px/1.1 var(--f-display); letter-spacing: .08em; text-transform: uppercase; }
.math p { margin: 0; max-width: 78ch; color: var(--ink-2); }
.math .eq { font: 13px/1.6 var(--f-data); background: var(--bg); border-left: 3px solid var(--tab); padding: 8px 10px; overflow-x: auto; white-space: pre-wrap; color: var(--ink); }
code, pre { font-family: var(--f-data); }
pre.cmd { margin: 0; background: var(--bg); padding: 10px; font-size: 12.5px; overflow-x: auto; border-left: 3px solid var(--tab); }
#tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--sheet); font: 12px/1.4 var(--f-data);
  padding: 6px 8px; border-radius: 3px; max-width: 260px; z-index: 20; opacity: 0; transition: opacity .08s; white-space: pre-line; }
#tip.on { opacity: 1; }
[data-tip] { cursor: default; }
@media (prefers-reduced-motion: reduce) { #tip { transition: none; } }
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(e){var t=e.target.closest('[data-tip]'); if(!t){tip.classList.remove('on');return;}
    tip.textContent=t.getAttribute('data-tip'); tip.classList.add('on');
    var x=e.clientX+14, y=e.clientY+14, w=tip.offsetWidth, h=tip.offsetHeight;
    if(x+w>innerWidth-8) x=e.clientX-w-14; if(y+h>innerHeight-8) y=e.clientY-h-14;
    tip.style.left=x+'px'; tip.style.top=y+'px';}
  document.addEventListener('pointermove',show);
  document.addEventListener('pointerleave',function(){tip.classList.remove('on')});
  document.addEventListener('focusin',function(e){var t=e.target.closest('[data-tip]'); if(!t)return; var r=t.getBoundingClientRect();
    tip.textContent=t.getAttribute('data-tip'); tip.classList.add('on'); tip.style.left=(r.left)+'px'; tip.style.top=(r.bottom+6)+'px';});
  document.addEventListener('focusout',function(){tip.classList.remove('on')});
})();
"""

esc = html.escape


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def pp(x, d=1):
    return f"{100 * x:+.{d}f} pts"


def short(label):
    """'Bijan Robinson (RB, ATL)' -> 'B. Robinson'"""
    name = label.split(" (")[0]
    parts = name.split()
    return name if len(parts) < 2 else f"{parts[0][0]}. {' '.join(parts[1:])}"


def risk_chip(p):
    cls, word = ("good", "low") if p < 0.08 else (("warn", "watch") if p < 0.15 else ("crit", "high"))
    return f'<span class="chip {cls}"><i></i>{pct(p)} {word}</span>'


def nice_ticks(lo, hi, n=4):
    span = hi - lo
    raw = span / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    step = min((s * mag for s in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    t = (int(lo / step) + (1 if lo % step else 0)) * step if lo > 0 else 0
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 6))
        t += step
    return out


# ------------------------------------------------------------------ charts (inline SVG)
def range_chart(rows, lo, hi):
    """rows: [(label, quantiles dict, color var, tip)] -> horizontal p10..p90 ranges, p25..p75 thick, median tick."""
    W, H, L, R = 620, 40 + 34 * len(rows), 128, 16
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Score ranges">']
    for t in nice_ticks(lo, hi, 6):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="8" y2="{H - 22}" stroke="var(--grid)" stroke-width="1"/>'
                   f'<text x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{t:g}</text>')
    for i, (label, q, color, tip) in enumerate(rows):
        y = 24 + 34 * i
        out.append(f'<g data-tip="{esc(tip)}" tabindex="0">'
                   f'<rect x="{L - 120}" y="{y - 14}" width="{W - L + 120}" height="28" fill="transparent"/>'
                   f'<text x="{L - 10}" y="{y + 4}" text-anchor="end" class="lbl">{esc(label)}</text>'
                   f'<line x1="{x(q["0.1"]):.1f}" x2="{x(q["0.9"]):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                   f'<rect x="{x(q["0.25"]):.1f}" y="{y - 6}" width="{x(q["0.75"]) - x(q["0.25"]):.1f}" height="12" rx="3" fill="{color}" opacity=".9"/>'
                   f'<line x1="{x(q["0.5"]):.1f}" x2="{x(q["0.5"]):.1f}" y1="{y - 10}" y2="{y + 10}" stroke="var(--ink)" stroke-width="2"/>'
                   f'<text x="{x(q["0.5"]):.1f}" y="{y - 13}" text-anchor="middle" class="lbl-strong">{q["0.5"]:g}</text></g>')
    out.append("</svg>")
    return "".join(out)


def line_chart(series, xlab, ylab, x_dom, y_dom, W=560, H=250, marks=None, fmt_y=lambda v: f"{100 * v:.0f}%", fmt_x=lambda v: f"${v:g}", xticks=5):
    """series: [(name, color var, [(x, y, tip)])]. 2px lines, 8px dots with a surface ring, hover tips."""
    L, R, T, B = 46, 14, 14, 34
    x = lambda v: L + (v - x_dom[0]) / (x_dom[1] - x_dom[0]) * (W - L - R)  # noqa: E731
    y = lambda v: T + (1 - (v - y_dom[0]) / (y_dom[1] - y_dom[0])) * (H - T - B)  # noqa: E731
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(ylab)} by {esc(xlab)}">']
    for t in nice_ticks(y_dom[0], y_dom[1], 4):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="var(--grid)"/>'
                 f'<text x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{fmt_y(t)}</text>')
    for t in nice_ticks(x_dom[0], x_dom[1], xticks):
        o.append(f'<text x="{x(t):.1f}" y="{H - 16}" text-anchor="middle">{fmt_x(t)}</text>')
    o.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="var(--rule)"/>')
    o.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 2}" text-anchor="middle" class="lbl">{esc(xlab)}</text>')
    for m in marks or []:
        if m[0] == "h":
            o.append(f'<line x1="{L}" x2="{W - R}" y1="{y(m[1]):.1f}" y2="{y(m[1]):.1f}" stroke="var(--ink-2)" stroke-width="1"/>'
                     f'<text x="{W - R}" y="{y(m[1]) - 4:.1f}" text-anchor="end" class="lbl">{esc(m[2])}</text>')
    for name, color, pts in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(a):.1f},{y(b):.1f}" for i, (a, b, _) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for a, b, tip in pts:
            o.append(f'<g data-tip="{esc(tip)}" tabindex="0"><circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="11" fill="transparent"/>'
                     f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4" fill="{color}" stroke="var(--sheet)" stroke-width="2"/></g>')
    for m in marks or []:
        if m[0] == "pt":
            o.append(f'<circle cx="{x(m[1]):.1f}" cy="{y(m[2]):.1f}" r="6.5" fill="none" stroke="var(--ink)" stroke-width="2"/>'
                     f'<text x="{x(m[1]):.1f}" y="{y(m[2]) - 11:.1f}" text-anchor="middle" class="lbl-strong">{esc(m[3])}</text>')
    o.append("</svg>")
    return "".join(o)


def spark(vals, me, W=120, H=26):
    x = lambda i: 2 + i / max(len(vals) - 1, 1) * (W - 4)  # noqa: E731
    y = lambda v: 2 + (1 - v) * (H - 4)  # noqa: E731
    d = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    c = "var(--you)" if me else "var(--muted)"
    return (f'<svg viewBox="0 0 {W} {H}" style="width:{W}px" aria-hidden="true"><line x1="2" x2="{W - 2}" y1="{H - 2}" y2="{H - 2}" stroke="var(--grid)"/>'
            f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/></svg>')


def heat_cell(v, lo, hi):
    steps = ["--h5", "--h4", "--h3", "--h2", "--h1"]
    k = 0 if hi <= lo else min(4, int((v - lo) / (hi - lo + 1e-12) * 5))
    ink = "var(--ink)" if k < 2 else "var(--sheet)"
    return f'background:var({steps[k]});color:{ink}'


# ------------------------------------------------------------------ page
def render(r, path=None):
    path = path or os.path.join(OUT, "cheatsheet.html")
    me = r["me"]
    st = r["standings"]
    mine = next(s for s in st if s["me"])
    by_chop = sorted(st, key=lambda s: s["p_chop_now"])
    safe_rank = [s["owner"] for s in by_chop].index(me) + 1
    wk = r["week"]
    lab = r["labels"]
    D = r["dynamic"]
    bud = sorted(D["budgets"], key=lambda d: d["budget"])
    bud_s = sorted(D["budgets_stress"], key=lambda d: d["budget"])
    top_b = max(bud, key=lambda d: d["budget"])
    # "free" spending: lowest budget still within 1 pt of the full-budget P(win)
    free_floor = min((d["budget"] for d in bud if d["p_win"] >= top_b["p_win"] - 0.01), default=top_b["budget"])
    below = [d for d in bud if d["budget"] < free_floor]
    slope = None
    if below:
        b1 = max(below, key=lambda d: d["budget"])
        f1 = next(d for d in bud if d["budget"] == free_floor)
        slope = (f1["p_win"] - b1["p_win"]) / max(f1["budget"] - b1["budget"], 1) * 10
    best = D["best"]
    used = D.get("used", best)
    worst_aggr = min((d for d in D["sweep"] if d["m_early"] >= 1.5), key=lambda d: d["p_win"], default=None)
    lus = r["lineups"]
    cur = next((l for l in lus if l["is_current"]), None)
    best_lu = lus[0]
    best_no_fa = next((l for l in lus if not l["uses_fa"]), None)
    fa = [f for f in r["free_agents"] if f["d_win"] > 0.002]
    scen = r["scenarios"]
    sims = r["sims"]

    # ---- power + league money
    pw = r["power"]
    me_pw = next(x for x in pw if x["me"])
    me_rank = pw.index(me_pw) + 1
    pace = r["faab_pace"]
    hist = pace["history"]
    fc = pace["forecast"]
    B0 = pace["start_total"] // (pace["n_auctions"] + 2)  # per-team starting budget
    usd = lambda x: round(x * B0 / 200)  # noqa: E731  (a $200-league dollar amount, in this league's FAAB)
    last = hist[-1] if hist else None
    burned_tot = sum(h["burned"] for h in hist)
    died_full = sum(1 for h in hist if h["burned"] >= B0)
    spent_hist = [h["spent"] for h in hist]
    n_left_auctions = pace["n_auctions"] - len(hist)
    i_next = 0 if r.get("pending") else 1  # forecast index of the next waiver run
    nxt = fc["top_price"][i_next]
    later = fc["top_price"][min(i_next + 3, len(fc["top_price"]) - 1)]

    # ---- the calls
    calls = []
    others = [x for x in pw if not x["me"]][:2]
    if me_rank == 1:
        lead = (f"<b>You.</b> {pct(me_pw['p_win'], 0)} to win the league if everyone, you included, keeps bidding the way they have; "
                f"next are {esc(others[0]['owner'])} ({pct(others[0]['p_win'], 0)}) and {esc(others[1]['owner'])} ({pct(others[1]['p_win'], 0)}).")
    else:
        lead = (f"<b>{esc(pw[0]['owner'])}</b> ({pct(pw[0]['p_win'], 0)} to win if everyone keeps bidding the way they have). "
                f"You're #{me_rank} at {pct(me_pw['p_win'], 0)}.")
    calls.append(("Best shape", lead + f" Bid the recommended way and you're at <b>{pct(me_pw['p_win_policy'], 0)}</b>."))
    calls.append(("This week",f"Chop risk <b>{pct(mine['p_chop_now'])}</b>, #{safe_rank} safest of {len(st)}. "
                  f"The score to beat (lowest rival) has a median of <b>{r['chop_line']['0.5']:g}</b> "
                  f"(9 weeks in 10 under {r['chop_line']['0.9']:g}). Your median is <b>{r['my_score']['0.5']:g}</b>."))
    if cur and best_no_fa:
        if cur["players"] == best_no_fa["players"] or abs(cur["p_chop"] - best_no_fa["p_chop"]) < 0.002:
            txt = f"Your current lineup is already the safest of the {r['n_lineups']} legal lineups from your roster ({pct(cur['p_chop'])} chop)."
        else:
            ins = [short(lab[p]) for p in best_no_fa["players"] if p not in cur["players"]]
            outs = [short(lab[p]) for p in cur["players"] if p not in best_no_fa["players"]]
            txt = f"Start <b>{', '.join(ins)}</b> over {', '.join(outs)}: chop risk {pct(cur['p_chop'])} → <b>{pct(best_no_fa['p_chop'])}</b>."
        if best_lu["uses_fa"] and best_lu["p_chop"] < best_no_fa["p_chop"] - 0.002:
            ins = [short(lab[p]) for p in best_lu["players"] if p not in best_no_fa["players"]]
            outs = [short(lab[p]) for p in best_no_fa["players"] if p not in best_lu["players"]]
            txt += f" Adding <b>{', '.join(ins)}</b> and starting him over {', '.join(outs)} gets it to <b>{pct(best_lu['p_chop'])}</b>."
        day = {p["pid"]: p["day"] for p in r["roster"]}
        early = [short(lab[p]) for p in best_lu["players"] if day.get(p) == "Thu" and p not in (r.get("live") or {}).get("locked", [])]
        if early:
            txt += f" {', '.join(early)} play{'s' if len(early) == 1 else ''} Thursday and lock{'s' if len(early) == 1 else ''} at kickoff."
        calls.append(("Lineup", txt))
    if fa:
        f0 = fa[0]
        extra = f" You have {r['open_spots']} open roster spot{'s' if r['open_spots'] != 1 else ''}, so no drop needed." if r["open_spots"] else " You'll need to drop your weakest bench player."
        have = [p for p in r["roster"] if p["pos"] == f0["pos"] and not p["ir"]]
        if f0["pos"] == "QB" and have:
            extra += (f" He's injury cover for {esc(short(have[0]['player']))}. The sim keeps rosters frozen and can't grab a replacement after an injury"
                      " the way you can, so read that number as an upper bound.")
        calls.append(("Free pickup", f"Add <b>{esc(f0['player'])}</b> now. He costs $0 (waivers have cleared) and adds <b>{pp(f0['d_win'])}</b> to your chance to win the league.{extra}"))
    when = "all season" if used["m_early"] == used["m_late"] else f"before week 12, {used['m_late']:g}× after"
    money = f"Bid about <b>{used['m_early']:g}×</b> what a typical manager in this league would bid for the same lineup gain, {when}."
    dflt = D.get("default") or {}
    d0 = next((d for d in D["sweep"] if (d["m_early"], d["m_late"]) == (dflt.get("m_early"), dflt.get("m_late"))), None)
    if d0 is not None and (d0["m_early"], d0["m_late"]) != (used["m_early"], used["m_late"]):
        money += (f" That's not the usual {d0['m_early']:g}×: in this league's sims it clearly wins more often "
                  f"({pct(used['p_win'])} vs {pct(d0['p_win'])} to win the league).")
    if worst_aggr and used["p_win"] - worst_aggr["p_win"] >= 0.005:
        money += f" Bidding 1.5× early drops your win chance by up to {100 * (used['p_win'] - worst_aggr['p_win']):.0f} pts."
    segs = []
    marks = [d for d in sorted(bud, key=lambda d: -d["budget"]) if d["budget"] in (top_b["budget"], top_b["budget"] - usd(40), top_b["budget"] - usd(80), top_b["budget"] - usd(120))]
    for a, b in zip(marks, marks[1:]):
        segs.append(f"${a['budget']}→${b['budget']} left costs <b>{100 * (a['p_win'] - b['p_win']):.1f} pts</b>")
    if segs:
        money += " Spending gets more expensive the lower you go: " + "; ".join(segs) + " of win chance."
    calls.append(("Money", money))
    if last:
        gap = last["linear_left"] - last["left"]
        calls.append(("League pace", (
            f"{'Ahead of' if gap > 0 else 'Behind'} the expected burn: the {last['alive']} alive teams hold <b>${last['left']:,}</b>, "
            f"${abs(gap):,.0f} ({abs(gap) / last['linear_left']:.0%}) {'below' if gap > 0 else 'above'} an even-spending pace. "
            f"Chopped teams took <b>${burned_tot:,}</b> out unspent ({died_full} of {len(hist)} with their full budget). "
            f"${pace['left_now']:,} over {n_left_auctions} auctions is ${pace['left_now'] / max(n_left_auctions, 1):.0f} each vs "
            f"${sum(spent_hist) / len(spent_hist):.0f} spent per auction so far, so prices fall: forecast top price "
            f"<b>${nxt:.0f}</b> at the next run, about ${later:.0f} three runs later.")))
    for sc in scen[:3]:
        rows = [p for p in sc["players"] if p.get("bid")]
        if not rows:
            continue
        p0 = max(rows, key=lambda p: p["bid"]["best_p_win"] - p["bid"]["p_win_if_zero"])
        b = p0["bid"]
        lift = b["best_p_win"] - b["p_win_if_zero"]
        head = f"{sc['owner']} was chopped" if sc["known"] else f"If {sc['owner']} is chopped ({pct(sc['p_chop'], 0)})"
        if b["best_bid"] > 0 and lift >= 0.005:
            calls.append(("Auction", f"{head}: bid <b>${b['best_bid']}</b> on <b>{esc(short(p0['player']))}</b> and walk away above ${b['walk_away']} "
                          f"(+{100 * lift:.1f} pts vs skipping). Rivals' top bid on him: median ${p0['rival_top']['0.5']:.0f}, 90th pct ${p0['rival_top']['0.9']:.0f}."))
        else:
            big = max(sc["players"], key=lambda p: p["d_win_static"])
            if r["faab_me"] < usd(10):
                calls.append(("Auction", f"{head}: with ${r['faab_me']} FAAB you can't outbid anyone, so put in $0 claims and hope players clear. "
                              f"{esc(short(big['player']))} would add {pp(big['d_win_static'])} if he got through."))
            elif big["d_win_static"] >= 0.01:
                calls.append(("Auction", f"{head}: rivals price these players out. {esc(short(big['player']))} would add {pp(big['d_win_static'])} "
                              f"if free, but no bid beats skipping by even half a point once you pay what it takes. $0–{usd(10)} flyers only."))
            else:
                calls.append(("Auction", f"{head}: nobody there is worth real money. $0–{usd(10)} flyers only: no bid on "
                              f"{esc(short(big['player']))} beats skipping by even half a point, because your lineup barely improves."))

    def call_html():
        return "".join(f'<li><div><span class="k">{esc(k)}</span><span class="v">{v}</span></div></li>' for k, v in calls)

    # ---- week box
    q_me, q_line = r["my_score"], r["chop_line"]
    lo = max(0, min(q_line["0.1"], q_me["0.1"]) - 10)
    hi = max(q_line["0.9"], q_me["0.9"]) + 10
    rc = range_chart([
        (f"{me} (you)", q_me, "var(--you)", f"Your week {wk} score\n10th pct {q_me['0.1']:g} · median {q_me['0.5']:g} · 90th {q_me['0.9']:g}"),
        ("Lowest rival", q_line, "var(--rival)", f"Lowest rival score = the chop line\n10th pct {q_line['0.1']:g} · median {q_line['0.5']:g} · 90th {q_line['0.9']:g}"),
    ], lo, hi)

    def lu_row(l, i):
        tags = []
        if l["is_current"]:
            tags.append('<span class="tag you">current</span>')
        if l["is_proj_opt"]:
            tags.append('<span class="tag">max points</span>')
        if l["uses_fa"]:
            tags.append(f'<span class="tag">adds {esc(short(lab[l["uses_fa"][0]]))}</span>')
        names = ", ".join(f"<b>{esc(short(lab[p]))}</b>" if p in locked else esc(short(lab[p])) for p in l["players"])
        cls = ' class="me"' if l["is_current"] else ""
        return (f'<tr{cls}><td class="num">{i}</td><td class="num r">{pct(l["p_chop"], 2)}</td>'
                f'<td class="num r">{l["mean"]:.1f} ± {l["sd"]:.0f}</td><td style="min-width:260px">{names}</td><td>{" ".join(tags)}</td></tr>')

    lu_rank = {id(l): i + 1 for i, l in enumerate(lus)}
    lv = r.get("live")
    locked = set(lv["locked"]) if lv else set()
    live_box = ""
    if lv:
        n = lv["n"]
        when = dt.datetime.fromtimestamp(lv["as_of"]).strftime("%a %-I:%M %p")
        prog = f" ({esc(', '.join(lv['in_progress']))})" if lv["in_progress"] else ""
        live_rows = "".join(
            f'<tr{" class=\'me\'" if t["me"] else ""}><td>{esc(t["owner"])}</td><td class="num r">{t["pts"]:.1f}</td>'
            f'<td class="num r">{t["left"]}</td><td class="num r">{pct(t["p_chop"])}</td></tr>' for t in lv["teams"])
        live_box = (f'<section class="box"><h2>Week {wk} so far <small>scores as of {when}</small></h2><div class="inner">'
                    f'<p class="note">{n["post"]} of {sum(n.values())} games final, {n["in"]} in progress{prog}. Every sim below starts from these scores: '
                    f'finished players keep their points, players in a live game add the unplayed share of a full-game draw, and lineups are locked '
                    f'the way Sleeper locks them (bold in the lineup table).</p>'
                    f'<div class="tbl"><table><thead><tr><th>Team</th><th class="r">Pts so far</th><th class="r">Starters left</th>'
                    f'<th class="r">Chop risk</th></tr></thead><tbody>{live_rows}</tbody></table></div></div></section>')
    lu_html = "".join(lu_row(l, lu_rank[id(l)] if lu_rank[id(l)] <= 10 else "—") for l in lus)

    roster_rows = "".join(
        f'<tr><td>{esc(p["player"])}{" <span class=\'tag\'>IR</span>" if p["ir"] else ""}</td>'
        f'<td>{esc(p["inj"] or "")}</td><td>{esc(p["day"] or "bye")}</td>'
        + "".join(f'<td class="num r">{v:g}</td>' for v in p["proj"]) + "</tr>"
        for p in sorted(r["roster"], key=lambda p: -sum(p["proj"])))

    fa_rows = "".join(
        f'<tr><td>{esc(f["player"])}</td><td class="num r">{f["proj_now"]:g}</td><td class="num r">{f["ros_avg"]:g}</td>'
        f'<td class="num r">{pp(f["d_win"])}</td><td class="num r">{pp(f["d_survive_now"], 2)}</td><td>{esc(f["inj"] or "")}</td></tr>'
        for f in r["free_agents"][:10])

    # ---- standings
    st_rows = "".join(
        f'<tr{" class=\'me\'" if s["me"] else ""}><td>{esc(s["owner"])}</td><td class="num r">${s["faab"]}</td>'
        f'<td>{risk_chip(s["p_chop_now"])}</td><td class="num r"><b>{pct(s["p_win"])}</b></td>'
        f'<td class="num r">{s["proj_now"]:g}</td><td class="num r">{s["season_pts"]:g}</td><td>{spark(s["survival"], s["me"])}</td></tr>'
        for s in st)

    # ---- auction board
    def scen_html(sc, first):
        rows = sc["players"][:7]
        trs = []
        for p in rows:
            b = p.get("bid")
            rec = (f'<b>${b["best_bid"]}</b> <span class="num" style="color:var(--muted)">(≤ ${b["walk_away"]})</span>' if b else "—")
            gain_line = f'+{100 * (b["best_p_win"] - b["p_win_if_zero"]):.1f} pts' if b else ""
            rg = "; ".join(f'{g["owner"]} {g["gain"]:g}/wk (${g["faab"]})' for g in p["rival_gains"][:3])
            trs.append(f'<tr><td data-tip="{esc("Top rival gains: " + rg)}" tabindex="0">{esc(p["player"])}{" · " + esc(p["inj"]) if p.get("inj") else ""}</td>'
                       f'<td class="num r">{p["gain_me"]:g}</td><td class="num r">{pp(p["d_win_static"])}</td>'
                       f'<td class="num r">${p["rival_top"]["0.5"]:.0f} / ${p["rival_top"]["0.9"]:.0f}</td>'
                       f'<td class="num r">{pct(p["p_no_rival_bid"], 0)}</td><td class="num r">{rec}</td><td class="num r">{gain_line}</td></tr>')
        minis = []
        for p in rows:
            b = p.get("bid")
            if not b:
                continue
            pts = [(g["bid"], g["p_win"], f"Bid ${g['bid']} on {short(p['player'])}\nP(win league) {pct(g['p_win'])} ± {pct(g['se'])}") for g in b["grid"]]
            ser = [("You", "var(--you)", pts)]
            sb = p.get("stress_bid")
            if sb:
                ser.append(("Rivals bid harder", "var(--rival)", [(g["bid"], g["p_win"], f"Rivals bid harder · bid ${g['bid']}\nP(win league) {pct(g['p_win'])}") for g in sb["grid"]]))
            ys = [g["p_win"] for g in b["grid"]] + ([g["p_win"] for g in sb["grid"]] if sb else [])
            ylo, yhi = max(0, min(ys) - 0.02), min(1, max(ys) + 0.03)
            ch = line_chart(ser, "your bid", "P(win league)", (0, max(g["bid"] for g in b["grid"])), (ylo, yhi), W=240, H=170, xticks=3,
                            marks=[("h", b["p_win_if_zero"], "skip him"), ("pt", b["best_bid"], b["best_p_win"], f"${b['best_bid']}")])
            leg = ('<div class="legend"><span><i style="background:var(--you)"></i>rivals as fitted</span>'
                   '<span><i style="background:var(--rival)"></i>rivals bid harder</span></div>') if sb else ""
            minis.append(f'<div class="mini"><h4>{esc(p["player"])}</h4><p class="verdict">Best bid <b>${b["best_bid"]}</b> '
                         f'(anything ${b["good_range"][0]}–${b["good_range"][1]} is within noise) · walk away above <b>${b["walk_away"]}</b>'
                         f'{" · if rivals bid harder: $" + str(sb["best_bid"]) if sb else ""}</p>{leg}{ch}</div>')
        head = (f"Auction: {esc(sc['owner'])} was chopped" if sc["known"] else f"Auction: if {esc(sc['owner'])} is chopped")
        sub = (f"on waivers now · {sims['grid_paths']:,} season paths per bid" if sc["known"]
               else f"{pct(sc['p_chop'])} likely in week {wk - 1}, assumed chopped · {sims['grid_paths']:,} season paths per bid"
               if sc.get("assumed")
               else f"{pct(sc['p_chop'])} likely this week · {sims['grid_paths']:,} paths per bid")
        return (f'<section class="box"><h2>{head} <small>{sub}</small></h2><div class="inner">{auction_note if first else ""}'
                f'<div class="tbl"><table><thead><tr><th>Player</th><th class="r">Gain to you /wk</th>'
                f'<th class="r">Win chance if free</th><th class="r">Top rival bid med / p90</th><th class="r">No rival bids</th>'
                f'<th class="r">Bid (walk-away)</th><th class="r">vs skipping</th></tr></thead><tbody>{"".join(trs)}</tbody></table></div>'
                + (f'<div class="minis">{"".join(minis)}</div>' if minis else "") + "</div></section>")

    auction_note = ('<p class="note"><b>Gain to you</b> is lineup points per week over the next 4 weeks. <b>Win chance if free</b> is the static boost '
                    'with no future auctions. <b>Bid</b> is the bid that maximized P(win league) in full season-with-auctions sims; rivals bid from '
                    'their fitted models and your later auctions are played out too. <b>Walk-away</b> is the highest bid that still leaves you better '
                    'off than skipping him. Hover a player for who else wants him.</p>')
    scen_blocks = "\n".join(scen_html(sc, i == 0) for i, sc in enumerate(scen[:5]))

    # ---- money charts
    bmax = max(d["budget"] for d in bud)
    pts1 = [(d["budget"], d["p_win"], f"${d['budget']} left · rivals as fitted\nP(win league) {pct(d['p_win'])} ± {pct(d['se'])}") for d in bud]
    pts2 = [(d["budget"], d["p_win"], f"${d['budget']} left · rivals bid harder\nP(win league) {pct(d['p_win'])}") for d in bud_s]
    ys = [d["p_win"] for d in bud + bud_s]
    money_chart = line_chart([("rivals as fitted", "var(--you)", pts1), ("rivals bid harder", "var(--rival)", pts2)],
                             "FAAB you start the auctions with", "P(win league)", (0, bmax), (0, min(1, max(ys) + 0.05)), W=560, H=260)
    ms = sorted({d["m_early"] for d in D["sweep"]})
    ml = sorted({d["m_late"] for d in D["sweep"]})
    vals = [d["p_win"] for d in D["sweep"]]
    lo_v, hi_v = min(vals), max(vals)
    heat = ["<table class='heat'><thead><tr><th>Weeks before 12 ↓ / after →</th>" + "".join(f"<th class='r'>{b:g}×</th>" for b in ml) + "</tr></thead><tbody>"]
    for a in ms:
        heat.append(f"<tr><th>{a:g}×</th>")
        for b in ml:
            d = next(d for d in D["sweep"] if d["m_early"] == a and d["m_late"] == b)
            ring = ";outline:2px solid var(--you);outline-offset:-2px" if (a, b) == (used["m_early"], used["m_late"]) else ""
            heat.append(f'<td style="{heat_cell(d["p_win"], lo_v, hi_v)}{ring}" data-tip="{esc(f"Bid {a:g}× typical before week 12, {b:g}× after\nP(win league) {pct(d["p_win"])} ± {pct(d["se"])}\nAvg spent ${d["avg_spent"]:.0f}")}" tabindex="0">{pct(d["p_win"], 1)}</td>')
        heat.append("</tr>")
    heat.append("</tbody></table>")
    top_txt = ("It is also the top cell this run." if best is used or (best["m_early"], best["m_late"]) == (used["m_early"], used["m_late"]) else
               f"The top cell this run ({best['m_early']:g}× / {best['m_late']:g}×, {pct(best['p_win'])}) is within noise of it: "
               "on fresh simulations these cells reshuffle, and 0.75× all season held up best.")

    # ---- rivals
    bmr = r["bid_model"]
    riv_rows = "".join(
        f'<tr{" class=\'me\'" if m["owner"] == me else ""}><td>{esc(m["owner"])}</td><td class="num r">${m["faab"]}</td>'
        f'<td class="num r">{m["n_bids"]} / {m["n_chances"]}</td><td class="num r">{m["bids_often"]:+.2f}</td><td class="num r">{m["bids_hard"]:+.2f}</td></tr>'
        for m in sorted(bmr["managers"], key=lambda m: -m["faab"]))
    pb = r["price_backtest"]
    pb_in = sum(1 for x in pb if x["p10"] <= x["actual"] <= x["p90"])
    pb_rows = "".join(f'<tr><td class="num">{x["leg"]}</td><td>{esc(x["player"])}</td><td class="num r">${x["actual"]}</td>'
                      f'<td class="num r">${x["median"]:.0f}</td><td class="num r">${x["p10"]:.0f}–${x["p90"]:.0f}</td></tr>'
                      for x in sorted(pb, key=lambda x: -x["actual"])[:8])

    # ---- math / model card
    mr = r["model_report"]
    cop = mr["copula"]
    bt = r.get("backtest")
    bt_txt = ""
    if bt:
        hits = sum(1 for w in bt["weeks"] if w["model_rank_of_chopped"] == 1)
        avg_p = sum(w["p_model_actual_starters"] for w in bt["weeks"]) / len(bt["weeks"])
        avg_u = sum(w["p_uniform"] for w in bt["weeks"]) / len(bt["weeks"])
        bt_txt = (f"Replaying weeks 1–{len(bt['weeks'])} with the rosters and starters each team actually used, the model gave the team that really got "
                  f"chopped {pct(avg_p)} on average (a uniform guess gives {pct(avg_u)}) and ranked them most likely {hits} of {len(bt['weeks'])} weeks. "
                  f"Log-loss {bt['logloss_model']:.3f} vs {bt['logloss_uniform']:.3f} for uniform. Chops are noisy; this is an edge, not a crystal ball.")
    sz, pa = bmr["size"], bmr["participation"]
    if bmr.get("prior"):
        fit_src = (f"This league has only {bmr['n_bids']} bids so far, too few to fit, so these are a prior fit on another "
                   "league's bids, with budgets scaled to this league's FAAB and no per-manager effects.")
        r2_txt = ""
    else:
        fit_src = "Fit on every bid this league has placed, winners and losers, with each manager's budget at the time."
        r2_txt = f" R² = {sz['r2']}."
    gen = dt.datetime.fromtimestamp(r["generated_at"]).strftime("%a %b %-d, %-I:%M %p")
    fet = dt.datetime.fromtimestamp(r["fetched_at"]).strftime("%a %b %-d, %-I:%M %p")
    pend = ("" if not r.get("pending")
            else f' · peek: assumes {esc(r["pending"]["owner"])} is chopped in week {wk - 1}' if r["pending"].get("assumed")
            else f' · auction pending for {esc(r["pending"]["owner"])}')
    weeks = r["weeks"]


    # ---- power rankings box
    pw_rows = "".join(
        f'<tr{" class=\'me\'" if x["me"] else ""}><td class="num">{i}</td><td>{esc(x["owner"])}'
        f'{f"<br><small class=sub>{pct(x["p_win_policy"])} with the recommended bids</small>" if x["me"] else ""}</td>'
        f'<td class="num r"><b>{pct(x["p_win"])}</b></td>'
        f'<td class="num r">{pct(x["p_win_frozen"])}</td><td class="num r">${x["faab"]}</td>'
        f'<td class="num r">{x["ros_proj"]:g}</td>'
        f'<td class="num r" data-tip="{esc(f"Weakest projected lineup: week {x["worst_week"]}, {x["worst_proj"]:g} pts (byes/injuries)")}" tabindex="0">wk {x["worst_week"]} · {x["worst_proj"]:.0f}</td>'
        f'<td>{risk_chip(x["p_chop_now"])}</td><td class="num r">{x["e_weeks"]:.1f}</td></tr>'
        for i, x in enumerate(pw, start=1))
    broke = [x['owner'] for x in pw if x['faab'] <= usd(15)]
    broke_txt = (', '.join(esc(b) for b in broke) + f' have ${usd(15)} or less') if broke else 'nobody is broke yet'
    power_box = f"""<section class="box"><h2>Power rankings <small>who wins with the auctions played out · {r['sims']['dynamic_paths']:,}+ season paths</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>#</th><th>Team</th><th class="r">Win league</th><th class="r">Frozen rosters</th><th class="r">FAAB</th><th class="r">Lineup proj /wk</th><th class="r">Weakest week</th><th>Chop wk {wk}</th><th class="r">Weeks survived of {len(weeks)}</th></tr></thead><tbody>{pw_rows}</tbody></table></div>
  <p class="note"><b>Win league</b> plays every remaining week and auction, with every manager, you included, bidding the way they have all season. That's the fair ranking; under your name is your chance if you bid the recommended way instead. <b>Frozen rosters</b> is the same race with no more auctions: it rewards teams that already spent ({broke_txt}) and ignores what money will buy. Lineup proj is the average projected best lineup over weeks {weeks[0]}–{weeks[-1]}; the weakest week is the bye-week hole most likely to get them chopped.</p>
</div></section>"""

    # ---- FAAB pace box
    N_teams = pace["n_auctions"] + 2
    act = [(0, pace["start_total"], f"Season start\n{N_teams} teams · ${pace['start_total']:,}")]
    act += [(h["leg"], h["left"], f"After week {h['leg']}\n{h['alive']} alive hold ${h['left']:,}\nspent ${h['spent']} · burned ${h['burned']} · top price ${h['top_price']}") for h in hist]
    lin = [(kk, (N_teams - kk) * B0 * max(0.0, 1 - kk / pace["n_auctions"]),
            f"Even pace after {kk} auctions\n${(N_teams - kk) * B0 * max(0.0, 1 - kk / pace['n_auctions']):,.0f}") for kk in range(0, pace["n_auctions"] + 1)]
    fpts = []
    if last:
        cur_left = pace["left_now"]
        x0 = last["leg"]
        fpts.append((x0, cur_left, f"Now\n${cur_left:,}"))
        if r.get("pending"):
            cur_left -= fc["league_spend"][0]
        for i in range(1, len(weeks)):
            cur_left -= fc["burned"][i] + fc["league_spend"][i]
            fpts.append((weeks[i - 1], max(cur_left, 0),
                         f"Forecast after week {weeks[i - 1]}\n${max(cur_left, 0):,.0f} left\nspend ${fc['league_spend'][i]:.0f} · burned ${fc['burned'][i]:.0f} · top ${fc['top_price'][i]:.0f}"))
    pace_chart = line_chart([("actual", "var(--s3)", act), ("even-spending pace", "var(--muted)", lin), ("forecast", "var(--s4)", fpts)],
                            "week", "FAAB held by alive teams", (0, r["final_week"]), (0, pace["start_total"]), W=560, H=260,
                            fmt_y=lambda v: f"${v / 1000:g}K" if v >= 1000 else f"${v:g}", fmt_x=lambda v: f"{v:g}")
    pace_rows = "".join(
        f'<tr><td class="num">{h["leg"]}</td><td class="num r">{h["alive"]}</td><td class="num r">${h["spent"]}</td>'
        f'<td class="num r">${h["burned"]}</td><td class="num r">${h["top_price"]}</td><td class="num r">{h["buyers"]}</td>'
        f'<td class="num r">${h["left"]:,}</td><td class="num r">${h["linear_left"]:,.0f}</td></tr>' for h in hist)
    tn = pace.get("trade_net_me", 0)
    traded = f" (you received ${tn})" if tn > 0 else f" (you sent ${-tn})" if tn < 0 else ""
    pace_box = f"""<section class="box"><h2>League FAAB pace <small>${pace['start_total']:,} at the start</small></h2><div class="inner">
  <div class="legend"><span><i style="background:var(--s3)"></i>actual</span><span><i style="background:var(--muted)"></i>even-spending pace</span><span><i style="background:var(--s4)"></i>model forecast</span></div>
  {pace_chart}
  <div class="tbl"><table><thead><tr><th>Wk</th><th class="r">Alive</th><th class="r">Spent</th><th class="r">Burned</th><th class="r">Top price</th><th class="r">Buyers</th><th class="r">Left</th><th class="r">Even pace</th></tr></thead><tbody>{pace_rows}</tbody></table></div>
  <p class="note"><b>Burned</b> is the unspent FAAB a team takes with it when it's chopped. <b>Even pace</b> means every team spends its ${B0} evenly over the season's {pace['n_auctions']} auctions. FAAB sent in trades is counted{traded}. The forecast comes from the power-ranking sims.</p>
</div></section>"""

    page = f"""<title>{esc(me)} Chop Sheet</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="top">
  <div><div class="sub">{esc(r['league'])} · guillotine · week {wk} of {r['final_week']}{pend}</div>
  <h1>{esc(me)} chop sheet</h1></div>
  <div class="meta">FAAB ${r['faab_me']} · data pulled {fet}<br>sims run {gen} on {r['threads']} threads</div>
</header>

<div class="board">
{live_box}
<section class="box"><h2>The calls <small>start here</small></h2><div class="inner"><ol class="calls">{call_html()}</ol></div></section>

{power_box}

<section class="box"><h2>Week {wk}: the chop line <small>{sims['season']:,} sims</small></h2><div class="inner">
  <div class="legend"><span><i style="background:var(--you)"></i>you</span><span><i style="background:var(--rival)"></i>lowest of the other {len(st) - 1}</span><span>bar = middle half, line = 10th–90th pct, tick = median</span></div>
  {rc}
  <p class="note">You're chopped when your score falls below the lowest rival's. Your chop risk this week: {risk_chip(mine['p_chop_now'])}</p>
</div></section>
<section class="box"><h2>Week {wk}: lineup <small>{r['n_lineups']} legal lineups · {sims['lineups']:,} sims each</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>#</th><th class="r">Chop risk</th><th class="r">Pts ± sd</th><th>Starters</th><th></th></tr></thead><tbody>{lu_html}</tbody></table></div>
  <p class="note">Every lineup is scored on the same simulated weeks, so even small gaps between rows are real differences, not noise. Inactive starters are covered by your best eligible bench player.</p>
</div></section>

<section class="box"><h2>Free agents for your open spot <small>$0, first come</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>Player</th><th class="r">Wk {wk}</th><th class="r">ROS /wk</th><th class="r">Win league</th><th class="r">Survive wk {wk}</th><th></th></tr></thead><tbody>{fa_rows}</tbody></table></div>
  <p class="note">Change in your chances from adding the player and changing nothing else (opponents' rosters frozen). Players dropped since the last waiver run are left out because they still need a claim.</p>
</div></section>
<section class="box"><h2>Your roster <small>projected points, weeks {weeks[0]}–{weeks[-1]}</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>Player</th><th>Status</th><th>Wk {wk}</th>{''.join(f'<th class="r">{w}</th>' for w in weeks)}</tr></thead><tbody>{roster_rows}</tbody></table></div>
</div></section>

{scen_blocks}

<section class="box"><h2>What a FAAB dollar is worth <small>{D['budgets'][0]['n']:,} paths per point</small></h2><div class="inner">
  <div class="legend"><span><i style="background:var(--you)"></i>rivals as fitted</span><span><i style="background:var(--rival)"></i>rivals bid ~2× as often, ~35% higher</span></div>
  {money_chart}
  <p class="note">Same rest of season, different starting budget. The curve is flat at the top, so a little spending is free. It falls fast near the bottom: money is what lets you buy the studs that come loose when strong teams get chopped late.</p>
</div></section>
<section class="box"><h2>How hard to bid <small>× the league-typical bid for the same gain</small></h2><div class="inner">
  <div class="tbl">{''.join(heat)}</div>
  <p class="note">P(win league) for each bidding style, all on the same {D['sweep'][0]['n']:,} season paths. Darker is better. The sheet bids <b>{used['m_early']:g}×</b> early and <b>{used['m_late']:g}×</b> after week 12 (outlined, {pct(used['p_win'])}). {top_txt} Overbidding early burns the budget on players who only marginally upgrade you.</p>
</div></section>

{pace_box}
<section class="box"><h2>Rival bidders <small>{bmr['n_bids']} bids over {bmr['n_rows']:,} chances</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>Manager</th><th class="r">FAAB</th><th class="r">Bids / chances</th><th class="r">Bids often</th><th class="r">Bids hard</th></tr></thead><tbody>{riv_rows}</tbody></table></div>
  <p class="note"><b>Bids often</b> and <b>bids hard</b> are each manager's fitted effects (log-odds and log-dollars vs league average, shrunk toward 0). The dangerous ones have money <i>and</i> a positive score. Price check: {pb_in} of {len(pb)} contested players this season cleared inside the model's 10th–90th range.</p>
  <div class="tbl"><table><thead><tr><th>Wk</th><th>Player</th><th class="r">Cleared</th><th class="r">Model median</th><th class="r">Model 80% range</th></tr></thead><tbody>{pb_rows}</tbody></table></div>
</div></section>

<section class="box"><h2>The math: scores <small>back of the sheet, 1 of 3</small></h2><div class="inner math">
  <h3>1 · The chop</h3>
  <p>Each week, the lowest score among the alive teams is chopped. Your weekly risk is the probability that your total is below the minimum of the others, which depends on the whole joint distribution, not just averages:</p>
  <div class="eq">P(chop) = P( S_you &lt; min_j S_j )
S_team  = Σ_slots score(starter, or first available bench sub)</div>
  <h3>2 · One player's week</h3>
  <p>Fitted on {cop['n_obs']:,} projected-vs-actual player weeks (last season plus this one). For each position and projection level, the empirical distribution of actual ÷ projected gives the quantile function, so busts, in-game injuries and boom weeks all come straight from data. Projections were unbiased in every bucket.</p>
  <div class="eq">score_i = proj_i · R_pos(u_i ; proj_i)
          R = empirical quantiles of actual ÷ projected
u_i     = Φ( a_pos · F_team + √(1 − a_pos²) · ε_i )
F_team  = √g · G_game + √(1 − g) · T_team
a = QB {cop['loadings']['QB']}, RB {cop['loadings']['RB']}, WR {cop['loadings']['WR']}, TE {cop['loadings']['TE']}     g = {cop['game_share']}</div>
  <p>Measured residual correlations: QB–WR {cop['same_team_corr'].get('QB-WR')}, QB–TE {cop['same_team_corr'].get('QB-TE')}, RB–RB {cop['same_team_corr'].get('RB-RB')} (committees cannibalize), opposing QBs {cop['opp_team_corr'].get('QB-QB')} (shootouts). Future weeks add a random-walk drift in each player's level (sd 0.10 per week, log scale) and a 2% weekly chance of a multi-week injury (mean 3 weeks).</p>
</div></section>

<section class="box"><h2>The math: bids <small>back of the sheet, 2 of 3</small></h2><div class="inner math">
  <h3>3 · Rivals' bids</h3>
  <div class="eq">gain_ji = mean over the next 4 wks of
          max(0, proj_iw − weakest starter i could replace on team j)
P(j bids)      = logistic({pa['a0']} + {pa['a_log1p_gain']}·ln(1+gain) {pa['a_log_budget']:+}·ln(budget) + v_j)
ln(bid | bids) = {sz['b0']} + {sz['b_gain']}·gain + {sz['b_log_budget']}·ln(budget) + {sz['b_leg']}·week + u_j + e
                 sd(e) = {sz['sigma']}</div>
  <p>{fit_src} Each extra lineup point a week raises a typical bid by about {100 * (2.71828 ** sz['b_gain'] - 1):.0f}%, and bids rise about {100 * (2.71828 ** sz['b_leg'] - 1):.0f}% per week as the season goes on.{r2_txt}</p>
  <h3>4 · Your bid</h3>
  <p>Sleeper FAAB is a sealed first-price auction: highest bid wins and pays its own bid. If F(b) is the chance your bid beats every rival and V is what the player adds to your win chance, the textbook answer is the bid that maximizes F(b) · (V − λb), where λ is what a dollar is worth to you later. Here λ isn't constant (the money curve bends), and V depends on what you'd buy later instead. So the sheet doesn't use the formula directly. It plays the whole rest of the season with every candidate bid and picks the one with the highest P(win league):</p>
  <div class="eq">b* = argmax_b  P(win league | bid b on player i,
                then your best policy in every later auction)</div>
</div></section>

<section class="box"><h2>The math: checks <small>back of the sheet, 3 of 3</small></h2><div class="inner math">
  <h3>5 · Why Monte Carlo, and why not just more of it</h3>
  <p>The quantities here are minimums of correlated, skewed, zero-inflated sums run through lineup logic and elimination, so there's no clean closed form. Past about a million sims the sampling error (±{100 * (0.05 * 0.95 / max(sims['season'], 1)) ** 0.5:.2f} pts on a 5% risk) is far below the model error in the projections. So the CPU goes to <b>common random numbers</b> instead: every lineup, pickup and bid is scored on the same simulated weeks, which makes differences between choices sharper than more draws would. The rest goes into the {len(D['sweep'])}-policy and per-bid season sweeps.</p>
  <h3>6 · Does it work?</h3>
  <p>{bt_txt}</p>
  <h3>Known gaps</h3>
  <p>Frozen-roster numbers ignore the auctions (the dynamic sims don't). Rivals are assumed to start their max-projection lineup. Bid behavior is fit from about 7 weeks of history, so the stress-test line shows how the answer moves if rivals bid harder. The roster cap drops the weakest rest-of-season player automatically.</p>
</div></section>

<section class="box"><h2>Re-run it <small>do this Tuesday after the chop, before bids lock</small></h2><div class="inner">
  <pre class="cmd">cd ~/Desktop/"Fantasy 2026"/guillotine
.venv/bin/python fetch.py && .venv/bin/python run.py
    # ~7 min on 8 threads, runs at low priority
GSIM_THREADS=4 .venv/bin/python run.py   # fewer cores
.venv/bin/python run.py --quick          # 1/8 of the sims, ~1 min</pre>
  <p class="note">After the chop, the auction boxes switch from "if X is chopped" to the real roster on waivers and spend their sims on that one auction. Timings for this run: {', '.join(f'{k} {v}s' for k, v in r['timings'].items())}.</p>
</div></section>
</div>
</div>
<div id="tip" role="tooltip"></div>
<script>{JS}</script>
"""
    with open(path, "w") as f:
        f.write(page)
    return path


if __name__ == "__main__":
    with open(os.path.join(OUT, "results.json")) as f:
        render(json.load(f))
    print("-> out/cheatsheet.html")
