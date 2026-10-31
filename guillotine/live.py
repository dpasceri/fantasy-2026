"""Live chop sheet: serves out/cheatsheet.html with an update panel that re-pulls everything and re-runs every sim.

    .venv/bin/python live.py [--port 8765]        # then open http://localhost:8765

Update = fetch.py (Sleeper rosters, lineups, projections and live box scores, plus ESPN game clocks), then the
full run.py, the same run as on Tuesdays (no --quick, nothing cached between runs). It goes through bench/cpuq on
the physical cores idle at that moment (8-12, one thread each; 15 cores ran slower than 12). The panel shows each
stage, its progress inside the stage (run.py reports its worker pools with GSIM_PROGRESS=1), an ETA from the last
run's stage times scaled by this run's pace, per-core load and the games in progress. The page reloads itself when
the new sheet is written. "Keep updating" starts the next full run as soon as one finishes, for as long as a game
is in progress or about to kick off.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "bin", "python")
SHEET = os.path.join(HERE, "out", "cheatsheet.html")
RESULTS = os.path.join(HERE, "out", "results.json")
LIVE = os.path.join(HERE, "data", "live.json")
MIN_CORES, MAX_CORES = 8, 12
SOON = 30 * 60  # "Keep updating" also runs this long before a kickoff
TICK = re.compile(r"^\[\s*([\d.]+)s\] (.+)$")
PROG = re.compile(r"^@@progress (\d+) (\d+)$")
LABELS = [("fit models", "Fit score models"), ("season sim", "Season sims"), ("lineup sim", "Lineup sims"),
          ("auction values", "Auction values"), ("dynamic: aggression", "Bid-style sweep"),
          ("dynamic: budget", "Budget curve"), ("dynamic: power", "Power rankings"), ("dynamic: forced-bid", "Bid grids")]

S = {"running": False, "phase": "idle", "log": [], "started": None, "finished": None, "ok": None, "last_secs": None,
     "auto": False, "gen": 0, "runs": 0, "cores": None, "note": "", "steps": [], "cur": 0, "frac": 0.0, "step_t0": None,
     "load": [0.0] * 16, "use": [], "fetch_secs": 4.0}
LOCK = threading.Lock()
PROC = {"pid": None}


def label(raw):
    return next((nice for key, nice in LABELS if raw.startswith(key)), raw)


def plan():
    """Stages of a full update with their expected seconds (the last run's times)."""
    try:
        tm = json.load(open(RESULTS))["timings"]
    except (OSError, ValueError, KeyError):
        tm = {}
    steps = [{"name": "Pull live data", "exp": S["fetch_secs"], "act": None}]
    return steps + [{"name": label(k), "exp": float(v), "act": None} for k, v in tm.items() if k != "total"]


def cpu_times():
    with open("/proc/stat") as f:
        rows = [l.split() for l in f if l.startswith("cpu") and l[3].isdigit()]
    return {int(r[0][3:]): (sum(map(int, r[1:])), int(r[4]) + int(r[5])) for r in rows}


def busy(a, b):
    return {c: 1 - (b[c][1] - a[c][1]) / max(b[c][0] - a[c][0], 1) for c in b}


def idle_cores():
    """Physical cores (CPU n with its SMT sibling n+16) that are at least 75% idle over one second."""
    a = cpu_times()
    time.sleep(1.0)
    u = busy(a, cpu_times())
    n = sum(1 for c in range(16) if max(u.get(c, 0), u.get(c + 16, 0)) < 0.25)
    return max(MIN_CORES, min(MAX_CORES, n))


def pinned_cpus(pid):
    """CPUs the sim is pinned to: cpuq's child process's affinity."""
    try:
        child = open(f"/proc/{pid}/task/{pid}/children").read().split()[0]
        spec = next(l.split(":")[1].strip() for l in open(f"/proc/{child}/status") if l.startswith("Cpus_allowed_list"))
    except (OSError, IndexError, StopIteration):
        return []
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def sampler():
    a = cpu_times()
    while True:
        time.sleep(1.0)
        b = cpu_times()
        u = busy(a, b)
        a = b
        use = pinned_cpus(PROC["pid"]) if PROC["pid"] else []
        with LOCK:
            S["load"] = [round(max(u.get(c, 0), u.get(c + 16, 0)), 2) for c in range(16)]
            if use:
                S["use"] = use


def games():
    try:
        G = json.load(open(LIVE))["games"].values()
    except (OSError, ValueError, KeyError):
        return []
    out = []
    for g in G:
        k = g.get("kickoff") or ""
        t = dt.datetime.fromisoformat(k.replace("Z", "+00:00")).timestamp() if "T" in k else None
        out.append({"label": f"{g['away']} @ {g['home']}", "state": g["state"], "clock": g.get("clock"), "kickoff": t})
    return sorted(out, key=lambda g: (g["kickoff"] or 0, g["label"]))


def games_live():
    """Is a game in progress, or about to kick off? (from the last fetch)"""
    now = time.time()
    return any(g["state"] == "in" or (g["state"] == "pre" and g["kickoff"] and 0 <= g["kickoff"] - now <= SOON) for g in games())


def eta():
    """Seconds left: the current stage from its own rate once it reports progress, the rest from the plan x pace."""
    st, i, now = S["steps"], S["cur"], time.time()
    if not S["running"] or i >= len(st):
        return 0.0, 1.0
    done = [s for s in st[1:i] if s["exp"] >= 2 and s["act"] is not None]  # sim stages only; the fetch is network time
    pace = min(2.0, max(0.5, sum(s["act"] for s in done) / sum(s["exp"] for s in done))) if done else 1.0
    el, f = now - S["step_t0"], S["frac"]
    by_plan = max(st[i]["exp"] * pace - el, 0.0)
    by_rate = el / f * (1 - f) if f >= 0.02 else by_plan
    return f * by_rate + (1 - f) * by_plan + sum(s["exp"] * pace for s in st[i + 1:]), pace


def advance(secs, raw=None):
    i = S["cur"]
    if i >= len(S["steps"]):
        S["steps"].append({"name": label(raw or "?"), "exp": 0.0, "act": None})
    S["steps"][i]["act"] = round(secs, 1)
    S["cur"], S["frac"], S["step_t0"] = i + 1, 0.0, time.time()


def step(phase, cmd, env=None):
    with LOCK:
        S["phase"] = phase
    p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    PROC["pid"] = p.pid
    for line in p.stdout:
        line = line.rstrip()
        m, t = PROG.match(line), TICK.match(line)
        with LOCK:
            if m:
                S["frac"] = int(m[1]) / int(m[2])
            elif t:
                advance(float(t[1]), t[2])
            elif line:
                if "waiting for" in line:
                    S["phase"] = "waiting for free cores"
                S["log"] = (S["log"] + [line])[-12:]
    rc = p.wait()
    PROC["pid"] = None
    return rc


def job():
    while True:
        k = idle_cores()
        t0 = time.time()
        with LOCK:
            S.update(running=True, started=t0, log=[], cores=k, note="", steps=plan(), cur=0, frac=0.0, step_t0=t0, use=[])
        rc = step("pulling live data", [PY, "fetch.py"])
        if rc == 0:
            with LOCK:
                S["fetch_secs"] = round(time.time() - t0, 1)
                advance(S["fetch_secs"])
            env = dict(os.environ, GSIM_PROGRESS="1")
            rc = step("simulating", [PY, os.path.join("bench", "cpuq"), "-n", str(k), "--min", str(MIN_CORES), "--", PY, "run.py"], env)
        with LOCK:
            S.update(finished=time.time(), ok=rc == 0, phase="idle" if rc == 0 else f"failed (exit {rc})")
            if rc == 0:
                S["gen"] += 1
                S["runs"] += 1
                S["last_secs"] = round(time.time() - t0)
            again = S["auto"] and rc == 0 and games_live()
            if S["auto"] and not again:
                S["auto"] = False
                S["note"] = "Keep updating is off: " + ("the last run failed." if rc else "no game in progress or about to start.")
            S["use"] = []
            S["running"] = again  # stays True between back-to-back runs, so a click can't start a second one
        if not again:
            return


def start():
    with LOCK:
        if S["running"]:
            return False
        S["running"] = True
    threading.Thread(target=job, daemon=True).start()
    return True


def status():
    with LOCK:
        left, pace = eta()
        out = {k: S[k] for k in ("running", "phase", "log", "started", "finished", "ok", "last_secs", "auto", "gen", "runs",
                                 "cores", "note", "steps", "cur", "frac", "load", "use")}
        el = time.time() - S["started"] if S["running"] and S["started"] else 0.0
        out.update(now=time.time(), eta=round(left, 1), pace=round(pace, 2), elapsed=round(el, 1),
                   step_el=round(time.time() - S["step_t0"], 1) if S["running"] and S["step_t0"] else 0.0)
    out["games"] = games()
    return out


PANEL = r"""
<div id="lv" class="lv" role="region" aria-label="Live update">
  <div class="lv-head">
    <span class="lv-dot" aria-hidden="true"></span><span id="lv-state">Live update</span>
    <span id="lv-run" class="lv-dim"></span>
    <button id="lv-min" class="lv-icon" type="button" aria-label="Collapse panel" aria-expanded="true">&#8211;</button>
  </div>
  <div class="lv-body">
    <div class="lv-big">
      <div><div id="lv-eta" class="lv-eta">--:--</div><div class="lv-cap" id="lv-eta-cap">left</div></div>
      <div class="lv-right"><div id="lv-done" class="lv-done"></div><div class="lv-cap" id="lv-sub"></div></div>
    </div>
    <div id="lv-bar" class="lv-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></div>
    <ol id="lv-steps" class="lv-steps"></ol>
    <div class="lv-row"><span class="lv-lab">Cores</span><div id="lv-cores" class="lv-cores" aria-hidden="true"></div><span id="lv-cores-txt" class="lv-dim"></span></div>
    <div class="lv-row"><span class="lv-lab">Games</span><div id="lv-games" class="lv-games"></div></div>
    <div id="lv-msg" class="lv-msg" aria-live="polite"></div>
    <div class="lv-ctl">
      <button id="lv-go" class="lv-go" type="button">Update now</button>
      <label class="lv-tog"><input id="lv-auto" type="checkbox"><span>Keep updating</span></label>
    </div>
  </div>
</div>
<style>
.lv { position: fixed; right: 16px; bottom: 16px; z-index: 60; width: 380px; max-width: calc(100vw - 32px); max-height: calc(100vh - 32px);
  overflow: auto; box-sizing: border-box; background: var(--sheet); color: var(--ink); border: 1px solid var(--rule); border-radius: 12px;
  box-shadow: 0 12px 40px rgba(0,0,0,.28); font: 500 13px/1.35 var(--f-body); }
.lv-head { display: flex; align-items: center; gap: 8px; padding: 9px 10px 9px 12px; background: var(--tab); color: var(--tab-ink);
  font: 700 14px/1 var(--f-display); letter-spacing: .1em; text-transform: uppercase; position: sticky; top: 0; }
.lv-head .lv-dim { color: var(--tab-ink); opacity: .7; font-weight: 500; }
.lv-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--good); flex: none; }
.lv.run .lv-dot { background: var(--you); box-shadow: 0 0 0 0 var(--you); animation: lvpulse 1.4s ease-out infinite; }
.lv.fail .lv-dot { background: var(--crit); }
.lv.wait .lv-dot { background: var(--warn); }
@keyframes lvpulse { 0% { box-shadow: 0 0 0 0 rgba(57,135,229,.7); } 100% { box-shadow: 0 0 0 9px rgba(57,135,229,0); } }
.lv-icon { margin-left: auto; background: none; border: 0; color: var(--tab-ink); font: 700 18px/1 var(--f-display); width: 30px; height: 26px;
  border-radius: 6px; cursor: pointer; }
.lv-icon:hover, .lv-icon:focus-visible { background: rgba(255,255,255,.14); }
.lv-body { padding: 12px 12px 12px; display: grid; gap: 10px; }
.lv.min .lv-body { display: none; }
.lv.min { width: auto; }
.lv-big { display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; }
.lv-eta { font: 700 46px/0.9 var(--f-display); letter-spacing: .01em; font-variant-numeric: tabular-nums; }
.lv-done { font: 700 22px/1 var(--f-display); text-align: right; font-variant-numeric: tabular-nums; }
.lv-right { text-align: right; }
.lv-cap { font: 500 11.5px/1.3 var(--f-data); color: var(--muted); margin-top: 4px; }
.lv-bar { display: flex; gap: 2px; height: 12px; }
.lv-seg { position: relative; flex: 1 1 0; min-width: 4px; background: var(--grid); border-radius: 3px; overflow: hidden; }
.lv-seg i { position: absolute; inset: 0 auto 0 0; background: var(--you); border-radius: 3px; transition: width .4s linear; }
.lv-seg.cur i { background-image: repeating-linear-gradient(-45deg, rgba(255,255,255,.28) 0 6px, transparent 6px 12px);
  background-size: 17px 17px; animation: lvstripe .8s linear infinite; }
@keyframes lvstripe { to { background-position: 17px 0; } }
.lv-steps { list-style: none; margin: 0; padding: 0; display: grid; gap: 3px; font: 12.5px/1.3 var(--f-data); }
.lv-steps li { display: grid; grid-template-columns: 16px 1fr auto; gap: 8px; align-items: center; color: var(--muted); }
.lv-steps li.done { color: var(--ink-2); }
.lv-steps li.cur { color: var(--ink); font-weight: 600; }
.lv-steps b { font-weight: 600; color: var(--you); }
.lv-ic { width: 12px; height: 12px; border-radius: 50%; border: 2px solid var(--rule); box-sizing: border-box; }
.done .lv-ic { border-color: var(--good); background: var(--good); }
.cur .lv-ic { border-color: var(--you); border-right-color: transparent; animation: lvspin .8s linear infinite; }
@keyframes lvspin { to { transform: rotate(360deg); } }
.lv-row { display: flex; align-items: center; gap: 10px; min-width: 0; }
.lv-lab { font: 700 11px/1 var(--f-display); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); width: 44px; flex: none; }
.lv-cores { display: grid; grid-template-columns: repeat(16, 1fr); gap: 2px; height: 26px; flex: 1; align-items: end; }
.lv-cores span { display: block; background: var(--grid); border-radius: 2px; height: 100%; position: relative; overflow: hidden; }
.lv-cores span i { position: absolute; left: 0; right: 0; bottom: 0; background: var(--muted); transition: height .6s ease; }
.lv-cores span.use i { background: var(--you); }
.lv-cores span.use { outline: 1px solid var(--you); outline-offset: -1px; }
.lv-dim { color: var(--muted); font: 11.5px/1.2 var(--f-data); white-space: nowrap; }
.lv-games { display: flex; flex-wrap: wrap; gap: 4px; min-width: 0; }
.lv-g { font: 600 11px/1 var(--f-data); padding: 4px 6px; border-radius: 5px; background: var(--grid); color: var(--ink-2); white-space: nowrap; }
.lv-g.in { background: var(--you); color: #fff; }
.lv-g.post { opacity: .6; }
.lv-msg { font: 12px/1.4 var(--f-data); color: var(--ink-2); }
.lv-msg:empty { display: none; }
.lv-ctl { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.lv-go { font: 700 15px/1 var(--f-display); letter-spacing: .1em; text-transform: uppercase; padding: 11px 18px; border-radius: 8px;
  border: 0; background: var(--you); color: #fff; cursor: pointer; min-height: 40px; }
.lv-go:disabled { opacity: .5; cursor: progress; }
.lv-go:focus-visible, .lv-tog input:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
.lv-tog { display: inline-flex; align-items: center; gap: 7px; cursor: pointer; font-weight: 600; }
.lv-tog input { width: 16px; height: 16px; accent-color: var(--you); }
@media (max-width: 520px) { .lv { right: 8px; left: 8px; bottom: 8px; width: auto; max-width: none; } .lv.min { left: auto; } .lv-eta { font-size: 38px; } }
@media (prefers-reduced-motion: reduce) { .lv *, .lv { animation: none !important; transition: none !important; } }
</style>
<script>
(function(){
  var GEN = __GEN__, last = null, got = 0;
  var $ = function(id){ return document.getElementById(id); };
  var box = $('lv'), go = $('lv-go'), auto = $('lv-auto');
  function mmss(s){ s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + ('0' + s % 60).slice(-2); }
  function clock(t){ return new Date(t * 1000).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'}); }
  function esc(s){ return String(s).replace(/[&<>"]/g, function(c){ return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }
  try { if (localStorage.getItem('lv-min') === '1') setMin(true); } catch (e) {}
  try { var y = sessionStorage.getItem('lv-y'); if (y) { sessionStorage.removeItem('lv-y'); window.scrollTo(0, +y); } } catch (e) {}
  function setMin(on){ box.classList.toggle('min', on); $('lv-min').innerHTML = on ? '&#9650;' : '&#8211;';
    $('lv-min').setAttribute('aria-expanded', on ? 'false' : 'true'); $('lv-min').setAttribute('aria-label', on ? 'Expand panel' : 'Collapse panel');
    try { localStorage.setItem('lv-min', on ? '1' : '0'); } catch (e) {} }
  $('lv-min').addEventListener('click', function(){ setMin(!box.classList.contains('min')); });

  function render(){
    var s = last; if (!s) return;
    var drift = (Date.now() - got) / 1000, run = s.running, wait = run && s.phase.indexOf('waiting') === 0;
    box.classList.toggle('run', run && !wait); box.classList.toggle('wait', wait); box.classList.toggle('fail', !run && s.ok === false);
    $('lv-state').textContent = wait ? 'Waiting for cores' : run ? (s.phase === 'pulling live data' ? 'Pulling live data' : 'Simulating') :
      s.ok === false ? 'Update failed' : 'Up to date';
    $('lv-run').textContent = run ? '' : (s.runs ? s.runs + ' run' + (s.runs > 1 ? 's' : '') : '');
    var left = run ? Math.max(0, s.eta - drift) : 0, el = run ? s.elapsed + drift : 0;
    if (run) {
      $('lv-eta').textContent = mmss(left); $('lv-eta-cap').textContent = 'left · ' + mmss(el) + ' elapsed';
      $('lv-done').textContent = 'done ~' + clock(Date.now() / 1000 + left);
      $('lv-sub').textContent = s.cores + ' cores · pace ' + (s.pace > 1.05 ? 'slower' : s.pace < 0.95 ? 'faster' : 'on') + (s.pace >= 0.95 && s.pace <= 1.05 ? ' plan' : ' than last run');
    } else {
      $('lv-eta').textContent = s.finished ? clock(s.finished) : '--:--';
      $('lv-eta-cap').textContent = s.finished ? 'last update' : 'no update yet this session';
      $('lv-done').textContent = s.last_secs ? mmss(s.last_secs) : '~5:30';
      $('lv-sub').textContent = s.last_secs ? 'last full run' : 'a full run takes';
    }
    // segmented bar and stage list
    var tot = 0, i, st = s.steps || [];
    for (i = 0; i < st.length; i++) tot += Math.max(st[i].act || st[i].exp, 1);
    var bar = $('lv-bar'), segs = '', items = '', pct = 0;
    for (i = 0; i < st.length; i++) {
      var w = Math.max(st[i].act || st[i].exp, 1), state = !run ? (st[i].act != null ? 'done' : 'todo') : i < s.cur ? 'done' : i === s.cur ? 'cur' : 'todo';
      var fill = state === 'done' ? 1 : state !== 'cur' ? 0 : s.frac ? s.frac :
                 st[i].exp ? Math.min((s.step_el + drift) / (st[i].exp * s.pace), 0.97) : 0;  // no pool progress yet: by time
      pct += w * fill;
      segs += '<div class="lv-seg ' + state + '" style="flex-grow:' + w.toFixed(1) + '"><i style="width:' + (fill * 100).toFixed(1) + '%"></i></div>';
      var t = state === 'done' ? Math.round(st[i].act) + 's' : state === 'cur' ? Math.round(s.step_el + drift) + 's / ~' + Math.round(st[i].exp * s.pace) + 's' : st[i].exp ? '~' + Math.round(st[i].exp * (run ? s.pace : 1)) + 's' : '';
      items += '<li class="' + state + '"><span class="lv-ic"></span><span>' + esc(st[i].name) + (state === 'cur' && s.frac ? ' <b>' + Math.round(s.frac * 100) + '%</b>' : '') + '</span><span>' + t + '</span></li>';
    }
    bar.innerHTML = segs; $('lv-steps').innerHTML = items; $('lv-steps').style.display = st.length ? '' : 'none';
    var p = tot ? Math.round(100 * pct / tot) : 0; bar.setAttribute('aria-valuenow', run ? p : 100); bar.style.display = st.length ? '' : 'none';
    if (run) document.title = mmss(left) + ' · ' + p + '% · Chop Sheet';
  }
  function cores(s){
    var use = {}, n = 0, sum = 0;
    (s.use || []).forEach(function(c){ use[c % 16] = 1; });
    var html = '';
    for (var c = 0; c < 16; c++) { var l = s.load[c] || 0; if (use[c]) { n++; sum += l; }
      html += '<span class="' + (use[c] ? 'use' : '') + '" title="core ' + c + ': ' + Math.round(l * 100) + '%"><i style="height:' + Math.round(l * 100) + '%"></i></span>'; }
    $('lv-cores').innerHTML = html;
    $('lv-cores-txt').textContent = n ? n + ' in use · ' + Math.round(100 * sum / n) + '%' : Math.round(100 * s.load.reduce(function(a, b){ return a + b; }, 0) / 16) + '% busy';
  }
  function gamesHtml(s){
    var now = Date.now() / 1000, g = s.games || [], inp = g.filter(function(x){ return x.state === 'in'; }),
        fin = g.filter(function(x){ return x.state === 'post'; }).length, next = g.filter(function(x){ return x.state === 'pre'; });
    var h = inp.map(function(x){ return '<span class="lv-g in">' + esc(x.label) + ' ' + esc(x.clock || '') + '</span>'; }).join('');
    if (next.length) { var k = next[0].kickoff; h += '<span class="lv-g">next ' + esc(next[0].label) + (k ? ' ' + new Date(k * 1000).toLocaleString([], {weekday: 'short', hour: 'numeric', minute: '2-digit'}) : '') + '</span>'; }
    h += '<span class="lv-g post">' + fin + '/' + g.length + ' final</span>';
    $('lv-games').innerHTML = g.length ? h : '<span class="lv-dim">no live data yet</span>';
  }
  function show(s){
    if (s.gen > GEN) { try { sessionStorage.setItem('lv-y', String(window.scrollY)); } catch (e) {} location.reload(); return; }
    last = s; got = Date.now();
    go.disabled = s.running; go.textContent = s.running ? 'Updating…' : 'Update now'; auto.checked = s.auto;
    var m = s.note || '';
    if (!s.running && s.ok === false) m = 'Last update ' + s.phase + ': ' + (s.log.slice(-2).join(' ') || 'no output');
    if (s.auto && s.running) m = 'Keep updating is on: the next full run starts as soon as this one finishes.';
    $('lv-msg').textContent = m;
    cores(s); gamesHtml(s); render();
    clearTimeout(show.t); show.t = setTimeout(poll, s.running ? 1000 : 4000);
  }
  function post(u){ fetch(u, {method: 'POST'}).then(function(r){ return r.json(); }).then(show).catch(function(){}); }
  function poll(){ fetch('/status').then(function(r){ return r.json(); }).then(show).catch(function(){
    $('lv-msg').textContent = 'live.py is not running'; clearTimeout(show.t); show.t = setTimeout(poll, 5000); }); }
  go.addEventListener('click', function(){ post('/update'); });
  auto.addEventListener('change', function(){ post('/auto?on=' + (auto.checked ? 1 : 0)); });
  setInterval(render, 250);
  poll();
})();
</script>
"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/status"):
            return self._send(200, json.dumps(status()), "application/json")
        if self.path not in ("/", "/index.html"):
            return self._send(404, "not found", "text/plain")
        try:
            sheet = open(SHEET, encoding="utf-8").read()
        except OSError:
            sheet = "<p>No sheet yet. Press Update now.</p>"
        with LOCK:
            gen = S["gen"]
        head = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        self._send(200, head + sheet + PANEL.replace("__GEN__", str(gen)), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path.startswith("/update"):
            start()
        elif self.path.startswith("/auto"):
            on = self.path.endswith("on=1")
            with LOCK:
                S["auto"], S["note"] = on, ""
            if on:
                start()
        else:
            return self._send(404, "not found", "text/plain")
        self._send(200, json.dumps(status()), "application/json")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    with LOCK:
        S["steps"] = [st | {"act": st["exp"]} for st in plan()]  # the last run's stage times, until the first update
    threading.Thread(target=sampler, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"live chop sheet on http://localhost:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
