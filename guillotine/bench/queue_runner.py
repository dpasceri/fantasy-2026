"""Queue for full backtests. Start once in the background:

    .venv/bin/python bench/queue_runner.py

Jobs are JSON files dropped in bench/queue/pending/<key>.job:
    {"key": "gbm_quantile", "cmd": ["bench/candidates/gbm_quantile/run_full.sh"], "timeout_s": 1200}
`cmd` runs with the guillotine folder as its working directory. Jobs start oldest first, up to
bench/queue/max_jobs at once, each through cpuq with up to bench/queue/max_slots cores (both
re-read before every start); cpuq makes a job wait until enough cores are free. Output goes to
bench/logs/<key>.log and a status record to bench/queue/done/<key>.json. A job left in running/
by an earlier runner is adopted: it is recorded once cpuq's last line shows up in its log. The
runner exits once bench/queue/CLOSED exists and nothing is pending or running.
"""
import json
import os
import re
import subprocess
import sys
import time

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH)
Q = os.path.join(BENCH, "queue")
MIN_CORES = 4
CPUQ_DONE = re.compile(r"\[cpuq\] (\d+) core\(s\), ([\d.]+)s, exit (-?\d+)")


def setting(name, default, hi):
    try:
        return max(1, min(hi, int(open(os.path.join(Q, name)).read().strip())))
    except (OSError, ValueError):
        return default


def log_path(key):
    return os.path.join(BENCH, "logs", f"{key}.log")


def cpuq_result(key):
    """(cores, wall_s, exit) from cpuq's last line in the job's log, or None while it runs."""
    try:
        m = CPUQ_DONE.findall(open(log_path(key), errors="replace").read())
    except OSError:
        return None
    return (int(m[-1][0]), float(m[-1][1]), int(m[-1][2])) if m else None


def finish(key, cmd, rc, wall, run, note=None):
    res = cpuq_result(key)
    rec = {"key": key, "cmd": cmd, "cores": res[0] if res else None, "exit": rc, "wall_s": round(wall, 1),
           "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "log": f"bench/logs/{key}.log"}
    if note:
        rec["note"] = note
    json.dump(rec, open(os.path.join(Q, "done", f"{key}.json"), "w"), indent=1)
    os.remove(run)
    print(f"[queue] done  {key}: exit {rc}, {rec['wall_s']}s on {rec['cores']} cores", flush=True)


def main():
    for d in ("pending", "running", "done"):
        os.makedirs(os.path.join(Q, d), exist_ok=True)
    os.makedirs(os.path.join(BENCH, "logs"), exist_ok=True)
    procs = {}  # key -> (Popen, t0, cmd, log file, running path)
    adopted = {f[:-4]: (os.path.join(Q, "running", f), time.time())
               for f in os.listdir(os.path.join(Q, "running")) if f.endswith(".job")}
    print(f"[queue] up, watching {Q}/pending" + (f"; adopted {sorted(adopted)}" if adopted else ""), flush=True)
    while True:
        for key, (p, t0, cmd, log, run) in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                log.close()
                finish(key, cmd, rc, time.time() - t0, run)
                del procs[key]
        for key, (run, t0) in list(adopted.items()):
            res = cpuq_result(key)
            if res:
                try:
                    cmd = json.load(open(run))["cmd"]
                except (OSError, ValueError, KeyError):
                    cmd = None
                finish(key, cmd, res[2], res[1], run, note="adopted from an earlier runner; wall_s is cpuq's")
                del adopted[key]
        jobs = sorted((os.path.getmtime(os.path.join(Q, "pending", f)), f)
                      for f in os.listdir(os.path.join(Q, "pending")) if f.endswith(".job"))
        if not jobs and not procs and not adopted and os.path.exists(os.path.join(Q, "CLOSED")):
            print("[queue] closed and empty, exiting", flush=True)
            return
        if not jobs or len(procs) + len(adopted) >= setting("max_jobs", 1, 4):
            time.sleep(3)
            continue
        _, fn = jobs[0]
        src, run = os.path.join(Q, "pending", fn), os.path.join(Q, "running", fn)
        os.replace(src, run)
        try:
            job = json.load(open(run))
            key = job.get("key") or fn[:-4]
            cmd = job["cmd"] if isinstance(job["cmd"], list) else ["bash", "-c", job["cmd"]]
        except (ValueError, KeyError) as e:
            json.dump({"job": fn, "error": f"bad job file: {e}"}, open(os.path.join(Q, "done", fn[:-4] + ".json"), "w"))
            os.remove(run)
            continue
        k = setting("max_slots", 6, 16)
        timeout = float(job.get("timeout_s", 1200))
        print(f"[queue] start {key} on up to {k} cores (timeout {timeout:.0f}s)", flush=True)
        # Start once MIN_CORES are free rather than waiting for all k: agents' tuning runs hold
        # slots on and off, and waiting for k at once can starve the queue.
        log = open(log_path(key), "w")
        p = subprocess.Popen([sys.executable, os.path.join(BENCH, "cpuq"), "-n", str(k), "--min", str(min(k, MIN_CORES)),
                              "--timeout", str(timeout), "--", *cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        procs[key] = (p, time.time(), cmd, log, run)


if __name__ == "__main__":
    main()
