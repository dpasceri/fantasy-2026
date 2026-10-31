"""Simulation optimization of my FAAB bid policy on TUNING seeds (>= 1000 only, through policy.tune()).

    bench/cpuq -n 2 -- .venv/bin/python -m bench.candidates.sim_optimization.tune [--stage all|bank|ocba|...|freeze]

Every evaluation goes through bench.harness.tracks.policy.tune(), which refuses seeds < 1000, so no
evaluation path (seeds 100-399) is ever simulated here. Work unit = one policy on one CRN seed block
(200 paths) in all three conditions; the objective is the mean P(win) over the three conditions,
always as a paired difference against the reference (= the baseline, 1.0/0.75) on the same block.

Stages (each cached in tune_out/, so reruns resume):
  bank   run.py's 16-point (m_early, m_late) grid on 40 blocks (seeds 1000-1039), main condition only:
         the data for the R&S efficiency study (pcs)
  ocba   variant "ocba": sequential OCBA under CRN over a 42-point step-multiplier grid refined around the
         bank's optimum (m_early 0.5-1.0, m_late 0.5-1.25; seeds 2000+)
  cma_global  CMA-ES (pycma) over the 6-D rich family from the incumbent, sigma0 0.2, popsize 8, 16 generations,
         2 blocks per evaluation, CRN within each generation (seeds 3000+), then a GP-smoothed shortlist
         and an OCBA final selection (seeds 3500+). Kept as a record: under the measured noise it wandered
         to worse regions (final estimate -0.004 +- 0.006), so it is not a variant.
  cma    variant "cma": the noise-adapted CMA-ES, a local search from the same incumbent with sigma0 0.08,
         popsize 6, 12 generations and 4 blocks per evaluation (seeds 5000+), same shortlist + OCBA final
         (seeds 5500+)
  bo     variant "bo": batch Bayesian optimization, noisy GP (Matern 5/2 ARD + white noise) with batch
         Thompson sampling, CRN within each batch (seeds 4000+), same shortlist + OCBA final (seeds 4500+)
  pcs    bootstrap-world study on the bank: P(correct selection), P(good selection) and expected
         opportunity cost vs simulation budget for run.py's uniform sweep, OCBA-CRN and KN
  freeze write tuned.json (the frozen parameters the policy track evaluates) from the stage files
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time
import warnings

from bench import harness  # noqa: F401
from bench.harness import slots
from bench.harness.tracks import policy as T
from bench.candidates.sim_optimization.policy import SimOpt, TUNED

import numpy as np

warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tune_out")
CONDS = ("", "_stress", "_bmboot")
REF = {"m_early": 1.0, "m_late": 0.75}                 # the baseline: anchor of every paired difference
CORE_S = 0.72                                          # measured core-s per policy x block x condition
GRID16 = [(a, b) for a in (0.5, 0.75, 1.0, 1.5) for b in (0.75, 1.0, 1.5, 2.0)]   # = run.py's sweep
GRID42 = [(a, b) for a in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0) for b in (0.5, 0.6, 0.75, 0.9, 1.0, 1.25)]
SEEDS = {"bank": 1000, "ocba": 2000, "cma_global": 3000, "cma_global_final": 3500, "bo": 4000, "bo_final": 4500,
         "cma": 5000, "cma_final": 5500}
N_BANK = 40
GENS, POP, BLOCKS = 16, 8, 2                           # BO batches, batch size, blocks per evaluation
CMA_CFG = {"cma_global": {"sigma0": 0.2, "pop": 8, "gens": 16, "blocks": 2},
           "cma": {"sigma0": 0.08, "pop": 6, "gens": 12, "blocks": 4}}
FINAL_BUDGET, FINAL_N0 = 72, 4                         # final OCBA: policy-blocks incl. the reference


# ---------------------------------------------------------------- rich family: [0,1]^6 <-> theta
NAMES = ("lm0", "lm1", "gam", "gmin", "kap", "alpha")
LO = np.array([math.log(0.25), -1.2, -1.0, 1.0, -1.0, -1.0])
HI = np.array([math.log(2.0), 1.2, 1.5, 5.0, 1.0, 5.0])


def decode(x):
    v = LO + np.clip(np.asarray(x, float), 0, 1) * (HI - LO)
    th = {k: round(float(z), 4) for k, z in zip(NAMES, v)}
    th["alpha"] = round(1 / (1 + math.exp(-th["alpha"])), 4)
    return th


def encode(th):
    v = [th["lm0"], th["lm1"], th["gam"], th["gmin"], th["kap"], math.log(th["alpha"] / (1 - th["alpha"]))]
    return np.clip((np.array(v) - LO) / (HI - LO), 0, 1)


X0 = encode({"lm0": math.log(0.75), "lm1": 0.0, "gam": 0.0, "gmin": 2.0, "kap": 0.0, "alpha": 0.88})


def step(a, b):
    return {"m_early": a, "m_late": b}


def key(th):
    return json.dumps(th, sort_keys=True)


# ---------------------------------------------------------------- parallel evaluator (through tune())
def _init():
    os.environ["CPUQ_SLOTS"] = "1"   # tune() inside a worker runs serially on that worker's core


def _work(task):
    seed, thetas, conds = task
    r = T.tune([SimOpt(variant="tune", theta=th) for th in thetas], seeds=[seed], conditions=conds, jobs=1)
    return seed, [{"w": [float(np.sum(x["arrays"]["win" + c])) for c in conds],
                   "wk": [float(np.sum(x["arrays"]["weeks" + c])) for c in conds],
                   "sp": [float(np.sum(x["arrays"]["spend" + c])) for c in conds]} for x in r]


class Sim:
    """Cached, parallel block evaluations: get([(theta, seed)], conds) -> {(key, seed): record}."""

    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.path = os.path.join(OUT, "cache.jsonl")
        self.sig = T.env()["sig"]
        self.db = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:   # a line cut short by an interrupted run: recomputed
                    continue
                if r.get("sig") == self.sig:   # blocks from another data snapshot are never reused
                    self.db[r["k"], r["s"], r["c"]] = r
        T.env()
        self.n = slots()
        self.pool = mp.get_context("fork").Pool(self.n, initializer=_init) if self.n > 1 else None
        self.blocks = 0       # policy-blocks simulated in this process (x conditions = runs)
        self.runs = 0

    def get(self, pairs, conds=CONDS):
        c = "".join(conds) or "main"
        todo = {}
        for th, s in pairs:
            assert s >= T.TUNE_MIN_SEED
            if (key(th), s, c) not in self.db:
                todo.setdefault(s, {})[key(th)] = th
        tasks = []
        for s, ths in sorted(todo.items()):
            ths = list(ths.values())
            per = max(1, math.ceil(len(ths) / max(1, math.ceil(len(ths) / 3))))  # ~3 policies per task
            tasks += [(s, ths[i:i + per], conds) for i in range(0, len(ths), per)]
        if tasks:
            res = self.pool.imap(_work, tasks) if self.pool else map(_work, tasks)  # ordered: zips with tasks
            with open(self.path, "a") as f:
                for (s, recs), (_, ths, _) in zip(res, tasks):
                    for th, r in zip(ths, recs):
                        rec = {"k": key(th), "s": s, "c": c, "sig": self.sig} | r
                        self.db[rec["k"], s, c] = rec
                        f.write(json.dumps(rec) + "\n")
                        self.blocks += 1
                        self.runs += len(conds)
        return {(key(th), s): self.db[key(th), s, c] for th, s in pairs}

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


def pw(rec):
    """Per-block P(win) per condition and the objective (their mean)."""
    w = np.asarray(rec["w"]) / T.C
    return w, float(w.mean())


# ---------------------------------------------------------------- OCBA under common random numbers
def ocba_targets(means, s2b, n, add, dmin=0.002):
    """CRN-OCBA allocation of `add` more blocks. means: current estimates; s2b[i] = per-block variance of
    X_b - X_i on common blocks (b = argmax). With nested CRN blocks, Var(dhat_b - dhat_i) ~ s2b[i] / n_i,
    so the asymptotic large-deviation allocation is n_i ~ s2b_i / delta_i^2 for i != b, and the best only
    needs to be paired with everyone: n_b = max_i n_i (Chen et al. 2000 ratio with Fu et al. 2007's
    correlated-sampling view). Returns integer increments summing to `add`."""
    k = len(means)
    b = int(np.argmax(means))
    d = np.maximum(means[b] - np.asarray(means), dmin)
    w = np.where(np.arange(k) == b, 0.0, np.maximum(s2b, 1e-5) / d ** 2)
    w[b] = w.max()
    tot = np.sum(n) + add
    target = tot * w / w.sum()
    deficit = np.maximum(target - np.asarray(n), 0)
    if deficit.sum() <= 0:
        deficit = w.copy()
    inc = np.floor(add * deficit / deficit.sum()).astype(int)
    for i in np.argsort(-(add * deficit / deficit.sum() - inc))[: add - inc.sum()]:
        inc[i] += 1
    return inc


def ocba_real(sim, thetas, seed0, budget, n0, add, tag):
    """Sequential OCBA over policies (anchor = REF on every block used). Budget counts policy-blocks incl. REF."""
    k = len(thetas)
    n, nref = np.zeros(k, int), 0
    Y = [[] for _ in range(k)]              # per-block objective difference vs REF
    rounds = []

    def extend(target):
        nonlocal nref
        top = max(int(target.max()), nref)
        pairs = [(thetas[i], seed0 + j) for i in range(k) for j in range(n[i], target[i])]
        pairs += [(REF, seed0 + j) for j in range(top)]           # cached ones come back without a rerun
        got = sim.get(pairs)
        for i in range(k):
            for j in range(n[i], target[i]):
                Y[i].append(pw(got[key(thetas[i]), seed0 + j])[1] - pw(got[key(REF), seed0 + j])[1])
        n[:] = target
        nref = top

    extend(np.full(k, n0))
    t0 = time.time()
    while n.sum() + nref < budget:
        means = np.array([np.mean(y) for y in Y])
        b = int(np.argmax(means))
        s2b = np.array([np.var(np.subtract(Y[b][:min(n[b], n[i])], Y[i][:min(n[b], n[i])]), ddof=1)
                        if i != b and min(n[b], n[i]) > 1 else 0.0 for i in range(k)])
        a = int(min(add, budget - n.sum() - nref))
        if a <= 0:
            break
        inc = ocba_targets(means, s2b, n, a)
        extend(n + inc)
        rounds.append({"best": b, "n": n.tolist(), "nref": nref})
        print(f"[{tag}] round {len(rounds)}: best {b} d {means[b]:+.4f}, blocks {int(n.sum()) + nref}/{budget} "
              f"({time.time() - t0:.0f}s)", flush=True)
    means = np.array([np.mean(y) for y in Y])
    se = np.array([np.std(y, ddof=1) / math.sqrt(len(y)) for y in Y])
    b = int(np.argmax(means))
    return {"best": b, "theta": thetas[b], "d": means.round(5).tolist(), "se": se.round(5).tolist(),
            "n": n.tolist(), "nref": int(nref), "blocks": int(n.sum() + nref), "rounds": rounds,
            "seeds": [seed0, seed0 + int(max(n.max(), nref)) - 1]}


# ---------------------------------------------------------------- GP surrogate (noisy) for BO and shortlists
def gp_fit(X, y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(np.full(X.shape[1], 0.5), (0.05, 20.0), nu=2.5) \
        + WhiteKernel(0.5, (1e-3, 2.0))
    gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=4, random_state=0)
    return gp.fit(X, y)


def shortlist(X, y, rec, k=3, sep=0.05):
    """Recommendation + the top-k evaluated points by GP posterior mean (de-duplicated), + two ablations of
    the recommendation (no competition cap; no strength term)."""
    gp = gp_fit(X, y)
    mu = gp.predict(X)
    pts = [np.asarray(rec)]
    for i in np.argsort(-mu):
        if len(pts) > k:
            break
        if min(np.max(np.abs(X[i] - p)) for p in pts) > sep:
            pts.append(X[i])
    ths = [decode(p) for p in pts]
    r = dict(ths[0])
    ths.append({kk: v for kk, v in r.items() if kk != "alpha"})        # ablation: no competition cap
    ths.append(r | {"kap": 0.0})                                       # ablation: no strength term
    labels = ["rec"] + [f"gp_top{j}" for j in range(1, len(pts))] + ["rec_no_cap", "rec_no_strength"]
    return ths, labels, gp, mu


def eval_batch(sim, xs, seeds):
    """Paired objective (vs REF, mean over the batch's blocks and conditions) for points of the rich family."""
    ths = [decode(x) for x in xs]
    got = sim.get([(th, s) for th in ths + [REF] for s in seeds])
    ref = np.array([pw(got[key(REF), s])[1] for s in seeds])
    ys = [float(np.mean([pw(got[key(th), s])[1] for s in seeds] - ref)) for th in ths]
    return ths, ys


# ---------------------------------------------------------------- stages
def stage_bank(sim):
    pairs = [(step(a, b), SEEDS["bank"] + j) for a, b in GRID16 for j in range(N_BANK)]
    got = sim.get(pairs, conds=("",))
    X = np.array([[got[key(step(a, b)), SEEDS["bank"] + j]["w"][0] / T.C for j in range(N_BANK)] for a, b in GRID16])
    np.savez(os.path.join(OUT, "bank.npz"), X=X, grid=np.array(GRID16), seeds=np.arange(N_BANK) + SEEDS["bank"])
    return {"grid": GRID16, "means": X.mean(1).round(5).tolist(), "n_blocks": N_BANK, "conds": ["main"],
            "seeds": [SEEDS["bank"], SEEDS["bank"] + N_BANK - 1], "blocks": int(X.size)}


OCBA_BUDGET, OCBA_N0, OCBA_ADD = 520, 4, 16
PCS_BUDGETS, PCS_R = [64, 128, 256, 400, 800], 1000


def stage_ocba(sim):
    ths = [step(a, b) for a, b in GRID42]
    r = ocba_real(sim, ths, SEEDS["ocba"], budget=OCBA_BUDGET, n0=OCBA_N0, add=OCBA_ADD, tag="ocba")
    r["grid"] = GRID42
    # diagnostic only (the variant is OCBA's own pick): a GP over the grid, each point's noise = its se^2
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    Z = np.log(np.array(GRID42))
    se2 = np.maximum(np.array(r["se"]) ** 2, 1e-6)
    gp = GaussianProcessRegressor(ConstantKernel(1e-4, (1e-6, 1e-1)) * Matern([0.3, 0.3], (0.05, 5.0), nu=2.5),
                                  alpha=se2, random_state=0).fit(Z, np.array(r["d"]))
    mu = gp.predict(Z)
    r["gp_smoothed"] = {"best": list(GRID42[int(np.argmax(mu))]), "d": mu.round(5).tolist(), "kernel": str(gp.kernel_)}
    return r


def stage_cma(sim, tag="cma"):
    import cma
    cf = CMA_CFG[tag]
    es = cma.CMAEvolutionStrategy(X0.tolist(), cf["sigma0"], {"popsize": cf["pop"], "bounds": [0, 1],
                                                              "seed": 20261029, "verbose": -9})
    X, y, gens = [], [], []
    for g in range(cf["gens"]):
        xs = es.ask()
        seeds = [SEEDS[tag] + cf["blocks"] * g + j for j in range(cf["blocks"])]
        _, ys = eval_batch(sim, xs, seeds)
        es.tell(xs, [-v for v in ys])
        X += [np.asarray(x) for x in xs]
        y += ys
        gens.append({"gen": g, "best": max(ys), "mean": float(np.mean(ys)), "sigma": float(es.sigma),
                     "xmean": np.round(es.mean, 4).tolist()})
        print(f"[{tag}] gen {g}: batch best {max(ys):+.4f} mean {np.mean(ys):+.4f} sigma {es.sigma:.3f} "
              f"mean-theta {decode(es.mean)}", flush=True)
    r = finish(sim, tag, np.array(X), np.array(y), es.result.xfavorite, gens, cf["pop"], cf["blocks"])
    r["config"] = cf | {"x0": decode(X0)}
    return r


def stage_bo(sim):
    from scipy.stats import qmc
    rng = np.random.default_rng(20261029)
    sob = qmc.Sobol(6, scramble=True, seed=7).random(2 * POP - 1)
    init = [X0] + list(sob)
    X, y, gens = [], [], []
    for g in range(GENS):
        seeds = [SEEDS["bo"] + BLOCKS * g + j for j in range(BLOCKS)]
        if g < 2:
            xs = init[g * POP:(g + 1) * POP]
        else:
            gp = gp_fit(np.array(X), np.array(y))
            mu = gp.predict(np.array(X))
            top = np.array(X)[np.argsort(-mu)[:4]]
            cand = np.vstack([qmc.Sobol(6, scramble=True, seed=100 + g).random(1024)]
                             + [np.clip(t + 0.05 * rng.standard_normal((64, 6)), 0, 1) for t in top])
            S = gp.sample_y(cand, POP, random_state=int(rng.integers(2 ** 31)))   # batch Thompson sampling
            xs, used = [], set()
            for j in range(POP):
                for i in np.argsort(-S[:, j]):
                    if i not in used and all(np.max(np.abs(cand[i] - x)) > 0.01 for x in xs):
                        used.add(i)
                        xs.append(cand[i])
                        break
        _, ys = eval_batch(sim, xs, seeds)
        X += [np.asarray(x) for x in xs]
        y += ys
        gens.append({"gen": g, "best": max(ys), "mean": float(np.mean(ys))})
        print(f"[bo] batch {g}: best {max(ys):+.4f} mean {np.mean(ys):+.4f}", flush=True)
    gp = gp_fit(np.array(X), np.array(y))
    cand = np.vstack([qmc.Sobol(6, scramble=True, seed=999).random(4096), np.array(X)])
    rec = cand[int(np.argmax(gp.predict(cand)))]                     # argmax of the posterior mean
    r = finish(sim, "bo", np.array(X), np.array(y), rec, gens, POP, BLOCKS)
    r["config"] = {"pop": POP, "gens": GENS, "blocks": BLOCKS, "init": "x0 + 15 scrambled Sobol"}
    return r


def finish(sim, tag, X, y, rec, gens, pop, blocks):
    ths, labels, gp, mu = shortlist(X, y, rec)
    r = ocba_real(sim, ths, SEEDS[f"{tag}_final"], budget=FINAL_BUDGET, n0=FINAL_N0, add=8, tag=f"{tag}-final")
    r["labels"], r["thetas"] = labels, ths
    r["label"] = labels[r["best"]]
    np.savez(os.path.join(OUT, f"{tag}_trace.npz"), X=X, y=y, mu=mu)
    curve = []                                   # best-so-far GP posterior mean after each generation (anytime)
    for g in range(2, len(gens) + 1, 2):
        m = gp_fit(X[: g * pop], y[: g * pop]).predict(X[: g * pop])
        curve.append({"evals": g * pop, "blocks": g * (pop + 1) * blocks, "gp_best_mean": round(float(m.max()), 5),
                      "theta": decode(X[int(np.argmax(m))])})
    return {"search": {"evals": len(y), "blocks": len(gens) * (pop + 1) * blocks, "gens": gens,
                       "rec": decode(rec), "y_best": round(float(np.max(y)), 5), "curve": curve,
                       "gp_kernel": str(gp.kernel_)}, "final": r, "theta": r["theta"]}


# ---------------------------------------------------------------- R&S efficiency study (bootstrap world)
def _uniform(X, T_, rng, R):
    k, B = X.shape
    n = max(1, T_ // k)
    idx = rng.integers(0, B, (R, n))
    return np.argmax(X[:, idx].mean(2), 0), np.full(R, n * k)


def _ocba_bank(X, T_, rng, R, n0=2, add=None):
    k, B = X.shape
    add = add or k
    picks, used = np.zeros(R, int), np.zeros(R, int)
    for r in range(R):
        order = rng.integers(0, B, max(T_, n0) + 1)
        n = np.full(k, n0)
        while True:
            a = int(np.argmax(n))
            ref = X[a, order[:n[a]]].mean()
            means = np.array([np.mean(X[i, order[:n[i]]] - X[a, order[:n[i]]]) for i in range(k)]) + ref
            if n.sum() + 1 > T_:
                break
            b = int(np.argmax(means))
            s2b = np.array([np.var(X[b, order[:min(n[b], n[i])]] - X[i, order[:min(n[b], n[i])]], ddof=1)
                            if i != b else 0.0 for i in range(k)])
            inc = ocba_targets(means, s2b, n, int(min(add, T_ - n.sum())))
            if inc.sum() == 0:
                break
            n = n + inc
        picks[r], used[r] = int(np.argmax(means)), n.sum()
    return picks, used


def _kn(X, T_, rng, R, delta, alpha=0.05, n0=5):
    """Kim & Nelson (2001) fully sequential procedure with CRN, truncated at the budget."""
    k, B = X.shape
    eta = 0.5 * ((2 * alpha / (k - 1)) ** (-2 / (n0 - 1)) - 1)
    h2 = 2 * eta * (n0 - 1)
    picks, used = np.zeros(R, int), np.zeros(R, int)
    for r in range(R):
        order = rng.integers(0, B, T_ + n0 * k)
        obs = X[:, order[:n0]]
        S2 = np.array([[np.var(obs[i] - obs[l], ddof=1) for l in range(k)] for i in range(k)])
        alive, rr, spent = np.arange(k), n0, n0 * k
        sums = obs.sum(1)
        while len(alive) > 1:
            m = sums[alive] / rr
            W = np.maximum(0, delta / (2 * rr) * (h2 * S2[np.ix_(alive, alive)] / delta ** 2 - rr))
            ok = m[:, None] >= m[None, :] - W
            np.fill_diagonal(ok, True)
            alive = alive[ok.all(1)]
            if len(alive) == 1 or spent + len(alive) > T_:
                break
            sums[alive] += X[alive, order[rr]]
            rr += 1
            spent += len(alive)
        picks[r] = alive[int(np.argmax(sums[alive]))]
        used[r] = spent
    return picks, used


def stage_pcs(R=None):
    """P(correct / good selection) and expected opportunity cost vs budget, in the bootstrap world of the bank
    (truth = the bank means; procedures resample whole CRN blocks with replacement, all alternatives on the
    same block sequence). Budget in policy-blocks (main condition, 200 paths each)."""
    R = R or PCS_R
    d = np.load(os.path.join(OUT, "bank.npz"))
    X = d["X"]
    mu = X.mean(1)
    best = int(np.argmax(mu))
    gap = mu[best] - mu
    rng = np.random.default_rng(5)
    budgets = PCS_BUDGETS
    rows = []
    for T_ in budgets:
        for name, fn in (("uniform (run.py)", lambda: _uniform(X, T_, rng, R)),
                         ("ocba_crn", lambda: _ocba_bank(X, T_, rng, R // 2)),
                         ("kn_d0.005", lambda: _kn(X, T_, rng, R // 2, 0.005)),
                         ("kn_d0.01", lambda: _kn(X, T_, rng, R // 2, 0.01))):
            if name.startswith("kn") and T_ < 5 * len(mu):
                continue                                   # KN's first stage alone (n0 = 5) exceeds the budget
            p, u = fn()
            rows.append({"budget_blocks": T_, "paths_per_alt_uniform": T_ * T.C // len(mu), "core_s": round(T_ * CORE_S),
                         "procedure": name, "pcs": round(float(np.mean(p == best)), 4),
                         "pgs_0.005": round(float(np.mean(gap[p] <= 0.005)), 4),
                         "eoc": round(float(np.mean(gap[p])), 5), "used_blocks": round(float(u.mean()), 1)})
            print(f"[pcs] {T_:4d} blocks {name:18s} PCS {rows[-1]['pcs']:.3f} PGS {rows[-1]['pgs_0.005']:.3f} "
                  f"EOC {rows[-1]['eoc']:.4f} used {rows[-1]['used_blocks']}", flush=True)
    return {"truth": {"grid": [list(g) for g in d["grid"]], "means": mu.round(5).tolist(), "best": d["grid"][best].tolist(),
                      "gaps_sorted": np.sort(gap).round(5).tolist()[:6]}, "R": R, "rows": rows,
            "note": f"bootstrap world of a {X.shape[1]}-block bank ({X.shape[1] * T.C} paths per alternative, main "
                    "condition): truth = the bank means"}


def freeze():
    sig = T.env()["sig"]
    st = {s: json.load(open(os.path.join(OUT, f"{s}.json"))) for s in STAGES + ("pcs",)
          if os.path.exists(os.path.join(OUT, f"{s}.json"))}
    stale = [s for s, v in st.items() if v.get("state_sig") != sig]
    assert not stale, f"stage files {stale} come from another data snapshot than {sig}: rerun tune.py"
    var = {}
    if "ocba" in st:
        o = st["ocba"]
        var["ocba"] = {"theta": o["theta"], "family": "step", "tune_d_vs_baseline": o["d"][o["best"]],
                       "tune_se": o["se"][o["best"]], "tune_blocks": o["n"][o["best"]], "seeds": o["seeds"]}
    for s in ("cma", "bo", "cma_global"):
        if s in st:
            f = st[s]["final"]
            var[s] = {"theta": st[s]["theta"], "family": "rich", "pick": f["label"],
                      "tune_d_vs_baseline": f["d"][f["best"]], "tune_se": f["se"][f["best"]],
                      "tune_blocks": f["n"][f["best"]], "seeds": f["seeds"]}
    blocks = {s: v.get("blocks", v.get("search", {}).get("blocks", 0) + v.get("final", {}).get("blocks", 0))
              for s, v in st.items() if s != "pcs"}
    searches = {"cma_global": var.pop("cma_global")} if "cma_global" in var else {}
    out = {"state_sig": T.env()["sig"], "ref": REF, "conditions": list(CONDS), "variants": var, "searches": searches,
           "compute": {"policy_blocks": blocks, "conditions_per_block": {s: 1 if s == "bank" else 3 for s in blocks},
                       "wall_s": {s: v.get("wall_s") for s, v in st.items() if s != "pcs"},
                       "slots": {s: v.get("slots") for s, v in st.items() if s != "pcs"},
                       "note": "wall_s/slots are this cache's last build of each stage (blocks already cached cost "
                               "nothing then); a block in all 3 conditions costs ~2.5 core-s"},
           "pcs": st.get("pcs")}
    with open(TUNED, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[tune] wrote {TUNED}: {json.dumps(var)}", flush=True)


STAGES = ("bank", "ocba", "cma_global", "bo", "cma")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all",) + STAGES + ("pcs", "freeze"))
    ap.add_argument("--quick", metavar="DIR", help="plumbing test: tiny budgets, outputs in DIR, tuned.json untouched")
    a = ap.parse_args()
    if a.quick:
        g = globals()
        g.update(CMA_CFG={k: {"sigma0": 0.1, "pop": 3, "gens": 2, "blocks": 1} for k in CMA_CFG})
        g.update(OUT=a.quick, GENS=3, POP=3, BLOCKS=1, N_BANK=3, FINAL_BUDGET=10, FINAL_N0=1, OCBA_BUDGET=8,
                 OCBA_N0=1, OCBA_ADD=2, GRID42=GRID42[:4], GRID16=GRID16[:4], PCS_BUDGETS=[8, 16], PCS_R=20,
                 TUNED=os.path.join(a.quick, "tuned.json"))
    stages = STAGES + ("pcs",) if a.stage == "all" else (a.stage,)
    sim = None
    for s in stages:
        if s in ("pcs", "freeze"):
            continue
        p = os.path.join(OUT, f"{s}.json")
        if os.path.exists(p) and json.load(open(p)).get("state_sig") == T.env()["sig"]:
            print(f"[tune] {s}: cached ({p})", flush=True)
            continue
        sim = sim or Sim()
        t0, b0 = time.time(), sim.blocks
        r = {"bank": stage_bank, "ocba": stage_ocba, "cma_global": lambda m: stage_cma(m, "cma_global"),
             "cma": stage_cma, "bo": stage_bo}[s](sim)
        r["wall_s"], r["new_blocks"], r["slots"] = round(time.time() - t0, 1), sim.blocks - b0, sim.n
        r["state_sig"] = sim.sig
        with open(p, "w") as f:
            json.dump(r, f, indent=1)
        print(f"[tune] {s}: done in {time.time() - t0:.0f}s on {sim.n} core(s)", flush=True)
    if sim:
        sim.close()
    if "pcs" in stages and os.path.exists(os.path.join(OUT, "bank.npz")):
        t0 = time.time()
        r = stage_pcs()
        r["wall_s"], r["state_sig"] = round(time.time() - t0, 1), T.env()["sig"]
        with open(os.path.join(OUT, "pcs.json"), "w") as f:
            json.dump(r, f, indent=1)
    freeze()


if __name__ == "__main__":
    main()
