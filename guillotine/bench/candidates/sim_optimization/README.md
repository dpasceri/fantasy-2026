# sim_optimization (policy track)

**What.** Simulation optimization of my FAAB bid policy inside the season-with-auctions sim. Each
variant is the policy that one search procedure picked on **tuning seeds only** (>= 1000, through
`bench.harness.tracks.policy.tune()`), frozen in `tuned.json`, then scored by the policy track on
the held-out evaluation seeds (smoke 100-104, full 100-399) like any other candidate.

| variant | family | search procedure (tuning seeds) | frozen parameters |
|---|---|---|---|
| `ocba` | step multiplier (the baseline's own family): `round(m x typical(g))`, `m = m_early` before week 12 and `m_late` after, no bid when g < 2 | sequential OCBA under common random numbers over a 42-point grid (m_early in {0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0} x m_late in {0.5, 0.6, 0.75, 0.9, 1.0, 1.25}); n0 = 4 blocks each, then rounds of 16; 521 policy-blocks in all (seeds 2000-2050) | m_early 0.9, m_late 0.75 |
| `cma` | rich 6-D family (below) | noise-adapted local CMA-ES (pycma) from the incumbent: sigma0 0.08, popsize 6, 12 generations, 4 blocks per evaluation, CRN within each generation (seeds 5000-5047). Then a GP shortlist and an OCBA final selection on fresh seeds (5500-5519). 412 policy-blocks | m = 0.905 at week 8, sliding to 0.743 by week 15; near-zero tier tilt (gam 0.04: x0.96 at g = 2.3 to x1.05 at g = 20); gmin 2.29; near-zero strength term (kap 0.02: at most x1.024 either way, but it is on, so z is computed); cap alpha 0.90 |
| `bo` | the same rich family | global batch Bayesian optimization over the whole box: noisy GP (Matern 5/2 ARD + white noise), batch Thompson sampling, 16 batches of 8 with 2 blocks per evaluation, CRN within each batch (seeds 4000-4031). Same shortlist + OCBA final (seeds 4500-4518). 363 policy-blocks | m = 1.97 sliding up to 6.5 (so the bid is nearly always the cap), cap alpha 0.56, gam 0.22, gmin 1.0, kap -0.66 |

A fourth search, `cma_global`, is kept as a record in `tune_out/cma_global.json` and is not a variant.
It was the first CMA-ES configuration: sigma0 0.2, popsize 8, 16 generations and 2 blocks per
evaluation (seeds 3000-3031, final 3500-3515). Under the measured noise it random-walked into a
low-multiplier, stud-only region and ended at -0.004 ± 0.006 on its fresh final seeds. That failure is
why the `cma` variant uses the noise-adapted settings. Both were chosen on tuning seeds only.

**Rich family** (`policy.py:SimOpt`, one class for every point of both families):

    bid = min( m(week) x (g/6)^gam x exp(kap * z) x typical(g),  cap_alpha )      no bid when g < gmin

- `m(week) = exp(lm0 + lm1 * t)`: log-linear multiplier by week, t = 0 at the auction after week 8,
  1 at the last auction (week 15). Replaces the baseline's step at week 12.
- `gam`: player-tier exponent. Positive spends relatively more on studs, negative on depth.
- `kap`, `z`: team strength. `z` in [-1, 1] is 2 x (share of alive rivals whose projected lineup for
  the next week is below mine) - 1. It is computed lazily, once per auction, only when a bid needs it.
- `cap_alpha`: competition-aware shading. It is the smallest whole-dollar bid `b` with
  P(every rival claim <= b - 1) >= alpha under the point BidModel `state.bm`. It uses the same
  formula as `State.rival_cdf`, with scalar math and bisection: 13 us a call, checked against
  `rival_cdf` on 1,448 captured cases. The cap never exceeds the richest rival's budget + 1, so
  money nobody can contest is never spent.
- Search box (CMA-ES and BO work in [0,1]^6): m at week 8 in [0.25, 2], lm1 in [-1.2, 1.2],
  gam in [-1, 1.5], gmin in [1, 5], kap in [-1, 1], logit(alpha) in [-1, 5] (alpha 0.27-0.993).

**Why it might beat the baseline.**

1. Better selection inside the baseline's own family. run.py picks its multiplier with a uniform
   16-point sweep on 10,000 paths of one seed and the main condition only. Its argmax (1.0/0.75) was
   sweep noise: the full baseline run shows that 0.75/0.75 beats it by +0.006. Ranking and selection
   with CRN spends the simulation where the decision is close, and it uses the robust objective (mean
   P(win) over main, stress and bmboot), so it does not tune to the exact rivals in `state.bm`.
2. A richer policy. Bidding the league-typical amount ignores who is actually competing for a
   player. The rivals' gains, their participation odds and above all their budgets (5 of 9 rivals
   hold $60 or less) set the price that wins. In a 4-seed probe the cap at alpha = 0.9 left the wins
   unchanged and cut spend from $127 to $98. The search then decides whether to redeploy that money,
   through a higher multiplier, a tier tilt or a time trend.
3. Compute. Uniform sweeps waste most of their paths on clearly bad settings. Under CRN, OCBA reaches
   the same probability of a good pick with far fewer paths (study below), so run.py could search a
   finer grid, or the robust objective, at the same cost.

**Protocol and leakage.**

- Every simulation in tune.py goes through `policy.tune()`, which refuses seeds below 1000. The
  evaluation seeds (100-399) are never simulated during tuning, and the variant choice is fixed
  before any evaluation run.
- Disjoint seed ranges per stage, so stages never reuse each other's paths:
  - bank: 1000-1039
  - OCBA grid: 2000+
  - `cma_global`: 3000+, final 3500+
  - BO batches: 4000+, final 4500+
  - `cma` (local): 5000+, final 5500+

  Every search draws fresh seeds for each generation or batch.
- The objective is always a **paired** difference against the reference (= the baseline, 1.0/0.75)
  on the same CRN blocks. The reference is simulated on every block that any alternative uses.
- Deterministic: fixed CRN seeds, pycma seed 20261029, GP `random_state=0`, a fixed Thompson-sampling
  generator and fixed Sobol seeds. Every block result is cached in `tune_out/cache.jsonl`, keyed by
  parameters, seed, conditions and data-snapshot signature, so a rerun resumes. Checked: deleting
  the ocba, cma and bo stage files and rerunning them from the cache, on 1 core instead of 2,
  reproduced every pick, estimate and trace exactly, with 0 new simulations.
- Cache and stage files from another data snapshot are never reused. After the next `fetch.py`
  (state != `0549ea074558`), `tune.py` recomputes everything, and `freeze` refuses mixed snapshots.
- The policies themselves use only `state` (point BidModel, budgets, rosters, projections) and keep
  nothing across paths: `reset()` clears the per-auction strength cache.

**Files.**

| file | role |
|---|---|
| `policy.py` | `SimOpt`, the policy family (the track's candidate class) |
| `tune.py` | the search: stages `bank`, `ocba`, `cma_global`, `bo`, `cma`, `pcs`, then `freeze` -> `tuned.json`. Run it as `bench/cpuq -n 2 -- .venv/bin/python -m bench.candidates.sim_optimization.tune` (about 40 min on 2 cores from an empty cache); `--quick DIR` is a 2-minute plumbing test that leaves `tuned.json` alone |
| `check.py` | SimOpt on 1.0/0.75 equals `Multiplier("baseline")` bit for bit (seed 1000, all conditions); `cap()` agrees with `State.rival_cdf`; every frozen variant builds and bids. `run_smoke.sh` runs it first |
| `evaluate.py` | `policy.run()` on the frozen variants. Also stores the tuning summary and the R&S study in the result JSON |
| `tuned.json` | frozen parameters, tuning-seed estimates, compute, R&S study |
| `tune_out/` | per-stage JSON, search traces (`*_trace.npz`), the bank (`bank.npz`), the block cache, the log |
| `run_smoke.sh`, `run_full.sh` | per the protocol (they evaluate; they do not retune) |

## Results

Everything below is in-sim. State `0549ea074558`: week 8, 10 alive, my budget $165, no pending
chop. "d" is the paired P(win) difference against the baseline (1.0/0.75), averaged over the three
conditions unless a column says otherwise. "Block" means 200 CRN paths.

### Tuning (seeds >= 1000; 2 cores; about 40 min wall, about 4,600 core-s)

| stage | policy-blocks | wall | pick | d on its own fresh seeds |
|---|---|---|---|---|
| bank: run.py's 16-point grid, main condition only | 640 | 212 s | 0.75/0.75 (0.4328), then 0.75/1.0 (0.4314); run.py's 1.0/0.75 is 3rd (0.4249); m_early 1.5 is 0.34 | n/a (uniform) |
| `ocba`: 42-point grid | 521 | 563 s (two passes) | 0.9/0.75 | +0.0043 ± 0.0020 (51 blocks) |
| `cma_global` (record) | 360 | 540 s | `rec` | -0.0041 ± 0.0063 (16) |
| `bo` | 363 | 480 s | `gp_top2` | -0.0058 ± 0.0044 (19) |
| `cma` (local) | 412 | 512 s | `gp_top3` | +0.0087 ± 0.0022 (20) |

Notes on the tuning runs:

- **Noise level.** A paired difference against the baseline on one block has a standard deviation
  of 0.016 for step policies and 0.023 for rich ones. The three conditions are nearly uncorrelated
  (r = 0.01-0.24).
  - Resolving a 0.005 effect therefore takes about 40-80 blocks per policy, versus 2-4 per
    evaluation inside a search.
  - Inside 0.7-1.0 early, the whole landscape is flat to about ±0.005.
- **Global searches.** `cma_global` and `bo` spent most evaluations in clearly bad regions. Their
  per-batch means sat around -0.01 to -0.04, and neither one's pick beat the baseline on fresh seeds.
- **Local search.** `cma` started at the incumbent with small steps and 4 blocks per evaluation, where
  CRN noise is lowest, and it moved steadily. Its mean went from 0.75 flat to about 0.98 at week 8,
  sliding to 0.64 by week 15. The pick (`gp_top3`) slides from 0.905 to 0.743. Both have gmin about
  2.3 and the alpha 0.9 cap on.
- **Ablations** in the `cma` final selection. Each has only 4 blocks, so these are hints, not results:
  - no cap: +0.003 ± 0.004, against +0.006 ± 0.003 for the same point with the cap;
  - no strength term: +0.001 ± 0.009.
- **Winner's curse.** The final picks are maxima over 6 alternatives, so their estimates run
  optimistic by a few 1e-3. The OCBA grid pick is a maximum over 42.
- **The local GP overfit.** Its white noise sits at the lower bound. The `cma` shortlist is therefore
  close to "the best observed points", and the fresh-seed OCBA is what really selects.

### R&S efficiency: P(correct selection) vs CPU budget (run.py's 16-point grid, main condition)

Bootstrap world of the 40-block bank (8,000 paths per alternative):

- Truth is the bank means: best 0.75/0.75; 0.75/1.0 is 0.0014 behind, then 1.0/0.75 at 0.008 behind.
- Each macro-replication resamples whole CRN blocks with replacement, and every alternative uses the
  same block sequence.
- 1,000 replications for uniform, 500 for the others.

Metrics: PCS picks the exact best; PGS picks within 0.005 of it; EOC is the expected regret in
P(win). Budget is in policy-blocks, where one main-condition block is about 0.72 core-s.
run.py's sweep is 800 blocks (16 x 10,000 paths, about 576 core-s).

| budget (core-s) | uniform = run.py: PCS / PGS / EOC | OCBA-CRN | KN (delta 0.005) | KN (delta 0.01) |
|---|---|---|---|---|
| 64 (46) | 0.47 / 0.74 / 0.0033 | 0.48 / 0.75 / 0.0031 | n/a (first stage alone is 80) | n/a |
| 128 (92) | 0.56 / 0.82 / 0.0019 | **0.63 / 0.89 / 0.0015** | 0.54 / 0.79 / 0.0024 | 0.54 / 0.81 / 0.0020 |
| 256 (184) | 0.66 / 0.90 / 0.0011 | **0.73 / 0.97 / 0.0007** | 0.70 / 0.91 / 0.0010 | 0.66 / 0.94 / 0.0009 |
| 400 (288) | 0.69 / 0.93 / 0.0009 | **0.82 / 0.98 / 0.0004** | 0.73 / 0.95 / 0.0007 | 0.72 / 0.98 / 0.0005 |
| 800 (576) | 0.82 / 0.99 / 0.0003 | **0.93 / 0.99 / 0.0003** | 0.88 / 1.00 / 0.0002 | 0.77 / 0.99 / 0.0003 (stops at 566) |

What the table shows:

- OCBA under CRN matches the uniform sweep's PCS with about half the compute: 0.82 at 400 blocks,
  against uniform's 0.82 at 800. Its PGS reaches 0.97 at 256 blocks, a third of run.py's budget.
- At run.py's own budget, PCS rises from 0.82 to 0.93.
- KN sits between the two. Its indifference-zone stopping rule rarely fires within budget, because
  h²S²/delta² is about 1,000 blocks, so it acts mostly as early elimination of the clear losers.
- At 800 blocks, uniform picks something worse than 0.005 only 1.4% of the time. That makes run.py's
  actual pick of 1.0/0.75 (0.008 behind in this world, 0.006 behind in the full baseline run) an
  unlucky draw rather than a typical outcome.
- Suggested change for run.py: sequential OCBA (n0 = 2 chunks per setting, then 16 more chunks per
  round, allocated as above) gets the same reliability for about half its sweep's CPU. It also frees budget to sweep the robust (three-condition) objective
  instead of the main condition only.

### Smoke (seeds 100-104, 1,000 paths, 5 blocks: a plumbing check only)

`bench/cpuq -n 1 -- bash bench/candidates/sim_optimization/run_smoke.sh` takes 37-42 s on 1 core
(sim 32-36 s depending on box load, check.py about 5 s). Output:
`bench/results/sim_optimization.smoke.json`, validated. Two runs gave identical numbers.

| variant | p_win | stress | bmboot | p_win_min | e_weeks | spend | buys | d_p_win vs baseline (main / stress / bmboot) |
|---|---|---|---|---|---|---|---|---|
| baseline 1.0/0.75 (cached `_baseline_policy.smoke`) | 0.433 | 0.436 | 0.404 | 0.404 | 6.98 | $143 | 7.5 | reference |
| m0.75 (baseline file) | 0.454 | 0.435 | 0.417 | 0.417 | 6.90 | $127 | 7.9 | +0.021 / -0.001 / +0.013 |
| `ocba` | 0.446 | 0.445 | 0.413 | 0.413 | 6.92 | $137 | 7.7 | +0.013 ± 0.012 / +0.009 ± 0.017 / +0.009 ± 0.006 |
| `cma` | 0.438 | 0.445 | 0.416 | 0.416 | 6.87 | $111 | 7.3 | +0.005 ± 0.008 / +0.009 ± 0.019 / +0.012 ± 0.010 |
| `bo` | 0.438 | 0.437 | 0.402 | 0.402 | 7.02 | $122 | 10.3 | +0.005 ± 0.023 / +0.001 ± 0.021 / -0.002 ± 0.008 |

With 5 blocks, none of these differences is resolved. `cma` is within noise of `ocba` while spending
$26 less, because the cap does that.

### Full run

Not run by this agent. Enqueue it with:

    bench/queue/pending/sim_optimization.job = {"key": "sim_optimization", "cmd": ["bash", "bench/candidates/sim_optimization/run_full.sh"], "timeout_s": 1200}

It evaluates the frozen `tuned.json` variants on seeds 100-399 (60,000 paths, 3 conditions). The
rule of thumb (10 x the smoke's 1-core sim_s) puts it at about 5-6 min on 6 cores. It pairs with
`_baseline_policy.json` (same state), and the leaderboard is `python -m bench.harness.compare policy`.
The paired resolution at that size is about ±0.003. The tuning-seed estimates above predict:

- `cma` about +0.006 to +0.009;
- `ocba` about +0.004;
- `bo` about -0.005, which is reported anyway as the global-BO result.

### Caveats

- **In-sim only.** The cap in `cma` and `bo` reads the point BidModel (`state.bm`), which is exactly
  what the main-condition rivals follow. Judge them on `p_win_bmboot`, `p_win_stress` and
  `p_win_min`. The objective already averaged all three conditions during tuning.
- **Snapshot drift.** Tuned on state `0549ea074558`. The family is a function of the state, so the
  frozen policies still run on a new snapshot (with a warning), but the parameters would need
  retuning after the week-8 chop and the next fetch.
- **No proof of global optimality.** The rich-family conclusions are limited by budget. The
  posterior on the 256 global-search evaluations puts the best point within +0.009 ± 0.017 of the
  baseline. The family may hold larger gains in regions that the noise hid.

## Sources

- Chen, Lin, Yücesan, Chick (2000), *Simulation budget allocation for further enhancing the
  efficiency of ordinal optimization*, Discrete Event Dynamic Systems 10(3):251-270: the OCBA
  allocation. Overview: https://mason.gmu.edu/~cchen9/ocba.html
- Fu, Hu, Chen, Xiong (2007), *Simulation allocation for determining the best design in the
  presence of correlated sampling*, INFORMS J. Computing 19(1):101-111. WSC 2004 version:
  https://www.informs-sim.org/wsc04papers/072.pdf. This is OCBA under CRN; the allocation
  n_i ~ Var(X_b - X_i)/delta_i², with n_b = max n_i so the best is paired with everyone, follows
  that view.
- Kim, Nelson (2001), *A fully sequential procedure for indifference-zone selection in
  simulation*, ACM TOMACS 11:251-273: procedure KN with CRN.
  https://www.cs.utexas.edu/~shivaram/readings/b2hd-KimNelson2001.html
- Hansen, Niederberger, Guzzella, Koumoutsakos (2009), *A method for handling uncertainty in
  evolutionary optimization...*, IEEE TEC 13(1):180-197: noise-aware CMA-ES (the rank-based
  view of noise that motivates the larger re-evaluations and smaller steps of `cma`).
  https://cs.utexas.edu/~shivaram/readings/b2hd-HansenNGK2009.html. Library: pycma 4.5.0
  (`pip install cma`).
- Letham, Karrer, Ottoni, Bakshy (2019), *Constrained Bayesian optimization with noisy
  experiments*, Bayesian Analysis 14(2):495-519 (arXiv 1706.07094): BO under heavy observation noise.
- Kandasamy, Krishnamurthy, Schneider, Póczos (2018), *Parallelised Bayesian optimisation via
  Thompson sampling*, AISTATS: batch Thompson sampling.
- Pan et al. (2020), *Bid shading by win-rate estimation and surplus maximization*, arXiv
  2009.09259: first-price bid shading against an estimated win-rate curve (the `cap_alpha` idea).
- Balseiro, Gur (2019), *Learning in repeated auctions with budgets: regret minimization and
  equilibrium*, Management Science 65(9): pacing multipliers on value under a budget (the
  multiplier-by-week family).
- scikit-learn 1.9 `GaussianProcessRegressor`; SciPy 1.18 `scipy.stats.qmc.Sobol`.
