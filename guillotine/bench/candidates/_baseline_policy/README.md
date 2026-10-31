# _baseline_policy

**What.** dynamic.py's bid rule as a policy-track candidate (`multiplier.py:Multiplier`). For each
player on offer with my lineup gain `g` (average weekly points over the next 4 weeks), bid

    round(m * typical(g)),  typical(g) = exp(mu(me, g, my budget, week) - u_me)

which is the BidModel's median bid for that gain at my budget, with my own "bids-hard" effect
taken out. `m = m_early` before week 12 and `m_late` from week 12 on. No bid when g < 2.

**Why.** This is what the pipeline does today, so every policy candidate is compared against it.
run.py sweeps 16 (m_early, m_late) pairs on 10,000 seed-11 paths and runs the argmax for its
P(win | recommended policy), budget curve and bid grid. On this snapshot that is 1.0 / 0.75
(P(win) 0.4277), tied with 0.75 / 0.75 (0.4273, se 0.005). The cheat sheet's text says
"bid 0.75-1x typical".

**Variants.**

| name | m_early | m_late | note |
|---|---|---|---|
| `baseline` | 1.0 | 0.75 | the reference: run.py's pick (`dynamic.best` in out/results.json, chosen on seed 11, disjoint from the eval seeds) |
| `m0.75` | 0.75 | 0.75 | the runner-up in run.py's sweep (a tie) |
| `x1.0` | 1.0 | 1.0 | bid the league-typical amount all season |

**Verification.** `bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates._baseline_policy.verify`
(about 30 s, tuning seeds only). It checks six things:

1. `dyn_policy.refit_bm(bm, bm.rows)` reproduces `BidModel(lg)` exactly.
2. `dyn_policy._paths_chunk` without a policy equals `dynamic._paths_chunk` on every chunk sum,
   for 0.75/0.75, 1.0/0.75 and the stress setting.
3. The policy hook running `Multiplier` reproduces those runs bit for bit.
4. The analytic `State.p_win` agrees with a Monte Carlo of `auction.top_rival_bids` within
   0.0035 over bids $1-120.
5. One seed's outcomes are identical in fresh processes under two different `PYTHONHASHSEED`
   values. The core `BidModel(lg)` fit drifts up to about 4e-3 with the hash seed, so the
   track refits it on canonically ordered rows.
6. Isolation. The track's per-(variant, condition) runs on fresh policy copies equal one joint
   `_paths_chunk` call, and a variant's paths don't depend on the other variants. A policy that
   keeps state across paths, or uses numpy's global RNG, still gets the same seed results
   whatever ran before it.

**Audit.** On run.py's own sweep paths (seed 11, 10,000 paths) this env, the policy hook and
`Multiplier` give 0.4272 / 0.4275 for 0.75/0.75 and 1.0/0.75, against 0.4273 / 0.4277 in
out/results.json. The 1-2 win gap is the core fit's hash-order drift.

**Runs.**

- `run_smoke.sh`: 5 seeds x 200 paths x 3 variants x 3 conditions, about 16 s on 2 cores and 27 s on 1.
- `run_full.sh`: the queued job `_baseline_policy`, 300 seeds = 60,000 paths, about 5 min on 6
  cores (estimated from the smoke run's 5.3 s per seed).
