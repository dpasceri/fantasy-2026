# _baseline_mc

**What.** Plain Monte Carlo: the core sampler and chop rule from `sim.py`, with no variance
reduction. `PlainMC.estimate(target, n, seed)` calls `bench.harness.tracks.mc.plain`. That
function draws `n` iid paths in chunks of 8000, and chunk `ci` comes from
`np.random.default_rng([seed, ci])`, the same way `sim.run_season` draws them.

- `week`: each path draws this week from `Universe.paths` and scores every alive team's
  projection-optimal lineup with bench subs. The lowest total is chopped. The estimate is the
  chop share for each team.
- `season`: each path plays weeks `lg.week..16` and chops the lowest alive team each week. The
  estimate is the share of paths each team survives to the end.

**Why.** This is the reference every mc-track candidate's efficiency is measured against.
`eff_rel` = (MSE × cost) of plain MC divided by the candidate's.

**Verification.** Run `bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates._baseline_mc.verify`.

- With the same seed and 16000 paths, `plain` on the season target equals
  `sim.run_season(U, plan, n, seed)["win"][0]` exactly (max |diff| = 0).
- `plain` on the week target equals `run_season`'s week-0 `chop` exactly. They make the same
  draws.
- When a reference is cached, it also prints z-scores of `out/results.json`'s standings against
  that reference. run.py's 640k paths come from a Universe that also holds free agents, so the
  numbers agree only within MC error, not draw for draw.

**Variants.** `baseline`.

**Runs.**

- `run_smoke.sh`: about 6 s with the smoke reference cached. About 35 s on 1 core the first time,
  when it also builds the smoke reference (1.6M week + 240k season paths).
- `run_full.sh`: the queued job `_baseline_mc`, about 7-8 min on 6 cores. It is the only job that
  builds the full reference (`--make-ref`): 64M week paths and 16M season paths, about 1700 core-s.
  It then runs 160 replications per target (n = 40000 / 32000), plus 40 calibration calls per
  target.

Smoke result (8 replications, n = 8000, `cpuq -n 2`): week rmse 0.00262, season rmse 0.00337,
cost 1.02 / 1.05, log_work -11.606. CPU per call is about 0.057 s (week) and 0.58 s (season) on
one of cores 8-15, or about 8 µs and 70 µs per path.
