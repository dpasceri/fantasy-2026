# _baseline_team

**What.** The core sim's one-week chop model, refit at every rolling origin on training rows only:

- marginal: `_baseline_player`'s bucketed actual/proj quantile tables (`fit_tables`, `ratio_grid`),
- dependence: model.fit's one-factor Gaussian copula, re-implemented in `copula.fit_copula` and run on
  the train items. That is: randomized-PIT z per bucket (seed 7), same-team and same-game residual
  correlations of players projected ≥ 3, loadings `a_pos` by least squares, and game share `g`,
- simulation: `sim.Universe.paths`' week-0 draw, verbatim (`copula.sample_chunk`).

For each test week, `WeekDraws` simulates every QB/RB/WR/TE projected > 0 that week:
96000 draws (the same count as backtest.py), in chunks of `sim.CHUNK`, with
`rng = default_rng([11, season, week, chunk])`. Every league of the week is scored on these same draws
(common random numbers). A team total is the sum of its starters' draws. A starter who is missing or
projected ≤ 0 scores 0, and there are no bench subs.

`predict` returns `p_chop` (the argmin share), `total_q` (empirical quantiles) and `line_q` (quantiles of
the per-draw minimum).

**Point-in-time choices.**

- No injury tags: `inj` is today's tag, so `p_out = 0`, as in backtest.py. Past-week projections
  are final, so players who were ruled out are already near 0.
- No level drift or injury hazard: the horizon is one week.

**Why.** This is the reference every team-track candidate is compared against.

**Verification.** Run `bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates._baseline_team.verify` (about 10 s).

1. `fit_copula` on `model.calibration_rows`, in their order, gives data/model.npz's loadings and game
   share exactly (max |diff| = 0).
2. The refit on the harness's rows, which are in table order, moves the TE loading by 0.017. Ties at a
   ratio of 0 get their uniform draws on different rows. This is reported, not asserted.
3. `sample_chunk` equals `sim.Universe.paths` bit for bit (max |diff| = 0). The check uses 2026 week 7,
   326 players and 8000 draws, with the same tables and seed.
4. With data/model.npz's parameters, the track's real leagues reproduce data/backtest.json's
   P(chopped team) for weeks 1-7 within ≤ 2.7 Monte Carlo SE of the difference. Log-loss is 2.382 here
   vs 2.384 in backtest.json. This confirms that the track's real items (teams, starters, chopped team)
   match the core backtest. Those parameters are in-sample, so this is not the rolling-origin score.

**Variants.** `baseline`.

**Runs.**

- `run_smoke.sh`: about 9 s on 2 cores (3 origins, 50 leagues).
- `run_full.sh`: the queued job `_baseline_team` (21 origins, 1015 leagues). It takes about 5.5 s
  per origin on 1 core, so under a minute on 6 cores. Peak memory is about 0.6 GB per worker.

**Smoke result.**

| Leagues | Log-loss (uniform) | Brier | Other |
|---|---|---|---|
| virtual, n = 48 | 2.076 (2.588) | 0.825 | rank_norm 0.229; crps_total 10.94, crps_line 5.54; ECE 0.024; team-total coverage 10/50/90 = 0.076 / 0.454 / 0.894 |
| real, n = 2 | 1.851 (2.553) | — | mean rank 2.5 |
