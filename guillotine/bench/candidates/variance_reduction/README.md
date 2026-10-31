# variance_reduction (mc track)

**What.** Conditional (product-form) Monte Carlo for the two mc-track numbers, P(chopped this week) and
P(win the league, frozen rosters), sampling exactly the core model (`sim.Universe.paths` + `sim.team_scores`).

The key structural fact: given one week's copula factors F (the game and NFL-team normals of `model.py`'s
one-factor copula), every alive team's total is independent of every other team's. Each rostered player sits
on exactly one roster, and everything else the sampler draws (idiosyncratic eps, week-0 availability, level
drift, injuries) is per player. So:

1. Draw the factors once per **group** and K inner draws per team on those factors.
2. **week**: score every one of the K^T team combinations of each group, exactly. The average of
   1{t lowest} over all combinations is mean_k prod_{s != t} S_s(X_t^k), with S_s the share of team s's draws
   above the value. One merged sort per group gives it as a running log-sum (`vr.chop_vstat`, checked against
   brute-force enumeration).
3. **season**: the elimination has no product form, so play R rounds of random per-team cyclic shifts within
   each group (`vr.recombine`): R·K combined seasons per group, each with the exact joint law given the
   group's factors.

This is the product-form / generalized U-statistic estimator of Kuntz, Crucinio & Johansen (2022), applied
conditionally on the factors, as in factor-copula credit-risk simulation (Glasserman & Li 2005). It is
unbiased for any K, R. Layers on top, each of which keeps every single path's law exact:

- **Outer factors.** Week: Owen-scrambled Sobol' points (scipy `qmc.Sobol`, LMS + digital shift), with the
  factor dims ordered by loading-weighted score variance so the best net coordinates go to the QB factors
  (QB loading is 0.95). Season: antithetic factor pairs (F, -F); Sobol' did not help over 9 x ~45 dims.
- **Inner stratification** (`Noise`).
  - *Latin hypercube*: each player's eps, and the week-0 availability uniforms of tagged players, are
    stratified over the K draws of a group. Each (group, player) gets one random permutation, taken from a
    per-call pool of 1024 and composed with an independent random cyclic shift. So every single draw is
    still uniform and players stay independent.
  - *Antithetic within the group*: draw k + K/2 = -draw k. This is cheaper than iid because it needs half
    the normals. The season uses it for level steps, and in `cmc_strat` also for eps, where LHS's extra
    `ndtri` did not pay on dev.
  - LHS removes the additive part of each team total's variance (Stein 1987). The combination statistic
    removes the cross-team interactions.
- **Player-level recombination** (`Rp`). Rp - 1 extra "virtual" draws per draw: every lineup component
  (week: slots linked by a shared bench sub; season: each player's whole 9-week path) reads its values
  from its own random cyclic shift of the group's K draws. Components share no player, so each virtual team
  draw has the exact law. All M = Rp·K draws per team enter the same combination statistic. This removes
  within-team interactions that LHS leaves, which is why it only pays together with LHS.

**Pure speedups** (same law, same float32 arithmetic up to summation order):

- Only lineup players (starters and bench subs) are scored each week.
- In week 0 (no injury hazard), a sub behind a starter with p_out = 0 can never play, so it is dropped.
- Inactivity is NaN-coded, so subs are looked up only where a starter sits.
- One complex64 gather does the quantile interpolation.
- Factors are drawn once per group, not once per path.
- Work runs in cache-sized chunks of 2048 draws, 1024 for the season `cmc_mix` (dev timing).

`verify.py` checks the re-implemented totals against `sim.team_scores`: exact for the season, and within
float32 summation order (1.5e-5) for the week's component sums. It also checks the statistic against
enumeration.

**Why it can beat plain MC.** The chop and win indicators are mostly noise from the idiosyncratic player
draws. The factor part of the variance is only 5.6% (week) and 1.1% (season) of p(1-p) (dev target,
measured). Plain MC pays p(1-p) per path. The product-form statistic pays only the first-order (Hajek)
terms, and LHS and player recombination shrink those further. Sobol' or antithetic factors keep the factor
term, which grows with K, small.

**Budget.** A call draws N = (0.95 n - a) / b inner paths, where a and b come from `calib.py` on the dev
target in plain-path units. So cost is about 0.95. The harness bills max(cost, 1), and overspending is billed.
N depends only on n, so calls are deterministic given the seed.

**Variants** (simplest first):

| variant | week | season |
|---|---|---|
| `cmc` | Sobol' factors, K=16, all-combination statistic, iid inner | antithetic factors, K=8, R=16 team recombination, iid inner |
| `cmc_strat` | + LHS on eps and availability | + LHS on week-0 availability, antithetic eps / level steps within the group |
| `cmc_mix` | + Rp=6 player-level recombination (M=96 draws per team per group) | LHS eps / availability, antithetic level steps, Rp=4, R=6 |

Every config is in `candidate.VARIANTS`.

**Tuning (dev target only, `mc.build_targets(dev=True)`, never the scored state).** `tune.py` reports
efficiency = (plain variance / variance) x (plain CPU / CPU). Variances are summed over teams, from 16-24
independent calls per config, interleaved with plain-MC timing calls. Dev results at n = 40000 (week) and
24000 (season), with about ±30% noise on each number:

| target | config | var reduction | cost | efficiency |
|---|---|---|---|---|
| week | plain (K=1) | 1 | 1 | 1 |
| week | iid factors, K=16, all-combination | 1.8 | 0.42 | 4.4 |
| week | antithetic factors, K=16, LHS, Rp=6 | 4.0 | 1.14 | 3.5 |
| week | Sobol', K=16, LHS (`cmc_strat`) | 4.4 | 0.54 | 8.1 |
| week | Sobol', K=16, LHS, Rp=4 / 6 / 8 / 16 | 10.6 / 15.3 / 15.2 / 18.1 | 0.97 / 1.10 / 1.20 / 1.77 | 10.9 / 13.9 / 12.7 / 10.2 |
| week | Sobol', K=8 / 32, LHS, Rp=8 / 6 | 15.0 / 12.6 | 1.35 / 1.06 | 11.1 / 11.9 |
| week | Sobol', K=16, no LHS, Rp=8 | 3.3 | 1.03 | 3.2 |
| season | antithetic factors, K=8, R=16 (`cmc`) | 2.4 | 0.94 | 2.6 |
| season | Sobol' factors, iid inner (vs `cmc`) | 2.1 | 0.97 | 2.1 |
| season | padded Sobol' factors, LHS eps, av, antithetic lvl | 4.25 | 1.08 | 3.9 |
| season | + LHS eps, av; antithetic lvl | 4.0 | 1.06 | 3.8 |
| season | + LHS av; antithetic eps, lvl (`cmc_strat`) | 4.5 | 0.83 | 5.4 |
| season | LHS eps, av; antithetic lvl; Rp=4, R=8 / R=6 (`cmc_mix`) | 10.8 / 12.0 | 1.81 / 1.70 | 6.0 / 7.1 |
| season | own-path conditional MC (P(t wins given others) over t's M paths) | 1.6-2.9 | ~1.1 | under 2.6 |

The factor share of the indicator variance (iid factors, K=2 covariance): 5.6% of p(1-p) for week and 1.1%
for season. Scrambled Sobol' over the 42 week factor dims is what makes large K pay on the week (anti 3.5x
vs Sobol' 13.9x at the same config). Costs above are before the chunk-size speedup. `calib.py` then gave
the final cost model (b = inner-path CPU in plain-path units): week 0.42 / 0.57 / 0.86 and season 0.84 /
0.73 / 1.33 for `cmc` / `cmc_strat` / `cmc_mix`.

`verify.py` (dev): the totals equal `sim.team_scores` (max |diff| 1.5e-5 from float32 summation order;
season 0). `chop_vstat` equals enumeration exactly. Bias against a 3.2M / 320k-path plain reference over
12 calls per variant: max |z| 2.68 / 1.55 / 1.90 (week) and 2.86 / 3.21 / 2.40 (season). The season z
pattern is the same for all three variants (+2.5, -2.9, -2.6 on the same teams), so it comes from the
reference's own error, not from the estimators.

A larger week check (dev, n = 40000, against a 6.4M-path plain reference) gave max |z| 1.24 / 2.07 / 1.92
over 80 fresh-seed calls per variant. Pairwise variant-minus-variant tests, which are free of reference
noise, gave max |z| 1.50-2.23 over 27 team comparisons. An earlier 40-call run had one pairwise 3.1, which
did not recur with fresh seeds.

**Sources.**

- J. Kuntz, F. R. Crucinio, A. M. Johansen, "Product-form estimators: exploiting independence to scale up
  Monte Carlo", Statistics and Computing 32 (2022). https://arxiv.org/abs/2102.11575
- P. Glasserman, J. Li, "Importance sampling for portfolio credit risk", Management Science 51 (2005).
  Conditional independence given the factors of a Gaussian copula.
  https://drupalgsb-test.paas.cc.columbia.edu/sites/default/files-efs/pubfiles/1368/Glasserman_importance_sampling.pdf
- P. L'Ecuyer, "Randomized quasi-Monte Carlo: an introduction for practitioners", MCQMC 2016 (Springer
  2018). https://hal.archives-ouvertes.fr/hal-01561550
- A. B. Owen, *Monte Carlo theory, methods and examples* (2013): conditioning, antithetics, LHS, scrambled
  nets, padding. https://artowen.su.domains/mc/
- M. Stein, "Large sample properties of simulations using Latin hypercube sampling", Technometrics 29
  (1987). LHS removes the additive component of the variance.
- S. Asmussen, D. Kroese, "Improved algorithms for rare event simulation with heavy tails", Adv. Appl.
  Prob. 38 (2006): conditional MC via exchangeability for sums and order statistics.
  https://www.cambridge.org/core/journals/advances-in-applied-probability/article/improved-algorithms-for-rare-event-simulation-with-heavy-tails/F5940116F912228BB80FC17A89518AF4
- scipy.stats.qmc.Sobol. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html

**Runs.**

- `run_smoke.sh`: `bench/cpuq -n 1 -- bash bench/candidates/variance_reduction/run_smoke.sh`, about 30 s.
- `run_full.sh`: queued only. On the scored state a plain-MC call at the full budget takes 0.44 s (week) and
  4.6 s (season) of CPU (`_baseline_mc` full calibration). So 3 variants x 160 reps x (0.46 + 4.2 s), plus 80
  calibration calls, is about 2500 core-s: about 7-8 min on 6 cores (`_baseline_mc` ran 400 calls of about
  1000 core-s in 190 s). The full reference is already built by `_baseline_mc`. (Reviewer correction: an
  earlier estimate of 1300 core-s used dev-target timings.)
- `verify.py`, `tune.py`, `calib.py`: dev-target checks and tuning (through cpuq).

**Smoke results** (scored state, R = 8, n = 8000 / 8000, against `_baseline_mc` on the same smoke split):

| variant | log_work (d vs baseline, 95% CI) | eff | week rmse / cost / eff / bias z max | season rmse / cost / eff / bias z max |
|---|---|---|---|---|
| `_baseline_mc` | -11.624 | 1.00 | 0.00262 / 1.02 / 1.0 / 2.1 | 0.00337 / 1.01 / 1.0 / 1.9 |
| `cmc` | -12.749 (d -1.12 [-1.41, -0.81]) | 3.08x | 0.00154 / 1.01 / 2.9x / 3.2 | 0.00188 / 0.93 / 3.3x / 1.4 |
| `cmc_strat` | -13.458 (d -1.83 [-2.20, -1.47]) | 6.26x | 0.00107 / 1.03 / 5.9x / 2.3 | 0.00131 / 0.93 / 6.7x / 2.1 |
| `cmc_mix` | -13.782 (d -2.16 [-2.62, -1.84]) | 8.65x | 0.00075 / 1.04 / 12.0x / 1.7 | 0.00136 / 0.90 / 6.2x / 1.3 |

Smoke run: 23 s wall on 1 core; candidate import 0.2 CPU s. With only 8 replications each eff carries
about a factor of 2 of noise. The smoke reference is small (240k season paths): its variance is 1/5 of
`cmc_strat`'s and `cmc_mix`'s season MSE (`mse_over_ref_var` 4.8 / 5.1), so their smoke season efficiencies
are understated. The full reference (16M) removes that.

**Caveats.**

- Smoke has R = 8, so its efficiencies carry about a factor of 2 of noise. The full run (R = 160) is the
  real test, including `bias_z_max`. On smoke, `cmc` week shows one |z| of 3.2 at R = 8 (7 degrees of
  freedom; about 1 such value expected among the 60 team-target z's). The dev checks above, at 12-80 calls per
  variant, show no bias.
- Cost is calibrated on the dev target (9 teams, 8 weeks). On the scored state (10 teams, 9 weeks) it can
  land a few percent off 0.95. Under 1 is billed as 1; over 1 is billed in full.
- An importance-splitting layer for the rare-chop teams (p about 0.01) was not needed: the combination
  statistic already turns rare indicators into smooth products.
- An "own-path" conditional estimator was tried and was worse at equal cost. It takes P(t wins | others' paths)
  as the share of t's M paths that beat the others-only chop lines.

**Independent review (dev target and synthetic inputs only).**

- z and the quantile interpolation match the `Universe.paths` formula on the same normals in every week
  (max diff 6e-6).
- `chop_vstat` matches a direct O(M^2 T) formula and brute force on tied, negative and zero totals,
  including `argmin`'s lowest-index tie rule (1e-16).
- `season_wins` matches the core elimination loop exactly.
- The per-team laws of every variant's real and virtual draws match core paths:
  - season: weekly-total mean and P(below q10) |z| max 2.4 over 72 cells, sd ratio 0.99-1.01, week-1 vs
    week-9 autocorrelation within 0.012;
  - week: mean and P(below q5) |z| max 1.9, cross-team correlation within 0.022.
