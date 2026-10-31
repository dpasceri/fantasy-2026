# projection_ensemble (player track)

**What.** This candidate combines Sleeper's projection with independent point-in-time signals. All
three variants share one base, then add a two-part stacked recalibration on top.

- **Base (`lpool`).** The baseline's bucket ratio tables (position × projection bucket, actual/proj
  quantiles on the 256 grid), with one change: neighbouring buckets are blended by a **linear pool**
  (CDF average, same log-projection weight) instead of quantile averaging.
- **Two-part stacked recalibration (`recal`, `stack`).** This is fitted on top of the base:
  - zero part: `logit P(y ≤ 0) = logit F_base(0) + X d`, a ridge logistic whose offset is the base's
    own mass at or below 0;
  - positive part: the base's PIT within its positive part is mapped through a probit
    location-scale `z ~ N(Xm b, exp(Xs g)²)`. This is fitted by weighted interval-censored maximum
    likelihood, so atoms and out-of-table outcomes are handled exactly.
  - The combined quantile at level τ is
    - `Q_base(F0 · τ / p0)` when τ < p0, which lands in the base's region at or below 0;
    - otherwise `Q_base(F0 + (1 − F0) Φ(μ + s Φ⁻¹((τ − p0) / (1 − p0))))`.
  - b = g = d = 0 gives back the base exactly. The ridge priors are centred there.
- **Level-1 data (cross-fitting).** Each training week is scored by a base fitted on every *other*
  training week (blocked leave-one-week-out, all before the test week). These PITs, and the
  player-history signals built from them, are the stacking level-1 data.
- **Signals (`signals.py`).** Each is computed causally week by week from `train_df`:
  - player EWMA of past probit-PIT residuals and of excess zeros (`1{y=0}` − base atom), with
    half-life 6 weeks, shrinkage 3 pseudo-games and 8 extra weeks across the season break;
  - `lneff`, the amount of history;
  - projection change vs the last projected game, a first-appearance flag, and team games missed
    since that game;
  - snap-share EWMA (snaps / team max);
  - opponent × position points allowed in PIT units (half-life 8 weeks, shrinkage 10);
  - team projected total and spread proxy (`team_proj − opp_proj`);
  - depth rank 2 / 3+;
  - the projection's stat mix: TD share, reception share, rush-yard share;
  - log projection.

  In `stack`, the signals enter the zero-part logit and the positive-part location with shared plus
  per-position slopes, and the positive-part log-scale with shared slopes. They are standardized on
  the training rows.
- **Stacking weight selection.** The ridge strength on all signal slopes is set per origin by
  forward-only inner validation. Each multiplier in {1, 3, 10, 30, 100, 1e4} is scored by mean CRPS
  over the last 4 training weeks, each fitted on the rows of earlier weeks only. The largest one
  effectively means "no signals". Training rows have recency weights with a 20-week half-life.

**Why.**

- The auditor found the baseline's zero mass too small (P(y ≤ 0) 0.128 predicted vs 0.186
  observed). That is exactly the quantile-averaging effect described by Lichtendahl et al.: averaging
  quantiles of two buckets keeps the smaller atom. The linear pool averages the atoms, and the
  stacked logit then corrects what is left.
- A first look showed that point-forecast stacking cannot help. A rolling ridge of actual on proj,
  the player EWMA of actuals and residuals, and opponent residual gave the same MSE as proj alone
  (30.94 vs 30.92 over 2025-05..2026-07), so Sleeper's mean is efficient.
- The signals do carry information about the *shape*. Within position × projection decile:
  - the player's past zero rate is rank-correlated 0.13-0.27 with this week's zero;
  - QB depth rank 2 correlates 0.54 with a zero;
  - snaps and past residuals shift TE/RB outcomes (about 0.1).

  So the combination is done on the distribution, in PIT space, rather than on the mean.

**Sources.**

- Lichtendahl, Grushka-Cockayne & Winkler (2013), "Is it better to average probabilities or
  quantiles?", *Management Science* 59(7):1594-1611. Quantile averaging vs the linear pool, used for
  the bucket blend.
- Gneiting & Ranjan (2013), "Combining predictive distributions", *Electron. J. Statist.*
  7:1747-1782, https://doi.org/10.1214/13-EJS823. Linear-pool dispersion and the beta-transformed
  linear pool; the recalibration here is a probit-normal, covariate-dependent cousin of the BLP.
- Yao, Vehtari, Simpson & Gelman (2018), "Using stacking to average Bayesian predictive
  distributions", *Bayesian Analysis* 13(3):917-1007, https://arxiv.org/abs/1704.02030. Stacking
  weights from out-of-sample proper scores; the reason for the inner CRPS validation.
- Sill, Takács, Mackey & Lin (2009), "Feature-weighted linear stacking",
  https://arxiv.org/abs/0911.0460. Blend weights that depend on meta-features.
- Nowotarski & Weron (2015), "Computing electricity spot price prediction intervals using quantile
  regression and forecast averaging", *Computational Statistics* 30(3):791-803 (QRA). Considered;
  linear QR on raw quantiles handles the zero atom badly, so a PIT-space model was used instead.
- Kuleshov, Fenner & Ermon (2018), "Accurate uncertainties for deep learning using calibrated
  regression", ICML, https://arxiv.org/abs/1807.00263. PIT recalibration of a regression CDF.
- Bassetti, Casarin & Ravazzolo (2018), "Bayesian nonparametric calibration and combination of
  predictive distributions", *JASA*, https://arxiv.org/abs/1502.07246.
- Wolpert (1992) and Breiman (1996). Stacked generalization, with cross-validated level-1 data.
- Gneiting, Raftery, Westveld & Goldman (2005), EMOS, *Mon. Wea. Rev.* 133:1098-1118. Location and
  scale both depend on predictors.

**Variants** (each one nests the previous):

| variant | what it adds | fitted per origin |
|---|---|---|
| `lpool` | linear pool of neighbouring bucket tables (the atom fix) | the tables only |
| `recal` | two-part recalibration by position and log projection: no information beyond Sleeper | 9 logit, 9 location, 8 log-scale coefficients |
| `stack` | the same, stacked on the 16 point-in-time signals; ridge strength chosen by inner validation | about 190 coefficients, heavily shrunk |

`recal` is the ablation for `stack`. It shows how much of the gain is calibration alone, which
other candidates also get, and how much comes from the signals.

**Leakage.**

- Only `train_df` and `test_df` are used. Nothing reads `data/`, `common.League` or `D.table()`.
- The cross-fitted PITs use other *training* weeks, which come both before and after the scored
  week but always before the test week. This is the stacking level-1 data. No test-week outcome
  enters.
- The inner penalty validation is forward-only in the rows it fits on (weeks before the validation
  week), but the validation week's base and the earlier rows' PITs and player-history features come
  from the leave-one-week-out bases, which also saw training weeks after the validation week (never
  the test week). So it is "before the test week", not strictly forward within training.
- Reviewer caveat (researcher degrees of freedom): the 16-signal list was picked after an exploratory
  within-bucket correlation scan over all items through 2026-07, and the LOWO switch and the wider
  penalty grid came after one smoke run (smoke weeks 2025-10, 2026-03, 2026-07 = blocks t 9, 20, 24
  are inside the full set). `lpool` and `recal` do not depend on the scan. Read `stack`'s full-run
  gain as an upper bound, and check it with those three blocks left out.
- The run is deterministic: no RNG, L-BFGS from zero. The npz files are byte-identical between
  `cpuq -n 1` and `-n 2`.

**Design history** (honest account):

1. I first used a location-scale probit on the full PIT. It cannot move the zero atom and the
   positive part independently, so I replaced it by the two-part form before trusting any number.
2. Inner rolling-origin PITs were replaced by cross-fitted ones after a training-only check. On
   2026-07's training weeks, the QB probit sd of the rolling PITs is 1.076 overall and 1.013 over the
   last 10 weeks, against 0.998 cross-fitted. The early bases (2-5 weeks of data) are much noisier
   than the final base, so they taught `recal` to over-widen QBs.
3. The penalty grid was first 0.3-10. The training-only validation curve kept choosing its top edge
   (at 2025-08 it falls monotonically up to 1e5), so it was widened.

Changes 2 and 3 were made after one smoke run. Each was justified by a training-only diagnostic,
but the smoke weeks are part of the full set.

**Smoke** (3 weeks, 908 items; 6 s on 1 core). The baseline is on the same split.

| | crps | pinball | cov10/50/90 | cov80 | cov_err | pit_maxdev | width80 | P(q ≤ 0) |
|---|---|---|---|---|---|---|---|---|
| baseline | 2.7708 | 1.3854 | 0.118/0.497/0.884 | 0.767 | 0.0176 | 0.0222 | 12.10 | 0.143 |
| `lpool` | 2.7705 | 1.3852 | 0.105/0.496/0.883 | 0.779 | 0.0118 | 0.0222 | 12.11 | 0.180 |
| `recal` | 2.7729 | 1.3864 | 0.101/0.493/0.887 | 0.785 | 0.0090 | 0.0222 | 12.31 | 0.191 |
| `stack` | 2.7621 | 1.3811 | 0.104/0.493/0.894 | 0.791 | 0.0063 | 0.0233 | 12.27 | 0.190 |

The observed share of y ≤ 0 on the smoke split is 0.191. Weekly crps for `stack` is 2.8261, 2.7365
and 2.7268, against the baseline's 2.8261, 2.7373 and 2.7539. The paired difference vs the baseline
is −0.0087, with a 3-block CI of [−0.0271, −0.0000]; with only 3 blocks, treat it as a plumbing
check.

**Runs.**

- `run_smoke.sh`: about 6 s on 1 core.
- `run_full.sh`: 21 origins × 3 variants. `stack` takes about 2-3 s per origin at the largest
  training sets, so the run should take about 1 minute on 6 cores. It is queued only.
