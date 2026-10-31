# hier_bayes (player track)

Hierarchical / empirical-Bayes player effects on top of a hurdle version of the baseline. Module
`hb.py`, class `HierBayes(variant)`. The three variants run with
`bench/cpuq -n 1 -- bash bench/candidates/hier_bayes/run_smoke.sh` (smoke) and
`run_full.sh` (full, queue only).

## What

1. **Base (the position x projection prior).** The baseline's own buckets (position, equal-count
   projection buckets, `nb = clip(n // 300, 5, 14)`). Each bucket is split at the atom:
   - its zero share `P(y <= 0)`,
   - a 256-level quantile table of `actual / proj` among the positive scores only.

   Both are interpolated in log projection between bucket medians, exactly as the baseline
   interpolates its tables. The atom is therefore interpolated explicitly. The baseline's mixed
   tables keep only the smaller of the two neighbouring atoms: its predicted `P(y <= 0)` is 0.143
   on smoke against 0.191 observed, and this base gives 0.186. `y | y <= 0` uses the position's
   pooled empirical non-positive scores, mostly an exact `0.0`.

2. **Effects** (variants `eb` and `dlm`). Every effect is fitted by empirical Bayes on
   decay-weighted history, using only the training frame:
   - **Zero propensity.** A random-intercept logistic model:
     `logit p = logit p0_base + b_player`, with `b ~ N(0, sig^2)`.
     - `sig` is the maximum of a 16-node Gauss-Hermite marginal likelihood over the
       undiscounted history.
     - The player posterior uses the decay-discounted likelihood. The predictive `P(y <= 0)` is
       integrated over that posterior by quadrature.
   - **Location.** Crossed normal-normal random effects on the probit PIT of positive scores
     under the base, `z = Phi^-1(F_pos(y))`:
     `z = player + team-season + opponent-season-position + eps`.
     - Each `tau^2` is the marginal-likelihood estimate from the groups' weighted means, with
       sampling variance `s2 * sum w^2 / (sum w)^2`.
     - Each effect is the James-Stein-type shrunk mean `tau^2 / (tau^2 + v) * mean`.
     - Fitted by 3 backfitting sweeps.
   - **z-scale level.** The residual variance of z by position. The interpolated base is a
     little too narrow even in-sample: z variance runs 1.06-1.18 for QB and about 1.0-1.1 for
     the other positions.
   - **Player scale** (`dlm` only). Per-player weighted residual variances are squeezed toward
     an inverse-gamma prior with limma's `fitFDist`/`squeezeVar` (log-moment matching for the
     prior df `d0` and scale `s0^2`). The residuals are first divided by their position's
     z-scale level, and the squeezed variances are normalized so their row-weighted mean is 1 within
     each position, so the position level comes only from the z-scale level (review fix, below).
   - **Predictive.** `z ~ N(m, s^2)`, where `s^2` includes the effects' posterior variances.
     Quantile at level tau:
     - above the atom, `q(tau) = proj * R_pos(Phi(m + s * Phi^-1(v)))` with
       `v = (tau - p0) / (1 - p0)`;
     - below the atom, the non-positive quantile at `tau / p0`.

3. **Time decay.** Each effect type weights its rows by `2^(-age / h)`.
   - Player rows from the previous season are aged by 8 more weeks.
   - Team and opponent effects are keyed by season, so they reset every offseason.

## Variants

| name | what changes |
|---|---|
| `hurdle` | Base only, no player identity. This is the control for the atom fix. |
| `eb` | Base + zero propensity + crossed location effects + per-position z-scale. All half-lives are fixed a priori at 8 weeks. |
| `dlm` | `eb` + limma player-scale squeeze. Per origin, it picks the half-lives (player in {4, 8, 16, inf}, team/opp in {4, 8, inf}, zero in {2, 4, 8, 16, inf}) and the variance-prior multiplier `d0 x {1, 3, 10, inf}` by one-step-ahead (prequential) likelihood over the last 16 training weeks. Each of those weeks is predicted only from the rows before it. This is the discount-factor reading of a local-level state-space model: exponential forgetting is the steady state of a local-level filter, and the forgetting rate is estimated rather than assumed. |

Everything is deterministic: there are no random draws. Each origin fits only on `train_df`.
Hyperparameters (`tau^2`, `sig`, `d0`, `s0^2`, z-scale and the `dlm` choices) are re-estimated
inside every fit. No `data/*.json` or `common.League` access. A `dlm` fit takes ≤ 1.5 s. The
whole full run (63 tasks) should take well under a minute on 6 cores.

## Why it might beat the baseline

Before designing anything, I measured persistence on 2025 only, using rolling one-step-ahead
correlations of shrunk past means with the next outcome, relative to the baseline PIT:
- per-player zero excess (`1[y <= 0] - p0`): 0.09 (WR 0.13). The split-half correlation was 0.26.
- team-season z: 0.06 (QB 0.12). The split-half correlation over teams was 0.55.
- opponent x position z: 0.05.
- player |z| (volatility): 0.03.
- player z (bias): 0.02.
- lag-1 autocorrelation of a player's z: 0.02.

So Sleeper's projection already absorbs the player's level. The usable hierarchy is the zero
propensity and the team and opponent context, plus the base's own calibration (atom and scale).
EB agrees: inside the fits, the player-location `tau^2` is 0 at every origin checked. Team
`tau^2` is about 0.005-0.015 (sd 0.07-0.12 in z), opponent `tau^2` is 0-0.015, and the zero
`sig` is about 0.3-0.47 on the logit scale. Year-over-year "consistency" is known to be weakly
sticky (sources below), so heavy shrinkage of player scale is expected.

## Dev results (design only: rolling origin on 2025 weeks 5-18, in-process)

These are CRPS differences against the baseline (2.7989), with a block bootstrap over the 14
weeks.

| config | Δcrps | 95% CI |
|---|---|---|
| hurdle, bucket atom (= `hurdle`) | -0.0005 | [-0.0015, +0.0003] |
| hurdle, smooth logistic atom (dropped) | +0.0089 | [+0.0043, +0.0139] |
| + zero propensity, gamma-Poisson (Clayton-Kaldor) | ±0 | `a` → ∞: the Poisson variance hides the heterogeneity under Bernoulli under-dispersion, so I replaced it |
| + zero propensity, logistic-normal | -0.0021 | [-0.0031, -0.0010] |
| + location: player only / team only / opp only | -0.0018 / -0.0057 / -0.0019 | |
| + player scale, limma d0 x1 / x3 | +0.0027 / -0.0003 | |
| + nested team x position effect (dropped) | 0 | `tau^2` = 0 |
| **eb** | **-0.0089** | [-0.0141, -0.0036] |
| **dlm** | **-0.0072** | [-0.0132, -0.0009] |

The gains are small but consistent: 11 of 14 dev weeks improve, and they grow late in the
season as history accumulates. The prequential choices in `dlm` jump between grid points from
origin to origin (team h 4 → inf → 4), and so far that costs a little against the fixed
half-lives.

## Smoke (908 items: 2025-10, 2026-03, 2026-07; `bench/cpuq -n 1`, 4.5 s)

| | crps | pinball | cov10/50/90 | cov80 | cov_err | pit_maxdev | P(y<=0) pred (obs 0.191) |
|---|---|---|---|---|---|---|---|
| baseline | 2.7708 | 1.3854 | 0.118/0.497/0.884 | 0.767 | 0.0176 | 0.0222 | 0.143 |
| hurdle | 2.7690 | 1.3845 | 0.107/0.498/0.885 | 0.779 | 0.0112 | 0.0211 | 0.186 |
| eb | 2.7641 | 1.3821 | 0.106/0.491/0.884 | 0.779 | 0.0129 | 0.0163 | 0.190 |
| dlm | 2.7604 | 1.3802 | 0.106/0.488/0.885 | 0.780 | 0.0132 | 0.0189 | 0.189 |

## Leakage notes and caveats

- All fitting and tuning happens inside each origin's `fit(train_df)`. The `dlm` choices use only
  training weeks.
- The design choices (which effects to keep, the a-priori 8-week half-life, the season reset for
  team effects, the bucket atom over the logistic atom) were made on the 2025 dev rolling
  origins. Those weeks are also test blocks of the full run. That is meta-level selection over a
  handful of coarse options, not parameter tuning, but read the 2025 blocks with that in mind.
  The 2026 weeks 1-7 were not used for design. The one exception is a first debug pass on the
  smoke split (2026-03 and 2026-07), run before the dev ablations.
- The effects are keyed by `pid`, `team` and `opp`, which are point-in-time on every item
  (README leakage section). `pos` is current metadata, as for every candidate.

## Sources

- Efron & Morris (1975), "Data analysis using Stein's estimator and its generalizations", JASA 70.
  This is the James-Stein / EB shrinkage of player averages.
- Brown (2008), "In-season prediction of batting averages: a field test of empirical Bayes and
  Bayes methodologies", Ann. Appl. Stat., https://arxiv.org/abs/0803.3697.
- Pericchi & Perez, heavy-tailed priors for robust EB player shrinkage (NEJSDS),
  https://nejsds.nestat.org/journal/NEJSDS/article/116/read.
- Smyth (2004), "Linear models and empirical Bayes methods for assessing differential expression
  in microarray experiments", SAGMB 3. This is `squeezeVar`/`fitFDist` (inverse-gamma variance
  prior). See also Phipson et al. (2016), robust hyperparameter estimation,
  https://arxiv.org/abs/1602.08678, and Lu & Stephens (2016), vash,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5181563.
- Clayton & Kaldor (1987), "Empirical Bayes estimates of age-standardized relative risks",
  Biometrics 43 (gamma-Poisson; tried, then replaced by the logistic-normal GLMM).
- Glickman (1999), "Parameter estimation in large dynamic paired comparison experiments", JRSS C 48,
  https://ideas.repec.org/a/bla/jorssc/v48y1999i3p377-394.html. Also West & Harrison (1997),
  *Bayesian Forecasting and Dynamic Models* (discount factors, local-level models).
- Egidi & Gabry (2018), "Bayesian hierarchical models for predicting individual performance in
  soccer", JQAS, https://arts.units.it/handle/11368/2929586.
- On how weakly weekly fantasy consistency carries from one year to the next:
  https://www.footballguys.com/article/2026-dynasty-in-theory-week06-13-theses-on-variance and
  https://www.playerprofiler.com/article/is-wide-receiver-consistency-predictable/.

## Independent review (before queueing)

- **Leakage.** No `data/*.json`, `League`, `P_OUT` or `inj` access. Every fit, hyperparameter and
  `dlm` choice uses only `train_df` (prequential evals use rows `t < T` of training weeks). The
  test frame carries no actual columns. Predictions are deterministic and invariant to the test
  rows' order (checked).
- **Fix: player-scale normalization (`dlm`).** `rel = post / s0^2` double-counted the position
  level (QB residual variance is about 1.1-1.2 against a global `s0^2`) and averaged above 1,
  because the mean of an inverse-gamma-squeezed variance exceeds its log-moment scale. At
  `d0 x 1`, seen players' predictive variance was inflated by 3-26% (by position, at 2025 weeks
  5-8). The fix divides residuals by the position's z-scale level before squeezing, then
  normalizes `rel` to a row-weighted mean of 1 within each position. It changes only the origins
  where the fitted `d0` is finite (2025 weeks 5-8). From 2025 week 9 on, `ev <= 0`, so `d0 = inf`
  and the player-scale term is a no-op. The smoke metrics are unchanged.
- **Holdout split.** The design choices were made on 2025 weeks 5-18, which are 14 of the 21
  full test blocks. `holdout.py` therefore runs after the backtest and adds a `holdout` field
  to the result JSON. It holds paired CRPS and pinball differences vs the baseline, split into
  `2025_design` and `2026_holdout` blocks. Read the 2026 split as the cleaner out-of-sample
  evidence.
