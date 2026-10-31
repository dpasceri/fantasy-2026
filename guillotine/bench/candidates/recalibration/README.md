# recalibration (team track)

**What.** This candidate keeps the team baseline's forecast exactly as it is (`_baseline_team.copula.CopulaSim`:
model.py marginal plus the team/game copula, refit per origin, with 96000 CRN draws of the test week). It then
applies post-hoc maps to three outputs: the chop probabilities, the team-total quantiles and the chop-line
quantiles. The maps are fit per origin on a **calibration set** made only of virtual weeks s < origin:

- For each such week s, the baseline is refit on every training week **except** s. This is
  leave-one-week-out cross-fitting, so it uses only data from before the origin, and week s stays out of
  sample.
- That refit is simulated on week s's pre-game frame (`S_CAL` = 48000 draws, seed `(11, season, week)`).
- Each of week s's virtual leagues then contributes:
  - its `p_chop`,
  - the randomized PIT of every team's actual total,
  - the PIT of the actual chop line (the week's minimum).

Real leagues are never used for calibration. The real test weeks get the same maps.

With identity maps the output equals the baseline's (`verify.py`):
- `total_q` and `line_q` match bit for bit;
- `p_chop` matches except that zero-count teams get the floor `FLOOR` = 1/96000 before renormalizing.

**Why.** It is the cheapest test of whether the baseline is over- or under-confident:
- a temperature `beta` < 1 means its chop probabilities are too sharp, `beta` > 1 too timid;
- the level maps l(0.1) / l(0.5) / l(0.9) show the direction of any bias in the team-total and chop-line
  quantiles.

The maps add only 1-5 parameters, fit on hundreds to thousands of out-of-sample leagues. So they can fix a
systematic miscalibration without the overfitting risk of a new model. They cannot add skill (ranking): with
`temp`, `rank_norm` cannot move.

Leagues of the same week share NFL outcomes, so their errors are correlated. The chop line in particular is
close to one draw per week. To keep a few weeks from dominating, every fit is shrunk toward the identity using
the week-cluster effective sample size:
- **Quantile maps:** weight n_eff / (n_eff + 20), where n_eff is the Kish effective size from the one-way
  ANOVA ICC of the probit PITs.
- **Chop maps:** the log-likelihood is divided by the design effect of its score. The design effect is
  tr(J_week) / tr(J_league), which makes this a magnitude-adjusted composite likelihood. The prior is
  N(identity, 0.25²).

## Variants

| Variant | Chop map (`p~ = max(p, FLOOR)`) | Team-total and chop-line quantiles |
|---|---|---|
| `temp` | temperature scaling: p' ∝ p~^beta (1 parameter) | baseline (unchanged) |
| `tmix_qq` | p' = (1-eps)·softmax(beta·log p~) + eps/N: temperature plus a uniform floor against tail surprises (2 parameters) | probit level map: l(tau) = Phi(a + b·Phi⁻¹(tau)), with (a, b) the normal-QQ line of the calibration PITs' probits (2 parameters each for totals and line) |
| `clogit_dcp` | conditional logit, i.e. Dirichlet calibration with exchangeable classes plus covariates: softmax(b0·log p~ + b1·log p~·(N-13.5)/3.5 + g1·zproj + g2·dlogsd), then the eps floor (5 parameters). `zproj` is the team's projected total as a z-score within its league; `dlogsd` is the log of its simulated sd minus the league mean. | distributional conformal / Kuleshov recalibration: l(tau) is the empirical tau-quantile of the calibration PITs, which is the pinball-optimal level for each tau |

For both quantile maps, the new quantile is read from the test week's simulated draws at level l(tau), with no
interpolation of the baseline's 99 quantiles. Each map moves only its own metrics:
- the chop map moves `logloss`, `brier`, `rank*` and `ece`;
- the totals map moves `crps_total` and `cov*_total`;
- the line map moves `crps_line`.

So the gains can be attributed per metric even though `tmix_qq` and `clogit_dcp` change all three outputs.
The fitted maps of every (variant, origin) are dumped to `bench/results/recalibration/params_<mode>.json`
by `params.py`, and the per-task lines `[recal] {...}` appear in the log. Those dumps answer the
over/under-confidence question.

## Sources

- C. Guo, G. Pleiss, Y. Sun, K. Weinberger, *On Calibration of Modern Neural Networks*, ICML 2017.
  Temperature scaling. https://arxiv.org/abs/1706.04599
- M. Kull et al., *Beyond temperature scaling: well-calibrated multiclass probabilities with Dirichlet
  calibration*, NeurIPS 2019. Here: log-probability features through a linear layer and softmax; temperature
  scaling is the scaled-identity special case. https://arxiv.org/abs/1910.12656
- V. Kuleshov, N. Fenner, S. Ermon, *Accurate Uncertainties for Deep Learning Using Calibrated Regression*,
  ICML 2018. Recalibrating a predictive CDF through the empirical CDF of its PITs.
  https://arxiv.org/abs/1807.00263
- V. Chernozhukov, K. Wüthrich, Y. Zhu, *Distributional conformal prediction*, PNAS 2021. PIT ranks as
  conformity scores, valid under misspecification and time-series data. https://arxiv.org/abs/1909.07889
- Y. Romano, E. Patterson, E. Candès, *Conformalized Quantile Regression*, NeurIPS 2019.
  https://arxiv.org/abs/1905.03222
- T. Gneiting, A. Raftery, A. Westveld, T. Goldman, *Calibrated probabilistic forecasting using ensemble
  model output statistics and minimum CRPS estimation*, MWR 2005. Post-processing a Monte Carlo ensemble's
  location and spread. https://sites.stat.washington.edu/raftery/Research/PDF/gneiting2005.pdf
- R. Ranjan, T. Gneiting, *Combining probability forecasts*, JRSS-B 2010. Parametric (beta) maps of the PIT
  scale; the probit map here is the Gaussian analogue. https://ideas.repec.org/a/bla/jorssb/v72y2010i1p71-91.html
- A. Rumack, R. Tibshirani, R. Rosenfeld, *Recalibrating probabilistic forecasts of epidemics*, 2021. PIT
  recalibration from retrospective forecasts; the log-score gain equals the PIT entropy.
  https://arxiv.org/abs/2112.06305

## Leakage

- Everything is fit inside the harness's `horizon(origin)` and uses only:
  - `train_df` (rows with t < origin);
  - the training league views, including their outcomes;
  - `data.week_frame(..., pre=True)` of weeks s < origin.
- The cross-fit refit for week s uses the training weeks after s as well, but all of them are before the
  origin. This is ordinary K-fold inside the training window and is allowed by the rolling-origin rule.
- The test week is touched only through the league views and `week_df`, both pre-game.
- The only state shared across tasks is the calibration set, and it is shared only among the 3 variant
  tasks of the **same origin** (`cached_calibration_set`). The set never depends on the variant, so the
  first task of an origin builds it under a file lock in `$RECAL_CACHE` and the other two load it.
  The cache key hashes this module's and the baseline's code, K/R/S, the training rows' t/pid/actual and
  the training league ids, so another origin can never hit it. The run scripts clear the directory
  (`.calcache_<mode>/` here) before and after each run. `verify.py` check 5 shows that a cached set is
  bit-identical to a rebuilt one, and the smoke output is bit-identical with and without the cache.
- The per-task parameter dumps are written, never read.
- Seeds:
  - sim draws are seeded by `(11, season, week)`;
  - PIT tie-breaking uniforms are a blake2b hash of the league id (`metrics.item_uniform`);
  - the optimizer is deterministic.

## Compute

- Per (variant, origin) task on 1 core, measured: about 1.1 s per calibration week plus about 8 s for the
  base fit and the test week. The heaviest full task (origin 2026-07, 24 calibration weeks, 1152 leagues)
  took 31 s; the lightest (2025-05, 4 weeks) took 8 s.
- The calibration set is built once per origin and shared by the 3 variants (see Leakage). That gives
  about 500 s of task time for the first variant plus about 180 s for each of the other two, about
  860 s in all. Rebuilding the set in every variant would cost about 1500 s.
- On 6 cores, with the ~2.7x per-task slowdown the team baseline showed under 6-way load (12 s per task
  vs 4.5 s solo), the full run should take about 6-8 min, well inside the 15 min budget. Memory is
  about 0.7 GB per worker (smoke max RSS 650 MB).
- Smoke (`RECAL_K=2` most recent weeks, `RECAL_R=16` replicates per week, to fit in 90 s): 33 s on 1 core with the shared calibration set (39-77 s before it); the output is identical on every run.
  The smoke calibration sets are tiny: 32 leagues from 2 weeks, where the ICC is close to unidentifiable. So
  smoke numbers test the pipeline, not the method. The quantile maps in particular are near noise there.

## Smoke result (50 items, the same split as `_baseline_team.smoke.json`)

| | virtual logloss | brier | rank_norm | crps_total | crps_line | ece | cov10/50/90 total | real logloss |
|---|---|---|---|---|---|---|---|---|
| baseline | 2.0761 | 0.8250 | 0.229 | 10.943 | 5.545 | 0.0241 | 0.076/0.454/0.894 | 1.851 |
| `temp` | 2.0736 | 0.8253 | 0.229 | 10.943 | 5.545 | 0.0271 | 0.076/0.454/0.894 | 1.826 |
| `tmix_qq` | 2.0777 | 0.8256 | 0.229 | 10.865 | 5.641 | 0.0243 | 0.090/0.480/0.914 | 1.826 |
| `clogit_dcp` | 2.0770 | 0.8273 | 0.229 | 10.870 | 5.610 | 0.0190 | 0.088/0.471/0.910 | 1.767 |

Paired against the baseline (leaderboard, 3 blocks):
- `temp`: logloss −0.0025 [−0.015, +0.014].
- `tmix_qq`: crps_total −0.077 [−0.129, −0.046].
- `clogit_dcp`: crps_total −0.073 [−0.124, −0.035].
- crps_line is worse in smoke (+0.10 / +0.06). That comes from a 2-week line map.

A within-training diagnostic at the last origin (2026-07, the calibration set of all 24 earlier weeks,
1152 leagues) shows the baseline is already close to calibrated:
- **Chop:** temperature b0 = 1.007, with deff 1.24 and eps = 0. The calibration log-loss goes from 2.2358 to
  2.2337.
- **Team-total PITs:** coverage 0.096/0.490/0.898, ICC 0.09.
- **Chop-line PITs:** coverage 0.095/0.497/0.897, ICC 0.26, n_eff 88 of 1152.

Expect small effects in the full run.

## Files

- `recal.py`: the candidate (`Recal`), its maps (`ChopMap`, `LevelMap`, `n_eff`), the transposed draws
  (`Draws`), `calibration_set` and its per-origin cache `cached_calibration_set`.
- `verify.py`: runs checks 1-5.
  1. `Draws` equals `WeekDraws`.
  2. Identity maps reproduce the baseline.
  3. The maps recover known synthetic miscalibration (beta 0.7 and eps 0.03; PIT shift 0.2 and sd 1.3), and
     n_eff collapses for clustered PITs.
  4. Calibration weeks are all before the origin.
  5. The per-run calibration-set cache returns exactly the uncached set.

  Run it with `bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates.recalibration.verify`.
- `params.py`: gathers the per-origin map dumps.
- `run_smoke.sh`, `run_full.sh`.
