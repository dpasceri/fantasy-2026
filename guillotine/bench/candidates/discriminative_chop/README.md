# discriminative_chop (team track)

## What

This candidate learns who gets chopped directly. Each league-week is a choice set, and the "chosen"
team is the one with the lowest total. A model on team features is fitted on every training league
before the origin: the 48 virtual replicates per week plus the real weeks. Only `p_chop` changes.
`total_q` and `line_q` come from the baseline sim itself: `CopulaSim` refit on all train items, with
96000 draws and rng `(11, season, week)`. They are bit-identical to `_baseline_team`, which was checked
on the smoke npz, so `crps_total` and `crps_line` have a paired difference of exactly 0.

**Team features**, one row per (league, team), all pre-game:

| Group | Features |
|---|---|
| Sim | `lp` = log P_sim(min), floored at 1e-4; mean, sd, q05 and q10 of the team total |
| Structure (starters) | TD share of projected points (pass_td × 4, rush/rec_td × 6 from `ps_*`); reception share; slots projected < 5; same-NFL-team pairs (stacks); opposing-team pairs in the same game; offense environment = proj-weighted `team_proj` |
| History (train items with t < s only) | Sum of the starters' shrunk projection bias Σr/(n+8); excess volatility: variance-weighted (vol − 1), with vol = (Σe² + 8)/(n + 8) and e² = r² / (c_pos (proj + 2)) |

The models see within-league relative versions of these: the gap to the league's lowest mean in
units of mean sd, its square, log(sd / mean sd), the q10 gap, and centered structural features.

**Cross-fitting.** For training week s, the sim features come from the baseline's marginal tables and
copula refit on every train item with t < origin except week s. That week's sim inputs therefore
never contain its own outcomes, and the feature quality matches the test week's, which uses all
weeks before the origin. Training draws: 8000 per week, rng `(13, season, week, chunk)`.

**Plackett-Luce stages.** League l's stage j is the bottom-up event "the j-th lowest realized team is
the minimum of the teams still in", with weight 0.5^j. Its offset is the sim's log P(min of that
set), recomputed from the draws for each stage, so there is no Harville shortcut. K = 1 is the plain
conditional logit on the chopped team. K ∈ {1, 3} is chosen by CV.

**Tuning** is rolling-origin and done inside `fit`:
- Folds: week-grouped, round-robin over the training weeks, at most 5 folds (leave-one-week-out at
  the first origin, which has 4 training weeks).
- Score: held-out top-1 chop log-loss per league, the same as the track's primary metric.
- Grid: ridge λ ∈ {0.3 … 1e5, ∞}, where ∞ means the prior exactly; ties go to more shrinkage.
- The chosen setting is refit on all training leagues.
- Every fitted number, including the variant's hyperparameters, uses only leagues and rows before
  the test week.

## Variants

| Name | Model |
|---|---|
| `clogit` | **Pure discriminative.** Ridge conditional / rank-ordered logit on the gap, gap², log sd ratio, q10 gap and the structural and history features. It does not use log p_sim. Prior β = 0, which is uniform. |
| `benter` | **Hybrid (Benter's second stage).** u = a · log p_sim + b · u_clogit, where u_clogit is `clogit`'s **out-of-fold** utility on each training league. Two coefficients, ridge toward (a, b) = (1, 0), λ by CV; λ = ∞ is the sim alone. |
| `gbm` | **Hybrid, boosted.** LightGBM with a custom softmax-per-league objective (grad w(p − y), hess w·p(1 − p)), init score = the stage offset log p_sim, depth-2 trees. It uses lp, relative sim features, the structural and history features, N and the team's rank by mean. (K, rounds ∈ {0, 5, …, 100}) by CV; 0 rounds is the sim. Deterministic, `num_threads = task_threads()`, `seed = task_seed()`. |

## Why it might beat the baseline

The baseline's P(chop) is a pure model forecast. Its marginal is bucketed by position and projection
level only, it ignores the composition of the projection (TD-dependent players are more volatile) and
player history, and it uses a one-factor copula.

A discriminative layer can:
1. correct systematic miscalibration of the sim's chop probabilities, such as temperature or the
   sensitivity to mean gap and sd;
2. add information the sim does not model.

Benter's result for horse racing is the template here. A fundamental model added to a strong
"market" forecast through a two-coefficient logit improved out-of-sample fit even though the
fundamental model alone was worse than the market. In our case the sim plays the market.

The risk is small data. There are 192 to 1158 training leagues per origin, and the replicates of one
week share the same player outcomes. Every variant therefore shrinks toward the baseline, and CV may
return exactly the baseline.

## Sources

- Bolton, R. N. & Chapman, R. G. (1986). Searching for positive returns at the track: a multinomial
  logit model for handicapping horse races. *Management Science* 32(8).
- Benter, W. (1994). Computer based horse race handicapping and wagering systems: a report. In
  *Efficiency of Racetrack Betting Markets* (World Scientific, 2008 ed.), ch. 19. The second-stage
  combination is the `benter` variant.
  https://www.gwern.net/doc/statistics/decision/1994-benter.pdf ; summary of the second-stage test:
  https://oddspapi.io/blog/?p=3174
- Beggs, S., Cardell, S. & Hausman, J. (1981). Assessing the potential demand for electric cars.
  *J. Econometrics* 16. Hausman, J. & Ruud, P. (1987). Specifying and testing econometric models for
  rank-ordered data. *J. Econometrics* 34. These cover the rank-ordered (exploded) logit, the
  Plackett-Luce stages, and down-weighting deeper ranks. See Stata `rologit`:
  https://www.stata.com/manuals15/rrologit.pdf
- Lo, V. S. Y. & Bacon-Shone, J. (1994). A comparison between two models for predicting ordering
  probabilities in multiple-entry competitions. *The Statistician* 43. They found Harville-type
  stage probabilities biased under normal "running times", which is why the stage offsets here are
  recomputed from the sim. https://hub.hku.hk/handle/10722/86405
- Cao, Z. et al. (2007). Learning to rank: from pairwise approach to listwise approach (ListNet,
  top-one softmax loss). ICML. This is the `gbm` objective.
- Shi, H. & Yin, G. (2018). Boosting conditional logit model. *J. Choice Modelling* 26. R package
  `clogitboost`: https://cran.r-project.org/web/packages/clogitboost/
- LightGBM custom objectives (callable `objective`; preds include `init_score`, checked):
  https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html

## Files and caching

- `features.py`: the cross-fitted training set and the team features.
- `models.py`: `ridge_cv`, `CLogit`, `Benter`, `Boost`.
- `candidate.py`: `DiscriminativeChop(variant)`.

**Caching.** The training set (the expensive part) is cached in
`bench/cache/discriminative_chop/train_t<tmax>_<key>.pkl`. `key` = sha1 of every `.py` file of this
candidate (so `candidate._sim`, which writes the test entries, is covered), the baseline sampler
(`_baseline_team/copula.py`) and marginal-table module (`_baseline_player/ratio.py`, including
`ratio_grid`), the core constants they use (`model.Q`, `model.CORR_MIN_PROJ`, `sim.CHUNK`,
`metrics.LEVELS`), the league scoring, `CFG`, the whole `train_df` (`pd.util.hash_pandas_object`) and
every training league's ids, starters and points.

Each test league's baseline-sim summary and raw features are cached in `test_<key>/<league_id>_<h>.pkl`,
where h covers the week frame and starters. Each test file holds pre-game data only: the sim
quantiles and the features, no outcomes.

A cache entry depends only on data before its origin, and a task only reads entries for its own
training set. The cache just lets the 3 variants of an origin share one computation; a flock means
the first task computes and the others wait. Cold and warm smoke runs give bit-identical npz files
(checked). `DC_NOCACHE=1` disables the cache, and the folder can be deleted at any time. The
independent review reran the smoke with a cold cache and with `DC_NOCACHE=1`: all three variants'
npz files are bit-identical to each other and to the cached run.

## Runtime

- Smoke on 1 core: 36 s with a cold cache, 11 s warm.
- Full: about 0.55 s per training week per origin (294 origin-weeks), plus the baseline's test sim
  (about 8 s per origin) and the model fits (about 6 s per origin over the 3 variants). That comes to
  about 550 core-seconds, or roughly 2-4 minutes on 6 cores.

## Smoke results

Same split as `_baseline_team.smoke.json`: 48 virtual and 2 real leagues; origins 2025-10, 2026-03
and 2026-07. "Diff" is the paired difference vs the baseline with the 3-block bootstrap CI from
`compare.py`. Three blocks is meaningless for inference.

| Variant | Virtual logloss | Diff [95% CI] | Brier | rank_norm | Real logloss |
|---|---|---|---|---|---|
| baseline | 2.0761 | — | 0.8250 | 0.229 | 1.851 |
| clogit | 2.0673 | −0.0088 [−0.047, +0.013] | 0.8250 | 0.229 | 1.746 |
| benter | 2.0717 | −0.0044 [−0.0096, −0.0016] | 0.8248 | 0.230 | 1.831 |
| gbm | 2.0778 | +0.0017 [−0.0000, +0.0045] | 0.8266 | 0.224 | 1.838 |

`crps_total` (10.943) and `crps_line` (5.545) are identical to the baseline by construction.

**Inner CV** on the training leagues (top-1 log-loss per league), printed as `[dc]` lines in the log.
"sim" is the baseline offset at 8000 draws.

| Origin | sim | clogit | benter (a, b) | gbm (rounds) |
|---|---|---|---|---|
| t < 9 | 2.2385 | 2.2659 | 2.2378 (1.34, −0.31) | 2.2378 (5) |
| t < 20 | 2.2708 | 2.2838 | 2.2708 (1.01, 0.00) | 2.2706 (5) |
| t < 24 | 2.2366 | 2.2390 | 2.2349 (0.92, 0.15) | 2.2358 (5) |

The plain conditional logit (K = 1) was chosen everywhere; the Plackett-Luce depth K = 3 never won.

The inner CV says the sim is hard to beat:
- **Pure model.** `clogit` is 0.002-0.03 worse than the sim in CV. Its smoke edge is probably noise.
- **Hybrids.** `benter` and `gbm` gain at most about 0.002 in CV.
- **Expectation for the full run.** Small gains or a tie for `benter` and `gbm`, and probably a small
  loss for `clogit`.

## Independent review (2026-10-29)

- **Fix.** The cache key hashed only `features.py`, `copula.py` and `fit_tables`. It missed
  `candidate.py` (whose `_sim` builds every test entry), `ratio_grid` (used by the sampler) and the core
  constants, so an edit there could have reused stale entries. It now hashes all of these (see
  Caching). Old entries are orphaned, not reused. Results are unchanged: the smoke npz files are
  bit-identical before and after.
- **Leakage checks, all passed.**
  - `fit` sees only `train_df` and `train_leagues` (t < origin, guarded by the harness). `predict` sees
    only the pre-game `week_df` and starters. No `data/` file, `common.League` or `inj` field is read.
  - All tuning is week-grouped CV inside `fit`.
  - Perturbing week s's actuals (×U(0.2, 2)) leaves every week-s training feature bit-identical except
    `h_vol`. That one moves through the pooled per-position scale `c_pos` (max change 0.087). This is a
    within-training effect of one scalar per position; it is not test-week information.
  - Benter's second stage is trained on stage-1 out-of-fold utilities whose models saw the other folds,
    including the second stage's own validation fold. This is the usual stacking approximation; it
    can only make the inner-CV choice of lambda slightly optimistic, and it uses training data only.
- **Comparability.**
  - Same items, blocks and outcome arrays as `_baseline_team`; compare.py lines them up.
  - `total_q` and `line_q` are bit-identical to the baseline (smoke npz and the full-mode origin
    2025-05).
  - At lambda = inf (benter) or 0 rounds (gbm), p_chop equals the baseline's to 1e-16 at 2025-05.
- **Design selection.** The benter variant replaced an earlier hybrid after the smoke numbers were
  seen. Weigh the 3 smoke origins (2025-10, 2026-03, 2026-07) with that in mind. The other 18 full
  origins are clean of it.
- **Budget.** Smoke: 39 s cold on 1 core (106 s with `DC_NOCACHE=1`), 0.6 GB peak RSS. Full mode at
  2025-05: 1.3 s fit + 7.3 s predict on 1 core. That extrapolates to about 350-700 core-seconds, or
  2-4 minutes on 6 cores.
