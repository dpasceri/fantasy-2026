# gbm_quantile (player track)

Tree-ensemble distributional regression for weekly player scores: a quantile regression forest,
the forest with a gradient-boosted zero-mass classifier, and LightGBM quantile regression. All three
work on the baseline's scale, r = actual / proj, and use the same pre-game features.

## What

**Features** (`features.py`, 20 columns, all pre-game). History features use only weeks strictly
before the row's own week, built on a pid × week grid with prefix sums shifted by one week.

| group | columns |
|---|---|
| projection | `pos`, `proj` |
| game environment | `team_proj`, `opp_proj`, `game_tot` (= team + opp), `spread` (= team − opp) |
| depth / role | `team_pos_rank`, `rel_top` (proj ÷ the team's top proj at his position), `share_team` (proj ÷ team_proj) |
| stat mix of the projection (league scoring) | `td_share` (TD points ÷ proj), `rec_share` (reception points ÷ proj), `rush_share` |
| own history | `h_n` (scored item-weeks so far), `h_e` (shrunk mean of (a − p) / (p + 3)), `h_z` (shrunk zero rate), `h_ewm` (EWM of the residual, half-life 3 weeks), `last_ratio` (last week's proj ÷ proj), `trend` (proj ÷ mean of the last 4 item-weeks' proj), `since` (weeks since last item-week) |
| opponent | `opp_res` (shrunk mean residual of the same position against this opponent) |

**Variants** (`candidate.py:GBMQuantile(variant)`):

- **`qrf`**: quantile regression forest (Meinshausen 2006).
  - 500 CART trees (sklearn `DecisionTreeRegressor`). Each tree is grown on a random half of the
    training **weeks**, because rows of one week share games, so the week is the sampling unit.
  - Trees split on the two-column target `[y / 8, 1{y ≤ 0}]`, which tracks the mean and the zero
    mass. They use `max_features = 1/3` and a minimum leaf of n_train / 40.
  - A test row's Meinshausen weights over **all** training rows (leaf co-membership ÷ leaf size,
    averaged over trees) define a weighted ECDF of r. The prediction is q_y(τ) = proj × Q_r(τ).
  - Zeros among the neighbours give an exact atom at 0. The zero mass is therefore estimated from
    the neighbourhood, instead of being interpolated between projection buckets (the baseline's
    known weakness).
- **`qrf_hurdle`**: the same forest used as a hurdle mixture.
  - P(y ≤ 0) is the mean of two estimates: the forest's weighted share of zeros, and a LightGBM
    binary classifier (learning rate 0.03, 7 leaves, rounds from early stopping on the last 20% of
    training weeks, then refit on all of them).
  - The y ≤ 0 and y > 0 parts are the forest's weighted ECDFs restricted to each side.
- **`lgbq`**: LightGBM quantile regression of r.
  - 15 levels from 0.02 to 0.98, each started from the training r quantile (`init_score`).
  - All levels share one round count: the median of the per-level early-stopping rounds on the last
    20% of training weeks. Then a refit on every training row.
  - Monotone rearrangement (sorting, Chernozhukov et al. 2010), then linear interpolation to the 99
    levels with linear tail extrapolation, clipped to the training r range.

Every variant snaps |q| < 0.05 to exactly 0, so the PIT sees ties with the zero atom.

## Why

- The baseline's ratio tables come from about 300 rows per position × projection bucket. They are
  noisy, and interpolating between bucket medians shrinks the zero atom (P(y ≤ 0) is 0.128
  predicted vs 0.186 observed).
- A QRF is an adaptive, many-partition version of the same "empirical ratio distribution of
  similar players" idea. Averaging 500 partitions smooths the tables, and the neighbourhoods can
  use depth, stat mix and history.
- Boosting is strong at classification, and the zero mass (did he play / get a touch) is where
  most of the feature signal sits (`rel_top`, `share_team`, `team_pos_rank`, `h_z`; see the
  Princeton thesis below). `qrf_hurdle` puts a boosted classifier exactly there.
- `lgbq` is the canonical boosted-quantile model. It is included to measure the family head-on.

## Evidence before the backtest (inner validation, training data only)

I fit on all training weeks except the last 4-6 and scored those weeks. This was done inside the
training sets of origins 2025-14, 2026-01 and 2026-07: inner validation weeks 2025 9-13,
2025 15-18 and 2026 1-6. CRPS (2 × mean pinball):

| inner split | baseline | qrf | qrf_hurdle (avg P0) | lgbq |
|---|---|---|---|---|
| 2025-14 (val 1487) | 2.7590 | 2.7428 | 2.7393 | 2.7679 |
| 2026-01 (val 1320) | 2.7860 | 2.7789 | 2.7757 | 2.7732 |
| 2026-07 (val 1900) | 2.7847 | 2.7702 | 2.7715 | 2.7879 |

The same splits also rejected these alternatives:

- **QRF with only pos + proj**: about 0.2-0.35% better than the baseline. So most of the QRF gain is
  smoothing; the features add about 0.1-0.35%.
- **PIT-space models**, which predict u = F_base(y) from cross-fitted (leave-one-week-out) baseline
  PITs and map back through the baseline: no better than the baseline. The forest's u histogram
  cannot fix noise in the test row's baseline grid.
- **IBUG** (LightGBM leaf-affinity kNN): 0.1-0.6% worse than the QRF.
- **proj-weighted LightGBM quantile loss** (= y-space pinball): worse than unweighted at 2 of 3
  splits.
- **ratio exponent β < 1** (y / proj^β): worse at 2 of 3.
- **time-decayed leaf weights**: no gain.
- **per-position forests**: worse at 2 of 3.
- **quantile-averaging QRF with the baseline**: no gain.

Fixed hyperparameters (leaf n/40, `max_features` 1/3, 500 trees, the split target) came from these
splits. They are not tuned on the smoke or full test weeks. LightGBM round counts are chosen per
origin by early stopping inside the training weeks.

## Smoke (bench/cpuq -n 1, 43 s, 908 items; baseline on the same split)

| | crps | Δ vs base [bootstrap 95% CI] | cov10/50/90 | cov80 | cov_err | pit_maxdev | crps QB / RB / WR / TE |
|---|---|---|---|---|---|---|---|
| baseline | 2.7708 | | 0.118 / 0.497 / 0.884 | 0.767 | 0.0176 | 0.0222 | 4.222 / 2.906 / 2.604 / 2.338 |
| qrf | 2.7648 | −0.0060 [−0.0136, −0.0024] | 0.100 / 0.502 / 0.883 | 0.783 | 0.0090 | 0.0222 | 4.277 / 2.889 / 2.588 / 2.336 |
| qrf_hurdle | 2.7639 | −0.0069 [−0.0189, −0.0013] | 0.102 / 0.499 / 0.883 | 0.781 | 0.0099 | 0.0200 | 4.275 / 2.890 / 2.588 / 2.333 |
| lgbq | 2.7825 | +0.0117 [−0.0083, +0.0257] | 0.116 / 0.503 / 0.884 | 0.769 | 0.0165 | 0.0262 | 4.341 / 2.928 / 2.590 / 2.340 |

Notes on the smoke:

- Predicted P(y ≤ 0) is about 0.184 for both QRF variants, against 0.186-0.198 observed. The
  baseline predicts 0.128-0.154.
- QB is worse on the smoke (about 90 QB items). It is better than the baseline in all three inner
  splits (4.65 vs 4.71, 4.45 vs 4.48, 4.41 vs 4.42), so treat it as smoke noise until the full run.
- The CIs come from a block bootstrap over only 3 weeks, so they are far too narrow to trust; the full run decides.
- Outputs are byte-identical across reruns.

## Runtime

- Smoke: 43 s on 1 core.
- Full: 63 tasks (21 origins × 3 variants) of 2-7 s each on 1 thread, so about 1-2 min on 6 cores.

## Leakage

- History, opponent and depth features use only weeks strictly before each row's week. The test
  week's features come only from `train_df` plus the test frame's pre-game columns.
- Early stopping and the hurdle classifier's rounds use only the last training weeks.
- No data/ files or `common.League` are read. `data.settings()` is used for the scoring weights.
- Seeds: the forest and LightGBM are seeded from `task_seed()`. LightGBM uses `deterministic=True`
  and `task_threads()` threads.

## Reviewer notes (independent review before queueing)

- **Per-origin leakage: none found.** Perturbation checks on `features.build` at origin 2026-03:
  - Changing the actuals of week t leaves every training row at weeks ≤ t unchanged, and changes
    only later rows and the test rows.
  - Changing week t0-1's projections leaves rows before t0-1 unchanged.
  - Shuffling the test frame permutes its features and nothing else.
  - Train and test feature means match at the last training week, so the depth features
    computed on the items-only test frame are on the training scale.
  - Early stopping, the P0 classifier and the round counts use training weeks only. Nothing
    reads `inj`, data/ files or `League.info`/`proj`.
- **Quantiles** are monotone before the harness sorts them (all 3 variants, 0 crossing rows).
  The snap to 0 at |q| < 0.05 keeps the order.
- **Budget:** at most 3.1 s per task on 1 core, at the largest origin (2026-07), so 63 tasks
  take about 1 min on 6 cores. Workers come from the harness, and LightGBM uses
  `task_threads()`.
- **Design-level selection (caveat).** The fixed choices listed above were made by scoring
  inner-validation weeks 2025 9-13, 2025 15-18 and 2026 1-6, and the smoke weeks were also
  seen. That covers 16 of the 21 full test blocks. Per origin this is not leakage, but the full
  leaderboard gain (expected 0.2-0.5% CRPS) is of the same order as plausible selection bias.
  - `clean_subset.py` runs after the full backtest. It writes `clean_subset.json`, which re-scores
    every variant against the baseline on the 5 blocks that design never scored: 2025-05..08 and
    2025-14 (2025-05 was run once as a crash check, with no tuning after it).
  - With only 5 blocks its bootstrap CI is coarse, so `crps_diff_by_block` lists the per-week
    differences too.

## Sources

- N. Meinshausen, *Quantile Regression Forests*, JMLR 7 (2006) 983-999. https://www.jmlr.org/papers/v7/meinshausen06a.html
- V. Chernozhukov, I. Fernández-Val, A. Galichon, *Quantile and Probability Curves Without Crossing*, Econometrica 78(3) (2010). https://arxiv.org/abs/0704.3649
- J. Brophy, D. Lowd, *Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees* (IBUG), NeurIPS 2022. https://arxiv.org/abs/2205.11412 (tested; worse here)
- T. Duan et al., *NGBoost: Natural Gradient Boosting for Probabilistic Prediction*, ICML 2020. https://arxiv.org/abs/1910.03225 (considered; parametric families fit this multimodal, zero-inflated target poorly)
- A. März, T. Kneib, *Distributional Gradient Boosting Machines* (LightGBMLSS, zero-adjusted families), 2022. https://arxiv.org/abs/2204.00778 (motivates the hurdle split)
- LightGBM parameters (quantile objective, `init_score`, early stopping). https://lightgbm.readthedocs.io/en/latest/Parameters.html
- Princeton senior thesis (2025) on quantile-head fantasy projections, which notes that much of the edge comes from learning whether a player plays. https://theses-dissertations.princeton.edu/entities/publication/d6981488-f30c-47ce-8f06-b6c3e98f2578

## Files

- `features.py`: the feature builder.
- `candidate.py`: `WeekForest`, `qrf_quantiles`, `qrf_hurdle_quantiles`, `lgb_quantiles` and
  `GBMQuantile`.
- `run_smoke.sh`, `run_full.sh`.
