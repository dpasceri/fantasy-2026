# rival_lineup (team track)

**What.** Opponent lineup behaviour. Real managers don't always start their max-projection lineup, and
sometimes they start players who are out. This candidate measures how, models it with shrinkage, and
tests whether simulating it changes chop prediction on the real weeks. Everything else is the baseline
(`_baseline_team.copula.CopulaSim`: marginal tables, copula, 96000 CRN draws), which this class subclasses.

Virtual leagues have no managers (their `roster` is the drafted starters), so **every variant returns the
baseline's prediction there, bit for bit.** The primary metric (virtual log-loss) therefore ties the
baseline by construction. All the signal is in the 7 real items (`real_*` metrics) and in the
lineup-level diagnostics below.

## What real managers did (2026 weeks 1-7, 98 team-weeks, 17 managers)

`lineup_eval.describe()` compares each week's actual starters with the max-projection lineup of the
roster the manager had that week (`common.best_lineup` logic, Sleeper's final projections). This is
descriptive only and is never fed to a model. 95% CIs resample whole weeks (7 blocks).

| Quantity | Value [95% CI] |
|---|---|
| Projected points left on the bench, per team-week | 1.32 [0.72, 2.08] |
| Share of team-weeks not at max projection | 57% [44%, 71%] |
| Share leaving > 3 projected points | 12% [4%, 22%] |
| Share starting someone projected 0 | 7.1% [1.9%, 13.8%] (7 events) |
| Actual points lost vs the max-projection lineup | 2.99 [0.92, 4.73] |
| Swaps vs max projection, per team-week | 0.66 [0.49, 0.84] |
| Status quo (weeks 2-7, 59 swaps): started-over player had started last week | 64%, vs 24% for the higher-projected player benched |
| Manager effect on bench points (moments estimator) | between-manager sd 0.0; within-week permutation p = 0.69 |

The 7 zero-projection starts were Bowers (wk 2), Barkley (wks 4, 5, 6 for the same manager), Chase (wk 4),
McConkey (wk 4) and Adams (wk 6). **3 of the 7 played and scored** (Chase 4.2, Barkley 1.5, Adams 5.4).
Across all players, only about 2.6% of recently projected players who are projected 0 while their team
plays score anything. So a manager's choice to start someone at projection 0 carries information. The
baseline scores every projection-0 starter as a sure 0.

## Model (`lineup.py`)

**ZeroModel (starters projected 0).**

- *Eligible zero*: a player whose last projection was ≥ 3 within the last 3 weeks (point-in-time, from
  training rows) and whose last team plays this week. Anyone else projected 0 still scores 0.
- *P(play)*: a Beta posterior. The prior mean is the population rate among eligible zeros in
  training (all players), with a prior strength of 2 pseudo-observations. It is updated with the real
  league's projection-0 starters from earlier weeks.
- *Score if he plays*: a baseline-marginal draw at projection h × his last projection. h is the training
  mean of actual / last projection, clipped to [0.2, 1].

**LineupFit (lineup choice).** Each week, manager m does one of two things:

- With probability λ_m (*inattention / status quo*), he leaves last week's lineup alone. Every last-week
  starter still on the roster stays in, even if he is now projected 0 or on bye. Open slots are filled
  by projection.
- Otherwise he sets the max-utility lineup with
  `u_i = log(phat_i + 0.5) + σ (r s_i + ε_i)`, `ε ~ N(0,1)`:
  - `s_i` = started last week (a status-quo bonus);
  - `phat_i` = projection, or γ × last projection for an eligible zero (game-time-decision optimism).

Slots are QB, RB×2, WR×2, TE, then 2 FLEX; greedy is exact for these nested slots, and the code is checked
against `common.best_lineup` on all 87 training rosters. r, σ and γ are shared across managers. Manager
effects are partially pooled: `λ_m ~ Beta(μκ, (1−μ)κ)`, with μ and κ on grids, and κ = 1e4 meaning
complete pooling.

**Fitting.**

- Likelihood: a tempered composite likelihood of who started. Each rostered player's start is a
  Bernoulli whose P(start) is simulated with 400 common-random-number draws per team-week. It has a
  2% lapse and weight 1/2, because one swap flips two indicators.
- The two lineup components mix per team-week.
- The posterior is computed exactly on the grid (5 r × 11 σ × 3 γ × 11 λ × 7 μ × 4 κ). The priors were
  fixed before any scoring and are skeptical: they lean toward max projection, with P(r) ∝ e^{−r/2},
  P(γ=0) = 1/2, P(μ) ∝ e^{−10μ}, σ uniform on 0.02-0.64 and κ uniform.
- With no earlier real lineups (2026 week 1), the model is exactly max projection.

**Prediction.** For each of the 96000 score draws, the model draws (r, σ, γ, λ_m) from manager m's
posterior, then the component and ε. So lineup uncertainty and parameter uncertainty both enter the
team totals, the chop probabilities and the chop line (`line_q` is the per-draw minimum).

**Posterior at origin 2026-07 (87 team-weeks).**

- r ≈ 1.5, σ ≈ 0.18, γ ≈ 0.25 (prior mean 0.19).
- μ ≈ 0.035, and P(complete pooling) ≈ 0.30.
- λ_m is shrunk hard toward μ. The largest is one rival at 0.077; most managers are 0.03-0.05.

## Variants

| Variant | Real-league lineups | Question it answers |
|---|---|---|
| `zero_start` | as played, as in the baseline, plus ZeroModel for projection-0 starters | Does the as-played chop model improve if a projection-0 starter is not a sure 0? It differs from the baseline only in weeks 4-6 (5 team-weeks). |
| `roster_max` | unknown: built from the roster at max projection (what run.py assumes for every rival) | Reference for `roster_rum` |
| `roster_rum` | unknown: drawn from LineupFit, with projection-0 starters scored by ZeroModel | Does simulating real lineup behaviour beat max projection when lineups aren't set yet (the live chop sheet)? |

For the `roster_*` variants, the baseline's as-played starters are an **oracle** (strictly more
information). Read `roster_rum − roster_max` (JSON `paired`), not `roster_rum − baseline`.

## Diagnostics in the result JSON

- `lineup_describe`: the table above.
- `lineup_eval`: rolling-origin lineup forecasts for each real test week. Each is fit inside
  `data.horizon` on earlier weeks only, then scored against the lineups actually set. It compares
  `max`, `pooled` (κ = 1e4) and `hier` on:
  - `crps_proj`: CRPS of the starters' projected total;
  - `nll`: negative composite log-likelihood of who started;
  - `zero_brier`: Brier score of "starts someone projected 0";
  - `d_pred` vs `d_obs`: projected bench points.

  Paired differences carry week-block bootstrap CIs. The per-team-week rows are in
  `bench/results/rival_lineup.lineup_rows[.smoke].npz`.
- `paired`: week-block bootstrap differences on the real metrics (`real_logloss`, `real_brier`,
  `real_rank`, `real_crps_total`, `real_crps_line`) and three virtual metrics for:
  - `roster_rum − roster_max`
  - `zero_start − baseline`
  - `roster_max − baseline`
  - `roster_rum − baseline`

## Leakage and honesty notes

- **Rolling origin.** Every parameter (lineup posterior, Beta P(play), h, population rate) is fit at each
  origin from `train_df` and `train_leagues` only. The `roster_*` variants never read the test week's
  `starters`. `zero_start` uses them exactly as the baseline does. All randomness is seeded from fixed
  seeds plus (season, week, roster or pid), so runs are deterministic.
- The test `roster` is the matchup's roster for that week, as the harness provides it. In-week adds
  after some games locked could in principle be in it.
- **Researcher degrees of freedom.**
  - The model *structure* was chosen after looking at the descriptive table above, which covers all 7
    real weeks: the status-quo flag, the inattention component and the zero-projection handling.
  - A first version (utility model only, with manager-specific σ) was replaced, because it structurally
    cannot keep an injured starter in a lineup. That flaw showed up on week 4.
  - The priors and grids were not tuned on outcomes, and nothing was changed after the v2 smoke.
  - Even so, the real-week numbers are not a fully blind test of the structure.
- **Power.** There are only 7 real items: 2 in smoke, with week-block CIs over 7 (or 2) blocks. The
  descriptive manager-effect test finds no evidence of manager differences, and the hierarchical
  posterior shrinks accordingly. Expect the chop-level differences to be within noise. The
  lineup-level diagnostics (98 team-weeks) are where the model can show anything.

## Sources

- K. Train, *Discrete Choice Methods with Simulation*, 2nd ed., Cambridge UP, 2009: random-utility
  probit choice and maximum simulated likelihood with common random numbers (ch. 5, 10).
  https://eml.berkeley.edu/books/train1201.pdf
- W. Samuelson and R. Zeckhauser, "Status quo bias in decision making", *J. Risk and Uncertainty* 1
  (1988) 7-59: the status-quo / default component.
  https://scholar.harvard.edu/files/rzeckhauser/files/status_quo_bias_in_decision_making.pdf
- Baseball Prospectus, "Fantasy Freestyle: A Behavioral Look at Lineup Setting": omission bias
  ("don't mess with success") in sit/start decisions.
  https://baseballprospectus.com/fantasy/article/24577/fantasy-freestyle-a-behavioral-look-at-lineup-setting
- Footballguys, "Hidden Yardage: Exploring the Bias in Fantasy Football": conservatism / inaction in the
  face of new information.
  https://www.footballguys.com/article/2024-hidden-yardage-exploring-the-bias-in-fantasy-football
- C. Varin, N. Reid and D. Firth, "An overview of composite likelihood methods", *Statistica Sinica* 21
  (2011) 5-42: independence composite likelihood and why it needs a temper.
  https://utstat.utoronto.ca/reid/research/varin_reid_firth.pdf
- P. Bissiri, C. Holmes and S. Walker, "A general framework for updating belief distributions", *JRSS B*
  78 (2016): tempered (general Bayes) posteriors from a loss. https://arxiv.org/abs/1306.6430
- B. Efron and C. Morris, "Data analysis using Stein's estimator and its generalizations", *JASA* 70
  (1975) 311-319, and Gelman et al., *Bayesian Data Analysis*, 3rd ed., ch. 5 (hierarchical
  beta-binomial): partial pooling of manager rates.
- F. Wichmann and N. Hill, "The psychometric function: I", *Perception & Psychophysics* 63 (2001)
  1293-1313: lapse rates that keep choice likelihoods robust to one-off decisions.

## Runs

- `bench/cpuq -n 1 -- bash bench/candidates/rival_lineup/run_smoke.sh`: 44 s on 1 core. It writes
  `bench/results/rival_lineup.smoke.json` (all 3 variants, 50 items) and the lineup diagnostics for
  weeks 3 and 7.
- `run_full.sh` is queued only:
  `{"key": "rival_lineup", "cmd": ["bash", "bench/candidates/rival_lineup/run_full.sh"], "timeout_s": 1200}`.
  - Estimated run time is about 4-5 min on 6 cores: 3 × the baseline's 49 s, plus about 15 s of lineup
    diagnostics and about 3-5 s of lineup fitting per real origin.
  - Memory is about 0.6 GB per worker, as for the baseline.

### Smoke result (same split as `_baseline_team.smoke.json`)

| | Virtual logloss (n = 48) | Real logloss (n = 2) | Real brier | Real crps_total | Real crps_line |
|---|---|---|---|---|---|
| baseline | 2.0761 | 1.8510 | 0.7928 | 10.702 | 3.765 |
| zero_start | 2.0761 (identical) | 1.8510 (no zeros in weeks 3, 7) | 0.7928 | 10.702 | 3.765 |
| roster_max | 2.0761 (identical) | 1.8044 | 0.7752 | 10.850 | 3.647 |
| roster_rum | 2.0761 (identical) | 1.8788 | 0.7957 | 10.811 | 4.022 |

Lineup level (weeks 3 and 7, 26 team-weeks; hier − max):

- crps_proj −0.093 [−0.278, +0.159]
- nll −2.40 [−3.10, −1.45]
- zero_brier +0.0012: no projection-0 starts happened in those weeks.
- hier − pooled ≈ 0 on every score.
