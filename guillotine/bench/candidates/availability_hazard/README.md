# availability_hazard

**What.** Learned replacements for the sim's two availability rules. Today, `common.P_OUT` /
`sim.Universe.p_out` sit Questionable players 25% and Doubtful players 80% of the time this week.
After that, `sim.INJ_HAZARD` starts a new absence with a flat 2% chance per week, and each
absence lasts a Geometric number of weeks with mean 3.

This candidate fits both rules on 10 seasons of nflverse injury reports joined to who actually
played (offense snaps), and scores them on the AVAILABILITY track (`bench/harness/tracks/availability.py`).

**Why it might beat the baseline.**
- The fixed tag odds are off. Over 2016-2026, relevant players tagged Questionable played 71% of
  the time, and Doubtful players played **0.7%** of the time, not 20%.
- Practice status also matters. Among Questionable players, those who did not practice played
  48% of the time, limited participants 73%, and full participants 88%.
- The flat 2% hazard is overconfident. Even a starter who played last week and has no tag
  misses 7% of next week's games and 18% at 8 weeks out. Missing time comes from injuries,
  injured reserve, benchings and cuts, and the risk is higher right after a return.
- Absences are not geometric. Across all relevant player-weeks, including injured reserve, the
  return rate is 33% after one missed game, 23% after two and 8% after seven. A geometric
  duration with mean 3 implies a flat 33%.

**Variants** (simplest first; one `Candidate(variant)` class in `models.py`):

| variant | gameday items (Q/D this week) | ahead items (h = 1..8 weeks) |
|---|---|---|
| `cells` | empirical-Bayes cell rates by tag x final practice status, shrunk to the tag rate (Beta prior, m = 20) | cell rates by h x tag at origin x missed-last-game x starter role, shrunk h x tag, then to h (m = 50) |
| `hazard` | L2 logit: tag, practice status, body part (10 groups), position, miss streak, games since return, previous report, IR-return flag, snap share, n weeks on the report, age, week, starter role | **structural discrete-time two-state model** in team games. The origin week uses the gameday logit, or an untagged logit for players with no tag. Each later team game uses an onset hazard `lambda(x, r)` (r = games since return) and a return hazard `mu(x, d)` (d = games missed so far). Both are person-period logits fitted on relevant player-weeks. A forward recursion over the states (A_r, U_d) gives P(plays at t+h). Step 1 uses the origin's report; later steps use hazards that leave the unknown future report out, since plugging in "no report" there was biased. |
| `gbm` | same logit as `hazard` | direct multi-horizon LightGBM on the ahead items: h, team games to target, every origin feature, plus depth rank / starter role. Heavily regularized (7 leaves, min 1000 per leaf, 150 rounds at 0.03, L2 10), because the 8 horizons of one player-origin are near-duplicates. |

`hazard` is the variant that drops straight into `sim.py`. Its per-player `lambda` and `mu`
replace `INJ_HAZARD` and the geometric draw on the `out_left` countdown, and the gameday logit
replaces `p_out`. `gbm` is the accuracy reference.

**Leakage.** Every fit sees only `train` (items whose target week is before the test week) and
`periods` (player-weeks before the test week). The harness features are causal: past snaps, plus
the pre-game report and roster moves of the origin week. Roster ACT and INA are merged because
INA is the game-day inactive list. The RES status is pre-game: RES players played in 0.03% of
weeks. The GBM settings and the marginal-hazard fix were chosen on 2019-2020 dev origins, which
come before the 2021+ test origins. Seeds are fixed, LightGBM runs `deterministic=True`, and the
parallel and serial runs are identical.

Review fixes (independent reviewer):
- `avail_data` fills a player's team backwards within a season, so an unsigned player who signs
  later "misses" his future team's games in `periods` (~0.5% of rows, all y = 0, and some depend
  on a roster row after the test week). `hazard` drops those rows (`_drop_backfilled`) before it
  fits `lambda` / `mu`. `gbm` and `cells` do not use `periods`.
- At a bye origin, step 1 of the recursion now uses the report of his last game (`rep_prev`), as in
  `periods`, not "no report". This affects almost no items, because weekly rosters have no rows
  for teams on bye, so bye-week players are out of scope.
- A truncation test rebuilt the grid from data cut at the origin: rosters and reports after t
  dropped, snaps from t on scrambled. It ran at 2021-05, 2023-09 and 2026-04. Predictions on the
  common items moved by at most 0.06 (mean of 5e-3 or less), and log-loss moved by 0.0014 or
  less, in both directions. What remains is a track-level static leak, not a candidate one:
  - The player universe and `pos` use every season's roster files, because pfr ids appear only
    in later files for some players.
  - So about 2% of the test items at 2021-05 exist only because of later files (0% at 2026-04).
  - It hits the baseline and the candidates identically.

**Results.**

Smoke run (3 origins, 3881 items: 40 gameday, 3841 ahead). Log-loss, lower is better:

| | logloss_bal | gameday ll | ahead ll | ahead brier | ahead mean p / rate |
|---|---|---|---|---|---|
| baseline (P_OUT + flat 2%) | 0.6991 | 0.7804 | 0.6178 | 0.1703 | 0.957 / 0.815 |
| cells | 0.5767 | 0.7284 | 0.4251 | 0.1308 | 0.815 |
| hazard | **0.4821** | 0.5593 | **0.4050** | 0.1234 | 0.802 |
| gbm | 0.4826 | 0.5593 | 0.4060 | 0.1236 | 0.809 |

Dev origins (2019-2020, 12 origins, before the test period): gameday ll 0.569 (baseline) /
0.508 (cells) / 0.440 (logit), and ahead ll 0.639 / 0.449 / 0.440 (hazard) / 0.432 (gbm).

**Sources.**
- Singer & Willett (2003), *Applied Longitudinal Data Analysis*, ch. 10-12: discrete-time hazard
  as a logit on person-period data. The `hapc` docs summarize the logit-hazard construction:
  https://hapc.readthedocs.io/en/latest/survival.html
- Tutz & Schmid (2016), *Modeling Discrete Time-to-Event Data* (Springer): duration-dependent
  hazards and competing transitions.
- Hamstring recurrence in the NFL, where the same-season reinjury risk is highest when players
  return within 2 weeks (motivates `r` in `lambda`): https://pmc.ncbi.nlm.nih.gov/articles/PMC6931147/
- Injury risk after return to play in pro football (soccer), where the excess risk decays toward
  baseline over a few weeks: https://pmc.ncbi.nlm.nih.gov/articles/PMC11787231/
- Empirical play rates by designation: Harvard Sports Analysis (2013),
  https://harvardsportsanalysis.org/2013/11/inaccuracies-in-the-injury-report-across-the-nfl/ ;
  TeamRankings, https://www.teamrankings.com/blog/nfl/nfl-injuries-how-often-do-probable-players-play-2-of-4
- Data: nflverse / nflreadr injuries, weekly rosters and snap counts,
  https://cloud.r-project.org/web/packages/nflreadr/vignettes/dictionary_injuries.html ;
  LightGBM, https://lightgbm.readthedocs.io

**Runs.**
- `run_smoke.sh`: about 11 s on 1 core. It writes `_baseline_availability.smoke.json`, then
  `availability_hazard.smoke.json`, then the smoke leaderboard.
- `run_full.sh`: queued only, at most a few minutes on 6 cores (91 origins, 111992 items). It
  writes `_baseline_availability.json` (the track baseline), then `availability_hazard.json`,
  then `leaderboard_availability.json`.
