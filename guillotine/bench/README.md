# bench: backtest bench for the guillotine sim

Everything here is new work layered on the core pipeline. The core files (`common.py`, `model.py`,
`sim.py`, `auction.py`, `dynamic.py`, `backtest.py`, `run.py`, `report.py`) and `data/` are
read-only: never edit them. All commands below run with cwd = the guillotine folder `G`
(`guillotine/` in this repo). If the path has a space, quote it.
Python is `.venv/bin/python` (3.14 with numpy, scipy, pandas, scikit-learn, lightgbm, statsmodels).

## CPU rule (hard)

At most 16 cores in total (one thread per physical core) for everything on this box, and no GPU.

- Anything heavier than a few seconds runs through the slot queue: `bench/cpuq -n 1 -- <cmd>`.
  Use `-n 1` for smoke tests and never more than `-n 2`.
- Never start a process or thread pool outside cpuq. Size any pool from `CPUQ_SLOTS`
  (`bench.harness.slots()`). Inside a harness task, use `bench.harness.task_threads()` threads
  (for example LightGBM `n_jobs`).
- Full backtests are never run by agents directly. They are queued (see below).

## cpuq

`bench/cpuq -n K [--timeout S] -- cmd args...` blocks until K of the 16 slots (physical cores
8-15, then 4-7, then 3-0) are free. It then pins the command to those cores and sets `CPUQ_SLOTS`, `GSIM_THREADS`
and the `*_NUM_THREADS` variables to K. It runs the command at nice 10 and kills its whole
process group on timeout (exit 124). Waiting for slots is normal, so give the shell a long
timeout.

## Queue for full runs

A runner (`bench/queue_runner.py`) is **already running**. Do not start another. To enqueue a
job, write `bench/queue/pending/<key>.job`:

```json
{"key": "<key>", "cmd": ["bash", "bench/candidates/<key>/run_full.sh"], "timeout_s": 1200}
```

Jobs run one at a time, oldest first. `cmd` runs with cwd = `G` under cpuq, with
`bench/queue/max_slots` cores (6 now; `CPUQ_SLOTS` says how many). Output goes to
`bench/logs/<key>.log` and a status record to `bench/queue/done/<key>.json`. A full run must
finish within 15 minutes on 6 cores. Each track baseline is its own job, `_baseline_<track>`
(for example `{"key": "_baseline_player", "cmd": ["bash", "bench/candidates/_baseline_player/run_full.sh"], "timeout_s": 1200}`).

## Packages

Install extras only under the lock, with `python -m pip` (pip's own script is broken):
`flock bench/.pip.lock .venv/bin/python -m pip install <pkg>`. Python 3.14 may lack wheels, so
fall back gracefully when an install fails.

## Layout

```
bench/
  cpuq, queue_runner.py, queue/{pending,running,done}, logs/
  harness/            shared harness (import as bench.harness; puts G on sys.path)
    data.py           player-week table, league settings, 2026 matchups
    metrics.py        quantile + probability scores, Metric specs, block bootstrap
    results.py        result JSON + per-item npz, validate / align / evaluate
    compare.py        leaderboard per track
    tracks/player.py  PLAYER track (others add tracks/<track>.py)
    tracks/availability.py  AVAILABILITY track (P(plays) this week and 1-8 weeks ahead)
    avail_data.py     nflverse injury / roster / snap grid for the AVAILABILITY track
    tracks/bids.py    BIDS track (rival FAAB bids and prices, point-in-time rosters, guard)
    tracks/policy.py  POLICY track (in-sim FAAB bid policy from today's state)
    dyn_policy.py     copy of dynamic.py with the policy hook (POLICY track)
  candidates/<key>/   one folder per candidate (see protocol)
  candidates/_baseline_player/   player baseline (= model.py marginal, refit per origin)
  candidates/_baseline_bids/     bids baseline (= auction.BidModel, refit per origin)
  candidates/_baseline_policy/   policy baseline (= dynamic.py's m x typical-bid rule)
  candidates/_baseline_availability/   availability baseline (= P_OUT + sim's flat 2%/wk hazard)
  candidates/_template/          copy me
  results/            <key>.json, <key>.smoke.json, <key>__<variant>[.smoke].npz, leaderboard_<track>.json
  cache/              harness caches (player_weeks_<signature>.pkl, rebuilt when data/ or data.py change)
  cache/nflverse/     nflverse source CSVs for the availability track (avail_data.fetch())
```

## Candidate protocol

Each candidate lives in `bench/candidates/<key>/` (key: letters, digits, `_`, `-`, `.`; use a
valid Python identifier so it imports as `bench.candidates.<key>.<module>`). It holds:

- its modules,
- `README.md`: what it is, why it might help, sources (papers, libraries), and the variants,
- `run_smoke.sh`: at most 90 s on 1 core. It writes `bench/results/<key>.smoke.json`. Run it as
  `bench/cpuq -n 1 -- bash bench/candidates/<key>/run_smoke.sh`.
- `run_full.sh`: at most 15 min on 6 cores. It writes `bench/results/<key>.json`. It is only
  ever run by the queue.

Both scripts `cd` to `G` themselves (`cd "$(dirname "$0")/../../.."`). See `candidates/_template/`.

### Results JSON

```json
{"key": "...", "track": "player", "mode": "smoke" | "full",
 "variants": [{"name": "...", "metrics": {...}, "per_item": "bench/results/<key>__<variant>.npz"}],
 "baseline": "bench/results/_baseline_<track>.json", "runtime_s": 12.3, "notes": "..."}
```

- The per-item npz holds:
  - `item_id`: harness ids, identical across candidates on a track,
  - `block`: test week or replication id, used by the block bootstrap,
  - one per-item loss array per metric, named after the metric.

  Tracks may add more arrays (the player track adds `pos`, `pit`, `proj`, `y` and `q`, the
  quantile matrix).
- Smoke runs write `<key>.smoke.json` and `<key>__<variant>.smoke.npz`, and point `baseline` at
  `_baseline_<track>.smoke.json`. A smoke run never clobbers a full run.
- Variants must be scored on exactly the harness's items, in the harness's order.
  `results.evaluate()` asserts this (`results.assert_aligned`). A track CLI that writes through
  `results.write()` gets this for free.
- Extra fields are allowed (the player track adds `by_week`, `fit_s`, `predict_s`, `origins`,
  `n_items`).
- `results.validate(res)` lists schema problems; the track CLIs assert it is empty after
  writing.
- The baseline is itself a result file, `bench/results/_baseline_<track>.json` (+ npz),
  produced by the queued job `_baseline_<track>`.

### Leaderboard

`python -m bench.harness.compare <track> [--mode full|smoke] [--B 2000]` scores every
`bench/results/*.json` of that track and mode against the baseline. The reference is the
baseline's variant named `baseline`, or its first variant. For each (key, variant) and each
metric it reports:

- the value, the baseline value and the mean paired difference (candidate − baseline),
- a 95% CI from a block bootstrap that resamples whole blocks with replacement, using the same
  resample for both sides,
- `p_worse`, the share of resamples in which the candidate is worse.

Rows are sorted by the track's `PRIMARY` metric, best first (descending diff when that metric
has `lower=False`, as the policy track's `p_win` does). A result is listed under `skipped`, with the
reason, if its items or blocks differ from the baseline's. The same goes for a result whose
outcomes `y` differ: data/ changed between the two runs (re-fetch, stat corrections), so rerun
the older one. A track module can name more per-item outcome arrays that must match the baseline's
in `OUTCOME_ARRAYS` (the team track: `n_teams`, `chopped`, `y_tot`, `line_y`). The output is
`bench/results/leaderboard_<track>.json` (`.smoke.json` in smoke mode). Numbers in it keep 9
significant digits, so tiny metrics such as the mc track's MSEs (~1e-6) are not rounded to zero.

Metric definitions come from `bench.harness.tracks.<track>.METRICS`, a list of
`metrics.Metric`. If the track module has none, compare falls back to the plain mean of every
per-item array that shares a name with a baseline metric, lower is better.

A `Metric` turns per-item arrays into per-item sufficient statistics `(n, k)` and maps their sum
to a value, so ratios such as PIT-histogram deviation or coverage error bootstrap correctly.
Helpers: `mean_of`, `group_mean`, `mean_abs_err`, `pit_dev`, `ece_metric`. `lower=None` marks a
metric with no better direction, such as raw coverage rates.

## Harness modules

### `bench.harness.data`

`table()` is the player-week frame over every cached week: 2025 weeks 1-18 and 2026 weeks 1-17,
with 2026 weeks 8+ unscored. It is cached and shared, so treat it as read-only. Columns:

| Columns | Meaning |
|---|---|
| `season`, `week`, `t` | `t` is the global chronological index over `WEEKS`: 2025-01 = 0 … 2026-17 = 34 |
| `pid`, `name`, `pos`, `team`, `opp`, `game` | Player and game identity; `team`/`game` are blank on rows with no projection that week (see Leakage rules) |
| `proj` | League-scoring projected points |
| `ps_<stat>` | Stat-level projections; absent means 0 |
| `item` | proj ≥ 0.5 and a game id |
| `team_proj`, `opp_proj` | Summed proj of the player's team and of the opponent that week |
| `team_pos_rank` | 1 = top proj at his position on his team |
| `final` | The week's stats file exists |
| `has_stats` | The player has a stats row |
| `actual` | League-scoring actual points |
| `as_<stat>` | Actual stats; NaN when the week isn't final |
| `inj_now`, `team_raw` | `LEAKY` (today's injury tag; Sleeper's raw team). Stripped from every frame a candidate gets; see below |

- `proj` and `actual` are `common.League.pts` on the cached stat dicts, the exact call that
  `model.calibration_rows` makes. They reproduce it bit for bit: same 7715 rows, same values.
- A projected player with no stats row scored 0.
- On all 784 2026 starters, `actual` equals Sleeper's `starters_points` to 1e-14.
- Sleeper's own half-PPR number is `ps_pts_half_ppr`. It has zero weight in league scoring.

Helpers:

- `t_of(season, week)`, `week_of(t)`
- `train_before(origin, items_only=False)`: every scored row strictly before `origin`
- `week_frame(season, week, items_only=True, pre=True)`
- `history(pid, before=(season, week))`
- `pregame(df)`: drops actuals and the `LEAKY` columns
- `horizon(origin)`: a context manager. Inside it, `table()` (so also `train_before`,
  `week_frame`, `history`, `starters`) is a private copy of the table as known at the test
  week's kickoff: later weeks dropped, the test week's actuals blanked (`final`/`has_stats`
  False, `actual`/`as_*` NaN), `LEAKY` columns removed. The player harness wraps every
  `fit`/`predict_quantiles` call in it.
- `item_ids(df)`
- `settings()`: slots, scoring, budget, weeks
- `matchups()`: 2026 per (week, roster): points, starters, starters_points, players, alive, final
- `starters()`: long frame of starters with Sleeper and league-scoring points
- `league()`: the core `League`

### `bench.harness.metrics`

**Quantile scores.** `LEVELS = np.linspace(0.01, 0.99, 99)`.

- `crps` = 2 × mean pinball over LEVELS. This approximates CRPS = 2∫ pinball_τ dτ. The mean over
  99 levels weights each level by 1/99 rather than 1/100 (+1%) and leaves out the outer 1% tails
  (−), so it reads about 1% high on a normal (audited: 1.0099 × the closed form) and a little less
  on heavy tails. It is a near-constant scale, and everyone uses the same items and levels, so
  paired differences are fair.
- `pinball`: mean pinball loss over LEVELS.
- `pit`: a randomized PIT. A tie with several quantiles (a point mass such as the zero for
  inactive players) is spread over every band it touches. The uniform draw is a fixed hash of
  `item_id`, identical for every candidate. A calibrated forecast hits nominal rates, atoms
  included (audited on a synthetic 17.7%-zero mixture: coverage 0.099 / 0.500 / 0.900 / 0.802,
  flat PIT). When the atom's mass sits exactly on a level (say 20%), the tie also covers that
  level's band, so the atom is spread one band too wide (cov10 0.096 there). Put atoms at
  exact values (0.0, not 1e-9) so ties are detected.
- `cov10`, `cov50`, `cov90`: share with PIT ≤ 0.1 / 0.5 / 0.9, which equals y ≤ q_level when
  nothing is tied.
- `cov80`: share inside the central 80% interval.
- `cov_err`: mean |coverage − nominal| over those four.
- `width80`: q90 − q10, a sharpness measure.
- `pit_maxdev`: max |share − 0.1| over a 10-bin PIT histogram.
- Per position: `crps_<POS>` and `pit_maxdev_<POS>`.

**Probability scores.** `logloss` (clipped to [1e-6, 1−1e-6]; multiclass rows renormalized),
`brier` (multiclass = sum over classes), `calibration_bins` (reliability table + ECE),
`prob_items`, `prob_metrics()`.

**Bootstrap.** `block_bootstrap(metric, arrays, blocks)` and
`paired_bootstrap(metric, cand, base, blocks)`.

### `bench.harness.results`

`write`, `read`, `load_arrays`, `validate`, `find`, `assert_aligned`, `align`, `evaluate`.
See the protocol above.

## Leakage rules

- Every evaluation is rolling-origin. Anything fitted or tuned may use only data from strictly
  before the test week: parameters, hyperparameters, calibration maps, feature selection, and
  the choice of variant. Test-week actuals are used only to score.
- Stacking or ensembling from other candidates' result files must also be rolling-origin. The
  npz files hold the test outcomes (`y`), so weights for week t may use only blocks < t.
- **`inj` in the proj files is today's injury tag.** It is identical in all 35 files of both
  seasons (checked), so it is never point-in-time. The harness keeps it only as
  `table().inj_now` and strips it from every frame a candidate receives. Don't use
  `common.League.info[...]["inj"]` or `common.P_OUT` in a backtest either.
- `pos` and `name` are also current metadata; no position changes were seen.
- The pid universe is today's: the same 3045 players in every file. 2025 rows therefore miss
  players who have since left Sleeper's player list (survivorship).
- `team`, `opp` and `game` are point-in-time **only on rows with a real projection** (`opp`
  set). Every item has one: 2025 midseason trades appear in the week they happened, and no game
  has more than two teams. A row with no projection that week (bye, injured, inactive, not yet
  signed: 93,700 rows, all with proj 0 and none of them items) has no `opp`. Sleeper fills its
  team with the player's *current* team and its game with that team's game. Examples: Jaylen
  Waddle tagged DEN in 2025 weeks 12 and 18, and Chris Rodriguez Jr. flipping WAS↔JAX. The
  audit therefore blanks `team`/`game` on those 16,330 rows (`opp` is already blank) and keeps
  the raw value as the `LEAKY` column `team_raw`. This changes no item, `team_proj`, `opp_proj`
  or item `team_pos_rank`.
- Past-week projections are Sleeper's final pre-game projections, so players ruled out are
  already near 0, as the core model assumes. Projections for weeks *after* the test week were
  made after its games (an injured player's next-week projection drops to 0), so they are
  future information. `horizon()` hides them from `table()`.
- **Guards.** The player harness enforces as much as it can (the team track has the same guards,
  plus `team.view()` refusing later weeks and test-week outcomes):
  - each (variant, origin) task runs in a freshly forked process, so module-level caches cannot
    carry information from a later origin to an earlier one,
  - `fit`/`predict_quantiles` run inside `data.horizon(origin)`, so `data.table()` and its
    helpers show nothing after the test week's kickoff, and writing into them cannot reach
    the shared table.

  It **cannot** guard `data/*.json` read directly, or `common.League`, whose `info` holds today's
  team and injury tag, whose `proj` holds every 2026 week, and which also holds `matchups`.
  Don't use them in a player-track candidate. For league data, use `data.settings()`.

## Track: player

Weekly score quantiles for every projected player-week (module `bench/harness/tracks/player.py`).

- **Items.** Player-weeks with proj ≥ 0.5 **and** a game id. This mirrors
  `model.calibration_rows`; in the cached data every row with proj ≥ 0.5 has a game, so the
  second condition never binds. `item_id = "<season>-<week:02d>-<pid>"`, `block = t` of the
  test week.
- **Origins.**
  - full: 2025 weeks 5-18 plus 2026 weeks 1-7, which is 21 test weeks and 6417 items;
  - smoke: 2025 week 10 and 2026 weeks 3 and 7, which is 908 items.
- **Rolling origin.** For each test week the harness:
  1. builds a **fresh** candidate,
  2. calls `fit(train_df)`, where `train_df = data.train_before(origin)` holds every scored
     player-week strictly before the test week. That is all players, not only items, with the
     `item` flag and the actual columns, minus `inj_now`.
  3. calls `predict_quantiles(test_df, levels)`, where `test_df` holds the test week's items,
     pre-game columns only, plus `item_id`, in harness order.

  Steps 1-3 run inside `data.horizon(origin)`, in a process forked for this one task. numpy's
  and `random`'s global RNGs are seeded per origin, and `bench.harness.task_seed()` returns the
  same seed (identical for every variant). Seed your own generators from it (for example
  LightGBM `seed=task_seed()`) so runs are reproducible whatever the job count.

  `predict_quantiles` must return `(len(test_df), 99)`, **rows in `test_df` order**. The harness
  sorts each row (monotone quantiles), rejects non-finite values, and asserts that the test
  frames' item ids equal the scored items. Quantiles may go negative, since fumbles and
  interceptions can make scores negative.
- **Candidate class.** `Class(variant=NAME)` if the constructor takes `variant` (or `**kw`),
  else `Class()`.
- **CLI.**

  ```
  python -m bench.harness.tracks.player --candidate pkg.module:Class --key KEY \
      --variant A [--variant B | --variant C=pkg.mod:Other] --mode smoke|full [--jobs N] [--notes "..."]
  ```

  A file path works in place of the module: `bench/candidates/x/m.py:Class`.
  - Origins × variants run as one forked process per task, `--jobs` at a time. The default and
    the cap are both `CPUQ_SLOTS`.
  - Each task gets `HARNESS_THREADS = CPUQ_SLOTS // jobs`, and its BLAS is capped through
    threadpoolctl.
  - `--jobs 1` (or running outside cpuq) runs one task at a time, still isolated.
  - `--jobs 0` runs everything in-process, for debugging only: there is no isolation.
- **Python API.** `bench.harness.tracks.player.run(key, [(name, factory)], mode, jobs, notes)`,
  where each factory is a zero-argument callable returning a fresh candidate.
- **Output.**
  - Results: `bench/results/<key>[.smoke].json`.
  - npz arrays: `item_id`, `block`, `pos`, `crps`, `pinball`, `pit`, `cov10`, `cov50`, `cov90`,
    `cov80`, `width80`, `proj`, `y`, and `q` (float32, n × 99).
  - Per-week CRPS: `variants[i].by_week`.
- **Metrics.** `PRIMARY = "crps"`. Also `pinball`, `pit_maxdev`, `cov_err`, coverage
  10/50/90/80, `width80`, and `crps_<POS>` / `pit_maxdev_<POS>`.
- **Baseline** (`candidates/_baseline_player/ratio.py:EmpiricalRatio`). It re-implements
  model.py's marginal, refit on training items only:
  - actual/proj ratio quantiles on the 256-level grid,
  - by position and equal-count projection bucket (`nb = clip(n // 300, 5, 14)`),
  - interpolated in log projection between bucket medians,
  - read at the harness levels with the same linear grid interpolation that `sim.Universe.paths`
    samples from.

  `python -m bench.candidates._baseline_player.verify` checks the reproduction. Fit on
  model.py's rows (2025 weeks 1-18 + 2026 weeks 1-7) it gives the same 7715 rows, and the
  centers and tables equal `data/model.npz` with max |diff| = 0. Its predictions equal
  `PlayerModel.quantiles` with max |diff| = 0.

  The audit added an independent check on a second row set. It ran the core `model.fit` (its
  output redirected to a scratch folder) on 2025 + 2026 weeks 1-3. Against the baseline fit on
  `train_before((2026, 4))`, the tables and the 2026-04 predictions both match with max
  |diff| = 0.

  The only difference from the core code: a bucket left empty by tied edges is skipped. That
  can only happen on tiny training sets and never on model.py's rows. The sim's week-0 draw also
  applies today's injury tags (`P_OUT`), which are leaky for past weeks, so the baseline leaves
  them out on purpose.

  Smoke result: crps 2.7708, pinball 1.3854, coverage 10/50/90 = 0.118 / 0.497 / 0.884,
  cov80 0.767, pit_maxdev 0.022, n 908.

  Full result (queued job `_baseline_player`, 1.8 s on 6 cores): crps 2.7920, pinball 1.3960,
  coverage 10/50/90 = 0.124 / 0.505 / 0.890, cov80 0.766, cov_err 0.018, pit_maxdev 0.024,
  width80 12.15, n 6417. By position, crps is QB 4.48, RB 2.94, WR 2.64, TE 2.19.

  Known weakness (a target for candidates). The PIT is slightly U-shaped (bottom bin 0.124, top
  0.110), and the zero mass is too small: the predicted P(y ≤ 0) is 0.128 against 0.186
  observed (WR 0.149 vs 0.219, TE 0.194 vs 0.272). Interpolating ratio tables between bucket
  medians gives an atom equal to the *smaller* of the two neighbouring buckets' zero masses.
- Leaderboard: `python -m bench.harness.compare player`.

## Track: team

Who gets chopped, and the distribution of every team total and of the chop line, the week's lowest
total (module `bench/harness/tracks/team.py`).

- **Items.** One item per league-week. Each item has a `kind`, and `block` = `t` of the test week.
  - **real** (`r-2026-03`): this league, 2026 weeks 1-7.
    - Every alive team, meaning its roster had players that week, ordered by roster id.
    - Each team uses the starters it **actually played**, from matchups, so lineups are
      point-in-time.
    - Total = the sum of the starters' league-scoring actuals. This equals Sleeper's points to 3e-14.
    - Chopped = argmin. It equals data/backtest.json's chopped owner every week (both are asserted
      when the cache loads).
    - Only 7 items, so these metrics are noisy.
  - **virtual** (`v-2025-05-07` = season-week-replicate): deterministic synthetic guillotine
    leagues, `GEN["r"]` = 48 per scored week (2025 weeks 1-18, 2026 weeks 1-7).
    - League size N = 10 + r % 8, so 6 leagues of each size 10..17.
    - Teams fill the league's slots (QB, RB, RB, WR, WR, TE, FLEX, FLEX) in a seeded randomized
      snake draft from that week's projections. Only QB/RB/WR/TE projected > 0 are eligible, and
      each player goes at most once per league.
    - Team k drafts off its own board: log(proj) + σ_k·ε, with σ_k = 0.30·exp(0.5·z_k). It takes the
      best player who fits an open slot and puts him in the most restrictive slot he fits.
    - This noise spreads team strength like the real league:

      | | Virtual | Real |
      |---|---|---|
      | CV of projected starter totals | 0.096 | 0.096 |
      | Mean projected total | 90 at N=17 to 104 at N=10 | 92 at 17 teams to 103 at 11 |
      | CV of actual totals | 0.231 | 0.248 |
      | Chopped team's projection rank fraction | 0.27 | 0.28 |

      The calibration used projections only, never outcomes.
    - Total = the sum of the starters' real actual points that week. Chopped = argmin.
  - Every league is cached in `bench/cache/team_leagues_<sig>.json`, so every candidate sees
    identical leagues. The signature covers `GEN`, the generator code, each week's draft pool and
    the real starters. Results record it as `leagues_sig`, and a different signature means the
    baseline must be rerun.
- **Origins.**
  - full: the player track's 21 test weeks (2025 weeks 5-18, 2026 weeks 1-7). Virtual
    replicates r < 48 plus the real week give 1008 + 7 = 1015 items.
  - smoke: 2025 week 10 and 2026 weeks 3 and 7. Virtual r < 16 plus 2 real weeks give 50 items.

  Items are ordered by origin: virtual r = 0.. first, then the real league (`team.items(mode)`).
- **Rolling origin.** For each test week the harness:
  1. builds a **fresh** candidate,
  2. calls `fit(train_df, train_leagues)`:
     - `train_df = data.train_before(origin)`, as in the player track;
     - `train_leagues` holds every league of earlier weeks: all 48 virtual replicates and the real
       weeks, with outcomes. This holds in smoke mode too.
  3. calls `predict(league)` once per test league, in harness order.
- **League views.** These are fresh numpy arrays, safe to mutate.

  | Field | Meaning |
  |---|---|
  | `league_id`, `kind`, `season`, `week`, `t`, `n`, `slots` | Identity; `n` = N teams |
  | `team_ids` | (N,): roster ids (real) or 0..N−1 (virtual) |
  | `starters`, `starter_pos`, `starter_proj` | (N, 8) pid str (`''` = empty slot), position, league-scoring projection (0 if none) |
  | `roster` | N pid lists. Real: the week's full roster, bench included; virtual: the starters |
  | `owners` | Real only: display names |
  | test only: `week_df` | The test week's pre-game frame (`data.week_frame(..., items_only=False)`), shared by that week's leagues |
  | train only: `starter_pts` (N, 8), `points` (N,), `chopped`, `line` | Outcomes |

  Use only these views. Never read `data/matchups_*.json` or `data/2026/stats_*.json` for a test
  week, or rebuild views yourself: they hold outcomes. Treat `week_df` as read-only (it is shared).
- **Guards** (the same as the player track's):
  - each (variant, origin) task runs in a freshly forked process (`maxtasksperchild=1`), so
    module-level caches cannot carry a later origin's fit into an earlier one;
  - `HARNESS_SEED` (`bench.harness.task_seed()`) and numpy's/random's global RNGs are seeded per
    origin with `player.origin_seed`, the same for every variant (common random numbers);
  - `fit` and `predict` run inside `data.horizon(origin)`: `data.table()` and its helpers show
    nothing after the test week's kickoff, and `team.view()` raises `RuntimeError` for a league of
    a later week or for the test week's outcomes (`outcome=True`).

  Reading `data/` files directly is not guarded.
- **Candidate class.** `Class(variant=NAME)` if the constructor takes `variant` (or `**kw`), else
  `Class()`. `predict(league)` returns a dict with:
  - `p_chop`: (N,), ≥ 0. It is renormalized to sum to 1.
  - `total_q`: (N, 99) team-total quantiles at `metrics.LEVELS`. Rows are sorted.
  - optional `line_q`: (99,) quantiles of the week's minimum total. Without it the harness derives
    the chop line from `total_q` as if teams were independent (`team.line_from_marginals`, tails
    extended linearly to levels 0 and 1) and counts the leagues in `line_from_marginals`. That
    ignores shared NFL players and games, and its outer levels are extrapolated (vs the exact
    minimum of 4 independent normals: within 0.43 points at levels 0.05-0.95, up to 4.7 at the
    1% tail), so return `line_q` from your joint model.
  - Non-finite values or wrong shapes raise.
- **CLI.**

  ```
  python -m bench.harness.tracks.team --candidate pkg.module:Class --key KEY \
      --variant A [--variant B | --variant C=pkg.mod:Other] --mode smoke|full [--jobs N] [--notes "..."]
  ```

  Origins × variants run over `--jobs` forked worker processes (default and cap `CPUQ_SLOTS`), one
  fresh process per task, exactly as in the player track (`HARNESS_THREADS`, threadpoolctl).
  `--jobs 0` runs everything in-process (debugging only: no isolation). The Python API is
  `bench.harness.tracks.team.run(key, [(name, factory)], mode, jobs, notes)`.
- **Per-item arrays (npz).**
  - Scores and outcomes: `item_id`, `block`, `kind`, `n_teams`, `chopped` (index), `logloss`,
    `brier`, `rank`, `rank_norm`, `crps_total`, `crps_line`, `p_chopped`, `log_n`.
  - `tot_hist`: (n, 10) team-total PIT decile counts.
  - `rel`: (n, 27). For each of 9 p_chop bins: Σp, Σy and the count.
  - Forecasts, NaN-padded to 17 teams: `p` (n, 17), `y_tot` (n, 17), `q_tot` (n, 17, 99) float32.
  - The chop line: `line_q` (n, 99), `line_y`, and `line_src` (1 = derived from the marginals).
  - Outcomes are in here, so stacking must be rolling-origin.
- **Per-league scores.**
  - `logloss` = −log p[chopped], with p clipped to 1e-6 and renormalized.
  - `brier` = Σ_k (p_k − 1{k = chopped})².
  - `rank` = mid-rank of the chopped team by p, where 1 = most likely. Ties count half, so a uniform
    forecast gets (N+1)/2 and can't win by tying.
  - `rank_norm` = (rank − 1)/(N − 1). This is 0.5 for uniform.
  - `crps_total` = the mean over teams of the CRPS of `total_q` (2 × mean pinball, as in
    `metrics.crps`).
  - `crps_line` = the CRPS of `line_q` against the actual minimum.
  - The team-total PIT is randomized, hashed on `<league_id>-<k>`.
- **Metrics (`METRICS`).** `PRIMARY = "logloss"`.
  - **Virtual-only** (unprefixed):
    - `logloss`, `brier`, `rank_norm`, `rank`, `p_chopped` (higher is better);
    - `crps_total`, `crps_line`;
    - `ece` over the reliability bins 0 / .025 / .05 / .075 / .1 / .15 / .2 / .3 / .5 / 1;
    - `pit_maxdev_total`, `cov_err_total`, and `cov10/50/90/80_total`, where coverages are
      weighted by team.
  - **Real-only** (`real_` prefix): `real_logloss`, `real_brier`, `real_rank` (comparable to
    backtest.json's `model_rank_of_chopped`), `real_rank_norm`, `real_crps_total`,
    `real_crps_line`, `real_ece`.

  Leagues weigh equally. Logloss values aren't comparable across N, but a paired difference is.
  The JSON adds `uniform` (the uniform forecast's logloss / brier / rank_norm, for both kinds),
  plus `by_week`, `by_size` (virtual logloss vs log N) and `reliability` tables per variant. ECE is
  only a diagnostic: the uniform forecast has ECE 0.
- **Baseline** (`candidates/_baseline_team/copula.py:CopulaSim`): the sim's own one-week chop
  model, refit per origin on train items.
  - Marginal: `_baseline_player`'s tables.
  - Dependence: model.fit's team/game Gaussian copula (`fit_copula`).
  - Simulation: `sim.Universe.paths`' week-0 draw, verbatim. It uses 96000 draws per week over
    every QB/RB/WR/TE projected > 0. The rng is `default_rng([11, season, week, chunk])`, and every
    league of the week shares the draws (common random numbers).
  - Totals are the sum of the starters' draws, with no bench subs. `p_out = 0`, because the inj tags
    are leaky and past projections already zero players ruled out. There is no drift.
  - It returns `line_q` from the per-draw minimum.
  - `python -m bench.candidates._baseline_team.verify` checks four things:
    - the copula fit on model.py's rows reproduces data/model.npz exactly;
    - the sampler equals `Universe.paths` bit for bit;
    - with model.npz's parameters, the real items reproduce backtest.json's P(chopped team), within
      ≤ 2.7 MC SE (log-loss 2.382 vs 2.384);
    - the harness-order refit is reported: its TE loading differs by 0.017, because tie draws land
      on other rows.
  - Smoke result:

    | Leagues | Logloss (uniform) | Brier (uniform) | Other |
    |---|---|---|---|
    | virtual, n = 48 | 2.0761 (2.5879) | 0.8250 (0.9237) | rank_norm 0.229; crps_total 10.94; crps_line 5.54; ECE 0.024; total coverage 10/50/90 = 0.076 / 0.454 / 0.894 |
    | real, n = 2 | 1.851 (2.553) | — | — |

  - Full result (queued job `_baseline_team`, 49 s on 6 cores, `leagues_sig` 41649c3a7b189438):

    | Leagues | Logloss (uniform) | Brier (uniform) | Other |
    |---|---|---|---|
    | virtual, n = 1008 | 2.2485 (2.5879) | 0.8586 (0.9237) | rank_norm 0.276; crps_total 11.66; crps_line 5.79; ECE 0.006; total coverage 10/50/90/80 = 0.101 / 0.480 / 0.885 / 0.783; pit_maxdev 0.015 |
    | real, n = 7 | 2.3696 (2.6287) | 0.8899 (0.9271) | mean rank 4.57; crps_line 2.99 |

  - Run time: the smoke takes 7-9 s on 2 cores. Full mode takes about 6 s per origin on 1 core (12 s
    with 6 running at once), 49 s in all on 6 cores, about 0.6 GB per worker.
- **Audit** (independent, before candidates):
  - Rolling-origin baseline on the 7 real weeks: log-loss 2.370 vs backtest.json's in-sample
    2.384 (uniform 2.629). P(chopped) per week is within 0.014 of backtest.json and ranks of the
    chopped team agree except week 4 (4 vs 3). The gap is the refit on fewer weeks plus MC noise.
  - The 0.017 TE-loading gap of the harness-order refit is tie-draw noise: refitting model.py's
    own rows with seeds 7-12 spans 0.017 in the TE loading and 0.014 in g.
  - Real items: the alive sets equal backtest.py's (players and points > 0), no empty starter
    slots, every starter in the table, mean projected and actual totals equal backtest.json's.
  - The baseline is deterministic (a rerun's npz is bit-identical) and unchanged by the guards.
  - The league cache holds no outcomes (ids, sizes, starters, real rosters/owners only); virtual
    leagues have no empty slots or duplicate players.
  - One full-mode item, `v-2025-18-20`, has a tie for the lowest total (58.28); `chopped` takes the
    lower team index. It is the same for every candidate.
- Leaderboard: `python -m bench.harness.compare team`.

## Track: bids

Rival FAAB bids (module `bench/harness/tracks/bids.py`): who bids on which player, how much,
and the top bid each offered player clears at.

- **Chances.** `bids.chances()` has one row per (waiver run, alive team with budget > 0,
  offered player the team does not hold), with **point-in-time** rosters and budgets:
  - `bids.rosters_at(T)` is the roster of every team just before time T (ms): today's
    `data/rosters.json` with every completed transaction at or after T rolled back. It equals the
    end-of-week matchup roster of every alive team in legs 3-4.
  - `auction.bid_history` (BidModel's rows) takes rosters from the end-of-week
    `matchups[leg].players` instead. Those snapshots keep players dropped after their game and, in
    legs 5-7, already hold some adds of that leg's post-chop waiver run (legs 5 / 6 / 7: 3 / 3 / 7
    teams differ from the roster at the chop). Two problems follow. It skips a claimant that already
    shows the claimed player, which drops 12 claims (three of them positive bids: $20, $7, $10).
    And the run's own outcome leaks into `gain`, because a winner's other chances are computed on
    a roster that already holds what he won. With point-in-time rosters, 64 of BidModel's test-leg
    rows (8 of them bids) get a different `gain`, by up to 2.3 points a week.
  - The point-in-time set is BidModel's 1564 rows plus 13 more (`repaired = True`, train only):
    the 12 dropped claims and one team that picked a claimed player up later in the week. Every
    claim with budget > 0 has a row. Budgets match `bid_history`'s on every shared row.
  - The outcome is `bid > 0`. A $0 claim counts as no bid, as in BidModel (38 such claims).
    `bid` never exceeds `budget`.
  - Offered players are the chopped roster (`pool`) in post-chop runs, plus every player someone
    claimed.
  - Waiver runs: every leg has one post-chop run (Wednesday after the chop: `post_chop`,
    `w0 = leg + 1`, `info_week = leg`). Legs 3-7 also have one or two pre-chop runs (Friday of
    week `leg`: `w0 = leg`, `info_week = leg − 1`), which offer only claimed players and drew no
    positive bid.
- **Groups (price items).** One per (run, offered player).
  - `top` = max bid over the group's chances, 0 = nobody bid. This is the clearing price of a
    sealed first-price auction. `contested` = `top > 0`.
  - `cleared` = the completed claim's bid, −1 = none. It differs from `top` in 4 contested test
    groups, each one where the top claim failed and nobody won. Price metrics score `top`.
- **Items.** The test legs' chances `c<leg:02d>-<run>-<pid>-<rid:02d>`, then their groups
  `p<leg:02d>-<run>-<pid>`; the `kind` array says `chance` or `price`. `block` = leg.
- **Origins.**
  - full: test legs 3-7, which is 1033 chances (98 positive bids) + 87 groups (36 contested)
    = 1120 items;
  - smoke: legs 5 and 7, which is 371 chances (35 bids) + 35 groups (13 contested) = 406 items.

  824 of the 1033 test chances, 96 of the 98 bids and 34 of the 36 contested groups are on pool
  players.
- **Rolling origin.** For each test leg the harness:
  1. builds a **fresh** candidate, in a freshly forked process (`maxtasksperchild=1`), with
     numpy's and `random`'s global RNGs and `HARNESS_SEED` (`bench.harness.task_seed()`) seeded
     per leg, the same for every variant;
  2. calls `fit(train_df)` once, where `train_df = bids.train_before(leg)` holds every chance of
     legs strictly before the test leg, outcomes included;
  3. calls `predict(test_df)` **once per waiver run of the leg**, in time order, where `test_df`
     holds that run's chances of `bids.test_frame(leg)` in harness order, without `bid`, `claim`,
     `won`, `repaired` and `gain_core`. (A repaired row exists only because of a claim.)
  - **Guard.** `fit` runs inside `bids.guard(leg, first run of the leg, leg)` and each `predict`
    inside `bids.guard(leg, run, info_week + 1)`. Inside it:
    - `bids.chances()`, `groups()`, `truth()`, `items()` and `train_before()` show only legs
      before the test leg;
    - `bids.rosters_at(T)` raises `bids.LeakError` for T after the run being predicted;
    - `data.table()` (and its helpers) is `data.horizon((2026, week))`. A predict call sees the
      actuals of weeks ≤ that run's `info_week` and nothing after week `info_week + 1`.
  - The CLI runs as the canonical module `bench.harness.tracks.bids`, so a candidate that imports
    it shares the guard.
- **Columns.**

  | Column | Meaning |
  |---|---|
  | `item_id`, `leg`, `run`, `rid`, `pid`, `name`, `pos` | Identity; `run` = waiver run timestamp (ms) |
  | `budget` | FAAB just before the run (traded FAAB included) |
  | `gain` | Avg weekly lineup gain over weeks w0..w0+3 on the point-in-time roster (`auction.gain`, the same call run.py makes live) |
  | `proj0`, `proj_h` | Player proj in week w0 / mean over the gain horizon |
  | `pool` | Player came from the chopped roster (public before the run) |
  | `post_chop`, `w0`, `info_week` | Run is after the leg's chop; first horizon week; last week final at the run |
  | `n_teams`, `faab_left` | Teams with budget > 0 in the run; their total FAAB |
  | train only: `bid` (0 = none), `claim`, `won` | Outcomes |
  | train only: `repaired`, `gain_core` | Row is not one of `bid_history`'s; BidModel's own `gain` on the end-of-week matchup roster (NaN on repaired rows). Kept so the baseline can fit exactly as the core does |

- **Candidate class.** `Class(variant=NAME)` if the constructor takes `variant` (or `**kw`),
  else `Class()`. `predict(test_df)` returns a dict with:
  - `p_bid`: shape (n,), P(bid > 0);
  - the bid given bid > 0, as **one** of:
    - `size_pmf`: (n, K), P(bid = k) for k = 0..K−1 dollars. Column 0 is ignored; mass past
      the grid goes to its top.
    - `size_q`: (n, 99), quantiles at `metrics.LEVELS`.
    - `size_samples`: (n, S), draws.

    `size_q` and `size_samples` are rounded to whole dollars, clipped to 1..budget and
    histogrammed, then mixed with 1% uniform over 1..budget. This keeps log scores finite but
    shifts scores a little, so return `size_pmf` for exact control.
  - The harness always caps mass above the budget onto the budget, and scores every pmf on
    dollars 0..KMAX (`bids.kmax()` = 242, the largest budget of any chance).
  - Optional `sample_joint(test_df, rng, n)`: called once per run, right after that run's
    `predict`, with `rng = np.random.default_rng([1000 + leg, k])` for the leg's k-th run. It
    returns (len(test_df), n) whole-dollar bids (0 = no bid), where column s is one joint draw
    over the run's chances. If it is present, the price distribution is the per-group max of
    `--draws` (default 4000) draws. Otherwise it is the **exact** independent product
    `CDF_top(k) = Π_j [(1 − p_j) + p_j F_j(k)]` of the marginals, which is what
    `auction.top_rival_bids` samples from.
- **CLI.**

  ```
  python -m bench.harness.tracks.bids --candidate pkg.module:Class --key KEY \
      --variant A [--variant B | --variant C=pkg.mod:Other] --mode smoke|full [--jobs N] [--draws S] [--notes "..."]
  ```

  Parallelism is the same as the player track. Legs × variants run as one forked process per
  task, `--jobs` at a time (default and cap `CPUQ_SLOTS`), with `HARNESS_THREADS` threads each.
  `--jobs 0` runs everything in-process, for debugging only: there is no isolation.
- **Python API.** `bench.harness.tracks.bids.run(key, [(name, factory)], mode, jobs, notes, draws=4000)`.
  The scoring helpers are importable too: `size_pmf`, `pmf_from_points`, `dcrps`, `dq`, `dpit`,
  `joint_pmf`, `price_pmf`, `price_pmf_draws`, `score`.
- **Scores.** Every bid is a whole dollar, so all CRPS values are **exact** discrete CRPS
  (`Σ_k (F(k) − 1{k ≥ y})²`), not the 99-quantile approximation. Log scores floor
  probabilities at 1e-6. PITs are randomized with `metrics.item_uniform`. The audit checked the
  joint log score, the CRPS (against numeric integration), `top_logloss`, `price_logs`,
  `price_crps` and the exact price pmf (against a 400k-draw Monte Carlo of the max) by brute force.
- **Output.**
  - Results: `bench/results/<key>[.smoke].json`, with `by_leg` per variant.
  - Mask arrays: `m_chance`, `m_bid` (chance with bid > 0), `m_group`, `m_contested`, `m_pool`.
  - Other per-item arrays: `item_id`, `block`, `kind`, `rid` (−1 for groups), `pid`, `pos`,
    `y` (bid, or the group's top), `y_bid`, `p` (p_bid, or the group's P(contested)), and
    every loss below. `size_q50` is the predicted median bid, `price_q10/50/90` and `price_p0`
    hold the price distribution, and `cleared` is the cleared price.
- **Metrics.** `PRIMARY = "logs_pool"`, the joint log score on chopped-roster chances. That is
  what `run.py` and `dynamic.py` price. Non-pool rows are offered only because someone claimed
  them, so they are kept out of the primary metric. Each metric below has a `_pool` twin
  restricted to pool players.
  - Per chance:
    - `logs`: the joint log score of the bid over {0, 1..budget}, = `logloss` + `size_logs` on
      bidders;
    - `logloss`: participation;
    - `brier`, `ece`, `bid_crps` (CRPS of the joint bid);
    - `p_mean` / `y_mean`: mean p and the observed bid rate.
  - Per bid, given bid > 0: `size_crps`, `size_logs`, and `size_pit` (mean; 0.5 when calibrated).
  - Per group:
    - `top_logloss`: P(anyone bids);
    - `top_crps`: unconditional top bid, 0 included.
  - Per contested group, with the price distribution conditioned on top > 0 (the items are
    selected on that):
    - `price_crps`, `price_logs`;
    - `price_in80`: q10 ≤ top ≤ q90 with integer quantiles, the inside-p10-p90 rule of
      `auction.price_backtest`. Unlike price_backtest, the distribution here is conditioned on
      top > 0 and every contested group counts, not only tops ≥ $15. Atoms push it above 0.8.
    - `price_cov80`: randomized-PIT coverage, exactly 0.8 when calibrated;
    - `price_cov_err` = |price_cov80 − 0.8|;
    - `price_pit`, `price_width80`.
- **Baseline** (`candidates/_baseline_bids/bidmodel.py:BidModelRefit`). This is BidModel's spec
  refit per origin; see that folder's README.
  - `baseline` (the reference) is the core pipeline as deployed. It fits on BidModel's own rows
    with BidModel's own feature (`repaired == False`, `gain_core`). That is exactly
    `auction.BidModel` on the league cut off before the test leg. It predicts from the
    point-in-time `gain`, as run.py does from today's rosters. P(bid > 0) is the sampler's: a
    draw rounding to $0 is no bid.
  - `logit`: the same fit, with the raw logistic `p_bid`.
  - `clean`: the same spec fit on the training frame a candidate sees (every chance,
    point-in-time `gain`). It isolates what cleaner training features alone buy.
  - `python -m bench.candidates._baseline_bids.verify` checks five things:
    - the 1564 rows, identical and in order;
    - the full-league fit: max |diff| = 0 on parameters, `p_bid` and `mu`;
    - at every test leg L, `auction.BidModel` on the league cut off before L, against the
      baseline fit on the same rows in `bid_history` order: max |diff| = 0. In harness order the
      test-leg `p_bid` agree within 3e-4 and `mu` exactly. The logistic stops elsewhere on the
      flat a0 / a_log_budget ridge, as BidModel itself does between processes;
    - the point-in-time rosters;
    - the exact price pmf against 200k `top_rival_bids` draws (max |CDF diff| 0.0028) and the
      guard.

    Results are identical across `PYTHONHASHSEED` values.

  Smoke result (`baseline`): logs_pool 0.8229, logs 0.6944, logloss 0.2674, size_crps 12.04,
  size_logs 4.527, price_crps 10.02, price_cov80 0.846, n 406 (371 chances, 35 bids,
  13 contested). Against it, `logit` scores logs_pool −0.0022 and `clean` −0.025 (CI
  [−0.051, +0.007] over 2 blocks; size_logs_pool −0.24). The smoke takes 1.4 s on 2 cores.

  Full result (queued job `_baseline_bids`, 2.0 s on 6 cores), `baseline`: logs_pool 0.8615,
  logs 0.7203, logloss 0.2938, brier 0.0766, ece 0.036, size_crps 11.73, size_logs 4.496,
  size_pit 0.660, price_crps 9.37 (pool 9.60), price_logs 4.04, price_cov80 0.667, price_in80
  0.694, price_pit 0.540, price_width80 42.6, top_logloss 0.645, n 1120. By leg 3-7, logs_pool
  is 1.154 / 0.718 / 1.035 / 0.726 / 0.558. Paired against it over 5 blocks:
  - `logit`: logs_pool −0.0013 [−0.0026, −0.0000];
  - `clean`: logs_pool −0.030 [−0.072, +0.007], size_logs_pool −0.27 [−0.62, +0.05].

  Known weakness (a target for candidates). Out of sample, bid sizes run high but swing by leg.
  In a diagnostic over legs 3-7:
  - the share of bids above the predicted median was 0.86 / 0.43 / 0.91 / 0.65 / 0.58
    (`clean`: 0.77 / 0.43 / 0.78 / 0.55 / 0.67);
  - the median predicted bid was $3 / 13 / 3 / 7 / 18, against realized medians of
    $18 / 8 / 15 / 12 / 31.

  With about 20 bids per leg, the lognormal size fit on a few dozen bids is unstable.
- **Track-specific leakage.**
  - **Look-ahead features.** `gain`, `proj0` and `proj_h` use Sleeper's *final pre-game*
    projections for weeks w0..w0+3, which postdate the run. The cache has no point-in-time
    projections, so this is unfixable. It is the same for every candidate, but it makes
    projection-heavy models look better than they will be live.
  - **Selected items.** A non-pool player is offered only because some team claimed him in
    that run, so every non-pool group holds exactly one claim. Learn this from training rows if
    you like, but the primary metric (`logs_pool`) ignores those rows.
  - **Rosters.** Use `bids.rosters_at(run)` (or earlier times) for roster features. **Never use
    `matchups[...]["players"]`** from `data.matchups()` or `lg.matchups`. For a post-chop run of
    leg L (`info_week = L`), week L's snapshot already holds some winners' claims from that very
    run. `starters`, `points` and `players_points` of weeks ≤ `info_week` are fine.
  - **The live league object holds everything.** `data.league()` has every leg's `tx`, today's
    `rosters` and `faab`, and the guard cannot see into it. Inside fit/predict:
    - never read `lg.tx`, `lg.rosters`, `lg.faab`, `bid_history`, `BidModel` or
      `price_backtest` for anything at or after the run being predicted (fit: the test leg's
      first run);
    - projections (`lg.proj`) are fine, with the look-ahead caveat above.
  - As everywhere, never use `inj`.
- Leaderboard: `python -m bench.harness.compare bids [--mode smoke]`.

## Track: policy

My FAAB bidding policy, scored inside the season-with-auctions sim (module
`bench/harness/tracks/policy.py`, sim `bench/harness/dyn_policy.py`).

**This track is in-sim only.** No real outcome is ever scored. Every path is a draw from the
core models (the player score model with its copula, and BidModel rivals), started from the
real current league state that `League()` reads: the week (now 8), the 10 alive teams, rosters,
budgets (mine is $165), and the pending chopped roster when there is one. So a number here says
how well a policy does against the sim's rivals. A policy that leans on the exact BidModel the
rivals follow is flattered. Read the two robustness conditions next to the main one.

- **Sim.** `dyn_policy.py` is a copy of `dynamic.py` (core stays untouched) with two opt-in
  setting keys:
  - `policy`: my bids come from a policy object;
  - `bm_rival`: rivals bid from a different BidModel than the one my policy sees.

  It also returns per-path outcomes. Without those keys it matches `dynamic._paths_chunk` bit
  for bit, and with the baseline policy object the hook reproduces dynamic.py's rule bit for bit.
  `python -m bench.candidates._baseline_policy.verify` checks both, plus the BidModel refit,
  `State.p_win`, determinism across processes and the isolation below (about 30 s on 1 core,
  tuning seeds only).
- **Fidelity (audited).** On run.py's own sweep paths (seed 11, 50 chunks = 10,000 paths) the
  track's env + policy hook + `Multiplier` give P(win) 0.4272 / 0.4275 for 0.75/0.75 and
  1.0/0.75, against 0.4273 / 0.4277 in `out/results.json` (e_weeks and spend agree to 3e-4 and
  $0.01). The leftover 1-2 wins in 10,000 are the core fit's hash-order drift.
- **Deterministic point model.** `auction.bid_history` builds its rows by iterating a set of pid
  strings, so the core `BidModel(lg)` fit changes with `PYTHONHASHSEED` from one process to the
  next. The participation terms move by up to about 4e-3, which would quietly break pairing
  across runs. The track therefore refits the same rows in a fixed order
  (`dyn_policy.canonical_bm`) and uses that as `state.bm` and as the main condition's rivals. Two
  smoke runs then give bit-identical npz files.
- **Items.** Season paths. Path `c` of CRN seed `s` has `item_id = "<state>-s<sss>-<ccc>"` and
  `block = s` (one seed is one chunk of `C = 200` paths). `<state>` is a 12-character content
  hash of the data snapshot (meta, league, users, rosters, schedule, model.npz, tx_*,
  matchups_*, the season's proj_*) plus the track version. After `fetch.py` changes data/, items
  change and old results land under `skipped` in the leaderboard, so rerun the baseline first.
  Seed `s` drives the same streams as `dynamic.run(seed=s)` chunk 0: scores, injuries, rival bid
  noise. Every variant and condition runs on the same paths (common random numbers).
- **Held-out seeds.**
  - Evaluation uses fixed seeds below 1000: smoke 100-104 (1,000 paths), full 100-399 (60,000
    paths). run.py's own sims use seeds 11, 12, 13 and 21, so no eval path was used to pick
    run.py's multipliers.
  - Candidates may tune **only** on seeds ≥ 1000, through
    `bench.harness.tracks.policy.tune(policies, seeds=range(1000, 1010), conditions=("",))`. It
    raises on any seed below 1000 and returns `p_win`, `se`, `e_weeks`, `avg_spend` and the
    per-path arrays for each policy object. It parallelizes over `CPUQ_SLOTS`, so call it under
    cpuq, for example from a candidate's constructor or run script before `run()`.
  - Tune against all three conditions (`conditions=("", "_stress", "_bmboot")`), or at least
    `""` and `"_bmboot"`. A policy tuned only on the main condition is tuned to the exact
    rivals it can read in `state.bm`, which is the optimism this track warns about.
  - Never call `dyn_policy._paths_chunk` or `_chunk` yourself on seeds below 1000.
- **Conditions.** All three run on the same paths. My policy always sees the point model in
  `state.bm` and is never told which condition it is in.

  | suffix | rivals bid from |
  |---|---|
  | (none) | the BidModel fit on all bid history (= dynamic.py) |
  | `_stress` | the same model with `rival_boost = (0.7, 0.3)`: about 2x as often and 35% higher (= run.py's STRESS) |
  | `_bmboot` | a BidModel refit on a cluster bootstrap of the bid history: whole lots (waiver run × player) resampled with replacement, one replicate per seed (`dyn_policy.boot_bm`), identical for every candidate |
- **Policy API.** `Class(variant=NAME)` (or `Class()`) is built once per variant in the parent
  process.
  - Required: `bid(player, gain, week, state) -> dollars`. The harness rounds the result to
    whole dollars and caps it at my budget, and anything below $1 is no claim.
  - `bid` is called for every player on offer in every auction I'm alive for with money left.
    `gain` is my lineup gain: the mean over the next ≤ 4 weeks of max(0, proj − the weakest
    starter he would replace). `week` is the NFL week whose chop fed the auction; the next week
    played is `week + 1`.
  - Optional: `reset()` runs at the start of every path. Carry nothing across paths.
  - Optional: `begin(state)` runs once per auction, before the bids, for policies that plan
    the whole pool at once (budget split, knapsack). Like `bid`, it is only called when I'm
    alive with money left.
  - Workers fork after construction, so heavy read-only setup in `__init__` is shared.
  - **Isolation.** Every (seed, variant, condition) runs on a fresh `copy.deepcopy` of the
    built object, with numpy's and `random`'s global RNGs reseeded per (seed, condition). The
    League, Universe, BidModel, PlayerModel and base state are shared, not copied. So results
    never depend on `--jobs`, worker scheduling or the other variants in the run (`--jobs 1`
    and `--jobs 2` give bit-identical npz files). Keep other big read-only tables at module or
    class level, or make them cheap to copy. An object that cannot be deep-copied is reused,
    with a warning. Use `state.rng` for randomness, not the global RNGs.
  - Never touch underscore attributes or `dyn_policy._G`. Never inspect the call stack. A policy
    must not see the sim's draws.
- **`state` (`dyn_policy.State`).** Plain values, rebuilt for every auction.

  | Field | Meaning |
  |---|---|
  | `me`, `week` | My roster id and the NFL week whose chop fed this auction |
  | `wi`, `weeks` | `week`'s index in `weeks` (−1 for a pending pool) and the sim's weeks, 8-16 |
  | `weeks_left`, `auctions_left` | Weeks still to play, and auctions still to come, after this one |
  | `alive`, `n_alive` | The alive teams (me included) and their count |
  | `chopped` | The roster on offer (None for a pending pool) |
  | `pool`, `gains` | The pids on offer, and `{pid: my gain}` |
  | `budget`, `budgets`, `rival_budgets` | My budget, every alive team's, and the rivals' |
  | `roster`, `rosters` | My roster, and `{rid: tuple}` for every alive team |
  | `fut` | The week indices behind the gains |
  | `bm` | The point BidModel |
  | `rng` | The policy's own `random.Random`, seeded per path. It never touches the sim's streams |
  | `min_gain` | 2.0, dynamic.py's cutoff |

  Helpers:
  - `proj(pid, wi)`, `ros(pid, wi=None)`: projected points, and the rest-of-season sum from the
    next week played by default;
  - `pos(pid)`;
  - `rival_gains(pid)`: `{rid: gain}` for rivals with money;
  - `typical(g, budget=None, week=None)`: the league-typical bid, `exp(mu(me, ...) − u_me)`;
  - `rival_cdf(pid, x)`: P(highest rival claim ≤ x) under the point model with no stress. Each
    rival is treated independently, which ignores money spent earlier in the same run.
  - `p_win(pid, b)`: P(bid b beats every rival claim), ties split evenly. It agrees with a Monte
    Carlo of `auction.top_rival_bids` within about 0.0035 (MC noise; verify.py asserts < 0.01).
- **Mechanics** (unchanged from dynamic.py):
  - claims are processed from the highest bid down, each winner paying its own bid;
  - a claim is skipped if the bidder can no longer afford it;
  - the roster cap is 15: the weakest rest-of-season player is dropped;
  - a chopped team's unspent budget is gone;
  - rivals draw their bids before seeing mine.
- **CLI.**

  ```
  python -m bench.harness.tracks.policy --candidate pkg.module:Class --key KEY \
      --variant A [--variant B | --variant C=pkg.mod:Other] --mode smoke|full [--jobs N] [--notes "..."]
  ```

  - Seeds run in parallel over `--jobs` forked workers (default and cap: `CPUQ_SLOTS`). Each
    task runs every variant and condition on one seed's paths.
  - `env()` hashes data/ before and after loading it and raises if a fetch changed it mid-load.
    A candidate run whose state differs from the baseline file's prints why it has no
    `vs_baseline`.
  - Python API: `bench.harness.tracks.policy.run(key, [(name, factory)], mode, jobs, notes)`.
- **Output.**
  - Arrays, per condition suffix: `item_id`, `block`, `win`, `weeks` (weeks survived, 0-9),
    `spend`, `buys` and `studs` (players bought with gain ≥ 8).
  - Each variant record carries `metrics` (below, plus `p_win*_se` = binomial se) and
    `vs_baseline`. The latter is the paired difference against the baseline's `baseline`
    variant on the same paths, for `d_p_win`, `d_e_weeks` and `d_avg_spend` in each condition,
    with a block (seed-cluster) se and the share of paths where the two disagree. It comes from
    the baseline file, or from this run when the run is the baseline.
  - Extra fields: `in_sim: true`, `state` (week, alive, budgets, pending), `state_sig`,
    `seeds`, `paths_per_seed` and `conditions`.
- **Metrics.**
  - Higher is better: `p_win`, `p_win_stress`, `p_win_bmboot`, `p_win_min` (the worst of the
    three) and `e_weeks{,_stress,_bmboot}`.
  - No direction: `avg_spend*`, `buys*`, `studs*`.
  - `PRIMARY = "p_win"`. The leaderboard shows paired differences with a block-bootstrap 95% CI
    over seeds, the honest CI for `_bmboot`, since each seed carries its own rival model.
- **Baseline** (`candidates/_baseline_policy/multiplier.py:Multiplier`). This is dynamic.py's
  rule: no bid when gain < 2, otherwise `round(m × typical(gain))`, with `m = m_early` before
  week 12 and `m_late` from it on. Variants:
  - `baseline`: 1.0 / 0.75, the reference. This is the policy the pipeline actually runs:
    run.py's sweep argmax on this snapshot (`dynamic.best` in `out/results.json`, picked on
    seed-11 paths, disjoint from the eval seeds). Its P(win) is the cheat sheet's
    `p_win_policy`, and it drives the budget curve and the forced-bid grid.
  - `m0.75`: 0.75 / 0.75, the runner-up in that sweep (0.4273 vs 0.4277, se 0.005, a tie). The
    cheat sheet's text says "bid 0.75-1x".
  - `x1.0`: 1.0 / 1.0.

  Smoke result (1,000 paths, state `0549ea074558`):

  | variant | p_win | stress | bmboot | e_weeks | spend | paired d_p_win |
  |---|---|---|---|---|---|---|
  | `baseline` (1.0/0.75) | 0.433 ± 0.016 | 0.436 | 0.404 | 6.98 | $143 | (reference) |
  | `m0.75` | 0.454 ± 0.016 | 0.435 | 0.417 | 6.90 | $127 | +0.021 ± 0.012 |
  | `x1.0` | 0.426 ± 0.016 | 0.427 | 0.405 | 6.98 | $145 | −0.007 ± 0.005 |

  Five blocks only, so treat the smoke numbers as a plumbing check. The 0.454 is seed luck:
  0.75/0.75 gives 0.4272 on tuning seeds 1000-1049 (10,000 paths). The smoke takes 16 s on 2
  cores and 27 s on 1.

  Full result: queued job `_baseline_policy`, 60,000 paths, 300 seeds, 287 s on 6 cores.

  | variant | p_win | stress | bmboot | e_weeks | spend | d_p_win [95% block CI] |
  |---|---|---|---|---|---|---|
  | `baseline` (1.0/0.75) | 0.4251 ± 0.0020 | 0.4378 | 0.4267 | 6.88 | $141 | (reference) |
  | `m0.75` | 0.4310 | 0.4369 | 0.4340 | 6.83 | $126 | +0.0059 [+0.0031, +0.0087] |
  | `x1.0` | 0.4208 | 0.4325 | 0.4215 | 6.88 | $143 | −0.0043 [−0.0058, −0.0028] |

  So run.py's argmax (1.0 early) was noise in its 10,000-path sweep. 0.75/0.75 is better in
  the main and bootstrap conditions and ties under stress. A paired difference of about ±0.003
  is the smallest effect a candidate can resolve at 60,000 paths.
- **Making a candidate.** Copy `candidates/_baseline_policy/run_*.sh` and point `--candidate` at
  your class. Run the smoke through cpuq. It must take at most 90 s on 1 core.
- **Compute budget.** The full run is 60x the smoke's paths on 6x the cores, so its wall time is
  about 10x the smoke's 1-core `sim_s`. Keep that `sim_s` under about 80 s.
  - The baseline costs about 3 ms per path per variant-condition, roughly 540 core-s per variant
    in the full run.
  - A full run makes about 16M `bid()` calls and 1.1M `begin()` calls per variant, because each
    path has about 6.3 auctions and 90 bids.
  - With 3 variants inside 15 min x 6 cores, keep `bid()` under about 50 µs on average, or
    `begin()` under about 1 ms.
  - Do expensive planning once per auction in `begin()`, then cache the bids.
  - Enqueue the full job like any other.
- Leaderboard: `python -m bench.harness.compare policy`.

## Track: mc

Estimator efficiency for the sim's two headline numbers (module `bench/harness/tracks/mc.py`).
The question here is how to get the same probabilities for less CPU. **This track is in-sim
only.** No real outcome is scored. The "truth" is the core model's own expectation, taken from a
very large plain-MC run. So an estimator must sample the same model; changing the model is a
question for the player and team tracks.

- **Targets.** `mc.build_targets()` builds both targets from today's real state, the one
  `League()` reads:
  - the week (now 8) and the 10 alive teams with their current rosters;
  - a `sim.Universe` over weeks 8-16 holding every rostered player of an alive team;
  - each team's projection-optimal lineup with two bench subs per slot (`Universe.lineup_idx`),
    the same lineups `run.py` hands `sim.run_season`.

  Week 8's `p_out` comes from today's injury tags. That is correct here, because this is the
  current week.

  | target | estimate (T = 10 values) | each path plays |
  |---|---|---|
  | `week` | P(chopped this week) per alive team (= run.py's `p_chop_now`) | week 8 only |
  | `season` | P(win the league), rosters frozen (= `sim.run_season`'s `win`, run.py's `p_win_frozen`) | weeks 8-16, with level drift and injury hazard |
- **Reference.**
  - It is plain MC (`mc.plain`): 64M week paths and 16M season paths in full mode, and 1.6M and
    240k in smoke mode. Chunk `ci` is drawn from `rng([REF_SEED, ci])`, seeds disjoint from
    every replication's.
  - It is built only by the queued `_baseline_mc` job (`--make-ref`, about 1700 core-s) and cached
    at `bench/cache/mc_ref_<mode>_<ref_sig>.json`. The cache holds p, the exact binomial SE, a
    batch-means SE (as a check), CPU time and the team order.
  - The build checkpoints finished batches to `mc_ref_<mode>_<ref_sig>.partial.json` every 15 s.
    If the job is killed (timeout), re-queue it: it resumes and produces the same counts, because
    each batch is keyed by its first chunk.
  - A full run of any other key exits if the reference is missing, and so does
    `--ref-only --mode full` without `--make-ref`. The smoke reference is built by whichever smoke
    run first needs it (about 30 core-s, deterministic).
  - Full-mode SE: at most 5.2e-5 (week) and 9.5e-5 (season).
  - Reference variance ÷ plain-MC MSE = n/N: 6e-4 (week) and 2e-3 (season). A candidate up to
    roughly 50× better than plain MC is therefore still resolved. `eff_rel_adj` subtracts the
    reference variance from both MSEs.
  - The smoke reference agrees with run.py's last standings (640k paths, from a Universe that
    also holds free agents) to max |z| = 2.1 over the 20 numbers.
- **Items.** One item per replication (`R`: full 160, smoke 8). `block = r`.
  - `item_id = "<sig>-r<rrr>"`. `<sig>` hashes the state files (meta, league, rosters, users,
    schedule, `model.npz`, the season's `proj_*`), `common.py`, `model.py`, `sim.py`, the source
    of mc.py's target and reference code (`Target`, `build_targets`, `week_counts`,
    `season_counts`, `plain`, `_ref_task`) and the mode's config. Editing any of these
    invalidates both the cached reference and old results.
  - `data/model.npz` is what run.py's `fit_model` writes. Refitting `model.fit` on today's data
    into a scratch folder reproduces it bit for bit (audited), so the targets are the cheat
    sheet's own model.
  - After `fetch.py` changes `data/` (next week: week 9, 9 teams), the items change and old
    results land under `skipped`. Re-queue `_baseline_mc` first: it rebuilds the reference for
    the new state.
- **Seeds.** Every candidate and variant gets the same seeds:
  - replication `r`: `100000 + r` (week) and `200000 + r` (season);
  - calibration call `k`: `300000 + k`.

  A candidate that draws chunk `ci` from `rng([seed, ci])`, as plain MC does, shares streams with
  the baseline, which tightens the paired CI.
- **Candidate API.** `Class(variant=NAME)` (or `Class()`).
  - Required: `estimate(target, n, seed) -> (T,)` estimates. They must be finite. The harness
    neither clips nor renormalizes them.
  - `n` is the budget in plain-MC paths: full 40,000 (week) / 32,000 (season), smoke 8,000 /
    8,000. It is a paths budget, not seconds, because the harness charges the call's real CPU. A
    candidate may spend it however it likes: fewer but costlier paths, a pilot run inside the
    call, QMC points.
  - **Spend the whole budget: aim for `cost` ≈ 1.** The primary metric bills
    `max(mean cost, 1)`. Spending less earns nothing, because MSE × cost only extrapolates for an
    unbiased estimator: a near-free biased answer, such as a normal approximation, would
    otherwise look many times better than plain MC. Overspending (cost > 1) is billed in full.
    Calibrate your paths-per-budget on the dev target.
  - `target` (`mc.Target`; shared, so treat it as read-only):

    | Field | Meaning |
    |---|---|
    | `name`, `horizon` | `'week'` or `'season'`; weeks simulated (1 or 9) |
    | `U` | `sim.Universe` over weeks 8-16. Column `U.n` is the empty slot |
    | `plan` | `sim.SeasonPlan`, with my roster as its single variant |
    | `idx` | `{wi: (3, T, 8)}` lineup indices for every team: starters, bench sub 1, bench sub 2 |
    | `teams`, `owners`, `me_t`, `T` | Roster ids in column order, display names, my column, team count |
    | `lg`, `pm` | The core League and PlayerModel |

  - Helpers:
    - `target.scores(score, avail, wi)`: (C, T) team totals;
    - `target.U.paths(C, rng, wi0)`: the core sampler;
    - `mc.plain(target, n, seed)`, `mc.week_counts(tg, C, rng)` and `mc.season_counts(tg, C, rng)`:
      plain MC, for example as a fallback for the target a method doesn't help.
  - Model internals for conditional, analytic or importance-sampling methods:
    - `U.qflat`: the (W, n·256) quantile tables;
    - `U.load` / `U.cload` and `pm.game_share`: copula loadings;
    - `U.team` / `U.game`: factor ids per week;
    - `U.p_out`;
    - `sim.LEVEL_STEP`, `sim.INJ_HAZARD`, `sim.INJ_MEAN_WEEKS`.
  - Exact reweighting (IS with likelihood ratios), control variates, stratification,
    conditioning, QMC and antithetics are all fair. Changing the distribution is not.
- **Rules.** These play the role of the leakage rules on this track.
  - Never read the reference cache or any result file from a candidate. The reference is for
    scoring only.
  - No state carries over between calls. Every call runs in a freshly forked worker
    (`maxtasksperchild=1`) with a fresh candidate, so in-memory caches would not survive anyway.
    Never write or read files to carry anything between calls or runs (no disk caches, no
    precomputed tables for the scored state). Everything that depends on the target happens
    inside `estimate()` and is charged: pilot runs, fitted control-variate coefficients, IS
    tilts, strata.
  - Enforced guards:
    - candidate construction is timed together with `estimate()`;
    - the CLI imports candidate modules in the parent, untimed, and refuses a candidate whose
      import burns more than 5 CPU s (`IMPORT_CPU_MAX`; the import CPU is stored as
      `import_cpu_s`), so no work can hide at import time;
    - a call that returns with child processes (running or unreaped) or extra Python threads
      still alive fails the run, since their CPU would go uncharged.
  - Fix hyperparameters a priori, or tune them on `mc.build_targets(dev=True)`. That is a practice
    state: the weakest projected team (me excepted) is removed, leaving 9 teams over weeks 9-16.
    Build your own plain-MC answer for it under cpuq. Never tune on the scored state.
  - Calls must be deterministic given `seed`, and single-threaded. Workers get
    `HARNESS_THREADS`, which is 1 when `--jobs` = `CPUQ_SLOTS`. CPU time counts every thread and
    every reaped child, so parallelism buys nothing.
- **Timing and cost.**
  - `cpu_<t>` is process CPU plus reaped children, measured around candidate construction plus
    `estimate()`. The target build and an untimed per-worker warm-up are excluded.
  - Background load on this box moves raw CPU by up to 2× between runs (measured), so raw seconds
    don't compare across jobs. Each run therefore spreads `calib` plain-MC calls at the same `n`
    (full 40, smoke 4 per target) among the candidate's calls.
  - `cost_<t> = cpu / median calibration CPU`, in units of one plain-MC call timed under the same
    conditions. Plain MC itself scores 1.00 ± 0.02.
  - Plain MC costs about 8 µs per path (week) and 70 µs per path (season) on one of cores 8-15:
    roughly 0.27 s and 2.2 s per full-mode call.
- **Metrics.** Everything is lower-is-better.
  - Per-item arrays: `mse_<t>` = mean over teams of (est − ref)², `cost_<t>`, `cpu_<t>`.
  - `PRIMARY = "log_work"` = the mean over the two targets of log(MSE_t × max(cost_t, 1)), where
    MSE_t and cost_t are means over replications (`COST_FLOOR = 1`; see the budget rule above).
    Per-target versions: `log_work_week`, `log_work_season`.
  - A leaderboard diff `d` in `log_work` means efficiency exp(−d) relative to plain MC, so
    −0.69 = 2× and −2.3 = 10×. `log_mse` and `log_cost` (also per target) split accuracy at the
    budget from compute.
  - **Treat |d| < 0.05 in `log_work` as a tie, whatever its CI says.** The bootstrap resamples
    replications, but each run's cost scale (the calibration median) is one number per run, so
    run-to-run timing noise of a few percent is not in the CI. In an audit smoke run, identical
    plain-MC code scored d = −0.027 [−0.039, −0.011] against the baseline, with `log_mse` exactly
    0. Real variance-reduction gains are much larger than this. `log_cost` is the raw, unfloored cost, so
    `log_work = log_mse + log_cost` holds exactly only when cost ≥ 1.
  - Also: `rmse_*`, `mse_*`, `cost_*`, `cpu_*` and `bias2_*`. `bias2_*` is the mean over teams of
    the squared mean error across replications, about var/R for an unbiased estimator.
  - Bias check (in `by_target`, not a leaderboard metric): `bias_z_by_team` = the mean error over
    replications ÷ sqrt(replication variance / R + reference SE²), and `bias_z_max` = its largest
    |z|. With 10 teams, a max |z| above about 3.5 in the full run (R = 160) means the estimator is
    biased, for example wrong IS weights.
- **Output.**
  - npz arrays: `item_id`, `block`, `est_<t>` and `err_<t>` (R × T), `mse_<t>`, `cost_<t>`,
    `cpu_<t>`, `wall_<t>`, `seed_<t>`.
  - Each variant record carries `eff_rel` (geometric mean over targets) and `by_target`. The
    latter holds `rmse`, `cpu_s`, `cost`, `cost_charged`, `eff_rel`, `eff_rel_adj`,
    `eff_rel_rawcpu`, `cost_ratio`, `mse_over_ref_var`, `mean_abs_err_by_team`, `bias_z_by_team`
    and `bias_z_max`. They are computed against the baseline file of the same mode when its items
    match, or against this run's `baseline` variant when the run is the baseline. `eff_rel` and
    `eff_rel_adj` use the charged cost. `eff_rel_rawcpu` (raw CPU seconds across the two runs,
    unfloored) is a diagnostic only.
  - Extra fields: `sig`, `R`, `teams`, `owners`, `me`, `reference_file`, `jobs`, `import_cpu_s`
    (CLI runs), and `targets` (per target: `n`, `horizon`, `weeks`, `reference`, `ref_se`,
    `ref_N`, `ref_var`, `calib_cpu_s`, `calib_k`, `calib_cpu_iqr`).
- **CLI.**

  ```
  python -m bench.harness.tracks.mc --candidate pkg.module:Class --key KEY \
      --variant A [--variant B | --variant C=pkg.mod:Other] --mode smoke|full [--jobs N] [--notes "..."]
  python -m bench.harness.tracks.mc --ref-only --mode smoke      # build (smoke) or show the reference
  ```

  - Calls run over `--jobs` forked workers (default `CPUQ_SLOTS`), season calls first. Variants
    are interleaved by replication (r0: A, B; r1: A, B; ...), so each variant sees the same box
    load over the run.
  - Python API: `bench.harness.tracks.mc.run(key, [(name, factory)], mode, jobs, notes)`.
  - Only the baseline job passes `--make-ref`.
- **Budget.**
  - A full run costs about 160 × (variants) × (cost_week × 0.27 s + cost_season × 2.2 s), plus
    about 100 core-s of calibration.
  - On 6 cores, keep variants × mean cost at about 8 or less to stay near 10 minutes.
  - The smoke (8 replications) takes about 5 s per plain-cost variant on 1 core, plus about 5 s
    of startup and calibration.
- **Baseline** (`candidates/_baseline_mc/plain.py:PlainMC`). This is plain MC, `mc.plain`,
  which equals `sim.run_season` bit for bit (`python -m bench.candidates._baseline_mc.verify`).
  - Smoke result (n = 8000, 8 replications): week rmse 0.00262, season rmse 0.00337, cost 1.0.
  - Expected full result (n = 40000 / 32000): rmse about 0.0012 (week) and 0.0017 (season).
  - The full run is the queued job `_baseline_mc`, about 7-8 min on 6 cores including the
    reference.
- Leaderboard: `python -m bench.harness.compare mc`.

## Track: availability

This track scores the probability that a fantasy-relevant player plays, both this week given his
injury tag and 1-8 weeks ahead. Module: `bench/harness/tracks/availability.py`. Data module:
`bench/harness/avail_data.py`.

- **Data.** The track uses nflverse, not `data/`, because `data/` holds only today's (leaky)
  injury tags and 1.5 seasons.
  - Sources: nflverse-data injuries, weekly rosters and snap counts, plus nfldata `games.csv`
    (the schedule).
  - Coverage: regular season 2016-2026, cached in `bench/cache/nflverse/`. `avail_data.fetch()`
    downloads only files that are missing, so the queued runs need no network.
  - Grid: the files are joined on a dense player × week grid. Players are gsis ids at
    QB/RB/WR/TE with a pfr id, about 1.7k players over 193 weeks.
  - Cache: `bench/cache/availability_<signature>.pkl`, rebuilt when the files or `avail_data.py`
    change. A build takes about 3 s.
  - `played` = at least one offense snap. On a 2025 check (5.5k player-weeks), every week
    nflverse says a player did not play, Sleeper's `off_snp` is also 0.
- **Items**, at an origin week t. Pre-game: t's final injury report and roster moves are known,
  t's games are not.
  - In scope: QB/RB/WR/TE whose mean offense share over his last 3 played games is ≥ 40%, who
    played within 18 weeks, is on the active roster (or designated to return from IR and tagged
    Q/D), and is not Out at t. Week-1 origins are excluded.
  - Kind 0, `gameday`: in-scope players tagged Questionable or Doubtful at t. The target is t.
  - Kind 1, `ahead`: every in-scope player. The targets are t + h for h = 1..8, in calendar
    weeks like `sim.py`, in the same season, only on weeks his origin team plays (byes are
    skipped), and only when the target week is final.
  - Out-at-origin players are excluded from `ahead`, because Sleeper's own future projections
    already zero many of them in the sim.
  - `item_id = "<season>-<week:02d>-<gsis_id>-h<h>"` (h0 = gameday).
  - `block = season * 100 + week` of the origin.
  - The target is `played` at the target week.
- **Origins.**
  - full: every final week from week 2 of 2021-2026, which is 91 origins (2026 through week 7)
    and 111992 items (1595 gameday, 110397 ahead).
  - smoke: 2023-09, 2025-06 and 2026-04, which is 3881 items (40 gameday).
  - 2016-2020 is training-only history (and dev ground).
- **Rolling origin.** For each test origin the harness builds a fresh candidate and calls
  `fit(train, periods)`:
  - `train` is every item of every earlier origin whose **target** week is before t, with `y`;
  - `periods` is the person-period table (relevant player × team-game week, every week before t)
    with pre-game features and outcome `y`, for discrete-time hazard fits.

  It then calls `predict_proba(test)` and expects `(len(test),)` values in [0, 1]. `test` holds
  the origin's items in harness order, with pre-game columns only and `y` dropped. The candidate
  class and CLI conventions are the same as the player track's.
- **Features** (causal, documented in `avail_data.py`):
  - origin report: `rep` (0/Q/D), `prac` (final practice: none/full/limited/DNP), `injg` (body
    part group), `ir_ret`, `rep_prev`, `prac_prev`, `n_rep4`;
  - history: `d` (team games missed in a row), `r` (team games played in a row), `pct_last`,
    `pct_avg3`, `rel`, `gap`, `games_s`, `miss_s`, `games_2y`, `miss_2y`;
  - player and calendar: `age`, `wk`, `last_wk`, `game0`;
  - depth: `pos_rank` (rank by role among his team's in-scope players at his position, so a
    starter returning from injury pushes his stand-in to 2) and `starter`
    (pos_rank ≤ QB1/RB2/WR3/TE1);
  - schedule: `k` = team games from t+1 to the target.
- **Leakage notes.**
  - Weekly-roster ACT and INA are merged because INA is the game-day inactive list.
  - RES (reserve) is pre-game: RES players played in 0.03% of RES weeks.
  - `game0` and `k` come from the schedule only.
  - nflverse report statuses are the final pre-game report for each game.
- **CLI.**

  ```
  python -m bench.harness.tracks.availability --candidate pkg.module:Class --key KEY \
      --variant A [--variant B=pkg.mod:Other] --mode smoke|full [--jobs N] [--notes "..."]
  ```

  Origins × variants run over `--jobs` forked workers, exactly as on the player track.
- **Output.**
  - npz arrays: `item_id`, `block`, `kind`, `h`, `grp` (gameday/ahead), `sgrp` (origin tag
    none/Q/D), `hgrp` (h0/h1/h2_4/h5_8), `pos`, `starter`, `p`, `y`, `logloss`, `brier`, plus
    the group columns `sgrp_gd`, `starter_ah` and `tag_ah`.
  - Per-origin log-loss: `variants[i].by_week`.
- **Metrics.**
  - `PRIMARY = "logloss_bal"` = ½ (mean gameday log-loss + mean ahead log-loss), so the ~1.4%
    of gameday items are not swamped. `brier_bal` is the same with Brier.
  - Also `logloss` and `brier` pooled over all items.
  - `logloss_gameday`, `brier_gameday` and `ece_gameday`, plus `logloss_gameday_Q` and
    `logloss_gameday_D`.
  - `logloss_ahead`, `brier_ahead` and `ece_ahead`, plus `logloss_ahead_starter`,
    `logloss_ahead_tagged`, `logloss_ahead_h1`, `logloss_ahead_h2_4` and `logloss_ahead_h5_8`.
  - `mean_p_*` and `rate_*` (no direction).
  - Log-loss clips at 1e-6, as in `metrics.logloss`.
- **Baseline** (`Baseline` in the track module; job `_baseline_availability`, folder
  `candidates/_baseline_availability/`).
  - Gameday: 1 − `common.P_OUT` (Q 0.75, D 0.20).
  - Ahead: the exact availability implied by `sim.Universe.paths`, a flat 2%/week hazard with
    Geometric absences of mean 3: 0.980, 0.967, 0.958, 0.953, 0.949, 0.947, 0.945, 0.944 for
    h = 1..8.
  - Smoke: logloss_bal 0.6991, gameday 0.7804, ahead 0.6178. Ahead mean p is 0.957 against an
    observed play rate of 0.815.
- **Observed rates** (all seasons):
  - Questionable players played 71%: 48% when they did not practice, 73% when limited, 88% when
    full.
  - Doubtful players played 0.7%.
  - An untagged starter who played his last game plays 92.6% one week ahead and 81.8% eight
    weeks ahead.
- Leaderboard: `python -m bench.harness.compare availability`.
