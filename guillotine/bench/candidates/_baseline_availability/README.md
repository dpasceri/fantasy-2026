# _baseline_availability

**What.** The core pipeline's availability rules, scored as probabilities on the AVAILABILITY
track. The class is `bench.harness.tracks.availability:Baseline`. It fits nothing.

- **Gameday items** (tagged Questionable or Doubtful this week): P(plays) = 1 − `common.P_OUT`,
  which is 0.75 for Questionable and 0.20 for Doubtful.
  - The values match `sim.Universe.p_out`.
  - The tag is the point-in-time nflverse game status, not Sleeper's leaky current tag.
- **Ahead items** (h = 1..8 weeks later): the exact P(available) implied by
  `sim.Universe.paths`. The model works like this:
  - at the current week, `out_left = 0`;
  - each week, a player with `out_left == 0` starts an absence with probability
    `INJ_HAZARD` = 0.02;
  - the absence lasts Geometric(1 / `INJ_MEAN_WEEKS` = 1/3) weeks, with support 1, 2, ...;
  - the player is available when `out_left == 0` after the draw;
  - `out_left` then counts down by one.

  This gives P = 0.980, 0.967, 0.958, 0.953, 0.949, 0.947, 0.945, 0.944 for h = 1..8
  (`availability.sim_ahead()`), the same for every player.

**Why.** It is the reference every availability-track candidate is compared against.

**Variants.** `baseline`.

**Runs.**
- `run_smoke.sh`: about 3 s.
- `run_full.sh`: the queued job `_baseline_availability`, under a minute.
- `availability_hazard/run_full.sh` also writes this baseline.

Smoke result: logloss_bal 0.6991, gameday log-loss 0.7804 (n = 40), ahead log-loss 0.6178
(n = 3841), and ahead mean p 0.957 against an observed rate of 0.815.
