# _baseline_player

**What.** The core model's weekly score marginal (`model.py`), re-implemented so it can be refit
at every rolling origin on training rows only:

- the empirical quantiles of actual/projected points, on model.py's 256-level grid,
- by position and equal-count projection bucket (`nb = clip(n // 300, 5, 14)`),
- interpolated in log projection between bucket medians.

The predicted quantiles are proj × ratio quantiles, read at the harness levels with the linear
grid interpolation that `sim.Universe.paths` uses when it samples. So this is the distribution
the sim actually draws from. There are no tuned hyperparameters.

**Why.** This is the reference every player-track candidate is compared against.

**Verification.** `python -m bench.candidates._baseline_player.verify` fits on model.py's rows
(2025 weeks 1-18 + 2026 weeks 1-7, 7715 rows). The centers and tables equal `data/model.npz`
exactly (max |diff| = 0), and the predictions equal `PlayerModel.quantiles` exactly.

**Difference from the core code.** A bucket left empty by tied edges is skipped. This is
possible only on tiny training sets, and never on model.py's rows.

**Variants.** `baseline`.

**Runs.**

- `run_smoke.sh`: about 2 s.
- `run_full.sh`: the queued job `_baseline_player`, well under a minute.
