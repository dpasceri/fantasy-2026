# _baseline_bids

**What.** The core rival-bid model (`auction.BidModel`), re-implemented so it can be refit at
every rolling origin on training legs only. It is a hurdle model:

- participation: `P(bid > 0) = logistic(a0 + a1 log(1 + gain) + a2 log(budget) + v_j)`, with
  manager effects `v_j` under a N(0, 1) penalty, fit by L-BFGS-B from BidModel's start point;
- size: `log(bid) | bid > 0 = b0 + b1 gain + b2 log(budget) + b3 leg + u_j + e`,
  `e ~ N(0, s^2)`, by least squares, with manager effects `u_j` shrunk by empirical Bayes.

The predicted bid is what `BidModel.sample` draws (and what `dynamic.py` / `run.py` use):
`min(round(exp(mu + s z)), budget)`. The harness gets it as an exact pmf over whole dollars,
not as samples. There are no tuned hyperparameters.

**Why.** This is the reference every bids-track candidate is compared against.

**Variants.**

- `baseline` (the reference): the core pipeline as deployed.
  - It fits on BidModel's own rows (`repaired == False`) with BidModel's own feature
    (`gain_core`, the lineup gain on the end-of-week matchup roster). That is exactly
    `auction.BidModel` on the league cut off before the test leg.
  - It predicts from the point-in-time `gain`, as `run.py` does from today's rosters.
  - `p_bid = logistic × P(round(exp(.)) >= 1)`, so a draw that rounds to $0 counts as no bid,
    as in `top_rival_bids` and `dynamic._auction`.
- `logit`: the same fit, with `p_bid` = BidModel.p_bid (the raw logistic). It differs from
  `baseline` by at most about 1% in p.
- `clean`: the same spec, fit on the training frame a candidate sees: all chances, including the
  12 claims `bid_history` drops, with point-in-time `gain`. It shows what cleaner training
  features alone buy, so a candidate's gain over `baseline` can be split into features and model.

**Verification.** `python -m bench.candidates._baseline_bids.verify` checks six things.

1. The track's unrepaired chances, with `gain_core`, are BidModel's 1564 rows, identical and in
   the same order (sorted by `raw_index`).
2. Fit on those rows, every parameter (a, v, beta, u, sigma, tau, r2), `p_bid` and `mu` equal
   `auction.BidModel(lg)`'s: max |diff| = 0.
3. At every test leg L, the baseline's training rows equal `auction.BidModel` on the league cut
   off before L (`lg.leg = L - 1`, tx and matchups truncated). Fit in `bid_history` order, every
   parameter matches (max |diff| = 0).
   - In harness order (what the run uses), the size model and `mu` match exactly. The test-leg
     `p_bid` agree within 3e-4.
   - The logistic parameters can sit up to about 0.02 apart. Early on nearly every budget is
     $200, so a0 and a_log_budget lie on a flat ridge, and L-BFGS-B stops at a point that
     depends on row order.
   - BidModel itself moves the same way between processes, because `bid_history`'s row order
     follows str-hash set order. Over PYTHONHASHSEED 1-4, its a0 spread reaches 0.0086.
   - The harness order is deterministic, so the baseline is reproducible. Results are
     bit-identical across PYTHONHASHSEED values.
4. Point-in-time rosters (`bids.rosters_at`) equal the end-of-week matchup rosters of every alive
   team in legs 3-4. In legs 5-7 the differences are counted: Sleeper's snapshot already holds
   some post-chop claims.
5. The exact independent-product top-bid pmf for every leg-7 group matches 200k draws of
   `auction.top_rival_bids`: max |CDF diff| = 0.0028, under the Monte Carlo tolerance of 0.005.
6. The guard: inside fit/predict, no chance of the test leg is visible, later rosters are
   refused, test frames carry no outcome column, and `data.table()` stops at the horizon.

**Runs.**

- `run_smoke.sh`: about 1.4 s on 2 cores.
- `run_full.sh`: the queued job `_baseline_bids`, 2.0 s on 6 cores. Full logs_pool: `baseline` 0.8615,
  `logit` 0.8602, `clean` 0.8311 (n 1120).
