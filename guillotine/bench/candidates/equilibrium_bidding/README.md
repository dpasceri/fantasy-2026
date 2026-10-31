# equilibrium_bidding (policy track)

**What.** A game-theoretic FAAB bid policy (`eqbid.py:EqBid`). Each waiver run is a sealed
first-price auction, so for a player worth `V` to me the right bid is the **best response to the
distribution `F` of the highest rival claim**, not a fixed fraction of the going rate:

    b* = argmax_b  P(win | b) * (V - b),     P(win | b) = F(b-1) + (F(b) - F(b-1)) / 2   (ties split)
    V  = k * pace * typical(g)               only for players with my lineup gain g >= 2 (as the baseline)
    pace = (rate / rate_0) ** gamma,         rate = budget / (auctions_left + 1), rate_0 = the path's first rate

- `typical(g)` is the BidModel's median bid for gain `g` at my budget (the baseline's anchor).
  So `V` is my reservation price, with money priced at `1/k` market dollars. This is the
  Lagrangian form `P(win)(v - lambda b)` of the budget-constrained bidding literature.
- `pace` is adaptive pacing (Balseiro & Gur). If I have more money left per remaining auction
  than at the start of the path, money is worth less to me, so `V` rises. If I have less, `V`
  falls. It is a one-exponent stand-in for the money curve's shadow price `lambda`.
- `F` is the product over rivals with money of each rival's claim CDF, exactly as the sim draws
  claims: P(claim) = `expit(eta)`, size `round(exp(mu + sigma z))`, capped at the rival's budget.
  It matches `state.rival_cdf` to machine precision, but is vectorized over the pool and the bid
  grid inside `begin()`.

The baseline bids `m * typical(g)` whatever the competition. A best responder bids $1-3 when no
rival with money wants the player (common late in the season, and whenever only the broke teams
are interested). It bids up to `V` when a rich rival is in. The money saved goes to studs.

**Why it might beat the baseline.** The money curve (out/results.json `dynamic.budgets`) is
steep: about +0.01 P(win) per $20 at today's $165. The baseline pays $10-36 for players nobody
else can bid on. In-sim the best responder spends about $106 a path instead of $126-141, and
buys more studs (gain >= 8): 0.77 a path against 0.61-0.65.

## Variants (same `k`, `gamma`; only the rival model `F` differs)

| name | rival model behind `F` | cost per `begin()` |
|---|---|---|
| `shade` | the point BidModel `state.bm` | ~0.1 ms |
| `bayes` | posterior predictive: equal mixture of `state.bm` and 8 BidModel refits on lot (waiver run x player) bootstraps of the bid history. Own tag 777001, never the eval condition's 20261029. This is the Bayes decision under rival-model uncertainty. | ~0.45 ms |
| `fp` | belief learning (fictitious play in the Brown / Fudenberg-Levine sense: best-respond to beliefs formed from observed play). A posterior over rival aggression `(d_logit, d_logsize) = s * (0.7, 0.3)`, with `s` in {-0.5, 0, 0.5, 1, 1.5} and prior {.1, .4, .25, .15, .1}. The prior mean is 0.375, a hedge that rivals adapt and bid harder. It is updated after every auction from what Sleeper shows publicly (budget drops and roster adds), and the bid is the best response to the posterior mixture. | ~0.25 ms |

**`fp` likelihood details** (`EqBid._learn`). Claims run highest first, an unaffordable claim is
skipped, and a win past the roster cap drops the weakest rest-of-season player, which can be a
player bought earlier in the same run. Let `y` be the price that cleared each player on offer:

- A rival that bought exactly one player reveals its claim, `y` = its budget drop (the pmf term).
- Every other accountable rival's claim lost to `y`, or, if that rival had bought once at
  `x > y`, fell into the skip window `(max(y, B - x), x)`.
- "Nobody took it" is used only when no buyer could have bought the player and then dropped it
  (its ROS is above every buyer's weakest kept player).
- Rivals with unidentified buys (two or more, or a possible buy-and-drop) are left out.

Leaving them out biases this partial likelihood, so it is corrected by **indirect inference**.
`tune.py --bind` measures the pooled MLE on tuning seeds 1000-1007: it peaks at -0.46 when rivals
truly bid at s = 0 (main) and at +0.71 when they bid at s = 1 (`_stress`). The likelihood is
therefore evaluated at `binding(s) = -0.46 + 1.17 s` (`eqbid.BIND`). After the correction the
average posterior mean of `s` from week 11 on is about 0.2 on main paths and about 0.66 on stress
paths, against 0.375 a priori. It learns in the right direction, with shrinkage from only about 6
auctions a path.

## Tuning (tuning seeds only, `tune.py`, results in `tuned.json`)

`bench/cpuq -n 2 -- .venv/bin/python -m bench.candidates.equilibrium_bidding.tune --grid` runs a
grid `k` in {2, 2.5, 3} x `gamma` in {0, 0.5, 1} for `shade`. It uses seeds 1000-1039 (8,000
paths) and all three conditions, paired with the baseline's `m0.75` on the same paths. The
objective is the mean P(win) over main, `_stress` and `_bmboot`. Run time is 629 s on 2 cores.
The argmax, `k = 2.5, gamma = 0.5`, is used by all three variants. `gmin = 2` (dynamic.py's
cutoff) and the `fp` prior were fixed by hand, not tuned.

| k | gamma | main | stress | bmboot | avg3 − m0.75 (± block se) | spend |
|---|---|---|---|---|---|---|
| m0.75 | | 0.4328 | 0.4298 | 0.4326 | 0 | $126 |
| 2.0 | 0 / 0.5 / 1 | .434 / .437 / .435 | .434 / .436 / .436 | .438 / .438 / .435 | +.0034 / +.0054 / +.0037 (±.0025) | $97 / 95 / 93 |
| 2.5 | 0 / **0.5** / 1 | .440 / **.445** / .442 | .437 / **.440** / .440 | .438 / **.443** / .440 | +.0065 / **+.0107** / +.0088 (±.0028) | $110 / 107 / 104 |
| 3.0 | 0 / 0.5 / 1 | .434 / .440 / .437 | .438 / .442 / .441 | .429 / .437 / .433 | +.0016 / +.0081 / +.0052 (±.0030) | $121 / 117 / 113 |

Earlier exploration on seeds 1000-1039 also showed the following. A larger `k` (4 or 6) overpays
for studs and loses 0.03-0.10. A convex money cost `beta (exp(b / beta) - 1)` (the concave money
curve, kept as the `beta=` option) was no better than linear plus pacing.

**Selection-free check on fresh tuning seeds 1040-1059** (4,000 paths, never used for any
choice), with differences paired against `m0.75`:

| policy | main | stress | bmboot | avg3 diff | spend |
|---|---|---|---|---|---|
| m0.75 | 0.4225 | 0.4375 | 0.4335 | 0 | $124 |
| baseline 1.0/0.75 | 0.4230 | 0.4377 | 0.4168 | −0.0053 ± 0.0034 | $140 |
| `shade` | 0.4340 | 0.4535 | 0.4343 | **+0.0094 ± 0.0032** | $106 |
| `bayes` | 0.4303 | 0.4535 | 0.4328 | +0.0077 ± 0.0034 | $106 |
| `fp` | 0.4278 | 0.4575 | 0.4300 | +0.0073 ± 0.0031 | $111 |

## Smoke (eval seeds 100-104, 1,000 paths, state 0549ea074558)

`bench/cpuq -n 1 -- bash bench/candidates/equilibrium_bidding/run_smoke.sh` took 48.5 s on 1
core (sim_s 46.7) and passed `results.validate`. On tuning seeds, `--jobs 1` and `--jobs 2` give
identical arrays. The baseline rows below are the cached `_baseline_policy.smoke.json` on the same
split.

| variant | p_win | stress | bmboot | p_win_min | e_weeks | spend | studs | d_p_win vs baseline [block-bootstrap 95% CI] |
|---|---|---|---|---|---|---|---|---|
| baseline 1.0/0.75 | 0.433 | 0.436 | 0.404 | 0.404 | 6.98 | $143 | | reference |
| m0.75 | 0.454 | 0.435 | 0.417 | 0.417 | 6.90 | $127 | | +0.021 [+0.000, +0.042] |
| `shade` | 0.444 | 0.449 | 0.427 | 0.427 | 6.97 | $108 | 0.81 | +0.011 [−0.023, +0.044]; bmboot +0.023 [+0.009, +0.035] |
| `bayes` | 0.450 | 0.444 | 0.420 | 0.420 | 6.99 | $109 | 0.82 | +0.017 [−0.015, +0.049]; bmboot +0.016 [+0.005, +0.023] |
| `fp` | 0.441 | 0.442 | 0.410 | 0.410 | 6.99 | $113 | 0.81 | +0.008 [−0.020, +0.032]; bmboot +0.006 [−0.002, +0.015] |

There are only five blocks, so the smoke is a plumbing check. The smoke's m0.75 row is seed luck
(see the track README). Judge the candidate on the full run.

## Full run

`run_full.sh` runs all three variants on seeds 100-399 (60,000 paths, 3 conditions). Seeds run in
parallel over the `CPUQ_SLOTS` workers, which is the track's default. Expected: about 8 min on 6
cores (10x the smoke's sim_s). Job file, not enqueued by me:

    bench/queue/pending/equilibrium_bidding.job = {"key": "equilibrium_bidding", "cmd": ["bash", "bench/candidates/equilibrium_bidding/run_full.sh"], "timeout_s": 1200}

## Leakage and isolation

- Every choice was made on CRN seeds >= 1000 through `policy.tune()` only: `k`, `gamma`, `BIND`,
  and the choice of variants. The eval seeds 100-399 were never used for a decision.
- The bootstrap refits use the bid history (completed tx, legs 1-8) with their own tag.
- The policy reads only `State` fields and helpers and the public `bm` attributes (`a`, `beta`,
  `sigma`, `v`, `u`, `rows`). `bayes` fits its refits lazily from `state.bm` on the first
  `begin()` in each process (about 0.5 s, cached at module level, deterministic). It never
  touches the sim's draws, underscore harness attributes, or the call stack.
  - Review fix: the constructor used to call `policy.env()`. Under
    `python -m bench.harness.tracks.policy` that imports a second copy of the track module, whose
    `env()` rebuilt the League, Universe and BidModel and re-pointed `dyn_policy._G` at them.
    They were numerically identical, so results did not change (the rerun smoke is bit-identical),
    but it was harness state a candidate must not touch.
- `fp` learns only from what a real Sleeper manager sees after a waiver run: budgets and rosters.
- All per-path state (`logw`, `last`, `plan`, `bids`) is reset in `reset()`. Randomness: none.

## Caveats

- **In-sim only, and partly flattered.** `shade` best-responds to the exact BidModel that the
  main-condition rivals follow. Read `_bmboot` (rivals from a different refit) and `_stress`
  (rivals about 2x as eager and 35% higher) next to main. On the fresh tuning seeds `shade` is
  ahead of m0.75 in all three conditions. Its bmboot edge is small (+0.001), and so is that of
  `bayes` and `fp` (about 0).
- **Real rivals can adapt.** A manager who keeps winning players at $1-3 will be noticed. The
  in-sim rivals never react to my policy, and `_stress` is the only "rivals bid harder" check.
  The `fp` prior leans toward harder bidding for this reason. It gives up some main-condition
  P(win) for the best stress result (+0.020 vs m0.75 on fresh seeds).
- **Approximations.** `F` treats rivals as independent and ignores money they spend earlier in
  the same run, as `state.p_win` does. My own claims in one run can exceed my budget. The sim
  then skips the lower ones, which acts as an implicit budget constraint. Two players at the same
  position on offer are valued separately.
- **One snapshot.** `k` and `gamma` were tuned on today's state: week 8, $165, team 13 with $200
  alive. After the next fetch, re-run `tune.py --grid` and `--bind` (on tuning seeds) before
  re-queuing.

## Sources

- Pan, Kitts, Zhou et al. (Verizon Media), AdKDD 2020, "Bid Shading by Win-Rate Estimation and Surplus
  Maximization", arXiv:2009.09259 (2020). Choose `b` to maximize `P(win | b)(v - b)` from an
  estimated win-rate curve. https://arxiv.org/abs/2009.09259
- Gligorijevic et al., "Bid Shading in The Brave New World of First-Price Auctions", CIKM 2020,
  arXiv:2009.01360. https://arxiv.org/abs/2009.01360
- Zhou et al., "An Efficient Deep Distribution Network for Bid Shading in First-Price Auctions",
  KDD 2021, arXiv:2107.06650. Two-step: model the highest competing bid's distribution, then
  maximize surplus. https://arxiv.org/abs/2107.06650
- "Double Distributionally Robust Bid Shading for First Price Auctions", arXiv:2410.14864.
  Robustness to win-rate estimation error, the motivation for `bayes` and `fp`.
  https://arxiv.org/abs/2410.14864
- Balseiro & Gur, "Learning in Repeated Auctions with Budgets: Regret Minimization and
  Equilibrium", Management Science 65(9), 2019. Lagrangian `v - lambda b` bidding and adaptive
  pacing, the source of the `pace` term.
  https://www.gsb.stanford.edu/faculty-research/publications/learning-repeated-auctions-budgets-regret-minimization-equilibrium
- Rabinovich, Naroditskiy, Gerding & Jennings, "Computing pure Bayesian-Nash equilibria in games
  with finite actions and continuous types", Artificial Intelligence 195 (2013). Fictitious play
  for auctions with discrete bids. https://eprints.soton.ac.uk/343596
- Bichler, Fichtl, Heidekrüger, Kohring & Sutterer, "Learning equilibria in symmetric auction
  games using artificial neural networks", Nature Machine Intelligence 3 (2021). Also Bichler,
  Fichtl & Oberlechner, "Computing Bayes Nash Equilibrium Strategies in Auction Games via
  Simultaneous Online Dual Averaging", arXiv:2208.02036. Both are equilibrium learning in
  discretized auction games. https://arxiv.org/abs/2208.02036
- Pitchik, "Budget-constrained sequential auctions with incomplete information", Games and
  Economic Behavior 66(2), 2009. Sequential first-price auctions with budgets, where the shadow
  value of money shapes the equilibrium. https://ideas.repec.org/p/tor/tecipa/tecipa-230.html
- Gourieroux, Monfort & Renault, "Indirect Inference", Journal of Applied Econometrics 8 (1993).
  A binding-function correction of a misspecified auxiliary estimator (`BIND`).
