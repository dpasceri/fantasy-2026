# adp_budget (policy track)

**What.** Approximate dynamic programming for my FAAB bids. A post-decision value function
`V(weeks left, my per-week strength, rivals' per-week strength, my budget, rivals' budgets) ≈ P(win)`
is fitted by regression on simulated continuations, on tuning seeds only. At each auction, the
policy bids by one-step lookahead on that V:

    b*(p) = argmax_b  pwin_p(b) * V(B - b, S + g_p) + (1 - pwin_p(b)) * V(B, S)   (no claim if b = 0 wins)

- `pwin_p(b)` is `state.p_win`'s formula, vectorised over every b at once (`pw.py`; checked equal
  to `state.p_win` to 6e-17).
- `S` is my projected best lineup in each remaining week.
- `g_p` is p's per-week lineup gain. Its 4-week mean equals the harness `gain` exactly (checked on
  14,228 offers).
- Players are planned from the highest gain down. Each one is planned on the expected budget the
  earlier ones leave (`B -= pwin * b`).

Files:

- `feats.py`: fast lineup sums and the basis functions.
- `vfa.py`: training-data generation and the value function.
- `policy.py`: `ADP` (the evaluated policy) and `Explore` (the behaviour policy).
- `pw.py`: the P(win) curve.
- `run.py`: the driver.

**Value function** (`vfa.VF`): one constrained logistic regression per weeks-left L = 1..8 on 8
basis functions.

- `ln1` = log P(I'm not lowest next week). Weekly team scores are taken as N(projected lineup, σ²),
  with σ = 24.6, the pooled sd measured on the sim's own Universe draws (own seed 4242).
- `lsr` = Σ over the later weeks of log P(not lowest). It assumes the weakest rivals go first, one
  per week (a static elimination model, 10-node Gauss-Hermite).
- Money: `sat = 1 - exp(-B/8)` (the first few dollars buy all the $1 claims once rivals are broke),
  `log1p(B)` and `B/100`.
- Rivals' money: `log1p(Σ rival budgets)` and `log1p(max rival budget)`.
- Constraints: the strength and own-money coefficients are ≥ 0 (a monotone V, as in Jiang &
  Powell). All money terms are 0 at L = 1, since no auction follows the last one and money is
  worthless there.
- Fit: backward induction over L (Longstaff-Schwartz style). Each row's target is TD(λ = 0.5):
  `0.5 * outcome + 0.5 * survived * V_{L-1}(next post-decision state)`. On a seed bootstrap this
  cut coefficient noise 3-5x against plain Monte Carlo returns.

**Training data** (`vfa.generate`, tuning seeds ≥ 1000 only, asserted):

- It runs the track's own sim (`dyn_policy._paths_chunk`) with `Explore`. That is a base policy
  (m0.75 multiplier) with probability 0.7. Otherwise it picks a random rule for that auction: skip,
  base plan × log-normal noise, a random multiple of `typical`, or a random P(win) target.
- All three rival conditions are used. `_bmboot` uses `dyn_policy.boot_bm` replicates of the
  tuning seeds, which never match the eval replicates.
- Inside the generating worker only, the auction routine is wrapped to record each post-decision
  state. With probability 0.25 it also resets my budget to U{0..165}. This randomised intervention
  identifies the value of money from exogenous variation, not from who happened to buy. The wrap is
  restored in `finally`, and evaluation never sees it.
- The evaluated `ADP` uses only the `State` API, the public BidModel attributes (`a, v, beta, u,
  sigma`), and projection tables copied from `env()["U"].proj`. Those are projections, never draws.

**Variants** (simplest first):

| name | V | rule |
|---|---|---|
| `fvi_cut` (review) | V0c: the same round-0 rows as V0, fit with traces cut at the budget interventions (see **Review**) | lookahead at every auction |
| `fvi_root` | V0: round-0 data (m0.75-based behaviour), 110 seeds × 3 conditions in full | lookahead when ≤ 7 weeks are left. The 8-week first auction uses the reference rule (1.0 × typical) |
| `api_root` | V1: one approximate-policy-iteration round. V refit on continuations of ε-greedy `fvi_root` (70 more tuning seeds × 3 conditions) | same root rule |
| `fvi` (not in the default run) | V0 | lookahead at every auction. `--variant fvi` still runs it |

**Why it might beat the baseline.**

- `typical(g)` ignores who else can bid. Once rivals are broke (most paths by week 10-12), the
  lookahead pays $1 instead of $10-40. It keeps that money for contested auctions and picks up
  extra depth: about 10.5 buys per path against 7.5.
- When rich rivals compete for a stud, it can pay more, because V prices strength against money.

**Tuning-seed evidence** (all paths on seeds ≥ 1000, disjoint from training; 10,000 paths × 3
conditions each; paired against the reference 1.0/0.75; ± is the block se over 50 seeds):

| seeds | policy | main | stress | bmboot |
|---|---|---|---|---|
| 2050-2099 | lookahead everywhere (V0) | +0.0065 ± 0.0047 | +0.025 ± 0.005 | −0.000 ± 0.005 |
| 2050-2099 | lookahead at the first auction only, reference afterwards | **−0.023** ± 0.005 | −0.017 | −0.035 |
| 2050-2099 | reference at the first auction, lookahead afterwards | **+0.014** ± 0.004 | +0.019 | +0.012 |
| 2100-2149 | lookahead everywhere (V0 + linear money term) | +0.004 ± 0.005 | +0.009 | +0.005 |
| 2100-2149 | same V, reference at the first auction (= `fvi_root`) | **+0.016** ± 0.005 | +0.012 | +0.023 |
| 2100-2149 | policy-iteration V1 (round 1 without the root rule), reference at the first auction | +0.011 ± 0.004 | +0.012 | +0.023 |

The fitted V misprices the 8-week first auction. With $165 and seven auctions still to come, it
overspends on studs. Lookahead at the first auction alone costs 0.02 or more, while lookahead
after it gains about +0.014 in every condition. That decomposition, replicated on two disjoint
seed sets, is the reason for the `root` rule. Earlier dev runs dropped three ideas:

- **Robust P(win).** A 1/3 stressed rival curve mixed into pwin cost 0.015 on bmboot.
- **Denial credit.** Crediting the rival who takes p if I don't brought no gain (−0.006).
- **4-week-mean basis.** A basis on 4-week mean strength gave a weaker V.

**Smoke** (`bench/cpuq -n 1`, 59.7 s wall, sim 49 s, fit 10 s; smoke seeds 100-104, 1,000 paths,
state 0549ea074558; the smoke V is fit on only 2 + 1 tuning seeds, so it is much noisier than
the full one):

| variant | p_win | stress | bmboot | e_weeks | spend | buys | d_p_win vs baseline |
|---|---|---|---|---|---|---|---|
| baseline 1.0/0.75 (cached `_baseline_policy.smoke.json`) | 0.433 | 0.436 | 0.404 | 6.98 | $143 | 7.5 | reference |
| m0.75 (same file) | 0.454 | 0.435 | 0.417 | 6.90 | $127 | 7.9 | +0.021 |
| `fvi` | 0.411 | 0.445 | 0.383 | 6.92 | $149 | 10.0 | −0.022 ± 0.019 |
| `fvi_root` | 0.449 | 0.453 | 0.419 | 7.02 | $135 | 10.3 | +0.016 ± 0.013 |
| `api_root` | 0.448 | 0.443 | 0.432 | 7.00 | $123 | 10.5 | +0.015 ± 0.011 |

**Compute.**

- A seed-condition costs about 1.33 s against 0.9 s for the multiplier.
- Full run: about 50 s for the round-0 fit, 50 s for round 1, then about 490 s of eval
  (10 × the smoke's sim_s), so about 10 min on 6 cores.
- Generation pools are sized from `CPUQ_SLOTS`. Results do not depend on `--jobs`: the training
  data is concatenated in job order and the fit is deterministic.

**Caveats.**

- **The root rule is state-specific.** Today's sim starts at week 8, so `weeks_left > 7` means
  exactly the first auction. After the next fetch (a week-9 state), the rule never fires and
  `fvi_root` becomes `fvi`. Whether the lookahead is sound at a 7-week root is untested; retune
  `ROOT` on tuning seeds then.
- **V is crude.** It is 8 basis functions per L, and its log-loss barely beats a per-L constant,
  because an 8-week outcome is mostly noise. Only its differences are used.
- **In-sim optimism.** pwin is exact for the main-condition rivals. The gains do hold on stress
  and bmboot on tuning seeds.
- **`api_root` is checked only in the smoke.** Its exact round-1 behaviour (ε-greedy `fvi_root`)
  was tested only there. The 10,000-path row above used round-1 data without the root rule.

**Sources.**

- W. B. Powell, *Approximate Dynamic Programming*, 2nd ed., 2011: post-decision states and
  regression VFA (§4.6-4.7). See also Powell, "ADP algorithms", EORMS 2009,
  https://castle.princeton.edu/Papers/EORMS-ADP_Algorithms_Dec72009.pdf
- F. Longstaff & E. Schwartz, "Valuing American options by simulation: a simple least-squares
  approach", RFS 14(1), 2001: regression on simulated continuations, backward induction.
- H. Cai et al., "Real-Time Bidding by Reinforcement Learning in Display Advertising", WSDM 2017,
  arXiv:1701.02490: an MDP over (auctions left, budget), bidding argmax P(win | b)(value + V).
- S. Balseiro & Y. Gur, "Learning in Repeated Auctions with Budgets", Management Science 2019:
  a budget's dual multiplier is the value of a dollar, used to shade bids.
- D. Bertsekas, rollout survey, https://www.mit.edu/~dimitrib/Rollouts_Survey.pdf: one-step
  lookahead on a base policy's value, and its cost-improvement property.
- D. Jiang & W. Powell, "An ADP algorithm for monotone value functions", arXiv:1401.1590:
  monotonicity constraints on V.

## Review (independent reviewer, 2026-10-30)

**Checked and sound.**

- **Leakage.** Every training and tuning seed is ≥ 1000 (`vfa.generate` asserts it). `team_sigma`
  uses seed 4242. The `_bmboot` training replicates are `boot_bm(bm, seed ≥ 1000)`, which never
  equal an eval replicate (100-399). The leaky proj `inj` field is not read. The evaluated
  policy reads only the `State` API, public BidModel attributes and the projection tables, never
  the draws.
- **Comparability.** Items, seeds, conditions and the npz are produced by
  `bench.harness.tracks.policy.run`, so they are the same paths as `_baseline_policy` (state
  `0549ea074558`). `results.validate` is empty.
- **Identities.** On 2,892 offers (tuning seed 3001, stress rivals), `pwin_curve` matches
  `state.p_win` to 1.1e-16, and the 4-week mean of `gains_by_week` equals the harness gain
  exactly.
- **The `_auction` wrap** happens only inside training workers, on tuning seeds, and is restored
  in `finally`. Evaluation never sees it.

**Bug found and fixed: the budget interventions biased the TD targets.**

- The wrap resets my budget to U{0..165} after 25% of auctions. That is fine for the row it
  records. But every *earlier* row on the same path bootstraps through that reset: its λ-return
  uses V and outcomes from a state whose budget was redrawn. So the earlier row's dependence on
  its own money is diluted by about 0.75 per later auction.
- The result is that V underprices money at long horizons. That is the "overspends at the
  8-week first auction" symptom the root rule was patching.
- **Fix.** `vfa.VF.fit(cut=True)`. When the next row was intervened on, the trace is cut and the
  earlier row bootstraps fully on V_{L-1} at the natural, pre-intervention state. That state is
  recorded as `Bpre`/`pert`, and the change does not alter any RNG stream. This is off-policy
  trace cutting. The default `cut=False` reproduces the old fit bit for bit.
- **Effect on V** (fit on tuning seeds 1000-1039 × 3 conditions; d logit / d$ at B = $165, uncut
  vs cut):

  | L | uncut | cut |
  |---|---|---|
  | 8 | 0.28e-3 | 0.87e-3 |
  | 7 | 0.40e-3 | 0.95e-3 |
  | 6 | 0.33e-3 | 0.67e-3 |
  | 2-3 | about the same | about the same |

- **Effect on the policy** (tuning seeds 2200-2229, 6,000 paths per condition, paired against
  1.0/0.75, block se):

  | policy | main | stress | bmboot | spend |
  |---|---|---|---|---|
  | `fvi` | +0.0035 ± 0.0071 | +0.016 | −0.008 | $140 |
  | `fvi_cut` | +0.0160 ± 0.0045 | +0.0085 | +0.000 | $115 |
  | `fvi_root` | +0.0158 ± 0.0050 | +0.013 | +0.003 | $127 |
  | root rule plus cut | +0.0160 ± 0.0053 | +0.008 | +0.002 | $114 |

  `fvi_cut − fvi` is +0.0125 ± 0.005 main and +0.008 bmboot, but −0.007 stress. `fvi_cut −
  fvi_root` is a tie: +0.000 main, −0.0045 stress, −0.003 bmboot.
- So the principled fix matches the hard-coded root rule without depending on today's
  snapshot. The root rule stops firing after the next fetch.
- **Default run changed.** It is now `fvi_cut, fvi_root, api_root`. `fvi` (known weak) was
  dropped to keep 3 variants and the same runtime. `fvi_root` and `api_root` are unchanged and
  keep the uncut V0/V1 they were tuned with.

**Budget.**

- Eval costs about 1.03 core-s per (seed, ADP variant, condition), measured here. That gives
  about 460 s for 2,700 units on 6 cores.
- Training is about 95 s on 6 cores: 330 + 210 seed-conditions at about 1 core-s each. Add
  about 15 s for setup.
- Total: about 9.5-10 min on 6 cores, inside 15 min. On only the queue's minimum of 4 cores it
  is about 14-15 min, still inside the job's 1200 s timeout.
