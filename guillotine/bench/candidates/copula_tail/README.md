# copula_tail (team track)

**What.** Same marginals as `_baseline_team`; only the dependence between players changes. Each origin
refits the baseline's pieces on train items: `_baseline_player` ratio tables, plus `fit_copula`'s loadings
`a_pos` and game share `g`. Every variant then draws a uniform `u` per player that is exactly U(0, 1)
marginally, and reads it off the baseline's 256-level quantile grid. So each player's score distribution
is the baseline's, and only the joint behaviour of team totals and their minimum moves. The module is
`tailcop.py:TailCopula`, with 96000 draws per week. Common random numbers: each chunk first draws the
baseline's normals in the baseline's order, with `default_rng([11, season, week, chunk])`, and draws any
extras after them.

| Variant | Dependence |
|---|---|
| `gt` | **Grouped t copula** on the baseline's team/game factor structure. Factor z → z·√(ν/W), W ~ χ²(ν), then the t_ν CDF. W is drawn either per game (`game`: each game is an independent t copula, so one game's players hit joint extremes, as in blowouts, shootouts and weather) or once per draw for the whole week (`slate`: tail dependence across games too). The group and ν ∈ {2 … 150} are chosen per origin by pairwise composite likelihood (CL) over every same-week pair of players projected ≥ 3. The Gaussian baseline is one of the candidates and wins ties; with it the draws are bit-identical to the baseline's. |
| `slot_t` | **Depth-chart (slot) copula with t tails.** Players map to slots QB1, RB1-3, WR1-4 and TE1-2 by projection rank on their NFL team that week. A game is a 20-dim vector (10 slots per side). Its correlation matrix holds the training games' normal-score correlations: a same-team block and an opponent block, with home/away symmetrized. Each entry is shrunk towards the factor model by empirical Bayes, w = τ²/(τ² + 1/n), where τ² is the spread of empirical-minus-factor correlations in excess of sampling noise. The matrix is then made positive definite. Unlike one positive factor, it can express QB1-WR1 stronger than QB-WR3, and negative RB1-RB2. Group and ν are refit by CL with this matrix. Unslotted players keep the baseline factor draw. |
| `shuffle` | **Empirical game-block copula.** This is a Schaake shuffle / ECC-style rank reordering. For each draw and each game, the variant copies the slot pattern of a random training game, with a random home/away orientation. A player's u is the randomized rank of that template team-week's residual among all templates' values for his slot. That is a checkerboard empirical copula, so u is exactly uniform. It keeps any tail asymmetry, negative pairs and opponent effects in the data. Slots a template lacks, and unslotted players, fall back to the `gt` draw. |

Residuals (z) are `fit_copula`'s randomized-PIT normal scores (seed 7, same buckets). Training games and
pairs come from `train_df` only, which is strictly before the test week. `train_leagues` is not used. No
inj tags are used.

**Why.** The chop is the minimum of 10-17 team totals, so it depends on how fantasy starters move together:

- Same-team and same-game pairs within a fantasy team widen or narrow its total and shape its left tail.
- Shared NFL teams across fantasy teams correlate their totals.

The baseline's one-factor Gaussian copula has no tail dependence and only positive same-team correlations.
Its QB loading sits at the 0.95 bound, because one factor has to carry every QB-pass-catcher pair. On the
training data before 2026 week 7 (`verify.py`), the template pool differs from the factor model:

| Pair | Pool rank correlation | Factor rank correlation | Lower-tail co-exceedance P(u < 0.1, u < 0.1)/0.1 |
|---|---|---|---|
| QB1-WR1 | +0.35 | +0.21 | 0.19 in both: the extra dependence is in the upper tail (shared touchdowns) |
| RB1-RB2 | −0.10 | 0.00 | 0.05 pool vs 0.11 factor |
| Opponent QB1-QB1 | +0.17 | +0.23 | 0.11 pool vs 0.20 factor |

Tail-dependent and empirical copulas are the standard fixes; see the sources below.

**What could overfit.** About 64 training games (2025 week 5 origin) to 365 (2026 week 7 origin) make a
raw empirical copula noisy, and that is `shuffle`'s risk. `slot_t` shrinks towards the factor model using a
data-driven τ. Because of that, it stays close to `gt` until about 2025 week 14: τ sits at its 0.01 floor
and the entry weights at about 0.01.

**Fitted values per origin** (printed as `[copula_tail] ...` in the log).

- CL always picks the `game` group, never `slate`, with ν = 40-150. That is very weak symmetric tail
  dependence: the CL gain is 0-11 nats over about 0.2-1.1M within-week pairs.
- So `gt` is close to the baseline by construction, and is the control that answers "do t tails help?".
- `slot_t` has τ = 0.01-0.035, median entry weight 0.01-0.28.

**Sources.**

- Clark, Gangopadhyay, Hay, Rajagopalan & Wilby (2004), "The Schaake Shuffle: a method for
  reconstructing space-time variability in forecasted precipitation and temperature fields", J.
  Hydrometeorology 5(1), 243-262. https://www.usgs.gov/publications/schaake-shuffle-a-method-reconstructing-space-time-variability-forecasted
- Schefzik, Thorarinsdottir & Gneiting (2013), "Uncertainty quantification in complex simulation models
  using ensemble copula coupling", Statistical Science 28(4), 616-640 (empirical-copula reordering, ECC).
  https://arxiv.org/abs/1302.7149
- Daul, De Giorgi, Lindskog & McNeil (2003), "The grouped t-copula with an application to credit risk",
  Risk 16(11). https://www.risk.net/risk-management/credit-risk/1530299/using-grouped-t-copula
  Demarta & McNeil (2005), "The t copula and related copulas", Int. Stat. Review 73(1) (t copula density,
  used for the CL).
- Oh & Patton (2017), "Modeling dependence in high dimensions with factor copulas", JBES 35(1), 139-154
  (factor copulas with fat-tailed or asymmetric factors; the baseline's structure is their Gaussian case).
  https://www.princeton.edu/~erp/erp%20seminar%20pdfs/Fall%202012/Oh_Patton_MV_factor_copula_apr12.pdf
- Krupskii & Joe (2013), "Factor copula models for multivariate data", JMVA 120, 85-101 (tail dependence
  and asymmetry in factor copulas). https://ideas.repec.org/a/eee/jmvana/v120y2013icp85-101.html
- Varin, Reid & Firth (2011), "An overview of composite likelihood methods", Statistica Sinica 21 (the
  pairwise CL used to choose ν and the group).

**Verification.** Run `bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates.copula_tail.verify [season week]`
(about 12 s). It uses training data and the pre-game frame only.

1. With the Gaussian group forced, the `gt` sampler gives the baseline's `WeekDraws` scores bit for bit.
   p_chop is identical, and quantiles agree to 3e-14.
2. Every variant's u is uniform for every player. The max KS distance over 326 players is 0.009-0.012,
   against the 1% critical value of 0.013 for that max.
3. `shuffle` reproduces the pool's slot rank correlations and lower-tail co-exceedances exactly. `slot_t`
   lies between the pool and the factor model.

**Runs.**

- `run_smoke.sh`: about 22 s on 1 core, idle machine (58 s on a fully loaded one).
- `run_full.sh`: queued only. It runs 21 origins × 3 variants, each origin taking about 2-8 s on 1 core, so
  about 2-4 min on 6 cores. Peak memory is about 0.6 GB per worker.

**Smoke** (50 items: 48 virtual, 2 real). Paired differences vs `_baseline_team.smoke.json`, on the same
items. With only 3 blocks, the smoke CIs mean nothing.

| | Virtual logloss | Brier | rank_norm | crps_total | crps_line | Real logloss |
|---|---|---|---|---|---|---|
| baseline | 2.0761 | 0.8250 | 0.229 | 10.943 | 5.545 | 1.851 |
| gt | 2.0752 (−0.0010) | 0.8249 | 0.226 | 10.943 | 5.545 | 1.850 |
| slot_t | 2.0732 (−0.0030) | 0.8242 | 0.230 | 10.935 | 5.549 | 1.847 |
| shuffle | 2.0799 (+0.0037) | 0.8255 | 0.228 | 10.940 | 5.537 | 1.857 |
