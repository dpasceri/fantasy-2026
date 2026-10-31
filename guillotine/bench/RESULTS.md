# Bake-off results (2026-10-30, state: week 8, 10 alive, dpasceri $165)

15 candidate algorithms against re-implemented production baselines. Full backtests:
`bench/results/`; leaderboards: `python -m bench.harness.compare <track>`. Each apparent winner
was then attacked by 2-3 independent verifiers (leaks/fairness, reproduction/statistics, simulator
transfer); their scripts and outputs are in `bench/verify/<claim>_<lens>/`. Nothing below has been
wired into the production pipeline.

| Track | Best candidate | Headline | After verification | Verdict |
|---|---|---|---|---|
| policy (your bids, in-sim) | equilibrium_bidding/shade, adp_budget/fvi_cut | P(win) 0.425 -> 0.441 / 0.444 | Replicates on fresh seeds; positive under 16/16 misspecified rival models. ~0.6 pt of it is just bidding 0.75x typical all season instead of 1.0x before week 12. Edge over 0.75x: about +1.0 pt (+0.7 under bootstrap rivals). Almost all of it comes from later auctions (pay $1-6 once rivals are broke or passive), so the policy must be re-run every week; this week's bids barely change. | holds, in-sim only |
| availability (injuries) | availability_hazard/gbm | logloss 0.654 -> 0.422 | Real, but ~69% is just refitting production's constants: Doubtful players play ~1% (sheet assumes 20%), healthy starters play ~93/86/82% at 1/4/8 weeks ahead (sheet ~98/95/94%). 3% of items leaked in through a future-built player crosswalk (gain -0.232 -> -0.217). Practice-report features only help after Friday, not for Tuesday bids. Lowers everyone's rest-of-season value ~10%, so relative bids barely move. Don't stack it on model.py's zero mass (double count). | holds, mostly a recalibration |
| player (score distributions) | gbm_quantile/qrf_hurdle | CRPS -0.45% | Real and robust (18/21 weeks, both seasons) but small: ~0.35-0.45%, mostly better zero-score odds. Moves chop odds ~0.2-0.5 pt per team; not significant on starters alone. | holds, small |
| bids (rival FAAB model) | structural_auction/hb | logs -0.094 | 88% of the gain is week 3, when the production BidModel had only 2 weeks of data. On weeks 4-7 it's a wash, and it would overprice chopped QBs. 5 blocks cannot support the CI. Cheap fixes worth keeping: point-in-time rosters ("clean"), drop the leg trend. | overstated: keep BidModel |
| mc (sim speed) | variance_reduction/cmc_mix | ~10x efficiency | Unbiased; 17x for this week's chop odds, 6x for frozen-roster P(win). But those steps are ~51 s of the 435 s run, and it does not apply to the dynamic auction sim, which makes the bid numbers. | holds, low value |
| team (who gets chopped) | none | all within +-0.006 logloss | The null is real for reshaping marginals/copula, but the track can't see gains below ~0.01 nats and only tests known, post-lock lineups. rival_lineup was effectively untested (virtual leagues have no bench). Pre-lock risks (rivals starting injured/bye players, ~7% of team-weeks) were never tested. | keep the copula |

Ranked by value for effort:
1. Bid multiplier 0.75x all season instead of 1.0x before week 12: one-line change, about +0.6 pt in-sim (a tie under the stress rivals).
2. Re-run a shade or fvi_cut policy at every auction: about +1.0 pt more in-sim. Needs shade's ~10 min tune (2 cores) after each fetch, and rival behavior should be watched (two rich rivals look passive on 1-2 bids each).
3. Recalibrate P_OUT (Doubtful ~0.99 out) and replace the flat 2%/week hazard with per-horizon play rates, net of the zero mass already in model.py's tables.
4. Everything else: not worth wiring in now.

## Applied to the pipeline (2026-10-30)

- **Item 1.** `run.py` bids a fixed `POLICY` of 0.75x all season instead of the sweep's argmax. The sweep is still
  shown on the sheet, with the cell in use outlined.
- **Item 3.** `sim.py` absences are refit to healthy starters (untagged, played last week) in nflverse 2016-2026,
  target weeks 1-16 only (weeks 17-18 add rested starters, and the league ends at 16):
  - Each week, an available player starts an absence with probability 6.4% (QB 8.6%).
  - 13% of absences are season-long (QB 20%); the rest are geometric with mean 2.1 weeks (QB 1.45).
  - The sim reproduces their play rates within 0.3 pt at 1-8 weeks ahead: 94/87/84% at 1/4/8 weeks (QB 91/84/79%).
- **Side effect: backup QBs.** A backup QB is now worth about 3 pts of P(win) for a one-QB roster. That is an upper
  bound, because rosters are frozen in the static sim and it can't stream a replacement after an injury. The sheet
  says so in the pickup advice.
- **P_OUT.** `common.P_OUT` Doubtful goes from 0.8 to 0.99, and `sim.py` now reads it from there. Questionable stays
  at 0.25: the sheet runs midweek, and only the Friday final-report rate (68-72% played) is measured.
- **Zero mass not netted out.** model.py's tables give a zero score 0-2% of the time at starter projections (QB up
  to 4%), and only part of that is not playing. So the overlap is at most that big.
- **Bench baselines.**
  - The bench's availability baseline is frozen at the old constants, so its leaderboard still reproduces.
  - The policy and mc tracks import the live `sim.py`, so re-running them now measures the new pipeline.
- **Not applied.** Item 2 (shade / fvi_cut re-planned every week).
