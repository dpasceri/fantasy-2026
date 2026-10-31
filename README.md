# Fantasy 2026

Fantasy football tools for [Sleeper](https://sleeper.com) leagues:

- **Guillotine simulator** (`guillotine/`): for guillotine leagues, where the lowest score each week is chopped and their roster goes to a FAAB auction. Monte Carlo sims of the rest of the season give chop odds, the best lineup, what each free agent is worth and how hard to bid. Output is a one-page cheat sheet.
- **Trade tools** (top level): a deterministic trade calculator and a brute-force search for win-win trades, using your league's scoring, best lineups week by week, and FantasyCalc market values.

## Setup

```bash
python3 -m venv guillotine/.venv
guillotine/.venv/bin/pip install -r requirements.txt
cp config.example.json config.json   # then put your league IDs in it
```

Your league ID is the number in the league's URL on sleeper.com. `config.json` stays on your machine (it's in `.gitignore`), and so does all the league data the tools download.

## Guillotine simulator

```bash
cd guillotine
.venv/bin/python fetch.py       # pull league, projections and live scores into data/
.venv/bin/python model.py       # fit player score distributions (data/model.npz)
.venv/bin/python run.py         # run every sim, write out/cheatsheet.html (~5-6 min on 8 cores)
.venv/bin/python run.py --quick # 1/8 of the sims, for a fast look
.venv/bin/python live.py        # local page at http://localhost:8765 that re-runs everything on demand
```

`GSIM_THREADS=4` caps the number of cores used. `--me <sleeper username>` picks your team.

How it works: each player's weekly score is drawn from distributions fit to projected-vs-actual history, correlated by NFL team and game (a Gaussian copula). Every manager plays the lineup they'd set from projections, the lowest alive team is chopped, and the chopped roster is auctioned with rival bids drawn from a model fit to the league's FAAB history. Every variant of your roster or lineup is scored on the same draws, so the differences are much less noisy than the numbers themselves.

`guillotine/bench/` is a backtest harness for testing model changes against past weeks; see its README.

## Trade tools

```bash
python3 sleeper_import.py       # league data into data/
python3 fantasycalc_import.py   # market values into data/market/
python3 trade_calc.py league                    # every team's projected lineup strength
python3 trade_calc.py report <team>             # one team's roster and weak spots
python3 trade_calc.py eval <team> --give "Player A" --get "Player B"
python3 find_trades.py                          # search for trades that help both sides
```

The trade tools use only the Python standard library.

## Data sources

Sleeper's public API (league data and projections), FantasyCalc (trade values), ESPN (live game clocks) and nflverse (historical rates). These are third-party services; use them within their terms and rate limits.

## License

MIT
