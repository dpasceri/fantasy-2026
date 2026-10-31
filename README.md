# Fantasy 2026

Fantasy football tools for [Sleeper](https://sleeper.com) leagues.

**[dpasceri.github.io/fantasy-2026](https://dpasceri.github.io/fantasy-2026/)**: paste a Sleeper league ID, pick your team, hit run. Monte Carlo sims of the rest of a guillotine league's season, run in your browser on all your cores: who gets chopped, what lineup to start, which free agents matter and what to bid in each FAAB auction. A full-precision run takes about 20 s on a 16-core desktop and under a minute on a recent phone.

Supported: guillotine leagues with FAAB waivers and QB / RB / WR / TE / FLEX slots. Not supported: kicker and defense slots, superflex and 2-QB, IDP, best ball.

What's here:

- **Web app** (`web/`): the simulator in plain JavaScript, served as a static site.
- **Guillotine simulator in Python** (`guillotine/`): the original, plus a backtest bench for model research.
- **Trade tools** (top level): a deterministic trade calculator and a brute-force search for win-win trades.

## How the sims work

Each player's weekly score is drawn from distributions fit to projected-vs-actual history in the league's own scoring, correlated by NFL team and game (a Gaussian copula). Every manager plays the lineup they'd set from projections, the lowest alive team is chopped, and the chopped roster is auctioned, with rival bids drawn from a model fit to the league's FAAB history (or a prior scaled to the league's budget when it has fewer than 20 bids). Every variant of your roster, lineup or bid is scored on the same simulated seasons, so the differences between them are much less noisy than the numbers themselves.

## Web app

```bash
python3 -m http.server 8790 --directory web     # then open http://localhost:8790
cd web && node --test test/*.test.js            # tests (Node 22+)
```

No build step and no dependencies: plain ES modules, one Web Worker per core. The browser pulls league data, projections and stats straight from Sleeper's public API and caches last season's files in IndexedDB.

It's a port of the Python version and checked against it: same results within sim noise, and the cheat sheet HTML is byte-identical to `report.py`'s for the same results. Speed work since the port has been held to byte-identical output. On the same 16-core desktop, a full-precision run takes 14 s here against about 7 minutes in Python.

Pushing to `main` runs the tests, stamps every file link with the commit (so browsers never mix cached old files with new ones) and deploys to GitHub Pages.

## Guillotine simulator (Python)

```bash
python3 -m venv guillotine/.venv
guillotine/.venv/bin/pip install -r requirements.txt
cp config.example.json config.json   # then put your league IDs in it
```

Your league ID is the number in the league's URL on sleeper.com. `config.json` stays on your machine (it's in `.gitignore`), and so does all the league data the tools download.

```bash
cd guillotine
.venv/bin/python fetch.py       # pull league, projections and live scores into data/
.venv/bin/python model.py       # fit player score distributions (data/model.npz)
.venv/bin/python run.py         # run every sim, write out/cheatsheet.html (~7 min on 10 cores)
.venv/bin/python run.py --quick # 1/8 of the sims, for a fast look
.venv/bin/python live.py        # local page at http://localhost:8765 that re-runs everything on demand
```

`GSIM_THREADS=4` caps the number of cores used. `--me <sleeper username>` picks your team.

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
