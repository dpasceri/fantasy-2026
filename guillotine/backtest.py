"""Backtest: replay weeks 1..leg-1 with the rosters and starters each team actually had.

Checks (a) projected vs actual team totals, (b) the chop line, (c) how much probability the
model put on the team that actually got chopped.
"""
from common import *  # noqa
from common import League, np
from model import PlayerModel
from sim import Universe, team_scores, run_week_lineups

import sim


def main(n_sims=96000):
    lg, pm = League(), PlayerModel()
    rows = []
    for w in range(1, lg.leg):
        ms = {m["roster_id"]: m for m in lg.matchups[w]}
        alive = [t for t, m in ms.items() if m.get("players") and (m.get("points") or 0) > 0]
        pids = [p for t in alive for p in ms[t]["players"]]
        U = Universe(lg, pm, pids, [w])
        U.p_out[:] = 0  # projections for past weeks are final, inactive players already zeroed
        idx = np.stack([U.fixed_lineup_idx(0, [p for p in ms[t]["starters"] if p in U.ix], ms[t]["players"]) for t in alive], 1)
        opt = np.stack([U.lineup_idx(0, ms[t]["players"]) for t in alive], 1)
        res = {}
        for name, I in (("actual_starters", idx), ("optimal", opt)):
            P = np.zeros(len(alive))
            for j in range(len(alive)):
                r = run_week_lineups(U, I, I[:, j:j + 1, :], j, n_sims, seed=10 + w)
                P[j] = r["p_chop"][0]
            res[name] = P / P.sum()
        proj_tot = [sum(U.proj[0, U.ix[p]] for p in ms[t]["starters"] if p in U.ix) for t in alive]
        act = [ms[t]["points"] for t in alive]
        chopped = int(np.argmin(act))
        rows.append({
            "week": w, "teams": len(alive),
            "proj_mean": round(float(np.mean(proj_tot)), 1), "actual_mean": round(float(np.mean(act)), 1),
            "chop_line_actual": round(float(min(act)), 1),
            "chopped": lg.owner(alive[chopped]),
            "p_model_actual_starters": round(float(res["actual_starters"][chopped]), 3),
            "p_uniform": round(1 / len(alive), 3),
            "model_rank_of_chopped": int((res["actual_starters"] > res["actual_starters"][chopped]).sum()) + 1,
        })
        print(rows[-1])
    ll_model = -np.mean([np.log(r["p_model_actual_starters"]) for r in rows])
    ll_unif = -np.mean([np.log(r["p_uniform"]) for r in rows])
    print(f"log-loss model {ll_model:.3f} vs uniform {ll_unif:.3f}")
    with open(os.path.join(DATA, "backtest.json"), "w") as f:
        json.dump({"weeks": rows, "logloss_model": ll_model, "logloss_uniform": ll_unif}, f, indent=1)


if __name__ == "__main__":
    main()
