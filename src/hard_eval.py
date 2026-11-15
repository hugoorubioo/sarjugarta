"""
Mode Difícil: validació creuada que imita el test (mateixes tres rèpliques que el mode estàndard).
Per a cada rèplica, models i paràmetres s'aprenen només amb la resta de dies.

Ús:  python src/hard_eval.py  ->  docs/results_dificil.json
"""
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from core import ROOT
from hard_core import hard_frame, load_hourly, oracle_hard, pseudo_test, run_hard
from hard_train import train_all

ZEROQ = [[[0.0] * 7] * 3] * 24


def _agent_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def eval_fold(first_day, extra=None):
    hard = _agent_module(ROOT / "submission_dificil/agent.py", "agent_dificil")
    std = _agent_module(ROOT / "submission/agent.py", "agent_estandard")
    df = load_hourly()
    fold = pseudo_test(df, first_day)
    models, params = train_all(df.drop(fold.index))
    d = hard_frame(fold)
    o = oracle_hard(d)
    out = {"hores": len(d), "oracle": {k: round(float(o[k]), 3) for k in ("score", "co2_pct", "cost_pct")}}

    class Zero:
        def act(self, obs):
            return 0.0

    def res(agent):
        r = run_hard(agent, d)
        r = {k: round(float(v), 3) for k, v in r.items()}
        r["temps_s"] = round(getattr(agent, "elapsed", 0.0), 1)
        return r

    out["agent_estandard"] = res(std.BatteryAgent())
    out["sense_models"] = res(hard.BatteryAgent({"models": False} if False else {**params, "models": None, "net_quantiles": None}) if False else _NoModels(hard, params))
    out["puntual"] = res(hard.BatteryAgent({**params, "models": models, "net_quantiles": ZEROQ}))
    out["escenaris"] = res(hard.BatteryAgent({**params, "models": models}))
    for k, cfg in (extra or {}).items():
        out[k] = res(hard.BatteryAgent({**params, "models": models, **cfg}))
    return first_day, out


def _NoModels(hard, params):
    a = hard.BatteryAgent({**params})
    a.gbm = None
    return a


if __name__ == "__main__":
    folds = [int(x) for x in sys.argv[1:]] or [1, 8, 15]
    with ProcessPoolExecutor(len(folds)) as ex:
        res = dict(ex.map(eval_fold, folds))
    json.dump(res, open(ROOT / "docs/results_dificil.json", "w"), indent=1)
    for k in res[folds[0]]:
        if k != "hores":
            print(f"{k:18s}", [res[f][k]["score"] for f in folds], "mitjana %.3f" % np.mean([res[f][k]["score"] for f in folds]),
                  [res[f][k].get("temps_s") for f in folds])
