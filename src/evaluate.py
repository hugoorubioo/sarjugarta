"""
Avaluació completa amb validació creuada que imita el test ocult.

El test són els últims 7 dies de cada mes. Aquí en fem 3 rèpliques amb dies del
train (dies 1-7, 8-14 i 15-21 de cada mes): per a cada rèplica s'aprenen TOTS els
paràmetres i models amb la resta de dies i s'avalua sobre la setmana reservada,
amb el simulador oficial. Així cap resultat fa servir informació del futur.

Ús:  python src/evaluate.py   ->  docs/results.json
"""
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from core import ROOT, load_hourly, oracle, pseudo_test, run, score_from_env

sys.path.insert(0, str(ROOT / "submission"))
from agent import BatteryAgent  # noqa: E402
from fit_params import fit_params  # noqa: E402
from train_forecaster import train  # noqa: E402

NO_GBM = {"models": {"co2": None, "preu": None}}


class _Fallback(BatteryAgent):
    """Només la política de reserva per quantils (sense optimització)."""
    def _solve(self, *a, **k):
        raise RuntimeError


class _Zero:
    def act(self, obs):
        return 0.0


def _starter():
    spec = importlib.util.spec_from_file_location("starter", ROOT / "starter_kit/agent_baseline.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.BatteryAgent()


def _res(agent, d, horizon=24):
    r, env = run(agent, d, return_env=True, horizon=horizon)
    cyc = float(np.abs(np.diff(np.r_[1000.0, env.soc_series])).sum() / 2 / 2000)
    return {"score": round(r["score_exacte"], 4), "co2_pct": r["co2_estalviat_pct"], "cost_pct": r["cost_estalviat_pct"],
            "co2_kg": r["co2_estalviat_kg"], "cost_eur": r["cost_estalviat_eur"], "infraccions": r["infraccions"],
            "co2_base_kg": r["co2_baseline_kg"], "cost_base_eur": r["cost_baseline_eur"], "cicles": round(cyc, 1),
            "temps_s": round(getattr(agent, "elapsed", 0.0), 1)}


def eval_fold(first_day):
    df = load_hourly()
    d = pseudo_test(df, first_day)
    tr = df.drop(d.index)
    P = fit_params(tr)
    M = train(tr)
    out = {"hores": len(d), "kappa": P["kappa"]}
    o = oracle(d)
    out["oracle"] = {"score": round(o["score"], 4), "co2_pct": round(o["co2_pct"], 2), "cost_pct": round(o["cost_pct"], 2)}
    out["sense_bateria"] = _res(_Zero(), d)
    out["plantilla"] = _res(_starter(), d)
    out["regles_quantils"] = _res(_Fallback({**P, **NO_GBM}), d)
    out["mpc"] = _res(BatteryAgent({**P, **NO_GBM}), d)
    # Mode difícil (3 h)
    out["dificil_sense_previsio"] = _res(BatteryAgent({**P, **NO_GBM, "plan_hours": 3}), d, horizon=3)
    out["dificil_perfils"] = _res(BatteryAgent({**P, **NO_GBM}), d, horizon=3)
    out["dificil_gbm"] = _res(BatteryAgent({**P, "models": M}), d, horizon=3)
    # Valor de la informació: quantes hores de previsió calen?
    out["horitzo"] = {str(h): _res(BatteryAgent({**P, **NO_GBM, "plan_hours": h}), d, horizon=h)["score"]
                      for h in (1, 2, 3, 4, 6, 8, 12, 16, 20, 24)}
    # Compromís CO2 <-> cost: escombrat del pes del preu
    out["kappa_sweep"] = {}
    for f in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 1000.0):
        r = _res(BatteryAgent({**P, **NO_GBM, "kappa": P["kappa"] * f}), d)
        out["kappa_sweep"][str(f)] = {k: r[k] for k in ("score", "co2_pct", "cost_pct")}
    # Valor de conèixer el consum: MPC amb consum net perfecte (cota) -> l'oracle amb horitzó 24 ja ho és
    return first_day, out


if __name__ == "__main__":
    with ProcessPoolExecutor(3) as ex:
        res = dict(ex.map(eval_fold, (1, 8, 15)))
    json.dump(res, open(ROOT / "docs/results.json", "w"), indent=1)
    for k in ("oracle", "plantilla", "regles_quantils", "mpc", "dificil_sense_previsio", "dificil_perfils", "dificil_gbm"):
        print(f"{k:24s}", [res[f][k]["score"] for f in (1, 8, 15)], "mitjana %.3f" % np.mean([res[f][k]["score"] for f in (1, 8, 15)]))
