"""Mode Difícil: bateria 6.000 kWh / ±1.500 kW, solar ×12, només 3 h de previsió, excedents a 0 €."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, diags, hstack, identity, vstack

from core import ALPHA, BETA, PENALTY, CampusBatteryEnv, load_hourly, pseudo_test  # noqa: F401

SPEC_HARD = {"capacity_kwh": 6000.0, "power_kw": 1500.0, "efficiency": 0.90, "soc_init": 0.50,
             "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1.0, "forecast_horizon": 3}
ETA = 0.9 ** 0.5


def hard_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["pv_kw"] = d["pv_kw"] * 12.0
    return d


def score_env(env) -> dict:
    g = np.asarray(env.grid); d = env.df
    netr = d["load_kw"].values - d["pv_kw"].values
    net = np.maximum(netr, 0)
    price, co2 = d["precio_eur_per_kwh"].values, d["co2_kg_per_kwh"].values
    d_cost = 1 - (g * price).sum() / (net * price).sum()
    d_co2 = 1 - (g * co2).sum() / (net * co2).sum()
    inf = env.clipped + env.violations
    return {"score": 100 * (ALPHA * d_co2 + BETA * d_cost) - PENALTY * inf, "co2_pct": 100 * d_co2,
            "cost_pct": 100 * d_cost, "infraccions": int(inf)}


def run_hard(agent, df_hard: pd.DataFrame, horizon=3, return_env=False):
    spec = dict(SPEC_HARD); spec["forecast_horizon"] = horizon
    env = CampusBatteryEnv(df_hard, spec=spec, export_price=0.0)
    obs, done = env.reset(), False
    while not done:
        obs, done = env.step(agent.act(obs))
    r = score_env(env)
    return (r, env) if return_env else r


def oracle_hard(df_hard: pd.DataFrame, cap=6000.0, pmax=1500.0):
    netr = df_hard["load_kw"].values - df_hard["pv_kw"].values
    net = np.maximum(netr, 0.0)
    price, co2 = df_hard["precio_eur_per_kwh"].values, df_hard["co2_kg_per_kwh"].values
    bco2, bcost = (net * co2).sum(), (net * price).sum()
    w = ALPHA * co2 / bco2 + BETA * price / bcost
    n = len(netr)
    I = identity(n, format="csr"); D = I - diags([np.ones(n - 1)], [-1], format="csr"); Z = csr_matrix((n, n))
    A_eq = hstack([-ETA * I, I / ETA, Z, D]).tocsr()               # c, d, g, s
    b_eq = np.r_[cap / 2, np.zeros(n - 1)]
    A_ub = hstack([I, -I, -I, Z]).tocsr(); b_ub = -netr           # g >= net + c - d
    res = linprog(np.concatenate([np.zeros(2 * n), w * 1e6, np.zeros(n)]), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, pmax)] * (2 * n) + [(0, None)] * n + [(0, cap)] * n, method="highs")
    g = res.x[2 * n:3 * n]
    d_co2 = 100 * (1 - (g * co2).sum() / bco2); d_cost = 100 * (1 - (g * price).sum() / bcost)
    return {"score": ALPHA * d_co2 + BETA * d_cost, "co2_pct": d_co2, "cost_pct": d_cost, "p": res.x[:n] - res.x[n:2 * n]}
