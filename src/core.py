"""Utilitats comunes: càrrega de dades, simulador oficial i òptim amb previsió perfecta (oracle)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, diags, hstack, identity, vstack

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter_kit"))
from environment import CampusBatteryEnv  # noqa: E402

CAP, PMAX, ETA = 2000.0, 500.0, 0.9 ** 0.5
ALPHA, BETA = 0.7, 0.3
EXPORT_PRICE = 0.04   # €/kWh, compensació d'excedents segons la plataforma web
PENALTY = 0.20        # punts per infracció segons la plataforma web


def load_hourly() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/processed/campus_hourly.csv", parse_dates=["Fecha"], index_col="Fecha")


def run(agent, df: pd.DataFrame, mask=None, return_env=False, horizon=24):
    """Executa un agent sobre el simulador oficial amb les regles de la plataforma web."""
    env = CampusBatteryEnv(df, export_price=EXPORT_PRICE, forecast_horizon=horizon)
    obs, done = env.reset(), False
    while not done:
        obs, done = env.step(agent.act(obs))
    r = env.evaluate_results(mask=mask)
    r["score_exacte"] = score_from_env(env)
    return (r, env) if return_env else r


def score_from_env(env) -> float:
    """Score sense arrodonir (el simulador arrodoneix els percentatges a 2 decimals)."""
    g, e = np.asarray(env.grid), np.asarray(env.exports)
    d = env.df
    netr = d["load_kw"].values - d["pv_kw"].values
    net = np.maximum(netr, 0)
    price, co2 = d["precio_eur_per_kwh"].values, d["co2_kg_per_kwh"].values
    bc = (net * price).sum() - (np.maximum(-netr, 0) * env.export_price).sum()
    d_cost = 1 - ((g * price).sum() - (e * env.export_price).sum()) / bc
    d_co2 = 1 - (g * co2).sum() / (net * co2).sum()
    return 100 * (ALPHA * d_co2 + BETA * d_cost) - PENALTY * (env.clipped + env.violations)


def oracle(df: pd.DataFrame, soc0=CAP / 2, export_price=EXPORT_PRICE):
    """Òptim global amb coneixement perfecte de tot el futur (programa lineal). Cota superior del Score."""
    netr = df["load_kw"].values - df["pv_kw"].values
    net = np.maximum(netr, 0.0)
    price, co2 = df["precio_eur_per_kwh"].values, df["co2_kg_per_kwh"].values
    base_cost = (net * price).sum() - (np.maximum(-netr, 0) * export_price).sum()
    base_co2 = (net * co2).sum()
    w = ALPHA * co2 / base_co2 + BETA * price / base_cost
    wexp = np.minimum(BETA * export_price / base_cost, w)
    n = len(df)
    I = identity(n, format="csr")
    D = I - diags([np.ones(n - 1)], [-1], format="csr")          # s_k - s_{k-1}
    Z = csr_matrix((n, n))
    # variables: c, d, g (import), e (export), s (SOC al final de l'hora)
    A_eq = vstack([hstack([-ETA * I, I / ETA, Z, Z, D]),          # dinàmica del SOC
                   hstack([I, -I, -I, I, Z])]).tocsr()            # balanç: net + c - d = g - e
    b_eq = np.concatenate([np.r_[soc0, np.zeros(n - 1)], -netr])
    cost = np.concatenate([np.zeros(2 * n), w, -wexp, np.zeros(n)]) * 1e6   # escalat numèric
    bounds = [(0, PMAX)] * (2 * n) + [(0, None)] * (2 * n) + [(0, CAP)] * n
    res = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    c, d, g, e = (res.x[k * n:(k + 1) * n] for k in range(4))
    d_co2 = 100 * (1 - (g * co2).sum() / base_co2)
    d_cost = 100 * (1 - ((g * price).sum() - (e * export_price).sum()) / base_cost)
    return {"score": ALPHA * d_co2 + BETA * d_cost, "co2_pct": d_co2, "cost_pct": d_cost, "p": c - d, "soc": res.x[4 * n:]}


def pseudo_test(df: pd.DataFrame, first_day=15, n_days=7) -> pd.DataFrame:
    """Imita l'estructura del test ocult (7 dies seguits de cada mes, concatenats) amb dies del train."""
    m = (df.index.day >= first_day) & (df.index.day < first_day + n_days)
    return df[m]
