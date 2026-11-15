"""
Mode Difícil: entrena els quatre models de predicció que fa servir l'agent.

  co2, preu  -> hores +3 … +23  (l'entorn només en dona 3)
  consum     -> hores +0 … +23  (l'hora actual tampoc es coneix: només la passada)
  solar ×12  -> hores +0 … +23

Cada model és un LightGBM directe multi-horitzó (l'antelació és una variable).
Les variables de memòria (última hora, ahir a la mateixa hora, mitjana de 24 h)
s'emmascaren imitant el test: cada setmana comença sense historial, així que la
posició dins del bloc se sorteja uniformement entre 0 i 167 hores.

També calcula els quantils de l'error de predicció del consum net, que l'agent
converteix en escenaris dins del programa lineal.

Ús:  python src/hard_train.py  ->  submission_dificil/{model_*.txt, params.json}
"""
import json

import numpy as np
import pandas as pd

from core import ALPHA, BETA, ROOT, load_hourly

LEADS = list(range(24))
FC = ["lead", "h_obj", "we_obj", "mes", "c0", "c1", "c2", "p0", "p1", "p2", "c_ahir", "p_ahir", "c_m24", "p_m24", "c_ult", "p_ult",
      "c_d", "p_d", "c_min24", "c_max24", "p_min24", "p_max24", "c_ahir2", "p_ahir2", "i_ult", "v_ult"]
FL = ["lead", "h_obj", "dow_obj", "we_obj", "mes", "l_ult", "l_ahir", "l_ult_ahir", "l_m24", "t_ult",
      "l_2", "l_3", "l_ahir2", "l_d", "i_ult"]
FV = ["lead", "h_obj", "mes", "doy", "v_ult", "i_ult", "v_ahir", "v_max24", "t_ult",
      "env_obj", "kt_ult", "v_2", "v_ahir2", "kt_ahir"]
QLEVELS = [(2 * i + 1) / 14 for i in range(7)]
PV_BINS = [20.0, 400.0]


def pv_envelope(df: pd.DataFrame) -> list:
    """Envolupant de cel clar: percentil 95 de la solar (×12) per mes i hora. Fila 0 = tot l'any."""
    v = df["pv_kw"] * 12.0
    year = v.groupby(df.index.hour).quantile(0.95).reindex(range(24)).fillna(0.0).values
    bym = {m: g.groupby(g.index.hour).quantile(0.95).reindex(range(24)).fillna(0.0).values for m, g in v.groupby(df.index.month)}
    out = [year]
    for m in range(1, 13):
        if m in bym:
            out.append(bym[m])
        else:
            neigh = [bym[k] for k in ((m - 2) % 12 + 1, m % 12 + 1) if k in bym]
            out.append(np.mean(neigh, axis=0) if neigh else year)
    return np.round(np.array(out), 1).tolist()


def build_dataset(df: pd.DataFrame, seed=0, env=None, mask=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    env = np.asarray(env if env is not None else pv_envelope(df))
    idx = df.index
    S = {"c": df["co2_kg_per_kwh"], "p": df["precio_eur_per_kwh"], "l": df["load_kw"], "v": df["pv_kw"] * 12.0,
         "i": df["irradiancia_wm2"], "t": df["temp_ambient_c"]}

    def at(s, hours):
        return s.reindex(idx + pd.Timedelta(hours=hours)).values

    lag = {k: np.stack([at(S[k], -j) for j in range(1, 50)]) for k in ("c", "p", "l", "v")}   # lag[k][j-1] = valor a t-j
    pos = rng.integers(0, 168, len(df)) if mask else np.full(len(df), 10 ** 6)   # hores d'historial disponibles

    def avail(x, dist):
        x = np.array(x, dtype=float, copy=True); x[pos < dist] = np.nan; return x

    base = pd.DataFrame(index=idx)
    for k in (0, 1, 2):                          # les 3 hores conegudes són les files següents (com al simulador)
        base[f"c{k}"] = S["c"].shift(-k).values; base[f"p{k}"] = S["p"].shift(-k).values
    base["mes"] = idx.month; base["doy"] = idx.dayofyear
    base["c_ult"], base["p_ult"] = avail(lag["c"][0], 1), avail(lag["p"][0], 1)
    base["l_ult"], base["v_ult"] = avail(lag["l"][0], 1), avail(lag["v"][0], 1)
    base["i_ult"], base["t_ult"] = avail(at(S["i"], -1), 1), avail(at(S["t"], -1), 1)
    base["c_m24"], base["p_m24"] = avail(lag["c"][:24].mean(0), 24), avail(lag["p"][:24].mean(0), 24)
    base["l_m24"], base["v_max24"] = avail(lag["l"][:24].mean(0), 24), avail(lag["v"][:24].max(0), 24)
    base["l_ult_ahir"] = avail(lag["l"][24], 25)
    base["c_d"] = avail((S["c"].values + S["c"].shift(-1).values + S["c"].shift(-2).values) / 3 - lag["c"][21:24].mean(0), 24)
    base["p_d"] = avail((S["p"].values + S["p"].shift(-1).values + S["p"].shift(-2).values) / 3 - lag["p"][21:24].mean(0), 24)
    base["c_min24"], base["c_max24"] = avail(lag["c"][:24].min(0), 24), avail(lag["c"][:24].max(0), 24)
    base["p_min24"], base["p_max24"] = avail(lag["p"][:24].min(0), 24), avail(lag["p"][:24].max(0), 24)
    base["l_2"], base["l_3"], base["v_2"] = avail(lag["l"][1], 2), avail(lag["l"][2], 3), avail(lag["v"][1], 2)
    base["l_d"] = avail(lag["l"][0] - lag["l"][24], 25)
    hour_prev = (idx - pd.Timedelta(hours=1)).hour
    env_prev = env[idx.month.values, hour_prev]
    base["kt_ult"] = np.where(env_prev > 50, base["v_ult"].values / np.maximum(env_prev, 1.0), np.nan)
    rows = []
    for h in LEADS:
        tgt = idx + pd.Timedelta(hours=h)
        d = base.copy()
        d["lead"] = h; d["h_obj"] = tgt.hour; d["dow_obj"] = tgt.dayofweek; d["we_obj"] = (tgt.dayofweek >= 5).astype(int)
        for k in ("c", "p", "l", "v"):
            d[f"{k}_ahir"] = avail(lag[k][24 - h - 1], 24 - h)
            d[f"{k}_ahir2"] = avail(lag[k][48 - h - 1], 48 - h)
        env_obj = env[idx.month.values, tgt.hour]
        d["env_obj"] = env_obj
        d["kt_ahir"] = np.where(env_obj > 50, d["v_ahir"].values / np.maximum(env_obj, 1.0), np.nan)
        d["y_co2"], d["y_preu"], d["y_load"], d["y_pv"] = at(S["c"], h), at(S["p"], h), at(S["l"], h), at(S["v"], h)
        rows.append(d)
    return pd.concat(rows).dropna(subset=["y_co2", "y_preu", "y_load", "y_pv", "c1", "c2"])


def _fit(X, y, n=500):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.04, num_leaves=31, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=1)
    m.fit(X, y)
    return m.booster_


def fit_models(data: pd.DataFrame) -> dict:
    mk = data[data["lead"] >= 3]
    return {"co2": _fit(mk[FC].values, mk["y_co2"].values), "preu": _fit(mk[FC].values, mk["y_preu"].values),
            "load": _fit(data[FL].values, data["y_load"].values), "pv": _fit(data[FV].values, data["y_pv"].values)}


def net_quantiles(models, data: pd.DataFrame) -> list:
    """Quantils de l'error (real − previst) del consum net, per antelació i nivell de solar prevista."""
    pv = np.maximum(models["pv"].predict(data[FV].values), 0.0)
    err = (data["y_load"].values - data["y_pv"].values) - (models["load"].predict(data[FL].values) - pv)
    b = np.digitize(pv, PV_BINS)
    out = []
    for h in LEADS:
        row = []
        for k in range(3):
            e = err[(data["lead"].values == h) & (b == k)]
            row.append(np.quantile(e, QLEVELS).round(1).tolist() if len(e) >= 30 else np.quantile(err[b == k], QLEVELS).round(1).tolist())
        out.append(row)
    return out


def train_all(df: pd.DataFrame, seed=0):
    """Retorna (models, params). Els quantils es calculen fora de mostra (20 % dels dies)."""
    env = pv_envelope(df)
    data = build_dataset(df, seed, env)
    days = pd.Series(data.index.normalize().unique())
    hold = set(days.sample(frac=0.2, random_state=seed))
    is_hold = data.index.normalize().isin(hold)
    q = net_quantiles(fit_models(data[~is_hold]), data[is_hold])
    models = fit_models(data)
    netr = df["load_kw"] - 12.0 * df["pv_kw"]; net = netr.clip(lower=0)
    kappa = (BETA / ALPHA) * (net * df["co2_kg_per_kwh"]).sum() / (net * df["precio_eur_per_kwh"]).sum()
    params = {"kappa": round(float(kappa), 5), "net_quantiles": q, "pv_bins": PV_BINS, "pv_env": env}
    return models, params


if __name__ == "__main__":
    models, params = train_all(load_hourly())
    out = ROOT / "submission_dificil"
    for k, b in models.items():
        b.save_model(str(out / f"model_{k}.txt"))
    json.dump(params, open(out / "params.json", "w"))
    print("kappa", params["kappa"], [f"{(out / f'model_{k}.txt').stat().st_size/1e6:.1f} MB" for k in models])
