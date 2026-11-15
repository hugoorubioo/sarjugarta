"""
Mode Difícil (només 3 h de previsió): model que prediu CO2 i preu de les hores +3 … +23.

Un LightGBM per senyal, amb l'antelació com a variable (model directe multi-horitzó).
Variables: hora i tipus de dia de l'hora objectiu, mes, les 3 hores conegudes de
tots dos senyals i la memòria recent de l'agent (ahir a la mateixa hora, mitjana
de les últimes 24 h, última hora). La memòria s'emmascara a l'atzar en entrenar
perquè al test cada setmana comença sense historial.

Ús:  python src/train_forecaster.py  ->  submission/forecaster_{co2,preu}.txt
"""
import numpy as np
import pandas as pd

from core import ROOT, load_hourly

LEADS = list(range(3, 24))
FEATS = ["lead", "h_obj", "we_obj", "mes", "c0", "c1", "c2", "p0", "p1", "p2",
         "c_ahir", "p_ahir", "c_m24", "p_m24", "c_ult", "p_ult"]


def build_dataset(df: pd.DataFrame, seed=0, mask_prob=0.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = df.index
    c = df["co2_kg_per_kwh"]; p = df["precio_eur_per_kwh"]

    def at(s, hours):  # valor exactament `hours` hores després (NaN si hi ha forat)
        return s.reindex(idx + pd.Timedelta(hours=hours)).values

    base = {"c0": c.values, "p0": p.values}
    # al simulador les "properes hores" són les files següents (travessa forats)
    for k in (1, 2):
        base[f"c{k}"] = c.shift(-k).values; base[f"p{k}"] = p.shift(-k).values
    # mitjana de les 24 h anteriors (NaN si la memòria no és completa)
    cm = np.stack([at(c, -j) for j in range(1, 25)]).mean(axis=0)
    pm = np.stack([at(p, -j) for j in range(1, 25)]).mean(axis=0)
    c_ult, p_ult = at(c, -1), at(p, -1)
    rows = []
    for h in LEADS:
        tgt = idx + pd.Timedelta(hours=h)
        d = pd.DataFrame(base, index=idx)
        d["lead"] = h
        d["h_obj"] = tgt.hour
        d["we_obj"] = (tgt.dayofweek >= 5).astype(int)
        d["mes"] = idx.month
        d["c_ahir"], d["p_ahir"] = at(c, h - 24), at(p, h - 24)
        d["c_m24"], d["p_m24"] = cm, pm
        d["c_ult"], d["p_ult"] = c_ult, p_ult
        d["y_co2"], d["y_preu"] = at(c, h), at(p, h)
        rows.append(d)
    out = pd.concat(rows).dropna(subset=["y_co2", "y_preu", "c1", "c2"])
    # emmascarem la memòria en una part de les mostres (inici de setmana al test)
    m = rng.random(len(out)) < mask_prob
    out.loc[m, ["c_ahir", "p_ahir", "c_m24", "p_m24", "c_ult", "p_ult"]] = np.nan
    return out


def train(df: pd.DataFrame, n_estimators=400):
    import lightgbm as lgb
    data = build_dataset(df)
    models = {}
    for name in ("co2", "preu"):
        m = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        m.fit(data[FEATS].values, data[f"y_{name}"].values)
        models[name] = m.booster_
    return models


if __name__ == "__main__":
    models = train(load_hourly())
    for name, b in models.items():
        path = ROOT / f"submission/forecaster_{name}.txt"
        b.save_model(str(path))
        print(path, f"{path.stat().st_size/1e6:.2f} MB")
