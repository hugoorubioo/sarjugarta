"""
Aprèn dels històrics de train els paràmetres que l'agent carrega de `params.json`:

* kappa: pes del preu dins del senyal de decisió  w = co2 + kappa·preu
* profiles: forma horària normalitzada de CO2 i preu per mes i tipus de dia
* net_profile: consum net típic per tipus de dia i hora (percentil baix, prudent)

Ús:  python src/fit_params.py   ->  submission/params.json
"""
import json

import numpy as np
import pandas as pd

from core import ALPHA, BETA, ROOT, load_hourly


def fit_params(df: pd.DataFrame) -> dict:
    netr = df["load_kw"] - df["pv_kw"]
    net = netr.clip(lower=0)
    kappa = (BETA / ALPHA) * (net * df["co2_kg_per_kwh"]).sum() / (net * df["precio_eur_per_kwh"]).sum()

    d = df.copy()
    d["hour"], d["month"], d["we"] = d.index.hour, d.index.month, (d.index.dayofweek >= 5).astype(int)
    d["date"] = d.index.normalize()
    profiles = {}
    for name, col in (("co2", "co2_kg_per_kwh"), ("preu", "precio_eur_per_kwh")):
        # forma = valor / mitjana del seu dia  (només dies complets)
        full = d.groupby("date")[col].transform("count") == 24
        dd = d[full].copy()
        dd["shape"] = dd[col] / dd.groupby("date")[col].transform("mean").clip(lower=1e-6)
        year = dd.groupby(["we", "hour"])["shape"].mean().unstack().values
        bym = {m: g.groupby(["we", "hour"])["shape"].mean().unstack() for m, g in dd.groupby("month")}
        out = {"0": year.tolist()}
        for m in range(1, 13):
            if m in bym and bym[m].shape == (2, 24) and not bym[m].isna().any().any():
                tab = bym[m].values
            else:  # mes sense dades (p. ex. novembre): mitjana dels veïns disponibles
                neigh = [bym[k].values for k in ((m - 2) % 12 + 1, m % 12 + 1) if k in bym and bym[k].shape == (2, 24)]
                tab = np.mean(neigh, axis=0) if neigh else year
            # suavitzem cap al perfil anual per robustesa (poques setmanes per mes)
            out[str(m)] = (0.75 * tab + 0.25 * year).tolist()
        profiles[name] = out

    ok = d[d["load_kw"] > 50]
    netp = (ok["load_kw"] - ok["pv_kw"]).groupby([ok["we"], ok["hour"]]).quantile(0.10).unstack().values
    return {"kappa": round(float(kappa), 5), "profiles": profiles, "net_profile": np.round(netp, 1).tolist()}


if __name__ == "__main__":
    params = fit_params(load_hourly())
    out = ROOT / "submission/params.json"
    json.dump(params, open(out, "w"))
    print("kappa =", params["kappa"], "->", out)
