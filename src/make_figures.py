"""Genera les figures de la memòria i la presentació a docs/figures/ (necessita docs/results.json)."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from core import ROOT, load_hourly, pseudo_test, run

sys.path.insert(0, str(ROOT / "submission"))
from agent import BatteryAgent  # noqa: E402
from fit_params import fit_params  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
    "lines.linewidth": 2, "figure.dpi": 200,
})
OUT = ROOT / "docs/figures"
df = load_hourly()
R = json.load(open(ROOT / "docs/results.json"))
FOLDS = ["1", "8", "15"]
mean = lambda key, f="score": float(np.mean([R[k][key][f] for k in FOLDS]))


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig); print(name)


# 1. Perfils diaris ---------------------------------------------------------
g = df.groupby(df.index.hour).mean(numeric_only=True)
fig, ax = plt.subplots(1, 4, figsize=(13, 2.9))
for a, col, title, c, scale in ((ax[0], "load_kw", "Consum del campus (kW)", BLUE, 1), (ax[1], "pv_kw", "Solar fotovoltaica (kW)", YELLOW, 1),
                                (ax[2], "precio_eur_per_kwh", "Preu PVPC (€/kWh)", ORANGE, 1), (ax[3], "co2_kg_per_kwh", "CO₂ de la xarxa (kg/kWh)", AQUA, 1)):
    a.plot(g.index, g[col], color=c); a.set_title(title); a.set_xticks([0, 6, 12, 18, 23]); a.set_ylim(0, g[col].max() * 1.15)
    a.set_xlabel("hora del dia")
save(fig, "01_perfils_diaris.png")

# 2. Consum net vs potència de la bateria -----------------------------------
net = (df.load_kw - df.pv_kw)
fig, a = plt.subplots(figsize=(6.5, 3))
a.hist(net, bins=np.arange(0, 2500, 50), color=BLUE, rwidth=0.9)
a.axvline(500, color=INK, lw=1.2, ls="--"); a.text(520, a.get_ylim()[1] * 0.9, "potència màx. bateria\n500 kW", color=INK2, fontsize=9, va="top")
a.set_title(f"El consum net supera els 500 kW el {100*(net>=500).mean():.1f}% de les hores"); a.set_xlabel("consum − solar (kW)"); a.set_ylabel("hores")
save(fig, "02_consum_net.png")

# 3. Una setmana d'operació --------------------------------------------------
d = pseudo_test(df, 15); P = fit_params(df.drop(d.index))
r, env = run(BatteryAgent({**P, "models": {"co2": None, "preu": None}}), d, return_env=True)
sl = slice(0, 168); x = d.index[sl]
fig, ax = plt.subplots(4, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.1, 1.1]})
ax[0].plot(x, d.co2_kg_per_kwh.values[sl], color=AQUA); ax[0].set_title("CO₂ de la xarxa (kg/kWh)")
ax[1].plot(x, d.precio_eur_per_kwh.values[sl], color=ORANGE); ax[1].set_title("Preu PVPC (€/kWh)")
p = np.diff(np.r_[1000.0, env.soc_series])[sl]; p = np.where(p > 0, p / 0.9 ** 0.5, p * 0.9 ** 0.5)
ax[2].bar(x, np.where(p > 0, p, 0), width=0.035, color=BLUE, label="càrrega"); ax[2].bar(x, np.where(p < 0, p, 0), width=0.035, color=ORANGE, label="descàrrega")
ax[2].set_title("Consigna de la bateria (kW)"); ax[2].legend(frameon=False, ncol=2, loc="upper right", fontsize=9); ax[2].set_ylim(-560, 700)
ax[3].fill_between(x, np.asarray(env.soc_series)[sl], color=BLUE, alpha=0.25, lw=0); ax[3].plot(x, np.asarray(env.soc_series)[sl], color=BLUE); ax[3].set_title("Energia emmagatzemada (kWh)")
ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m")); ax[3].set_ylim(0, 2100)
fig.tight_layout(); save(fig, "03_setmana_exemple.png")

# 4. Comparació d'estratègies -------------------------------------------------
rows = [("Plantilla oficial (regles fixes)", mean("plantilla"), GREY), ("Regles per quantils (pla B)", mean("regles_quantils"), GREY),
        ("El nostre agent (MPC + LP)", mean("mpc"), BLUE), ("Òptim teòric (futur perfecte)", mean("oracle"), INK2)]
fig, a = plt.subplots(figsize=(7.5, 2.6))
for i, (n, v, c) in enumerate(rows[::-1]):
    a.barh(i, v, color=c, height=0.55); a.text(max(v, 0) + 0.08, i, f"{v:.2f}".replace(".", ","), va="center", color=INK, fontsize=10)
a.set_yticks(range(4)); a.set_yticklabels([r[0] for r in rows[::-1]], color=INK); a.set_xlim(0, mean("oracle") * 1.15); a.grid(axis="y", visible=False)
a.set_title("Score mitjà en validació (mode estàndard, 24 h de previsió)"); a.set_xlabel("punts")
save(fig, "04_comparacio_score.png")

# 5. Valor de la informació ---------------------------------------------------
hs = [int(h) for h in R["1"]["horitzo"]]
vals = [np.mean([R[k]["horitzo"][str(h)] for k in FOLDS]) for h in hs]
fig, a = plt.subplots(figsize=(6.5, 3.2))
a.axhline(mean("oracle"), color=INK2, lw=1.2, ls="--"); a.text(24, mean("oracle") + 0.1, "òptim teòric", ha="right", color=INK2, fontsize=9)
a.plot(hs, vals, color=BLUE, marker="o", ms=5, label="només amb la previsió donada")
a.plot([3], [mean("dificil_gbm")], marker="D", ms=8, color=ORANGE, ls="none", label="3 h + la nostra predicció")
a.set_xticks(hs); a.set_ylim(0, mean("oracle") * 1.15); a.set_xlabel("hores de previsió de preu i CO₂ disponibles"); a.set_ylabel("Score")
a.legend(frameon=False, loc="lower right", fontsize=9); a.set_title("Quantes hores de futur calen?")
save(fig, "05_valor_informacio.png")

# 6. Compromís CO2 - cost ------------------------------------------------------
ks = list(R["1"]["kappa_sweep"])
co = [np.mean([R[k]["kappa_sweep"][f]["co2_pct"] for k in FOLDS]) for f in ks]
ct = [np.mean([R[k]["kappa_sweep"][f]["cost_pct"] for k in FOLDS]) for f in ks]
fig, a = plt.subplots(figsize=(5.6, 3.6))
a.plot(co, ct, color=GREY, lw=1.5, marker="o", ms=5, zorder=1)
i = ks.index("1.0"); a.scatter([co[i]], [ct[i]], s=90, color=BLUE, zorder=3)
a.annotate("el nostre pes\n(màxim Score)", (co[i], ct[i]), xytext=(-70, -38), textcoords="offset points", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2))
a.annotate("només CO₂", (co[0], ct[0]), xytext=(8, -4), textcoords="offset points", color=INK2, fontsize=9)
a.annotate("només cost", (co[-1], ct[-1]), xytext=(8, 0), textcoords="offset points", color=INK2, fontsize=9)
a.set_xlabel("reducció de CO₂ (%)"); a.set_ylabel("estalvi de cost (%)"); a.set_title("Compromís entre emissions i factura")
save(fig, "06_compromis_co2_cost.png")
