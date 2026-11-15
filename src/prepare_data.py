"""
Construeix la taula horària del campus a partir de la telemetria de 5 minuts.

Segueix exactament la recepta del tutorial oficial (mateixa definició de
`load_kw` i `pv_kw` que fa servir el simulador) i hi afegeix la meteorologia.

Ús:
    python src/prepare_data.py <carpeta data/train> [sortida.csv]
"""
import sys
from pathlib import Path

import pandas as pd


def build_hourly(data_dir: Path) -> pd.DataFrame:
    cont = pd.read_csv(data_dir / "contador.csv", parse_dates=["Fecha"]).set_index("Fecha")
    inv = pd.read_csv(data_dir / "inversor.csv", parse_dates=["Fecha"], low_memory=False).set_index("Fecha")
    met = pd.read_csv(data_dir / "meteo.csv", parse_dates=["Fecha"]).set_index("Fecha")
    pvpc = pd.read_csv(data_dir / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(data_dir / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

    # Consum: signe invertit de la potència exportada, acotat a [0, 5000] kW
    load_5 = (-cont["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(lower=0, upper=5000)
    load_h = load_5.resample("1h").mean().dropna()

    # Solar: suma de la potència activa AC dels 3 inversors
    cols = [
        [c for c in inv.columns if f"INVERSOR {i}" in c.upper() and "POTENCIA ACTIVA" in c.upper() and "FASE" not in c.upper()][0]
        for i in (1, 2, 3)
    ]
    pv_5 = sum(inv[c].fillna(0) for c in cols).clip(lower=0)
    pv_h = pv_5.resample("1h").mean().dropna()

    met_h = pd.DataFrame({
        "irradiancia_wm2": met["Atersa - IRRADIANCIA (W/m2)"],
        "temp_ambient_c": met["Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)"],
        "temp_panel_c": met["Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)"],
        "vent_ms": met["Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)"],
    }).resample("1h").mean()

    df = pd.DataFrame({
        "load_kw": load_h,
        "pv_kw": pv_h,
        "precio_eur_per_kwh": pvpc["precio_eur_per_kwh"],
        "co2_kg_per_kwh": co2["co2_kg_per_kwh"],
    }).dropna()
    df = df.join(met_h)
    df.index.name = "Fecha"
    return df


if __name__ == "__main__":
    data_dir = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "data/processed/campus_hourly.csv"
    df = build_hourly(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(out)
    print(f"{len(df)} hores -> {out}")
