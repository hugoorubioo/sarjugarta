"""
agent.py — Repte UPC Sostenible (Datathon 2026)
================================================
Control predictiu (MPC) de la bateria del campus amb programació lineal.

Idea en una frase
-----------------
A cada hora resolem el problema d'optimització EXACTE de la bateria sobre un
horitzó mòbil (les 24 h de preu i CO2 que ens dona l'entorn + una extensió
prevista amb perfils apresos de l'històric), apliquem només la primera
consigna i tornem a planificar a l'hora següent.

Peces
-----
1. Senyal únic de decisió. El Score és 0,70·ΔCO2(%) + 0,30·ΔCost(%); com que
   tots dos termes són lineals en l'energia importada, minimitzar el Score
   equival a minimitzar  Σ xarxa_t · w_t  amb
       w_t = co2_t + KAPPA · preu_t
   on KAPPA converteix €/kWh a "kg CO2 equivalents de Score"
   (KAPPA = 0,30/0,70 · CO2_base/Cost_base, estimat amb les dades de train).
2. Programa lineal (scipy/HiGHS) amb la física exacta de l'enunciat:
   rendiment 94,87 % per sentit, ±500 kW, 0–2.000 kWh, i el balanç al punt de
   connexió (descarregar per sobre del consum net només val 0,04 €/kWh).
3. Previsió més enllà de l'horitzó conegut: perfil horari per mes i tipus de
   dia (apresos al train) reescalat al nivell observat recentment.
4. Seguretat: clamping final que garanteix 0 infraccions, i una política de
   reserva sense dependències si el solver falla o anem justos de temps.

Només fa servir biblioteques permeses: numpy, scipy, json, pathlib, datetime.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

try:  # scipy és a la llista oficial de biblioteques; si faltés, tenim pla B
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, diags, hstack, identity, vstack
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

# ----------------------------------------------------------------------------
# Constants físiques (enunciat oficial)
# ----------------------------------------------------------------------------
CAP_KWH = 2000.0
P_MAX_KW = 500.0
ETA = 0.9 ** 0.5          # rendiment per sentit (round-trip 90 %)
EPS = 1e-4                # marge de seguretat numèric recomanat per l'organització

DEFAULT_PARAMS = {
    "kappa": 0.2517,           # kg CO2 per € : pes relatiu del preu dins del senyal w
    "export_price": 0.04,      # €/kWh de compensació d'excedents
    "plan_hours": 24,          # horitzó total del pla (conegut + previst)
    "terminal_factor": 0.0,    # valor de l'energia que queda al final del pla (× mediana de w)
    "outage_kw": 50.0,         # per sota d'això, la lectura de consum es considera caiguda de telemetria
    "time_budget_s": 40.0,     # si superem aquest temps acumulat, passem a la política ràpida
    "profiles": None,          # perfils horaris normalitzats [mes][tipus_dia][hora] per a co2 i preu
    "net_profile": None,       # consum net típic [tipus_dia][hora] (kW)
}


def _parse_ts(ts):
    """Converteix el timestamp de l'observació a datetime (o None si no es pot)."""
    try:
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        if isinstance(ts, datetime):
            return ts
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


class BatteryAgent:
    def __init__(self, params=None):
        p = dict(DEFAULT_PARAMS)
        try:
            with open(Path(__file__).parent / "params.json", encoding="utf-8") as f:
                p.update(json.load(f))
        except Exception:
            pass
        if params:
            p.update(params)
        self.p = p

        # Mode Difícil: models LightGBM (text) que prediuen CO2 i preu de les hores +3…+23
        self.gbm = p.pop("models", None)
        if self.gbm is None:
            try:
                d = Path(__file__).parent
                if (d / "forecaster_co2.txt").exists() and (d / "forecaster_preu.txt").exists():
                    import lightgbm as lgb
                    self.gbm = {k: lgb.Booster(model_file=str(d / f"forecaster_{k}.txt")) for k in ("co2", "preu")}
            except Exception:
                self.gbm = None      # sense models: perfils horaris (pla B)

        self.max_len_seen = 0      # longitud màxima del vector de previsió vista (24 normal, 3 mode difícil)
        self.hist = []             # historial contigu recent: (hora, co2, preu)
        self.prev_ts = None
        self.elapsed = 0.0
        self._lp_cache = {}
        self.debug = {}

    # ------------------------------------------------------------------
    # 1. Previsió de CO2 i preu més enllà de l'horitzó conegut
    # ------------------------------------------------------------------
    def _profile(self, name, month, daytype):
        prof = self.p.get("profiles")
        if not prof:
            return None
        try:
            return np.asarray(prof[name][str(month)][daytype], dtype=float)
        except Exception:
            try:
                return np.asarray(prof[name]["0"][daytype], dtype=float)   # perfil anual
            except Exception:
                return None

    def _extend(self, name, known, hour0, dow0, month, n_extra):
        """Prediu `n_extra` hores després del vector conegut.

        forma (perfil normalitzat del mes i tipus de dia)  ×  nivell recent,
        on el nivell és la mitjana de (valor observat / perfil) de les hores
        recents i de les ja conegudes del futur.
        """
        L = len(known)
        hours = (hour0 + np.arange(L + n_extra)) % 24
        days = (dow0 + (hour0 + np.arange(L + n_extra)) // 24) % 7
        shape = np.ones(L + n_extra)
        for dt in (0, 1):
            prof = self._profile(name, month, dt)
            if prof is None:
                continue
            sel = (days >= 5) == bool(dt)
            shape[sel] = prof[hours[sel]]
        # nivell: hores conegudes del futur + historial contigu recent
        ratios = [k / s for k, s in zip(known, shape[:L]) if s > 1e-9]
        idx = 1 if name == "co2" else 2
        for (h, d, c, pr) in self.hist[-24:]:
            prof = self._profile(name, month, int(d >= 5))
            s = prof[h] if prof is not None else 1.0
            if s > 1e-9:
                ratios.append((c if idx == 1 else pr) / s)
        level = float(np.mean(ratios)) if ratios else 0.0
        pred = level * shape[L:]
        if prof is None and L >= 24:      # sense perfils: "demà s'assembla a avui"
            pred = np.resize(np.asarray(known[-24:], dtype=float), n_extra)
        return pred

    def _extend_gbm(self, co2, preu, hour0, dow0, month):
        """Mode Difícil (3 h conegudes): prediu les hores +3…+23 amb els models apresos."""
        leads = np.arange(3, 24)
        n = len(leads)
        nan = float("nan")
        H = self.hist                               # H[-1] és l'hora actual
        c_hist = [x[2] for x in H]; p_hist = [x[3] for x in H]
        c_ult = c_hist[-2] if len(H) >= 2 else nan
        p_ult = p_hist[-2] if len(H) >= 2 else nan
        c_m24 = float(np.mean(c_hist[-25:-1])) if len(H) >= 25 else nan
        p_m24 = float(np.mean(p_hist[-25:-1])) if len(H) >= 25 else nan
        c_ahir = np.full(n, nan); p_ahir = np.full(n, nan)
        for i, h in enumerate(leads):
            back = 24 - h                           # ahir a la mateixa hora = fa `back` hores
            if len(H) >= back + 1:
                c_ahir[i] = c_hist[-1 - back]; p_ahir[i] = p_hist[-1 - back]
        X = np.column_stack([
            leads, (hour0 + leads) % 24, ((dow0 + (hour0 + leads) // 24) % 7 >= 5).astype(float),
            np.full(n, month if month else nan),
            np.full(n, co2[0]), np.full(n, co2[1]), np.full(n, co2[2]),
            np.full(n, preu[0]), np.full(n, preu[1]), np.full(n, preu[2]),
            c_ahir, p_ahir, np.full(n, c_m24), np.full(n, p_m24), np.full(n, c_ult), np.full(n, p_ult),
        ])
        # un sol fil: prediccions minúscules, i evitem dependre dels nuclis del servidor
        return self.gbm["co2"].predict(X, num_threads=1), self.gbm["preu"].predict(X, num_threads=1)

    # ------------------------------------------------------------------
    # 2. Previsió del consum net (només importa quan és < 500 kW)
    # ------------------------------------------------------------------
    def _net_forecast(self, last_load, last_pv, hour0, dow0, n):
        net = np.full(n, 1000.0)
        prof = self.p.get("net_profile")
        if prof:
            hours = (hour0 + np.arange(n)) % 24
            days = (dow0 + (hour0 + np.arange(n)) // 24) % 7
            wk = np.asarray(prof[0], dtype=float)[hours]
            we = np.asarray(prof[1], dtype=float)[hours]
            net = np.where(days >= 5, we, wk)
        last_net = last_load - last_pv
        mode = int(self.p.get("use_load", 2))
        if mode == 0:
            return np.full(n, 1000.0)
        if last_load < self.p["outage_kw"]:
            # Caiguda de telemetria/subministrament: mentre dura, el campus no importa res,
            # així que descarregar no estalvia ni CO2 ni cost. Les caigudes duren hores.
            net[: min(int(self.p.get("outage_hours", 3)), n)] = 0.0
        elif mode == 2:
            # la primera hora s'assembla molt a l'anterior (persistència)
            net[0] = min(net[0], last_net) if last_net < P_MAX_KW else max(last_net, P_MAX_KW)
        return net

    # ------------------------------------------------------------------
    # 3. Programa lineal sobre l'horitzó
    # ------------------------------------------------------------------
    def _matrices(self, n):
        if n not in self._lp_cache:
            eye = identity(n, format="csr")
            dif = eye - diags([np.ones(n - 1)], [-1], format="csr") if n > 1 else eye
            zero = csr_matrix((n, n))
            # variables: c (càrrega), d (descàrrega), g (import), e (export), s (SOC final d'hora)
            a_eq = vstack([
                hstack([-ETA * eye, eye / ETA, zero, zero, dif]),   # s_k - s_{k-1} = η·c_k - d_k/η
                hstack([eye, -eye, -eye, eye, zero]),               # net_k + c_k - d_k = g_k - e_k
            ]).tocsr()
            bounds = [(0.0, P_MAX_KW)] * (2 * n) + [(0.0, None)] * (2 * n) + [(0.0, CAP_KWH)] * n
            self._lp_cache[n] = (a_eq, bounds)
        return self._lp_cache[n]

    def _solve(self, soc, w, net, terminal_value):
        n = len(w)
        a_eq, bounds = self._matrices(n)
        w_exp = np.minimum(self.p["kappa"] * self.p["export_price"], w)
        tie = 1e-6 + 1e-7 * np.arange(n)      # evita cicles inútils i, a igualtat, actua abans
        cost = np.concatenate([tie, tie, w, -w_exp, np.zeros(n)])
        cost[-1] = -terminal_value
        b_eq = np.concatenate([[soc], np.zeros(n - 1), -net])
        res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError("LP no resolt")
        return float(res.x[0] - res.x[n])

    # ------------------------------------------------------------------
    # 4. Política de reserva (sense solver): llindars per quantils de la finestra
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback(w):
        if len(w) < 4:
            return 0.0
        lo, hi = np.quantile(w, 0.25), np.quantile(w, 0.75)
        if w[0] <= lo and hi > lo / 0.9:
            return P_MAX_KW
        if w[0] >= hi:
            return -P_MAX_KW
        return 0.0

    # ------------------------------------------------------------------
    # Decisió horària
    # ------------------------------------------------------------------
    def act(self, obs: dict) -> float:
        t0 = datetime.now()
        soc = float(obs.get("soc_kwh", CAP_KWH / 2))
        soc = min(max(soc, 0.0), CAP_KWH)
        try:
            consigna = self._decide(obs, soc)
        except Exception:
            consigna = 0.0

        # --- Clamping de seguretat: garanteix 0 infraccions físiques ---
        max_c = max(0.0, min(P_MAX_KW, (CAP_KWH - soc) / ETA - EPS))
        max_d = max(0.0, min(P_MAX_KW, soc * ETA - EPS))
        consigna = float(min(max(consigna, -max_d), max_c))
        if not np.isfinite(consigna) or abs(consigna) < 1e-3:
            consigna = 0.0
        self.elapsed += (datetime.now() - t0).total_seconds()
        return consigna

    def _decide(self, obs, soc):
        hour = int(obs.get("hour", 0))
        dow = int(obs.get("day_of_week", 0))
        ts = _parse_ts(obs.get("timestamp"))
        month = ts.month if ts is not None else 0

        co2 = [float(x) for x in (obs.get("co2_properes_24h") or [])]
        preu = [float(x) for x in (obs.get("preus_properes_24h") or [])]
        if not co2 or not preu:
            co2, preu = [float(obs["co2_actual"])], [float(obs["preu_actual"])]
        # Per contracte, el primer element és l'hora actual; si no ho fos, l'hi posem al davant
        if "co2_actual" in obs and "preu_actual" in obs:
            ca, pa = float(obs["co2_actual"]), float(obs["preu_actual"])
            if abs(co2[0] - ca) > 1e-9 or abs(preu[0] - pa) > 1e-9:
                co2, preu = [ca] + co2, [pa] + preu
        L = min(len(co2), len(preu))
        co2, preu = co2[:L], preu[:L]

        # Historial contigu (es buida quan el test salta a la setmana d'un altre mes)
        if ts is not None and self.prev_ts is not None:
            if abs((ts - self.prev_ts).total_seconds() - 3600.0) > 1.0:
                self.hist = []
        self.prev_ts = ts
        self.hist.append((hour, dow, co2[0], preu[0]))
        if len(self.hist) > 72:
            self.hist = self.hist[-72:]

        # Final de la simulació: el vector de previsió s'escurça -> no hi ha "demà"
        self.max_len_seen = max(self.max_len_seen, L)
        ending = L < self.max_len_seen

        n_extra = 0 if ending else max(0, int(self.p["plan_hours"]) - L)
        kappa = self.p["kappa"]
        w = np.asarray(co2) + kappa * np.asarray(preu)
        if n_extra > 0:
            co2_x = preu_x = None
            if L == 3 and self.gbm is not None:
                try:
                    co2_x, preu_x = self._extend_gbm(co2, preu, hour, dow, month)
                except Exception:
                    co2_x = None
            if co2_x is None:
                co2_x = self._extend("co2", co2, hour, dow, month, n_extra)
                preu_x = self._extend("preu", preu, hour, dow, month, n_extra)
            w = np.concatenate([w, co2_x + kappa * preu_x])
        w = np.maximum(w, 0.0)

        nd = obs.get("noves_dades_5min") or {}
        last_load = float(nd.get("ultim_consum_kw", 1000.0))
        last_pv = float(nd.get("ultima_solar_kw", 0.0))
        net = self._net_forecast(last_load, last_pv, hour, dow, len(w))

        terminal = 0.0 if ending else self.p["terminal_factor"] * float(np.median(w))
        self.debug = {"w": w, "net": net, "L": L}

        if _HAS_SCIPY and self.elapsed < self.p["time_budget_s"]:
            try:
                return self._solve(soc, w, net, terminal)
            except Exception:
                pass
        p = self._fallback(w)
        return min(p, 0.0) if net[0] <= 0 else p
