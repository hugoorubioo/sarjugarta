"""
agent.py — Repte UPC Sostenible (Datathon 2026) · MODE DIFÍCIL
===============================================================
Bateria de 6.000 kWh i ±1.500 kW, planta solar ×12 i només 3 hores de previsió
de mercat. Control predictiu (MPC) amb programació lineal ESTOCÀSTICA.

Què canvia respecte al mode estàndard
-------------------------------------
* Ara la bateria pot donar més potència (1.500 kW) que el consum net del campus
  (600–1.300 kW) i al migdia sobra solar. Descarregar de més o carregar quan hi
  ha excedent gratuït depèn del consum i de la solar, que NO es coneixen.
* El mercat només es coneix a 3 hores; la resta del dia s'ha de predir.

Peces
-----
1. Quatre models LightGBM (entrenats amb data/train) prediuen, hora a hora fins
   a +23 h: CO2 i preu de la xarxa, consum del campus i generació solar.
2. La incertesa del consum net es representa amb 7 escenaris per hora (quantils
   de l'error de predicció mesurats fora de mostra).
3. Un programa lineal minimitza el valor ESPERAT de  Σ xarxa_t · w_t  amb
   w_t = co2_t + KAPPA · preu_t  (el mateix senyal únic del mode estàndard),
   on la importació de cada hora es calcula escenari a escenari:
       importació = max(0, consum_net + càrrega − descàrrega)
   Així el pla ja descompta que descarregar de més es perd (excedent a 0 €)
   i que carregar amb excedent solar és gratuït.
4. S'aplica només la primera consigna i es replanifica a l'hora següent.
5. Clamp final de seguretat (0 infraccions) i política de reserva sense models.

Biblioteques: numpy, scipy, lightgbm, json, pathlib, datetime (totes permeses).
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, diags, hstack, identity, vstack
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

ETA = 0.9 ** 0.5
EPS = 1e-4
NAN = float("nan")

DEFAULT_PARAMS = {
    "kappa": 0.2617,
    "plan_hours": 24,
    "terminal_factor": 0.0,
    "time_budget_s": 40.0,
    "net_quantiles": None,      # [antelació][nivell solar][7 quantils] de l'error del consum net (kW)
    "pv_bins": [20.0, 400.0],
    "pv_env": None,             # envolupant de cel clar de la solar [mes][hora] (fila 0 = anual)
    "scen_scale_now": 1.3,      # amplitud dels escenaris de l'hora actual
    "scen_scale_later": 1.3,    # amplitud dels escenaris de les hores següents
    "net_shift": 0.0,           # desplaçament prudent del consum net previst (kW)
}


def _parse_ts(ts):
    try:
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        if isinstance(ts, datetime):
            return ts
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def _f(x, default=NAN):
    try:
        v = float(x)
        return v if v == v else default
    except Exception:
        return default


class BatteryAgent:
    def __init__(self, params=None):
        p = dict(DEFAULT_PARAMS)
        here = Path(__file__).parent
        try:
            with open(here / "params.json", encoding="utf-8") as f:
                p.update(json.load(f))
        except Exception:
            pass
        if params:
            p.update(params)
        self.gbm = p.pop("models", None)
        if self.gbm is None:
            try:
                names = ("co2", "preu", "load", "pv")
                if all((here / f"model_{k}.txt").exists() for k in names):
                    import lightgbm as lgb
                    self.gbm = {k: lgb.Booster(model_file=str(here / f"model_{k}.txt")) for k in names}
            except Exception:
                self.gbm = None
        self.p = p

        # Memòria del bloc actual (setmana contínua). Índex = hora dins del bloc.
        self.co2, self.preu = [], []            # conegudes fins a l'hora actual
        self.load, self.pv, self.irr, self.temp = [], [], [], []   # conegudes fins a l'hora anterior
        self.prev_ts = None
        self.prev_hour = None
        self.max_len_seen = 0
        self.elapsed = 0.0
        self._lp_cache = {}
        self.debug = {}

    # ------------------------------------------------------------------
    # Memòria
    # ------------------------------------------------------------------
    def _update_memory(self, obs, hour, co2_now, preu_now):
        ts = _parse_ts(obs.get("timestamp"))
        jump = False
        if self.prev_hour is None:
            jump = True                                   # primer pas de la simulació
        elif ts is not None and self.prev_ts is not None:
            jump = abs((ts - self.prev_ts).total_seconds() - 3600.0) > 1.0
        elif hour != (self.prev_hour + 1) % 24:
            jump = True
        self.prev_ts, self.prev_hour = ts, hour
        if jump:
            # Nova setmana de test: la telemetria "de l'última hora" és de fa setmanes -> no s'aprofita
            self.co2, self.preu, self.load, self.pv, self.irr, self.temp = [], [], [], [], [], []
        else:
            nd = obs.get("noves_dades_5min") or {}
            self.load.append(_f(nd.get("ultim_consum_kw")))
            self.pv.append(_f(nd.get("ultima_solar_kw")))
            self.irr.append(_f(nd.get("irradiancia_wm2")))
            self.temp.append(_f(nd.get("temp_ambient_c")))
        self.co2.append(co2_now)
        self.preu.append(preu_now)
        return ts

    @staticmethod
    def _back(series, dist, now_len):
        """Valor de fa `dist` hores (l'hora actual és l'índex now_len-1)."""
        i = now_len - 1 - dist
        return series[i] if 0 <= i < len(series) else NAN

    # ------------------------------------------------------------------
    # Prediccions (mateixes variables que src/hard_train.py)
    # ------------------------------------------------------------------
    def _forecast(self, co2, preu, hour, dow, ts):
        n_now = len(self.co2)                 # hores viscudes en aquest bloc, inclosa l'actual
        ell = n_now - 1                       # hores d'historial disponibles
        leads = np.arange(24)
        tgt = hour + leads
        h_obj = tgt % 24
        dow_obj = (dow + tgt // 24) % 7
        we_obj = (dow_obj >= 5).astype(float)
        mes = float(ts.month) if ts is not None else NAN
        doy = float(ts.timetuple().tm_yday) if ts is not None else NAN

        def ult(s):
            return self._back(s, 1, n_now) if ell >= 1 else NAN

        def m24(s, fn):
            if ell < 24:
                return NAN
            seg = s[n_now - 25:n_now - 1]
            return float(fn(seg)) if len(seg) == 24 and all(x == x for x in seg) else NAN

        def ahir(s):
            return np.array([self._back(s, 24 - h, n_now) if ell >= 24 - h else NAN for h in leads])

        def ahir2(s):
            return np.array([self._back(s, 48 - h, n_now) if ell >= 48 - h else NAN for h in leads])

        def lag(s, d):
            return self._back(s, d, n_now) if ell >= d else NAN

        c = list(co2) + [co2[-1]] * 3
        p = list(preu) + [preu[-1]] * 3
        ones = np.ones(24)
        c_d = (c[0] + c[1] + c[2]) / 3 - np.mean([lag(self.co2, d) for d in (22, 23, 24)]) if ell >= 24 else NAN
        p_d = (p[0] + p[1] + p[2]) / 3 - np.mean([lag(self.preu, d) for d in (22, 23, 24)]) if ell >= 24 else NAN
        env = self.p.get("pv_env")
        mrow = ts.month if ts is not None else 0
        if env:
            env_row = np.asarray(env[mrow], dtype=float)
            env_obj = env_row[h_obj]
            env_prev = env_row[(hour - 1) % 24]
        else:
            env_obj, env_prev = np.full(24, NAN), NAN
        v_ult, v_ahir = ult(self.pv), ahir(self.pv)
        kt_ult = v_ult / max(env_prev, 1.0) if env_prev == env_prev and env_prev > 50 else NAN
        kt_ahir = np.where(env_obj > 50, v_ahir / np.maximum(env_obj, 1.0), NAN)
        XC = np.column_stack([leads, h_obj, we_obj, mes * ones, c[0] * ones, c[1] * ones, c[2] * ones,
                              p[0] * ones, p[1] * ones, p[2] * ones, ahir(self.co2), ahir(self.preu),
                              m24(self.co2, np.mean) * ones, m24(self.preu, np.mean) * ones,
                              ult(self.co2) * ones, ult(self.preu) * ones,
                              c_d * ones, p_d * ones, m24(self.co2, np.min) * ones, m24(self.co2, np.max) * ones,
                              m24(self.preu, np.min) * ones, m24(self.preu, np.max) * ones,
                              ahir2(self.co2), ahir2(self.preu), ult(self.irr) * ones, v_ult * ones])
        XL = np.column_stack([leads, h_obj, dow_obj, we_obj, mes * ones, ult(self.load) * ones, ahir(self.load),
                              lag(self.load, 25) * ones, m24(self.load, np.mean) * ones, ult(self.temp) * ones,
                              lag(self.load, 2) * ones, lag(self.load, 3) * ones, ahir2(self.load),
                              (lag(self.load, 1) - lag(self.load, 25)) * ones, ult(self.irr) * ones])
        XV = np.column_stack([leads, h_obj, mes * ones, doy * ones, v_ult * ones, ult(self.irr) * ones,
                              v_ahir, m24(self.pv, np.max) * ones, ult(self.temp) * ones,
                              env_obj, kt_ult * ones, lag(self.pv, 2) * ones, ahir2(self.pv), kt_ahir])
        self._X = (XC, XL, XV)
        g = self.gbm
        co2_hat = g["co2"].predict(XC, num_threads=1)
        preu_hat = g["preu"].predict(XC, num_threads=1)
        load_hat = g["load"].predict(XL, num_threads=1)
        pv_hat = np.maximum(g["pv"].predict(XV, num_threads=1), 0.0)
        return co2_hat, preu_hat, load_hat, pv_hat

    def _forecast_simple(self, co2, preu, n):
        """Pla B sense models: persistència."""
        n_now = len(self.co2)
        last_load = self._back(self.load, 1, n_now)
        last_pv = self._back(self.pv, 1, n_now)
        load = np.full(n, last_load if last_load == last_load else 1000.0)
        pv = np.full(n, last_pv if last_pv == last_pv else 0.0)
        return np.full(n, float(np.mean(co2))), np.full(n, float(np.mean(preu))), load, pv

    # ------------------------------------------------------------------
    # Programa lineal estocàstic
    # ------------------------------------------------------------------
    def _matrices(self, n, s):
        key = (n, s)
        if key not in self._lp_cache:
            eye = identity(n, format="csr")
            dif = eye - diags([np.ones(n - 1)], [-1], format="csr") if n > 1 else eye
            # variables: c (n), d (n), soc (n), g (n·s: importació per hora i escenari)
            a_eq = hstack([-ETA * eye, eye / ETA, dif, csr_matrix((n, n * s))]).tocsr()
            rep = vstack([eye] * s).tocsr()                       # files ordenades per escenari
            a_ub = hstack([rep, -rep, csr_matrix((n * s, n)), -identity(n * s, format="csr")]).tocsr()
            self._lp_cache[key] = (a_eq, a_ub)
        return self._lp_cache[key]

    def _solve(self, soc, w, net_scen, cap, pmax, terminal):
        n, s = net_scen.shape                                     # hores × escenaris
        a_eq, a_ub = self._matrices(n, s)
        tie = 1e-6 + 1e-7 * np.arange(n)
        soc_cost = np.zeros(n); soc_cost[-1] = -terminal
        cost = np.concatenate([tie, tie, soc_cost, np.tile(w / s, s)])
        b_eq = np.concatenate([[soc], np.zeros(n - 1)])
        b_ub = -net_scen.T.reshape(-1)                            # g_{k,s} >= net_{k,s} + c_k - d_k
        bounds = [(0.0, pmax)] * (2 * n) + [(0.0, cap)] * n + [(0.0, None)] * (n * s)
        res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError("LP no resolt")
        return float(res.x[0] - res.x[n])

    # ------------------------------------------------------------------
    # Decisió horària
    # ------------------------------------------------------------------
    def act(self, obs: dict) -> float:
        t0 = datetime.now()
        cap = _f(obs.get("cap_max_kwh"), 6000.0)
        pmax = _f(obs.get("power_max_kw"), 1500.0)
        soc = min(max(_f(obs.get("soc_kwh"), cap / 2), 0.0), cap)
        try:
            consigna = self._decide(obs, soc, cap, pmax)
        except Exception:
            consigna = 0.0
        # --- Clamp de seguretat: garanteix 0 infraccions ---
        max_c = max(0.0, min(pmax, (cap - soc) / ETA - EPS))
        max_d = max(0.0, min(pmax, soc * ETA - EPS))
        consigna = float(min(max(consigna, -max_d), max_c))
        if not np.isfinite(consigna) or abs(consigna) < 1e-3:
            consigna = 0.0
        self.elapsed += (datetime.now() - t0).total_seconds()
        return consigna

    def _decide(self, obs, soc, cap, pmax):
        hour = int(obs.get("hour", 0))
        dow = int(obs.get("day_of_week", 0))
        co2 = [float(x) for x in (obs.get("co2_properes_24h") or [])]
        preu = [float(x) for x in (obs.get("preus_properes_24h") or [])]
        ca, pa = _f(obs.get("co2_actual")), _f(obs.get("preu_actual"))
        if not co2 or not preu:
            co2, preu = [ca], [pa]
        elif ca == ca and pa == pa and (abs(co2[0] - ca) > 1e-9 or abs(preu[0] - pa) > 1e-9):
            co2, preu = [ca] + co2, [pa] + preu
        L = min(len(co2), len(preu), 24)      # el pla és de 24 h com a màxim
        co2, preu = co2[:L], preu[:L]

        ts = self._update_memory(obs, hour, co2[0], preu[0])
        self.max_len_seen = max(self.max_len_seen, L)
        ending = L < self.max_len_seen                    # final de la simulació: no hi ha "demà"
        n = L if ending else max(L, min(24, int(self.p["plan_hours"])))

        if self.gbm is not None:
            co2_hat, preu_hat, load_hat, pv_hat = self._forecast(co2, preu, hour, dow, ts)
        else:
            co2_hat, preu_hat, load_hat, pv_hat = self._forecast_simple(co2, preu, 24)
        co2_plan = np.array(co2_hat[:n], dtype=float); preu_plan = np.array(preu_hat[:n], dtype=float)
        k = min(L, n)
        co2_plan[:k] = co2[:k]; preu_plan[:k] = preu[:k]   # el que es coneix, es fa servir tal qual
        w = np.maximum(co2_plan + self.p["kappa"] * preu_plan, 0.0)

        net_hat = (load_hat - pv_hat)[:n] + self.p["net_shift"]
        q = self.p.get("net_quantiles")
        if q:
            bins = np.digitize(pv_hat[:n], self.p["pv_bins"])
            net_scen = np.array([net_hat[i] + np.asarray(q[min(i, len(q) - 1)][int(bins[i])], dtype=float) for i in range(n)])
            # Les hores futures es tornaran a decidir amb més informació: la incertesa que cal
            # cobrir avui per a elles és menor que l'error de predicció brut.
            sc = np.full(n, float(self.p["scen_scale_later"])); sc[0] = float(self.p["scen_scale_now"])
            net_scen = net_hat[:, None] + (net_scen - net_hat[:, None]) * sc[:, None]
        else:
            net_scen = net_hat[:, None] + np.array([-250.0, -100.0, 0.0, 100.0, 250.0])[None, :]

        terminal = 0.0 if ending else self.p["terminal_factor"] * float(np.median(w))
        self.debug = {"w": w, "net_hat": net_hat, "load_hat": load_hat, "pv_hat": pv_hat, "co2_hat": co2_hat, "preu_hat": preu_hat}

        if _HAS_SCIPY and self.elapsed < self.p["time_budget_s"]:
            try:
                return self._solve(soc, w, net_scen, cap, pmax, terminal)
            except Exception:
                pass
        # Política de reserva: carrega amb excedent o a les hores bones, descarrega fins al consum net previst
        nn = float(net_hat[0])
        if nn < 0:
            return min(pmax, -nn)
        if len(w) >= 4 and w[0] >= np.quantile(w, 0.7):
            return -min(pmax, max(nn - 150.0, 0.0))
        if len(w) >= 4 and w[0] <= np.quantile(w, 0.2):
            return pmax
        return 0.0
