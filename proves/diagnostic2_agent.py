"""
Agent de DIAGNÒSTIC 2 (només per a la Validació Ràpida, no per al rànquing).
Comprova si les previsions a moltes hores vista canvien d'una hora a la següent
(és a dir, si el servidor les revisa) comparant els vectors successius.
Codis: 1.0 = tot coincideix; 2 + error x1000 = hi ha revisions.
"""


class BatteryAgent:
    def __init__(self):
        self.t = 0
        self.p = []          # vectors de preu rebuts a cada pas
        self.c = []          # vectors de CO2 rebuts a cada pas

    @staticmethod
    def _revisio(vectors, pas_antic, pas_nou):
        """Màxima diferència entre el que es va anunciar a `pas_antic` i el que s'anuncia a `pas_nou`."""
        antic, nou = vectors[pas_antic], vectors[pas_nou]
        salt = pas_nou - pas_antic
        pitjor, on = 0.0, 0
        for k in range(len(nou)):
            if k + salt < len(antic):
                d = abs(float(antic[k + salt]) - float(nou[k]))
                if d > pitjor:
                    pitjor, on = d, k + salt
        return pitjor, on

    def act(self, obs):
        t = self.t
        self.t += 1
        self.p.append(list(obs.get("preus_properes_24h") or []))
        self.c.append(list(obs.get("co2_properes_24h") or []))
        out = 0.0
        if t == 0:
            out = float(len(self.p[0]))
        elif t == 1:      # preu: vector d'ara contra el de fa 1 hora (totes les antelacions)
            d, _ = self._revisio(self.p, 0, 1)
            out = 1.0 if d < 1e-12 else 2.0 + min(98.0, 1000.0 * d)
        elif t == 2:      # CO2: vector d'ara contra el de fa 1 hora
            d, _ = self._revisio(self.c, 1, 2)
            out = 1.0 if d < 1e-12 else 2.0 + min(98.0, 1000.0 * d)
        elif t == 3:      # preu: vector d'ara contra el de fa 3 hores
            d, _ = self._revisio(self.p, 0, 3)
            out = 1.0 if d < 1e-12 else 2.0 + min(98.0, 1000.0 * d)
        elif t == 4:      # CO2: vector d'ara contra el de fa 4 hores
            d, _ = self._revisio(self.c, 0, 4)
            out = 1.0 if d < 1e-12 else 2.0 + min(98.0, 1000.0 * d)
        elif t == 5:      # antelació on hi ha la revisió més gran: preu + 0,01 x CO2 (0 = cap)
            dp, kp = self._revisio(self.p, 0, 5)
            dc, kc = self._revisio(self.c, 0, 5)
            out = (kp if dp > 1e-12 else 0) + 0.01 * (kc if dc > 1e-12 else 0) + 0.5
        return float(out)
