# Guia de lliurament i de presentació

## Lliurament a la plataforma (https://repteupc.datathon.cat/)

1. Pestanya **Lliurament** → **Validació Ràpida** amb `submission.zip` (48 passos, no gasta quota).
2. Si passa, **tramesa oficial** de `submission.zip`. En el primer lliurament es fixen el nom de l'equip, els
   integrants i el PIN.
3. Segona tramesa recomanada: `submission/agent.py` sol. En mode estàndard dona el mateix Score (5,856 en validació)
   i no depèn de cap fitxer auxiliar ni de LightGBM; és l'assegurança per si el servidor tingués problemes amb els models.
4. Abans del tancament, marqueu aquestes dues com a **solucions oficials per al jurat**.
5. Si la plataforma té rànquing de **Mode Difícil**, la tramesa bona és `submission.zip` (porta els models de predicció).

El desempat és per menys infraccions i després per data de registre més antiga: com que el Score té sostre i altres
equips hi poden arribar, **convé enviar aviat**.

## Guió de la presentació (uns 9 minuts)

| # | Diapositiva | Missatge | Temps |
|---|---|---|---|
| 1 | Portada | Una bateria, una decisió cada hora; la resolem de manera òptima i demostrable | 0:20 |
| 2 | El repte | 2 MWh / ±500 kW / 90 %; Score 70/30; 24 h de preu i CO₂ conegudes | 0:40 |
| 3 | Les dades | Forma diària de consum, solar, preu i CO₂ | 0:45 |
| 4 | Primera troballa | Consum net > 500 kW el 99,5 % de les hores → no cal predir el consum | 0:45 |
| 5 | Un sol senyal | `w = CO₂ + 0,25·preu`, deduït de la fórmula del Score | 0:50 |
| 6 | L'agent | MPC: observa, optimitza (LP), actua, repeteix | 1:00 |
| 7 | Una setmana real | El patró surt sol de l'optimització | 0:45 |
| 8 | Resultats | 5,86 punts = 99,8 % de l'òptim teòric, 0 infraccions | 1:00 |
| 9 | Valor de la informació | Amb 12 h n'hi ha prou; Mode Difícil amb LightGBM: 5,47 | 0:50 |
| 10 | Sensibilitat | κ és el màxim de Score i la corba és plana | 0:45 |
| 11 | Enginyeria | 0 infraccions, pla B, 6 s de 60, casos límit | 0:40 |
| 12 | Impacte | 44 t CO₂ i 91.000 € l'any; límits: degradació i mida | 0:45 |
| 13 | Tancament | No es pot fer millor amb aquesta bateria, i ho hem demostrat | 0:20 |

Les notes de l'orador de cada diapositiva porten el text complet.

## Preguntes que us poden fer

- **Per què no aprenentatge per reforç o una xarxa neuronal?** El problema és lineal i el futur rellevant (preu i CO₂
  a 24 h) és conegut. Un programa lineal dona l'òptim exacte, en 3 ms i de manera explicable. L'RL com a molt
  s'hi acostaria.
- **Com sabeu que és òptim?** Hem calculat el màxim amb tot el futur conegut (5,865) i l'agent fa 5,856.
- **I si el test és diferent del train?** En mode estàndard l'agent no fa servir cap patró après, només κ, i el Score
  és pla al voltant de κ (±25 % costa ~0,01–0,02 punts).
- **Per què no preveieu el consum?** Perquè no canvia cap decisió: el consum net és > 500 kW el 99,5 % de les hores.
  L'única informació útil del consum és detectar caigudes de telemetria, i això sí que ho fem.
- **D'on surt la diferència amb l'òptim?** D'una caiguda de telemetria de 14 hores a la rèplica 15–21: la primera
  hora no es pot preveure.
- **Què passa al final de la simulació?** El vector de previsió s'escurça; l'agent ho detecta i buida la bateria.
- **I la degradació?** No entra al Score. Es pot afegir un cost per cicle al mateix programa lineal; seria el pas
  següent per a una decisió d'inversió real.
- **Què heu fet servir d'IA?** Digueu-ho amb naturalitat si us ho pregunten: quines eines i per a què.
