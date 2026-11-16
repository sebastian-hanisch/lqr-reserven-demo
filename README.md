# Reserven: Wie viel Modellfehler hält ein LQR aus? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lqr-reserven-demo.streamlit.app/)**

Diese Demo gehört zum Themenfeld **Regelung (von LQR bis MPC)** auf der Website „Sebastian Hanisch – Operations Research und Machine Learning“ und baut auf
[`lqr-riccati-demo`](https://github.com/sebastian-hanisch/lqr-riccati-demo) auf: Dort liefert die Riccati-Gleichung den Gewinn K und Pole im Einheitskreis (R = 1: 0,378 ± 0,188 i, Spektralradius 0,422); **wie weit der Kreis
von der Instabilität entfernt ist, bleibt offen**. Diese Demo misst es, am selben Wagen (Doppelintegrator) und mit derselben Konvention (u = −K·x, Euler-Form, Q = I). Es trennt drei Fälle: den **stetigen** Entwurf, **denselben Entwurf abgetastet** und den
**Entwurf mit Schätzer** (LQG, Doyles Gegenbeispiel). Verwandte Demos: [`horizont-falle-demo`](https://github.com/sebastian-hanisch/horizont-falle-demo) (Gesetz mit endlichem Horizont, dessen Reserve hier nicht gemessen ist),
[`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo) und [`windup-pi-demo`](https://github.com/sebastian-hanisch/windup-pi-demo) (mit aktiver Stellgrenze ist die Schleife nichtlinear),
[`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo) (Verzug und Rechenzeit; hier nur die Verzugsreserve als Zahl) und [`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo) (Störungsmodell mit Beobachter). Das Filter des LQG ist das der Demo
[`kalman-filter-demo`](https://github.com/sebastian-hanisch/kalman-filter-demo) (erst vorhersagen, dann korrigieren); der Entwurf des Filters wird hier nicht gelehrt, nur seine Wirkung auf die Reserve gemessen. Aus den Demos zur Rückkopplung
([`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo), [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo)) kommt die Begriffsbrücke: Der Spektralradius ρ dort entspricht hier den Polen der Schleife;
neu ist der **Abstand der Ortskurve zu −1**. Das Lager ist hier nicht das Vehikel.

## Warum dieses Problem

Ruhige Pole sagen nichts darüber, wie viel Modellfehler ein Regelkreis aushält. **Der LQR am Wagen mit R = 1 hat den Spektralradius 0,422 und ist trotzdem instabil, wenn die Strecke 1,94-mal so stark wirkt wie modelliert.** Gemessen werden drei Reserven, immer mit Namen:
der **Nyquist-Abstand** min |1 + L(e^{jω})| (Schleife am Streckeneingang geöffnet, L(z) = K·(zI − A)⁻¹·B), das **Verstärkungsfenster** (Bereich des Faktors g in u = −g·K·x, in dem die Schleife stabil bleibt) und die **Phasenreserve** (180° + arg L bei |L| = 1, kleinste Durchtrittsstelle).
Alles ist **eigene Nachrechnung in diesem Modell** (Wagen, Ts = 1, Euler-Form B = (0, Ts)ᵀ, Q = I, ein Eingang). Für den abgetasteten LQR gibt keine gelesene Quelle eine Reservegarantie an; für den stetigen LQR steht sie bei Doyle (1978, als Scan-Bild gelesen: 60° Phasenreserve, Verstärkung nach unten bis auf die Hälfte (−6 dB) und nach oben unbegrenzt,
Zuschreibung an Safonov/Athans; im Original nicht gelesen). Eine Neuheitsaussage gibt es nicht: Die Rückführdifferenz-Gleichung und die Schranke in Befund B8 sind eigene Herleitung; ob sie in Lehrbüchern stehen, ist ungelesen.

## Modell

**Konvention (gilt für jede Zahl):** x⁺ = A·x + B·u, Regler u = −K·x, Kosten Σ xᵀQx + uᵀRu mit Q = I (kein Faktor ½), ein Eingang. Der Wagen hat A = [[1, Ts], [0, 1]]; **Euler-Form** B = (0, Ts)ᵀ (wie in den anderen Regelungs-Demos), **Halte-Form** B = (Ts²/2, Ts)ᵀ. Q und R
gemeinsam mit Ts zu skalieren gibt denselben Gewinn; die Kosten gelten je Periode. Die Voreinstellung ist Ts = 1, Euler-Form, R = 1.
- **Nyquist-Abstand** min |1 + L|: Gitter über ω in (0, π] mit Verfeinerung und dem Randpunkt ω = π einzeln; **Empfindlichkeitsspitze** 1/min |1 + L|.
- **Verstärkungsfenster** (g_unten, g_oben): am Wagen exakt nach Jury (vier lineare Bedingungen in g), zweiter Weg Bisektion auf dem Spektralradius, dritter Weg der Nyquist-Schnitt −1/L(−1). „Stabil“ heißt Spektralradius < 1 − 10⁻¹²; ein Fensterrand ist nie „genau“ getroffen (dort ist ρ = 1 ± Rundungsrauschen).
- **Verzugsreserve** = Phasenreserve / Durchtrittsfrequenz, in Perioden; nur eine Zahl, der Verzug selbst ist Thema von [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo). Deren kritische Rechenzeit gehört zu einem anderen Modell (B = [½, 1]ᵀ) und ist mit dieser Zahl nicht verglichen.
- **Streckenverstärkung m** (Modellfehler): die Strecke wirkt m-mal so stark wie modelliert, der Regler weiß es nicht (u wird zu m·u).
- **LQG am Wagen:** gemessen wird die Position, Messrauschen 1, Prozessrauschen W·G·Gᵀ; **Schätzform** (u_k mit der Messung k) und **Vorhersage-Form** (u_k mit Messungen bis k − 1); der Gewinn K ist der des Zustands-LQR.
- **Doyles Gegenbeispiel** (stetig): ẋ = [[1, 1], [0, 1]]·x + (0, 1)ᵀ·u + (1, 1)ᵀ·w, y = x₁ + v, Güte Q = q·(1, 1)ᵀ(1, 1), R = 1, Rauschintensitäten σ und 1, hier q = σ. Die Parameter stammen aus Doyles Aufsatz (Scan, als Bild gelesen) und sind hier nachgerechnet.
- Die Formeln stehen im Expander 📐 der App. Die Zufallssysteme der Schranke (Experiment 6) kommen aus einem ganzzahligen SplitMix64 mit exakten Logarithmus- und Kosinus-Routinen (`res_rng.py`); der Zufalls-Seed ist ein Regler.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Entwurf** (`res_dynamics.dlqr`) | DARE mit `scipy.linalg.solve_discrete_are`; zweiter Weg (nur in den Tests) die eigene Riccati-Rekursion aus P = Q | die Rekursion braucht Symmetrisierung in jedem Schritt (siehe „Hinweise zu den Zahlen“) |
| **Nyquist-Abstand** (`res_margins.min_return`) | Frequenzgang auf 20 001 Punkten, Verfeinerung um den Tiefstwert (Brent), Randpunkt π einzeln | Gitterverfahren; Lage des Tiefstwerts nicht vorausgesetzt (bei R = 10⁶ liegt er nicht genau bei π) |
| **Fenster** (`jury_window`) | Jury für Systeme zweiter Ordnung in Formeln; Skalar in Formeln; LQG und Doyle über Eigenwerte (Gitter 0,001 bis 8, dann Bisektion; Doyle: Hurwitz vierten Grades in Formeln) | LQG-Fenster nur um m = 1 (zusammenhängend vorausgesetzt, nicht für jeden Entwurf geprüft) |
| **Phasenreserve** (`phase_margins`) | Vorzeichenwechsel von \|L\| − 1 auf dem Gitter, Brent-Verfeinerung; kleinste über alle Durchtrittsstellen | klassische Reserve, keine Scheibenreserve |
| **Zeitbereich** (`D.settle`) | freie Bahn aus (1, 0), Schritte bis ‖x‖ dauerhaft unter 1 % der Spitze, beobachtet bis 20 000 Perioden | nahe der Fensterkante „klingt kaum ab“; bei ρ ≥ 1 ohne Zahl |
| **Zufallssysteme** (`bound_check`) | n = 2 bis 4, A, B, M normalverteilt (SplitMix64), Q = MMᵀ + 0,05·I, R von 0,01 bis 100, nur steuerbar (Rang und Kondition ≤ 10⁷) | Zählungen, nie einzelne Zahlen; DARE-Residuum < 10⁻¹⁰ filtert schlecht gelöste Fälle |

**Vorgerechnet** (`data/res_frozen.json`, erzeugt von `tools/make_frozen.py`, rund eine halbe Minute Laufzeit; in der App als „vorgerechnet“ gekennzeichnet): die 90 LQG-Entwürfe und die Zählungen an je 3 000 Zufallssystemen (Zufalls-Seed 15). Die App
rechnet beim ersten Aufruf rund 3 Sekunden (mit dem Import), danach je Regler 0,1 bis 0,5 Sekunden; die Zufallssysteme (bis 300, Standard 200) brauchen bis rund 2 Sekunden (Messung mit AppTest auf dem Entwicklungsrechner, nicht festgeschrieben und ohne das Zeichnen im Browser).

## Befunde (gemessen, keine Behauptungen)

Zähleinheit: Periode (Zeit) bzw. Radiant je Periode (Frequenz). Wagen = Ts = 1, Euler-Form, Q = I, wenn nicht anders gesagt. Jede Zahl steht in `tests/test_claims.py` (Tabelle unten); „vorgerechnet“ heißt: aus `data/res_frozen.json`, im Test neu gerechnet.

**B1: Nyquist-Abstand, stetig gegen abgetastet.** Am Wagen liegt der kleinste Abstand zu −1 bei **0,4836 (R = 1), 0,3720 (R = 0,1), 0,3485 (R = 0,01) und 0,3458 (R = 0,001)**, in 4 von 4 R bei ω = π; die Empfindlichkeitsspitze ist 2,068 / 2,688 / 2,870 / 2,892.
Der stetige Entwurf mit demselben R hat auf dem Gitter ω = 10⁻³ bis 10³ (10 001 Punkte) den kleinsten Wert 1,0000 (1,000001 bei R = 1, 1,000005 bei R = 0,1, 1,00005 bei R = 0,01), nie unter 1 − 10⁻⁹. Die stetige Garantie (Abstand ≥ 1) wird abgetastet in 3 von 3 gemessenen R (1 / 0,1 / 0,01) verfehlt.
**B2: Verstärkungsfenster.** Obere Kante 1,9363 / 1,5923 / 1,5349 / 1,5286 (5,74 / 4,04 / 3,72 / 3,69 dB) für R = 1 / 0,1 / 0,01 / 0,001, nach Jury, Bisektion und Nyquist-Schnitt in 4 von 4 R gleich (Abweichung unter 10⁻⁹, gemessen 7,7·10⁻¹³). Nach unten ist die Schleife für jedes g > 0 stabil
(die Bisektion endet an der Genauigkeit des Tests, etwa 10⁻¹¹): die Verstärkung darf beliebig sinken, aber nur um den Faktor 1,5 bis 1,9 steigen. Der Tiefstwert ist am Wagen gleich 1 − 1/g_oben (Ts = 1, 4 von 4 R; Abweichung unter 10⁻⁹, gemessen 0), und das auch bei Ts = 0,5 / 0,1 / 0,01 in beiden Formen von B (R = 1; relativ unter 10⁻⁶), jeweils mit dem Tiefstwert bei ω ≈ π.
**B3: Phasen- und Verzugsreserve.** Phasenreserve 39,6° / 34,4° / 33,2° / 33,1° (R = 1 / 0,1 / 0,01 / 0,001), Verzugsreserve 0,604 / 0,423 / 0,392 / 0,389 Perioden (unter einer Periode in 4 von 4 R); stetig 72,4° / 79,2° / 85,3° (R = 1 / 0,1 / 0,01).
**B4: Monotonie in R.** Am Wagen (Ts = 1, Q = I) fällt der Abstand mit kleiner werdendem R in 10 von 10 gemessenen R (0,001 bis 100) in beiden Formen von B (R = 100: 0,7958, R = 10: 0,6596). Schneller heißt hier nicht robuster: R = 0,1 hat die kleineren Pole (0,093; 0,362) als R = 1 und hält nur das 1,59-Fache aus.
Bei R = 10⁻⁴ und 10⁻⁶ ändert sich der Abstand kaum noch (0,3455), ein Grenzwert ist damit nicht bewiesen; bei R = 10⁶ ist K = (0,0010; 0,0447) und der Abstand 0,978: nahe 1, nicht gleich 1 (der Tiefstwert liegt dort nicht genau bei π).
**B5: Zeitbereich** (Anfangszustand (1, 0), R = 1, Streckenverstärkung m). m = 1: ρ = 0,422, 6 Perioden bis 1 %; m = 1,9: ρ = 0,953, 79 Perioden; m = 1,93: ρ = 0,9918, 462; m = 1,936: ρ = 0,99959, 9 161; m = 1,94: ρ = 1,0048, instabil; m = 0,05: ρ = 0,979, 220 Perioden.
Bei R = 0,1 ist m = 1,8 schon instabil (ρ = 1,336; nominal 6 Perioden): R = 0,1 hat die kleineren Pole und das kleinere Fenster. „Stabil“ nahe der Kante heißt „klingt kaum ab“. Die Spitze 1/|1 + L| = 2,068 (R = 1, ω nahe π) stimmt mit der stationären Amplitude einer simulierten Sinus-Störung am Streckeneingang auf 10⁻⁹ überein (Test; 8 Fälle mit R = 1 und 0,1).
**B6: der stetige Gewinn, abgetastet gehalten** (exakte Halte-Diskretisierung). K_c = (1; √3) bei R = 1 ist genau für Ts < 2/k₂ stabil: **Ts_max = 1,1547 (R = 1), 0,4950 (R = 0,1), 0,1826 (R = 0,01)**, ein einziger Wechsel stabil/instabil auf 50 000 Gitterpunkten (Ts 10⁻⁴ bis 5). Spektralradius bei Ts = 1: 0,6115 / 3,9875 / 14,30 (R = 1 / 0,1 / 0,01). Der stetige Entwurf braucht bei starkem Eingriff also eine kleine Abtastzeit.
**B7: Abtastzeit-Reihe** (R = 1; Euler | Halten): Abstand 0,4836 | 0,4858 (Ts = 1), 0,6698 | 0,6714 (0,5), 0,8117 | 0,8120 (0,25), 0,9182 | 0,9182 (0,1), 0,9579 | 0,9579 (0,05), 0,9914 | 0,9914 (0,01); obere Fensterkante Euler 1,936 / 3,029 / 5,311 / 12,22 / 23,77 / 116,1. Die beiden Formen von B unterscheiden sich bei Ts = 1 um 0,002; der Befund hängt nicht an der Form von B.
Der Abstand 1 − min |1 + L| ist für Ts ≤ 0,1 das 0,82- bis 0,86-Fache von Ts (nur diese Reihe, kein Satz). Der Abstand steigt in der gemessenen Reihe mit kleiner werdender Ts (in allen 5 Schritten der Reihe, beide Formen).
**B8: die Schranke** (eigene Herleitung, in diesem Modell geprüft). Aus der DARE folgt für einen Eingang (R + BᵀPB)·|1 + L|² = R + ‖Q^{1/2}(zI − A)⁻¹B‖², daraus **min |1 + L| ≥ r₀ = √(R/(R + BᵀPB))**, und det(A − BK) = det(A)·R/(R + BᵀPB). An 3 000 Zufallssystemen (Zufalls-Seed 15, SplitMix64, n = 2 bis 4, vorgerechnet):
3 000 steuerbar und lösbar, 2 990 mit DARE-Residuum < 10⁻¹⁰; dort Gleichung bis 3,3·10⁻¹⁰, Determinantenbeziehung bis 3,3·10⁻¹²; 0 Verletzungen der Schranke und 0 Verletzungen des daraus folgenden Intervalls (1/(1 + r₀), 1/(1 − r₀)) (je System drei Stichproben); der kleinste Quotient gemessener Abstand / Schranke ist 1,00005
(auf dem Gitter, also eine obere Näherung): die Schranke ist bei einzelnen Systemen fast scharf. Am Wagen ist sie für kleines R fast leer: r₀ = 0,4221 / 0,1832 / 0,0614 / 0,0195 gegen gemessen 0,4836 / 0,3720 / 0,3485 / 0,3458 (R = 1 / 0,1 / 0,01 / 0,001); garantiertes Intervall (0,703; 1,730) bei R = 1 und (0,942; 1,065) bei R = 0,01 gegen
tatsächlich (0; 1,936) und (0; 1,535). Am Wagen ist r₀² gleich dem Produkt der Polbeträge (0,17815 bei R = 1). Stetig: an 3 000 Zufallssystemen (gleicher Zufalls-Seed, vorgerechnet) 0 Verletzungen von min |1 + L| ≥ 1 − 10⁻⁹ (kleinster Wert 1,0000000000).
**B9: Doyles Gegenbeispiel, nachgerechnet.** Die Formeln aus Doyles Aufsatz stimmen: Reglergewinn g = f·(1, 1), f = 2 + √(4 + q); Filtergewinn k = d·(1, 1), d = 2 + √(4 + σ); die Koeffizienten c₁ = d + f − 4 + 2(m − 1)·d·f und c₀ = 1 + (1 − m)·d·f des charakteristischen Polynoms (Abweichung unter 10⁻⁹).
Fenster der Streckenverstärkung für q = σ = 1 / 5 / 12 / 100 / 1 596 / 10⁴ / 10⁶: (0,9208; 1,0557), (0,9150; 1,0400), (0,9144; 1,0278), **(0,9381; 1,0067)**, (0,9779; 1,00057), (0,99049; 1,000096), **(0,999005; 1,000001)**; die obere Kante ist exakt 1 + 1/(d·f), die untere kommt aus der Hurwitz-Bedingung vierten Grades (Abweichung gegen die Eigenwerte unter 10⁻⁶ der Fensterbreite).
Der reine LQ-Regler ist für jedes m > 2/f stabil (Spur 2 − m·f, Determinante 1; 0,472 bei q = 1, 0,164 bei q = 100, 0,002 bei q = 10⁶) und nach oben ohne Grenze. |1 + L(0)| = 1/(1 + d·f) (0,0385 bei q = σ = 5, 0,00668 bei 100, 9,96·10⁻⁷ bei 10⁶), die Empfindlichkeitsspitze also 26,0 / 149,8 / 1,0·10⁶ gegen 1 beim LQ-Regler.
**B10: LQG am abgetasteten Wagen** (Positionsmessung, Messrauschen 1, 90 Entwürfe, vorgerechnet: 3 Regler × 2 Formen × 3 Rauschrichtungen G × 5 Rauschstärken W). Nyquist-Abstand 0,1327 bis 0,5950, **unter dem des Zustands-LQR in 63 von 90 Entwürfen** (28 von 30, 21 von 30, 14 von 30 je Regler); die obere Fensterkante liegt in 32 von 90 sogar über der des Zustands-LQR;
kleinste Phasenreserve 9,0°. LQG kann also besser oder schlechter sein; **eine Garantie gibt es nicht**. Trennungsprinzip (Pole der Schleife = Pole von A − BK und des Filters): größte Abweichung 2,1·10⁻⁷ in 90 von 90; zweiter Rechenweg für den Abstand (Determinantenquotient) bis 6,7·10⁻⁶.

| Regler (Zustands-LQR: Abstand; Fenster) | Form, G = B | W | Fenster der Streckenverstärkung | Abstand | kleinste Phasenreserve |
|---|---|---|---|---|---|
| Q = I, R = 1 (0,4836; 0 bis 1,9363) | Schätzform | 0,01 | (0; 2,689) | 0,5283 | 36,2° |
| | Schätzform | 10⁴ | (0,335; 1,531) | 0,3433 | 29,0° |
| | Vorhersage | 10⁴ | (0,536; 1,383) | 0,2768 | 24,4° |
| Q = I, R = 0,1 (0,3720; 0 bis 1,5923) | Vorhersage | 10⁴ | (0,694; 1,308) | 0,2357 | 17,7° |
| Q = 100·(1, 1)ᵀ(1, 1), R = 1 (0,2617; 0 bis 1,3545) | Schätzform | 10⁸ | (0,786; 1,257) | 0,1987 | 12,7° |
| | Vorhersage | 10⁸ | (0,834; 1,153) | 0,1327 | 9,0° |

(Untergrenze „0“ heißt: stabil bis zum Gitterrand 0,001.) Mit wachsendem W wird das Fenster unten endlich: bei Q = I, R = 1 und G = B in der Schätzform ab dem Gitterwert W = 100 und in der Vorhersage-Form ab dem Gitterwert W = 100 (erster der fünf Rauschstärken 0,01 bis 10⁸, bei dem die Untergrenze über 0 liegt); bei G = (1, −3)ᵀ in der Vorhersage-Form schon ab W = 1, in der Schätzform bei G = (1, −3)ᵀ nicht.
Die Reserve ändert sich zwischen W = 10⁴ und 10⁸ nicht mehr (Q = I, R = 1, Vorhersage-Form, G = B: 0,2768 bei W = 10⁴ und 10⁸); ob sie darüber hinaus gegen einen Grenzwert geht, ist auf diesem Gitter nicht gezeigt. Bei G = B kehrt die Reserve auch für W = 10⁸ **nicht** auf die des Zustands-LQR zurück (0,3433 gegen 0,4836, Schätzform); ob das an CB = 0 (Relativgrad 2) liegt, ist nicht untersucht.
**B11: Handbeispiele von Hand** (Brüche in Q(√5)). Skalar x⁺ = x + u (Q = R = 1): p = (1 + √5)/2, K = 1/p = 0,618, Pol 0,382, kleinster Abstand (5 − √5)/4 = 0,691 bei z = −1, Fenster 0 < g < 1 + √5 = 3,236, Phasenreserve genau 72°. Skalar x⁺ = 2x + u: p = 2 + √5, K = (1 + √5)/2 = 1,618, Pol 0,382,
Fenster 0,618 < g < 1,854, kleinster Abstand (5 − √5)/6 = 0,461, (1 + p)·|1 + L(−1)|² = 10/9, Schranke 0,437, Phasenreserve 29,8°, Verzugsreserve 0,557 Perioden; mit einer Periode Verzug sind die Pole z² − 2z + K mit |z|² = K = 1,618 > 1 instabil. Wagen mit Deadbeat-Gewinn K = (1, 2): Fenster (0; 4/3), |1 + L(−1)| = 1/4.
Doyle q = σ = 5: Polynom s⁴ + 6s³ + 11s² + (6 + 50u)·s + (1 − 25u) mit u = m − 1, Fenster ((43 − 9√5)/25; 26/25) = (0,915016; 1,04).
**B12: LQG im Zeitbereich** (Schätzform, Q = I, R = 1, G = B, W = 10⁴; Anfangszustand (1, 0), Schätzer 0): m = 1: ρ = 0,422 (wie ohne Schätzer: die Pole der Schleife enthalten die des Zustands-LQR), 8 Perioden bis 1 %; m = 0,4: ρ = 0,946, 62 Perioden; m = 0,3: ρ = 1,028 (instabil); m = 1,5: ρ = 0,973, 167 Perioden; m = 1,54: ρ = 1,0076 (instabil).

### Zahl → Test

| Befund | Test (`tests/test_claims.py`, wenn nicht anders genannt) | Art |
|---|---|---|
| B1 | `test_b1_nyquist_distance_continuous_vs_sampled` | Band 10⁻⁴ bis 10⁻⁹; Zählung exakt |
| B2 | `test_b2_upper_gain_three_ways_and_db`, Orakel `test_window_upper_edge_three_ways_agree` | Band 10⁻⁹ |
| B3 | `test_b3_phase_and_delay_margins` | Band 0,05° bzw. 5·10⁻⁴ |
| B4 | `test_b4_monotone_in_r_and_limits`, Eigenschaftstest `test_wagen_reserve_does_not_shrink_when_r_grows` | Band; Ordnung exakt |
| B5 | `test_b5_time_domain` (Perioden als Band ± 2), Orakel `test_sine_disturbance_amplitude_equals_inverse_return_difference` | Band |
| B6 | `test_b6_held_continuous_gain` | Band 10⁻⁴; Zählung exakt |
| B7 | `test_b7_sampling_time_series` | Band 10⁻⁴ |
| B8 | `test_b8_bound_wagen_and_frozen_counts` (3 000 Systeme neu gerechnet) | Zählung ± 2, Floats als Band |
| B9 | `test_b9_doyle_windows`, Orakel `test_doyle_hurwitz_window_agrees_with_eigenvalue_window` | Band 10⁻⁶ der Fensterbreite |
| B10 | `test_b10_lqg90_counts_and_rows` (90 Entwürfe neu gerechnet) | Zählung ± 2, Fenster Band 2·10⁻³ |
| B11 | `tests/test_dynamics.py` (H1 bis H5) | exakt in Q(√5) bzw. Band 10⁻⁹ |
| B12 | `test_b12_lqg_time_domain` | Band ± 2 Perioden, ρ 10⁻⁴ |

Alles, was nicht in dieser Tabelle steht (Rechenzeit, Lesbarkeit der Abbildungen), ist Messung dieser README und nicht im Test. Zahlen aus Zufallssystemen stehen nur als Zählungen mit ihrer Zahl n und dem Zufalls-Seed; einzelne Werte daraus sind nicht belegt.

## Hinweise zu den Zahlen

- **Fensterränder des LQG:** Die Kanten werden von einem Gitter mit Schritt 0,001 abgelesen, danach per Bisektion verfeinert und auf drei Stellen gerundet. Ein bloßes Ablesen vom Gitter (abgerundet) kann um 0,001 niedriger liegen.
- **Zweiter Rechenweg des LQG:** größte Abweichung über die 90 Entwürfe 6,7·10⁻⁶ (Trennungsprinzip 2,1·10⁻⁷). Die Toleranz im Test steht bei 10⁻⁵.
- **Zufallssysteme:** gemessen mit dem ganzzahligen SplitMix64 (nicht dem NumPy-Generator): 3 000 von 3 000 steuerbar, 2 990 mit Residuum < 10⁻¹⁰, größte Abweichung der Gleichung 3,3·10⁻¹⁰, 0 Verletzungen; der kleinste Quotient ist 1,00005 (die Schranke ist bei einzelnen Systemen fast scharf), und die stetige Gegenprobe hat 0 Verletzungen in 3 000.
- **Doyle, untere Kante:** Die zunächst aus der Quelle notierte Formel für die untere Kante („1 − (d + f − 4)/(2df) < m“) ist nur die notwendige Bedingung c₁ > 0; die wirkliche Kante folgt aus der Hurwitz-Bedingung und liegt höher: 0,9381 statt 0,9315 bei q = σ = 100, 0,9150 statt 0,8800 bei q = σ = 5. Die obere Kante und Doyles Aussage (das Fenster wird beliebig klein) bleiben bestätigt. Die Rechnung widerspricht damit der Notiz, nicht der Quelle selbst.
- **Doyle, reiner LQ-Regler:** geschlossen gerechnet: stabil für jedes m > 2/f (Spur 2 − m·f, Determinante 1), nach oben ohne Grenze, nicht nur bis zu einem Gitterrand.
- **Phasenreserve stetig:** exakt per Nullstellensuche (72,38° / 79,23° / 85,25°), auf eine Stelle gerundet 72,4° / 79,2° / 85,3°.
- **Riccati-Rekursion als zweiter Weg:** Ohne Symmetrisierung in jedem Schritt lief sie bei Zufallssystemen mit instabilem A in einen Überlauf oder konvergierte nicht (13 von 25 Systemen des Zufalls-Seeds 4 bei Toleranz 10⁻¹² und höchstens 5 000 Schritten, alle mit instabilem A; die Zahl hängt an der Abbruchregel: bei Toleranz 10⁻¹⁰ sind es 9, bei 10⁻⁹ sind es 4; das Band 11 bis 15 steht im Test); mit Symmetrisierung konvergiert sie in 25 von 25 und stimmt mit SciPy überein (10⁻⁹ relativ im Test).
- **Rauschrichtung G = (1, 0)ᵀ:** Die Demo weist sie ab, weil das Paar (A, G) nicht steuerbar ist (das Rauschen erreicht die Geschwindigkeit nie).
- **Zuschnitt:** Die Schätzform ist der Standard, die Vorhersage-Form eine Variante; Doyle (stetig) hat ein eigenes Experiment, der Wagen bleibt das Vehikel; die Schranke steht hier mit „eigene Herleitung, geprüft“; die Verzugsreserve ist nur eine Zahl; die Loop-Transfer-Recovery im Diskreten bleibt offen.

## Ehrliche Grenzen

- **Eine Strecke.** Der Wagen (und ein Skalar als Handbeispiel), Q = I (beim LQG auch Q = 100·(1, 1)ᵀ(1, 1)), ein Eingang. Strecken mit Nullstelle oder mehreren Eingängen sind nicht gemessen; was die Messreihen über Zufallssysteme sagen (B8), gilt nur für die dort beschriebene Familie.
- **Keine Neuheitsaussage.** Die Rückführdifferenz-Gleichung und die Schranke sind eigene Herleitung, in diesem Modell an 3 000 Zufallssystemen und von Hand geprüft; ob sie in Lehrbüchern stehen, ist ungelesen. „Kalman-Ungleichung“ steht hier nur als gemessener Befund (stetig ≥ 1 auf dem Gitter), nicht als Beweis.
- **Quellen.** Doyle (1978) liegt als Scan vor, der als Bild gelesen wurde; die Parameter des Gegenbeispiels sind nachgerechnet, die 60° Phase und die Verstärkung von der Hälfte (−6 dB) bis unbegrenzt sind seine Zuschreibung an Safonov/Athans (im Original nicht gelesen). Für den abgetasteten LQR gibt keine gelesene Quelle eine Reservegarantie.
- **LQG nur mit Positionsmessung,** zwei Formen, ohne Polvorgabe. Warum die Reserve bei G = B und wachsendem W nicht auf die des Zustands-LQR zurückgeht (Loop-Transfer-Recovery im Diskreten, ob CB = 0 die Ursache ist), ist nicht untersucht. Die 90 Entwürfe sind eine Auswahl von Rauschstärken und -richtungen, keine Stichprobe aus einer Verteilung.
- **Reserven des Gesetzes mit endlichem Horizont und des MPC mit aktiver Stellgrenze sind nicht gemessen.** Mit aktiver Grenze ist die Schleife nichtlinear; ein Nyquist-Abstand gilt nur im linearen Gebiet.
- **Phasenreserve nur klassisch** (kleinste Durchtrittsstelle); keine Scheibenreserve, keine strukturierte Singulärwertgrenze. Verstärkungsfehler und Verzug werden nicht gemeinsam betrachtet; der Zeitbereich ist ohne Rauschen.
- **Zeitbereich nahe der Kante.** „Stabil“ am Rand des Fensters heißt „klingt kaum ab“ (m = 1,936: 9 161 Perioden); Aussagen darüber sind Bänder, nie ein Vergleich mit 1.
- **Tiefstwert bei ω = π** ist am Wagen für R von 0,001 bis 100 bei Ts = 1 gemessen und bei Ts = 1 / 0,5 / 0,1 / 0,01 (R = 1); für andere Strecken oder Q nicht vorausgesetzt.

## Verifikation

- **Handbeispiele und Kern:** `tests/test_dynamics.py` (Brüche in Q(√5): Skalar a = 1 und 2, Deadbeat-Wagen, Doyle q = σ = 5, stetiger Gewinn gehalten).
- **Orakel** (`tests/test_oracle_lqr_reserven.py`, anderer Rechenweg): DARE gegen eigene Riccati-Rekursion; stetiger Entwurf gegen die geschlossene Form und das CARE-Residuum; Fenster nach Jury gegen Bisektion gegen Nyquist-Schnitt; Abstand gegen 1 − 1/g_oben und gegen die Rückführdifferenz-Gleichung (aus P, nicht aus K);
  Sinus-Störung im Zeitbereich gegen 1/|1 + L|; LQG: Frequenzgang gegen Determinantenquotient und Trennungsprinzip; Doyle: Polynom aus `np.poly` gegen die Formeln und Hurwitz-Fenster gegen Eigenwert-Fenster.
- **Eigenschaftstests** (`tests/test_eigenschaften_lqr_reserven.py`, 20 bis 50 Zufallssysteme mit festem Zufalls-Seed): Q und R gemeinsam skalieren ändert K nicht; B → c·B mit K → K/c ändert L nicht; Ähnlichkeitstransformation; Schranke gilt für A innerhalb und außerhalb des Einheitskreises; stetig ≥ 1; am Wagen R wachsend und Ts-Reihe; g_oben = 1/(1 − Abstand).
- **Claims** (`tests/test_claims.py`): jede Zahl der Befunde B1 bis B12 mit Voraussetzung; die vorgerechneten Zählungen werden neu gerechnet und mit der Datei und der README verglichen.
- **Randfälle** (`tests/test_edge_cases.py`): R von 10⁻⁶ bis 10⁶, Ts → 0, Fensterkante nur als Band, q = σ = 10⁶, G = (1, 0)ᵀ und W = 0, stabile Skalarstrecke ohne Durchtrittsstelle, a → −a.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten; **Presets** (`tests/test_presets.py`): Wirkung und Zahlen der Hilfetexte; **Oberfläche:** AppTest-Rauchtest, Layout (Schnellstart in der Seitenleiste, fester Kopf, jeder Regler hebt das aktive Preset auf), Footer, Laufzeit-Importe.
- Plattformabhängiges: Zählungen aus Neu-Rechnung ± 2, Anteile ± 0,01, Gleitkommazahlen relativ 10⁻⁹ oder als Band; exakt nur, was in Brüchen gerechnet ist. Gleichstände an Schwellen gibt es hier nicht (Fensterkanten sind offen, ρ = 1 zählt als instabil).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis (Nyquist-Bild), Kernfragen, sechs Experimente, Grenzen, Formeln |
| `data/res_frozen.json` | vorgerechnet: 90 LQG-Entwürfe, Zählungen an je 3 000 Zufallssystemen |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, pandas, plotly |
| `res_constants.py` | Regler-Stufen, Voreinstellungen, Presets, Farben |
| `res_dynamics.py` | Wagen und Skalar, DARE und CARE, Riccati-Rekursion, Zeitbereich |
| `res_margins.py` | Frequenzgang, Fenster (Jury, Bisektion), Phasenreserve, Schranke, Zufallssysteme, Doyle, LQG |
| `res_exact.py` | Q(√5) mit Brüchen: Handbeispiele des Skalars und von Doyle q = σ = 5 |
| `res_evaluation.py` | Auswertungen der Experimente, Verdikte (reine Funktionen), vorgerechnete Messreihen |
| `res_layout.py` | Fester Kopf (Stylesheet) |
| `res_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) |
| `res_rng.py` | SplitMix64 mit exaktem Logarithmus und Kosinus (Kopie der Vorlage des Website-Repos) |
| `res_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/make_frozen.py` | erzeugt `data/res_frozen.json` (mit `--check` neu gerechnet und verglichen) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Claims, Randfälle, Verdikte, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Reserven für das Gesetz mit endlichem Horizont und für MPC mit Stellgrenzen** ([`horizont-falle-demo`](https://github.com/sebastian-hanisch/horizont-falle-demo) und [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo) behandeln diese Gesetze selbst); hier nur der Hinweis, dass die Aussage an der Stellgrenze endet.
- **Entwurf des Filters, Polvorgabe, Loop-Transfer-Recovery:** nicht gelehrt, nur die Reserve des fertigen LQG gemessen.
- **Scheibenreserve, strukturierte Singulärwertgrenzen, Mehrgrößenstrecken, Strecken mit Nullstelle.**
- **Verzug als Experiment:** nur die Verzugsreserve als Zahl (Phasenreserve / Durchtrittsfrequenz); wie ein ganzer Verzug den Entwurf kippt, misst die Demo [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo).
- **PDF-Export** (anders als bei Fall-Demos): eine Konzepte-Demo mit lebenden Abbildungen, kein Bericht.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regelung von LQR bis MPC: vom Plan zur Regel](https://sebastianhanisch.net/konzepte-regelung.html).
