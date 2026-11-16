"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, nicht Seeds. Damit prüft der Test die Grenzen (Abstand 1 − 1e-6, 0,5; ρ = 0,9; Rand des Fensters) und jede
genannte Zahl direkt. Die Verdikte der Presets stehen zusätzlich in tests/test_presets.py (kein Preset ohne Wirkung meldet eine Wirkung)."""

import math

import res_evaluation as ev

SUMMARY = {"count": 90, "below_lqr": 63, "min_return_min": 0.1327}


def test_nyquist_verdict_names_distance_location_peak_and_count():
    kind, text = ev.verdict_nyquist(1.0, 1.0, 0.4836, math.pi, 1.0, 3)
    assert kind == "warning"
    assert "Ts = 1, Q = I, R = 1" in text and "**0,484**" in text and "bei ω = π" in text and "Empfindlichkeitsspitze 2,07" in text
    assert "bei 1,000" in text and "in 3 von 3 gemessenen R (1 / 0,1 / 0,01)" in text


def test_nyquist_verdict_thresholds():
    assert ev.verdict_nyquist(1.0, 1.0, 0.4999, math.pi, 1.0, 3)[0] == "warning"
    assert ev.verdict_nyquist(1.0, 1.0, 0.5, math.pi, 1.0, 3)[0] == "info"            # genau 0,5: noch keine Warnung
    assert ev.verdict_nyquist(1.0, 0.01, 0.9914, math.pi, 1.0, 3)[0] == "info"
    assert ev.verdict_nyquist(1.0, 0.01, 1.0 - 1e-7, 3.0, 1.0, 0)[0] == "success"     # ≥ 1 − 1e-6: die stetige Garantie gilt
    kind, text = ev.verdict_nyquist(0.01, 0.05, 0.9579, 1.5, 1.0, 1)
    assert "bei ω = 1,500" in text and "Ts = 0,05" in text and "R = 0,01" in text and "in 1 von 3" in text


def test_gain_verdict_branches_and_numbers():
    window = (0.0, 1.9363)
    kind, text = ev.verdict_gain(1.0, 1.9, window, 0.953, 79)
    assert kind == "warning" and "0 < m < 1,9363" in text and "m = 1,90" in text and "ρ = 0,953" in text and "in 79 Perioden auf 1 %" in text
    assert ev.verdict_gain(1.0, 1.0, window, 0.422, 6)[0] == "success"
    assert ev.verdict_gain(1.0, 1.0, window, 0.9, 6)[0] == "warning"                  # genau 0,9: schon Warnung
    assert ev.verdict_gain(1.0, 1.0, window, 0.8999, 6)[0] == "success"
    kind, text = ev.verdict_gain(0.1, 1.8, (0.0, 1.5923), 1.336, None)
    assert kind == "error" and "ρ = 1,336" in text and "wächst" in text and "R = 0,1" in text
    kind, text = ev.verdict_gain(1.0, 1.9, window, 0.99999, None)
    assert kind == "warning" and "20 000 Perioden nicht auf 1 % abgeklungen" in text
    assert ev.verdict_gain(1.0, 1.0, window, 1.0, None)[0] == "error"                 # ρ = 1 zählt als instabil


def test_window_verdict():
    kind, text = ev.verdict_window(1.0, (0.0, 1.9363), 39.6, 0.604)
    assert kind == "info" and "**1,9363**" in text and "(5,74 dB)" in text and "Phasenreserve 39,6°" in text and "Verzugsreserve 0,604 Perioden" in text
    assert "keine Durchtrittsstelle" in ev.verdict_window(1.0, (0.0, 3.0), None, None)[1]
    assert ev.verdict_window(1.0, None, None, None)[0] == "error"
    assert "∞ dB" in ev.verdict_window(1.0, (0.0, math.inf), 70.0, 1.0)[1]


def test_ts_verdict_states_the_held_gain_and_the_limit():
    rows = [{"ts": 1.0, "min_euler": 0.4836}, {"ts": 0.1, "min_euler": 0.9182}, {"ts": 0.01, "min_euler": 0.9914}]
    kind, text = ev.verdict_ts(rows, 1.0, 3.9876, 0.4950, 0.1)
    assert kind == "info" and "Ts = 1: 0,484, Ts = 0,1: 0,918, Ts = 0,01: 0,991" in text and "R = 0,1" in text
    assert "bei Ts = 1 instabil (ρ = 3,988)" in text and "stabil nur für Ts < 0,4950 (= 2/k₂)" in text
    assert "bei Ts = 0,1 stabil (ρ = 0,902)" in ev.verdict_ts(rows, 0.1, 0.9016, 0.4950, 0.1)[1]


def test_doyle_verdict_branches():
    kind, text = ev.verdict_doyle(100.0, 1.02, (0.938072, 1.006721), 0.16396, 0.006676, False, True)
    assert kind == "warning" and "q = σ = 100" in text and "0,938072 < m < 1,006721" in text and "m > 0,1640" in text
    assert "Bei m = 1,020 ist der Kreis mit Filter instabil, der LQ-Regler allein stabil" in text and "|1 + L(0)| = 0,006676" in text
    kind, text = ev.verdict_doyle(1e6, 1.0, (0.999005, 1.000001), 0.001996, 9.96e-07, True, True)
    assert kind == "success" and "q = σ = 10⁶" in text and "mit Filter stabil" in text
    assert "9,96e-07" not in text                                                   # keine wissenschaftliche Schreibweise mit Punkt oder e in der Anzeige


def test_lqg_verdict_compares_with_the_state_feedback():
    res = {"window": (0.335, 1.531), "min_return": 0.3433, "lqr_window": (0.0, 1.9363), "lqr_min": 0.4836}
    kind, text = ev.verdict_lqg(res, 1e4, "schaetz", SUMMARY)
    assert kind == "warning" and "Schätzform, W = 10⁴" in text and "(0,335; 1,531)" in text and "**0,343**" in text and "(0; 1,936) und 0,484" in text
    assert "Über 90 vorgerechnete Entwürfe liegt der Abstand in 63 unter dem des Zustands-LQR" in text and "eine Garantie gibt es nicht" in text
    better = dict(res, min_return=0.5283, window=(0.0, 2.688))
    kind, text = ev.verdict_lqg(better, 0.01, "vorhersage", SUMMARY)
    assert kind == "info" and "Vorhersage-Form, W = 0,01" in text and "(0; 2,688)" in text
    assert "instabil bei m = 1" in ev.verdict_lqg(dict(res, window=None), 1.0, "schaetz", SUMMARY)[1]


def test_bound_verdicts():
    rows = [{"r": 1.0, "r0": 0.4221, "min_return": 0.4836, "guar_lo": 0.703, "guar_hi": 1.73, "win_lo": 0.0, "win_hi": 1.9363},
            {"r": 0.001, "r0": 0.01953, "min_return": 0.3458, "guar_lo": 0.9808, "guar_hi": 1.0199, "win_lo": 0.0, "win_hi": 1.5286}]
    kind, text = ev.verdict_bound_wagen(rows)
    assert kind == "info" and "bei R = 0,001 nur 0,0195 (gemessen 0,3458)" in text and "(0,981; 1,020) statt (0; 1,529)" in text
    check = {"violations_bound": 0, "ok": 200, "seed": 15, "good": 199}
    frozen = {"violations_bound": 0, "ok": 3000, "requested": 3000, "seed": 15}
    kind, text = ev.verdict_bound_random(check, frozen)
    assert kind == "success" and "0 Verletzungen der Schranke in 200 gültigen Zufallssystemen (Zufalls-Seed 15, 199 mit" in text
    assert "Vorgerechnet: 0 Verletzungen in 3 000 von 3 000 Zufallssystemen" in text
    assert ev.verdict_bound_random(dict(check, violations_bound=1), frozen)[0] == "error"


def test_garantie_verdict_names_all_three_cases():
    kind, text = ev.verdict_garantie(0.4836, 1.0, 1.0, (0.938072, 1.006721), SUMMARY)
    assert kind == "warning" and "Stetig: Abstand 1,000. Abgetastet mit Ts = 1: 0,484." in text and "0,938 < m < 1,0067" in text
    assert "in 63 von 90 vorgerechneten LQG-Entwürfen" in text and "kleinster Wert 0,133" in text
