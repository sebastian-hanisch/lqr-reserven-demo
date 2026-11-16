"""Claims-Tests: jede Zahl der Befunde B1 bis B12 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung überall: Wagen (Doppelintegrator), Ts = 1, Euler-Form B = (0, Ts)ᵀ, Q = I, ein Eingang, wenn nicht anders gesagt; Doyle stetig; LQG mit Positionsmessung. Zahlen, die auf anderen
Plattformen anders ausfallen können (Bisektions- und Gittergrößen, Zufallssysteme), stehen als Band (Zählungen ± 2, Anteile ± 0,01, Gleitkommazahlen relativ 1e-9 oder Band wie angegeben);
exakt nur, was in Brüchen gerechnet ist (tests/test_dynamics.py). Die vorgerechneten Messreihen (`data/res_frozen.json`) werden hier in voller Größe neu gerechnet und mit der Datei und der README verglichen."""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import res_constants as C
import res_dynamics as D
import res_evaluation as ev
import res_margins as M

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FROZEN = json.loads((ROOT / "data" / "res_frozen.json").read_text(encoding="utf-8"))
I2 = np.eye(2)
R4 = (1.0, 0.1, 0.01, 0.001)


def _readme_has(*texts):
    flat = README.replace(" ", " ")
    for t in texts:
        assert t in flat, t


def _w(r, ts=1.0, form="euler"):
    a, b, p, k = ev.design(r, ts, form)
    return a, b, p, k


# --- B1 -------------------------------------------------------------------------------------------------------------------------------

def test_b1_nyquist_distance_continuous_vs_sampled():
    for r, expected, peak in zip(R4, (0.4836, 0.3720, 0.3485, 0.3458), (2.068, 2.688, 2.870, 2.892)):
        a, b, p, k = _w(r)
        mn, wm = M.min_return(a, b, k)
        assert mn == pytest.approx(expected, abs=5e-5) and abs(wm - math.pi) < 1e-5
        assert 1 / mn == pytest.approx(peak, abs=5e-4)
    for r, expected in zip((1.0, 0.1, 0.01), (1.000001, 1.000005, 1.00005)):
        ac, bc, kc = ev.cont_design(r)
        cmin, _ = M.cont_min_return(ac, bc, kc)
        assert cmin >= 1 - 1e-9 and cmin == pytest.approx(expected, abs=2e-6)
    assert ev.misses_guarantee(1.0, "euler") == 3
    _readme_has("0,4836 (R = 1), 0,3720 (R = 0,1), 0,3485 (R = 0,01) und 0,3458 (R = 0,001)", "2,068 / 2,688 / 2,870 / 2,892",
                "1,000001 bei R = 1, 1,000005 bei R = 0,1, 1,00005 bei R = 0,01", "in 3 von 3 gemessenen R (1 / 0,1 / 0,01) verfehlt")


# --- B2 -------------------------------------------------------------------------------------------------------------------------------

def test_b2_upper_gain_three_ways_and_db():
    devs = []
    for r, hi_exp, db_exp in zip(R4, (1.9363, 1.5923, 1.5349, 1.5286), (5.74, 4.04, 3.72, 3.69)):
        a, b, p, k = _w(r)
        lo, hi = M.jury_window(a, b, k)
        assert hi == pytest.approx(hi_exp, abs=5e-5) and 20 * math.log10(hi) == pytest.approx(db_exp, abs=5e-3) and lo == 0.0
        devs.append(abs(hi - M.g_upper_bisect(a, b, k)))
        assert abs(hi - M.g_nyquist_pi(a, b, k)) < 1e-9
        assert M.g_lower_bisect(a, b, k) < 1e-9
        assert M.min_return(a, b, k)[0] == pytest.approx(1 - 1 / hi, abs=1e-9)
    assert max(devs) < 1e-9
    his = [M.jury_window(*_w(r)[:2], _w(r)[3])[1] for r in R4]
    assert round(min(his), 1) == 1.5 and round(max(his), 1) == 1.9                                  # „nur um den Faktor 1,5 bis 1,9“ (gerundet)
    _readme_has("1,9363 / 1,5923 / 1,5349 / 1,5286 (5,74 / 4,04 / 3,72 / 3,69 dB)", "um den Faktor 1,5 bis 1,9 steigen")


# --- B3 -------------------------------------------------------------------------------------------------------------------------------

def test_b3_phase_and_delay_margins():
    for r, pm_exp, d_exp in zip(R4, (39.6, 34.4, 33.2, 33.1), (0.604, 0.423, 0.392, 0.389)):
        a, b, p, k = _w(r)
        pms = M.phase_margins(a, b, k)
        assert len(pms) == 1
        assert pms[0]["pm_deg"] == pytest.approx(pm_exp, abs=0.06) and pms[0]["delay"] == pytest.approx(d_exp, abs=5e-4) and pms[0]["delay"] < 1
    for r, pm_exp in zip((1.0, 0.1, 0.01), (72.4, 79.2, 85.3)):
        ac, bc, kc = ev.cont_design(r)
        (m,) = M.phase_margins(ac, bc, kc, stetig=True)
        assert m["pm_deg"] == pytest.approx(pm_exp, abs=0.06)
    _readme_has("39,6° / 34,4° / 33,2° / 33,1°", "0,604 / 0,423 / 0,392 / 0,389 Perioden", "72,4° / 79,2° / 85,3°")


# --- B4 -------------------------------------------------------------------------------------------------------------------------------

def test_b4_monotone_in_r_and_limits():
    rs = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
    for form in ("euler", "halten"):
        vals = [M.min_return(*ev.design(r, 1.0, form)[:2], ev.design(r, 1.0, form)[3], n=4001)[0] for r in rs]
        assert len(vals) == 10 and all(x < y for x, y in zip(vals, vals[1:]))
    assert M.min_return(*_w(100.0)[:2], _w(100.0)[3])[0] == pytest.approx(0.7958, abs=5e-5)
    assert M.min_return(*_w(10.0)[:2], _w(10.0)[3])[0] == pytest.approx(0.6596, abs=5e-5)
    a1, b1, _, k1 = _w(1.0)
    a01, b01, _, k01 = _w(0.1)
    poles = sorted(abs(np.linalg.eigvals(a01 - b01 @ k01)))
    assert poles == pytest.approx([0.093, 0.362], abs=5e-4)                                  # R = 0,1: die kleineren Pole …
    assert D.spectral_radius(a01 - b01 @ k01) < D.spectral_radius(a1 - b1 @ k1)
    assert M.jury_window(a01, b01, k01)[1] == pytest.approx(1.59, abs=5e-3)                  # … und nur das 1,59-Fache
    for r in (1e-4, 1e-6):
        assert M.min_return(*_w(r)[:2], _w(r)[3])[0] == pytest.approx(0.3455, abs=2e-4)
    a, b, p, k = _w(1e6)
    assert k[0] == pytest.approx([0.0010, 0.0447], abs=1e-4) and M.min_return(a, b, k)[0] == pytest.approx(0.978, abs=5e-4)
    _readme_has("10 von 10 gemessenen R (0,001 bis 100)", "(R = 100: 0,7958, R = 10: 0,6596)", "(0,093; 0,362)", "das 1,59-Fache", "(0,3455)", "K = (0,0010; 0,0447)", "der Abstand 0,978")


# --- B5 -------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("r,m,rho,steps", [(1.0, 1.0, 0.42208, 6), (1.0, 1.9, 0.95279, 79), (1.0, 1.93, 0.9918, 462), (1.0, 1.936, 0.99959, 9161),
                                           (1.0, 1.94, 1.00478, None), (1.0, 0.05, 0.97924, 220), (0.1, 1.0, 0.36161, 6), (0.1, 1.8, 1.33581, None)])
def test_b5_time_domain(r, m, rho, steps):
    a, b, p, k = _w(r)
    mat = D.closed_loop(a, b, k, m)
    assert D.spectral_radius(mat) == pytest.approx(rho, abs=2e-5)
    got, _ = D.settle(mat, [1.0, 0.0])
    if steps is None:
        assert got is None
    else:
        assert got is not None and abs(got - steps) <= 2                                       # Zählung als Band ± 2


def test_b5_readme_numbers():
    _readme_has("m = 1: ρ = 0,422, 6 Perioden bis 1 %", "m = 1,9: ρ = 0,953, 79 Perioden", "m = 1,93: ρ = 0,9918, 462", "m = 1,936: ρ = 0,99959, 9 161", "m = 1,94: ρ = 1,0048, instabil",
                "m = 0,05: ρ = 0,979, 220 Perioden", "(ρ = 1,336; nominal 6 Perioden)", "Spitze 1/|1 + L| = 2,068")
    a, b, p, k = _w(1.0)
    assert 1 / abs(1 + M.L_of(np.exp(1j * (math.pi - 1e-4)), a, b, k)[0]) == pytest.approx(2.068, abs=1e-3)


# --- B6 -------------------------------------------------------------------------------------------------------------------------------

def test_b6_held_continuous_gain():
    grid = np.linspace(1e-4, 5.0, 50000)
    ad = np.zeros((len(grid), 2, 2))
    ad[:, 0, 0] = ad[:, 1, 1] = 1.0
    ad[:, 0, 1] = grid
    bd = np.stack([grid ** 2 / 2, grid], axis=1)[:, :, None]
    for r, tmax, rho1 in zip((1.0, 0.1, 0.01), (1.1547, 0.4950, 0.1826), (0.6115, 3.9875, 14.30)):
        ac, bc, kc = ev.cont_design(r)
        assert D.held_continuous_gain_ts_max(kc) == pytest.approx(tmax, abs=5e-5)
        stable = np.max(np.abs(np.linalg.eigvals(ad - bd @ kc[None])), axis=1) < 1
        assert int(np.sum(stable[1:] != stable[:-1])) == 1                                       # ein einziger Wechsel auf 50 000 Gitterpunkten
        assert grid[stable].max() < D.held_continuous_gain_ts_max(kc) < grid[~stable].min() + 1e-3
        assert D.held_continuous_gain_radius(kc, 1.0) == pytest.approx(rho1, abs=6e-3 if rho1 > 10 else 1e-4)
    _readme_has("Ts_max = 1,1547 (R = 1), 0,4950 (R = 0,1), 0,1826 (R = 0,01)", "0,6115 / 3,9875 / 14,30", "auf 50 000 Gitterpunkten")


# --- B7 -------------------------------------------------------------------------------------------------------------------------------

def test_b7_sampling_time_series():
    expected = {1.0: (0.4836, 0.4858), 0.5: (0.6698, 0.6714), 0.25: (0.8117, 0.8120), 0.1: (0.9182, 0.9182), 0.05: (0.9579, 0.9579), 0.01: (0.9914, 0.9914)}
    gup = {1.0: 1.936, 0.5: 3.029, 0.25: 5.311, 0.1: 12.22, 0.05: 23.77, 0.01: 116.1}
    data = ev.ts_rows(1.0)
    for row in data["rows"]:
        e, h = expected[row["ts"]]
        assert row["min_euler"] == pytest.approx(e, abs=6e-5) and row["min_halten"] == pytest.approx(h, abs=6e-5)
        assert row["gup_euler"] == pytest.approx(gup[row["ts"]], rel=6e-4)
    vals = [r["min_euler"] for r in data["rows"]]
    assert all(x < y for x, y in zip(vals, vals[1:])) and abs(data["rows"][0]["min_euler"] - data["rows"][0]["min_halten"]) == pytest.approx(0.002, abs=5e-4)
    for row in data["rows"]:
        if row["ts"] <= 0.1:
            assert 0.82 - 5e-3 <= (1 - row["min_euler"]) / row["ts"] <= 0.86 + 5e-3
    assert data["cont_min"] >= 1 - 1e-9
    _readme_has("0,4836 | 0,4858 (Ts = 1)", "0,6698 | 0,6714 (0,5)", "0,8117 | 0,8120 (0,25)", "1,936 / 3,029 / 5,311 / 12,22 / 23,77 / 116,1", "um 0,002", "das 0,82- bis 0,86-Fache von Ts")


# --- B8 -------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bound3000():
    return M.bound_check(C.FROZEN_SEED_BOUND, C.FROZEN_COUNT)


@pytest.fixture(scope="module")
def kalman3000():
    return M.kalman_check(C.FROZEN_SEED_BOUND, C.FROZEN_COUNT)


def test_b8_bound_wagen_and_frozen_counts(bound3000, kalman3000):
    rows = ev.bound_rows(1.0, "euler")
    by = {row["r"]: row for row in rows}
    for r, r0, mn in ((0.001, 0.0195, 0.3458), (0.01, 0.0614, 0.3485), (0.1, 0.1832, 0.3720), (1.0, 0.4221, 0.4836)):
        assert by[r]["r0"] == pytest.approx(r0, abs=6e-5) and by[r]["min_return"] == pytest.approx(mn, abs=5e-5) and by[r]["r0"] <= by[r]["min_return"]
        assert by[r]["poles_product"] == pytest.approx(by[r]["r0"] ** 2, rel=1e-9)               # det(A) = 1 am Wagen
    assert by[1.0]["poles_product"] == pytest.approx(0.17815, abs=5e-6)
    assert (by[1.0]["guar_lo"], by[1.0]["guar_hi"]) == pytest.approx((0.703, 1.730), abs=5e-4)
    assert (by[0.01]["guar_lo"], by[0.01]["guar_hi"]) == pytest.approx((0.942, 1.065), abs=5e-4)
    assert by[0.01]["win_hi"] == pytest.approx(1.535, abs=5e-4) and by[1.0]["win_hi"] == pytest.approx(1.936, abs=5e-4)
    # 3 000 Zufallssysteme: neu gerechnet und gegen die Datei und die README
    fb = FROZEN["bound"]
    assert (bound3000["seed"], bound3000["requested"]) == (15, 3000) == (fb["seed"], fb["requested"])
    assert abs(bound3000["ok"] - fb["ok"]) <= 2 and abs(bound3000["good"] - fb["good"]) <= 2 and 2985 <= bound3000["good"] <= 3000
    assert bound3000["ok"] >= 2998 and bound3000["violations_bound"] == 0 == bound3000["violations_gain"] == fb["violations_bound"] == fb["violations_gain"]
    assert bound3000["max_equation_good"] < 1e-8 and bound3000["max_det_good"] < 1e-8
    assert bound3000["tightest_ratio"] == pytest.approx(1.00005, abs=2e-4) and 1.0 <= bound3000["tightest_ratio"]
    fk = FROZEN["kalman"]
    assert kalman3000["violations"] == 0 == fk["violations"] and kalman3000["ok"] >= 2998 and kalman3000["smallest"] >= 1 - 1e-9
    _readme_has("3 000 steuerbar und lösbar, 2 990 mit DARE-Residuum < 10⁻¹⁰", "bis 3,3·10⁻¹⁰", "bis 3,3·10⁻¹²", "0 Verletzungen der Schranke und 0 Verletzungen des daraus folgenden Intervalls",
                "1,00005", "0,4221 / 0,1832 / 0,0614 / 0,0195", "(0,703; 1,730) bei R = 1 und (0,942; 1,065) bei R = 0,01", "(0; 1,936) und (0; 1,535)", "(0,17815 bei R = 1)", "0 Verletzungen von min |1 + L| ≥ 1 − 10⁻⁹")
    assert fb["good"] == 2990 and fb["ok"] == 3000 and round(fb["max_equation_good"] * 1e10, 1) == 3.3 and round(fb["max_det_good"] * 1e12, 1) == 3.3


# --- B9 -------------------------------------------------------------------------------------------------------------------------------

def test_b9_doyle_windows():
    expected = {1.0: (0.9208, 1.0557), 5.0: (0.9150, 1.0400), 12.0: (0.9144, 1.0278), 100.0: (0.9381, 1.0067), 1596.0: (0.9779, 1.00057), 1e4: (0.99049, 1.000096), 1e6: (0.999005, 1.000001)}
    for q, (lo_e, hi_e) in expected.items():
        lo, hi = M.doyle_window_hurwitz(q)
        digits_lo = len(str(lo_e).split(".")[1])
        assert lo == pytest.approx(lo_e, abs=0.6 * 10 ** -digits_lo) and hi == pytest.approx(hi_e, abs=6e-7 if hi_e < 1.0001 else 6e-5)
        assert hi == pytest.approx(1 + 1 / M.doyle_f(q) ** 2, rel=1e-12)
    # Quellenformel für die untere Kante ist nur die notwendige Bedingung: liegt darunter
    for q, formula, true_lo in ((100.0, 0.9315, 0.9381), (5.0, 0.8800, 0.9150)):
        f = M.doyle_f(q)
        lo_formula = 1 - (2 * f - 4) / (2 * f * f)
        assert lo_formula == pytest.approx(formula, abs=5e-5) and M.doyle_window_hurwitz(q)[0] == pytest.approx(true_lo, abs=5e-5) and M.doyle_window_hurwitz(q)[0] > lo_formula
    for q, lq, r0, ms in ((1.0, 0.472, None, 18.94), (100.0, 0.164, 0.00668, 149.8), (1e6, 0.002, 9.96e-7, 1.0e6)):
        assert M.doyle_lq_lower(q) == pytest.approx(lq, abs=5e-4)
        assert 1 / M.doyle_return0(q) == pytest.approx(ms, rel=5e-3)
        if r0:
            assert M.doyle_return0(q) == pytest.approx(r0, rel=5e-3)
    assert M.doyle_return0(5.0) == pytest.approx(0.0385, abs=5e-5) and 1 / M.doyle_return0(5.0) == pytest.approx(26.0, abs=1e-9)
    g, k = M.doyle_gains(5.0, 5.0)
    assert np.allclose(g, M.doyle_f(5.0)) and np.allclose(k, M.doyle_f(5.0))
    _readme_has("(0,9208; 1,0557), (0,9150; 1,0400), (0,9144; 1,0278), **(0,9381; 1,0067)**, (0,9779; 1,00057), (0,99049; 1,000096), **(0,999005; 1,000001)**",
                "0,9381 statt 0,9315 bei q = σ = 100, 0,9150 statt 0,8800 bei q = σ = 5", "0,472 bei q = 1, 0,164 bei q = 100, 0,002 bei q = 10⁶",
                "0,0385 bei q = σ = 5, 0,00668 bei 100, 9,96·10⁻⁷ bei 10⁶", "26,0 / 149,8 / 1,0·10⁶")


# --- B10 ------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lqg90():
    rows = ev.lqg90_rows()
    return rows, ev.lqg90_summary(rows)


def test_b10_lqg90_counts_and_rows(lqg90):
    rows, summary = lqg90
    froz = FROZEN["lqg90_summary"]
    assert summary["count"] == 90 == froz["count"] and abs(summary["below_lqr"] - 63) <= 2 and abs(froz["below_lqr"] - 63) <= 2
    for key, expected in (("I_1.0", 28), ("I_0.1", 21), ("rang1_1.0", 14)):
        assert summary["below_per_regler"][key][1] == 30 and abs(summary["below_per_regler"][key][0] - expected) <= 2
    assert abs(summary["upper_above_lqr"] - 32) <= 2
    assert summary["min_return_min"] == pytest.approx(0.1327, abs=2e-4) and summary["min_return_max"] == pytest.approx(0.5950, abs=2e-4)
    assert summary["pm_min"] == pytest.approx(9.0, abs=0.06)
    assert summary["separation_max"] < 1e-6 and summary["route2_max"] < 1e-5                         # gemessen 2,1e-7 und 6,7e-6
    assert all(r["min_return"] > 0 for r in rows) and all(r["lo"] is None or r["lo"] >= 0 for r in rows)
    # die Tabelle der README
    table = [("I", 1.0, "schaetz", 0.01, (0.0, 2.689), 0.5283, 36.2), ("I", 1.0, "schaetz", 1e4, (0.335, 1.531), 0.3433, 29.0), ("I", 1.0, "vorhersage", 1e4, (0.536, 1.383), 0.2768, 24.4),
             ("I", 0.1, "vorhersage", 1e4, (0.694, 1.308), 0.2357, 17.7), ("rang1", 1.0, "schaetz", 1e8, (0.786, 1.257), 0.1987, 12.7), ("rang1", 1.0, "vorhersage", 1e8, (0.834, 1.153), 0.1327, 9.0)]
    for lq, r, form, w, (lo, hi), mn, pm in table:
        row = next(x for x in rows if (x["lq"], x["r"], x["form"], x["g"], x["w"]) == (lq, r, form, "b", w))
        assert row["lo"] == pytest.approx(lo, abs=1e-3) and row["hi"] == pytest.approx(hi, abs=1e-3) and row["min_return"] == pytest.approx(mn, abs=2e-4) and row["pm_deg"] == pytest.approx(pm, abs=0.06)
        frozen_row = ev.lqg90_lookup(lq, r, form, "b", w)
        assert frozen_row["min_return"] == pytest.approx(row["min_return"], rel=1e-6)
    lqr_refs = {("I", 1.0): (0.4836, 1.9363), ("I", 0.1): (0.3720, 1.5923), ("rang1", 1.0): (0.2617, 1.3545)}
    for (lq, r), (mn, hi) in lqr_refs.items():
        row = next(x for x in rows if (x["lq"], x["r"]) == (lq, r))
        assert row["lqr_min"] == pytest.approx(mn, abs=5e-5) and row["lqr_hi"] == pytest.approx(hi, abs=5e-5)
    # Fenster unten endlich ab W: Q = I, R = 1, G = B
    def lo_of(form, g, w):
        return next(x for x in rows if (x["lq"], x["r"], x["form"], x["g"], x["w"]) == ("I", 1.0, form, g, w))["lo"]
    assert lo_of("schaetz", "b", 1.0) == 0.0 < lo_of("schaetz", "b", 100.0) and lo_of("vorhersage", "b", 1.0) == 0.0 < lo_of("vorhersage", "b", 100.0)
    assert lo_of("vorhersage", "m3", 1.0) > 0.0 and lo_of("vorhersage", "m3", 0.01) == 0.0 and all(lo_of("schaetz", "m3", w) == 0.0 for w in C.LQG_W_GRID)
    # Grenzwert und keine Rückkehr zum Zustands-LQR (G = B, Schätzform)
    assert next(x for x in rows if (x["lq"], x["r"], x["form"], x["g"]) == ("I", 1.0, "vorhersage", "b") and x["w"] == 1e8)["min_return"] == pytest.approx(0.2768, abs=2e-4)
    assert next(x for x in rows if (x["lq"], x["r"], x["form"], x["g"]) == ("I", 1.0, "schaetz", "b") and x["w"] == 1e8)["min_return"] == pytest.approx(0.3433, abs=2e-4)
    _readme_has("Nyquist-Abstand 0,1327 bis 0,5950", "in 63 von 90 Entwürfen** (28 von 30, 21 von 30, 14 von 30 je Regler)", "in 32 von 90 sogar über der", "kleinste Phasenreserve 9,0°",
                "2,1·10⁻⁷ in 90 von 90", "bis 6,7·10⁻⁶", "(0; 2,689)", "(0,335; 1,531)", "(0,536; 1,383)", "(0,694; 1,308)", "(0,786; 1,257)", "(0,834; 1,153)",
                "0,2768 bei W = 10⁴ und 10⁸", "(0,3433 gegen 0,4836, Schätzform)")
    assert FROZEN["lqg90_summary"]["below_per_regler"]["I_1.0"] == [28, 30] and froz["upper_above_lqr"] == 32


# --- B12 ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m,rho,steps", [(1.0, 0.42208, 8), (0.4, 0.94567, 62), (0.3, 1.02789, None), (1.5, 0.97272, 167), (1.54, 1.00758, None)])
def test_b12_lqg_time_domain(m, rho, steps):
    a, b, k, g, _, _ = ev.lqg_inputs(1.0, 1.0, "euler", "I", "b")
    res = M.lqg_analyse(a, b, k, g, 1e4, "schaetz", n_freq=2001)
    mat = res["m0"] + m * res["m1"]
    assert D.spectral_radius(mat) == pytest.approx(rho, abs=2e-5)
    got, _ = D.settle(mat, [1.0, 0.0, 0.0, 0.0])
    if steps is None:
        assert got is None
    else:
        assert got is not None and abs(got - steps) <= 2
    assert res["window"][0] == pytest.approx(0.335, abs=1e-3) and res["window"][1] == pytest.approx(1.531, abs=1e-3)       # m = 0,3 und 1,54 liegen außerhalb


def test_b12_readme_numbers():
    _readme_has("m = 1: ρ = 0,422 (wie ohne Schätzer", "8 Perioden bis 1 %", "m = 0,4: ρ = 0,946, 62 Perioden", "m = 0,3: ρ = 1,028 (instabil)", "m = 1,5: ρ = 0,973, 167 Perioden", "m = 1,54: ρ = 1,0076 (instabil)")


# --- Konsistenz der Dateien ---------------------------------------------------------------------------------------------------------------

def test_frozen_file_is_consistent_with_a_fresh_computation_of_the_lqg_part(lqg90):
    rows, summary = lqg90
    for old, new in zip(FROZEN["lqg90"], rows):
        assert (old["lq"], old["r"], old["form"], old["g"], old["w"]) == (new["lq"], new["r"], new["form"], new["g"], new["w"])
        assert old["min_return"] == pytest.approx(new["min_return"], rel=1e-6)
        assert (old["hi"] is None) == (new["hi"] is None) and (old["hi"] is None or old["hi"] == pytest.approx(new["hi"], abs=1e-4))
    assert FROZEN["lqg90_summary"]["below_lqr"] == summary["below_lqr"] or abs(FROZEN["lqg90_summary"]["below_lqr"] - summary["below_lqr"]) <= 2


def test_every_test_named_in_the_readme_exists():
    names = set(re.findall(r"`(test_[a-z0-9_]+)`", README))
    assert names
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py"))
    for name in names:
        assert f"def {name}(" in source, name


def test_readme_names_the_frozen_measurement_as_vorgerechnet_and_no_forbidden_words():
    assert "vorgerechnet" in README and "Zufalls-Seed" in README
    for word in ("trägt", "tragen", "Saatwert", "läuft lokal", "geplant"):
        assert word not in README, word
