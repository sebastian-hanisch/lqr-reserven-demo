"""Mini-Instanz-Tests von Hand (Handbeispiele H1 bis H5 des Plans, Brüche in Q(√5), `fractions`) für die Einheiten res_dynamics, res_exact und res_margins.

Jeder Erwartungswert steht als Rechnung im Test, nicht als vom Code abgeschriebene Zahl. Konvention: u = −K·x, Q = R = 1, Schleife am Streckeneingang geöffnet."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import res_dynamics as D
import res_exact as X
import res_margins as M

S5 = X.S5


def _close(x, y, tol=1e-9):
    return abs(float(x) - float(y)) < tol


# --- H1: Skalar x⁺ = x + u --------------------------------------------------------------------------------------------------------------

def test_h1_scalar_integrator_exact_and_numerical():
    ex = X.scalar_exact(1)
    p = ex["p"]
    assert p == (1 + S5) / 2 and p * p == p + 1                       # DARE p² = p + 1 (Handrechnung: p = 1 + p − p²/(1 + p))
    assert ex["K"] == 1 / p == (S5 - 1) / 2                           # K = p/(1 + p) = 1/p
    assert ex["pole"] == 1 / (p * p)                                  # Pol 1 − K = 0,382
    assert ex["min_return"] == (5 - S5) / 4                           # (1 + c)/2 bei z = −1: (5 − √5)/4 = 0,691
    assert (ex["g_lo"], ex["g_hi"]) == (QS0, 1 + S5)                  # Fenster 0 < g < 1 + √5 = 3,236
    a, b = D.skalar(1.0)
    pn, kn = D.dlqr(a, b, np.array([[1.0]]), 1.0)
    assert _close(pn[0, 0], float(p), 1e-12) and _close(kn[0, 0], float(ex["K"]), 1e-12)
    assert M.window(a, b, kn) == pytest.approx((0.0, float(1 + S5)), abs=1e-12)
    assert _close(M.min_return(a, b, kn)[0], float(ex["min_return"]), 1e-9)
    assert X.scalar_phase_margin_degrees(1) == pytest.approx(72.0, abs=1e-9)   # K/2 = sin 18°: Phasenreserve genau 72°
    pm = M.phase_margins(a, b, kn)
    assert len(pm) == 1 and pm[0]["pm_deg"] == pytest.approx(72.0, abs=1e-6)


QS0 = X.QS(0)


# --- H2: Skalar x⁺ = 2x + u (instabile Strecke) -----------------------------------------------------------------------------------------

def test_h2_scalar_unstable_exact():
    ex = X.scalar_exact(2)
    p = ex["p"]
    assert p == 2 + S5 and p * p - 4 * p - 1 == X.QS(0)               # p² − 4p − 1 = 0
    assert ex["K"] == (1 + S5) / 2                                    # K = 2p/(1 + p)
    assert ex["pole"] == (3 - S5) / 2                                 # Pol 2 − K = 0,382
    assert ex["g_lo"] == (S5 - 1) / 2 and ex["g_hi"] == 3 * (S5 - 1) / 2    # 0,618 < g < 1,854
    assert ex["min_return"] == (5 - S5) / 6                           # bei z = −1: (1 + c)/3
    assert ex["residual_lhs"] == F(10, 9) == ex["residual_rhs"]       # (1 + p)·|1 + L(−1)|² = 1 + 1/9
    assert ex["pole"] == 2 * ex["r0_squared"]                         # det(A − BK) = det(A)·R/(R + BᵀPB): Pol = a/(1 + p)


def test_h2_scalar_unstable_numerical_margins_and_delay():
    ex = X.scalar_exact(2)
    a, b = D.skalar(2.0)
    p, k = D.dlqr(a, b, np.array([[1.0]]), 1.0)
    assert _close(k[0, 0], float(ex["K"]), 1e-12)
    lo, hi = M.window(a, b, k)
    assert _close(lo, float(ex["g_lo"])) and _close(hi, float(ex["g_hi"]))
    assert _close(M.min_return(a, b, k)[0], float(ex["min_return"]))
    assert _close(M.bound_r0(b, p, 1.0), math.sqrt(float(ex["r0_squared"])))   # 0,437 ≤ 0,4607
    pm = M.phase_margins(a, b, k)[0]
    cos_w = (4 - float(ex["K"])) / 4                                  # |e^{jω} − 2| = K ⇒ cos ω = (4 − K)/4 = 0,5955
    assert _close(pm["w"], math.acos(cos_w))
    assert 29.7 < pm["pm_deg"] < 29.9 and 0.55 < pm["delay"] < 0.56 and pm["delay"] < 1
    assert X.scalar_phase_margin_degrees(2) == pytest.approx(pm["pm_deg"], abs=1e-6)
    # eine Periode Verzug: x⁺ = 2x − K·x_{k−1}, Polynom z² − 2z + K, Produkt der Pole K = 1,618 > 1: instabil
    assert not D.is_stable(np.array([[2.0, -float(ex["K"])], [1.0, 0.0]]))


def test_qs_formatting_and_arithmetic():
    assert X.qs_str((1 + S5) / 2) == "1/2 + 1/2·√5" and X.qs_str(2 + S5) == "2 + √5" and X.qs_str(X.QS(F(3, 2))) == "3/2"
    assert X.qs_str((S5 - 1) / 2) == "−1/2 + 1/2·√5"
    assert (1 + S5) * (1 - S5) == X.QS(-4) and (S5 - 2) > X.QS(0) and (S5 - 3) < X.QS(0)     # √5 = 2,236…


# --- H3: Wagen, Jury-Fenster -----------------------------------------------------------------------------------------------------------

def test_h3_deadbeat_gain_in_fractions():
    """Rationale Gegenprobe: Deadbeat K = (1, 2) (A − BK ist nilpotent): Fenster (0, 4/3), |1 + L(−1)| = 1/4."""
    k1, k2 = F(1), F(2)
    assert min(2 / (k2 - k1), 4 / (2 * k2 - k1)) == F(4, 3) and 1 - (2 * k2 - k1) / 4 == F(1, 4)
    a, b = D.wagen(1.0, "euler")
    kd = np.array([[1.0, 2.0]])
    assert np.allclose(np.linalg.matrix_power(a - b @ kd, 2), 0.0)
    lo, hi = M.jury_window(a, b, kd)
    assert lo == 0.0 and hi == pytest.approx(4.0 / 3.0, abs=1e-12)
    assert M.min_return(a, b, kd)[0] == pytest.approx(0.25, abs=1e-9)
    assert D.is_stable(D.closed_loop(a, b, kd, 1.333)) and not D.is_stable(D.closed_loop(a, b, kd, 1.334))


def test_h3_wagen_r1_jury_formula_by_hand(wagen_r1):
    a, b, p, k = wagen_r1
    k1, k2 = k[0]
    assert k1 == pytest.approx(0.42208, abs=5e-6) and k2 == pytest.approx(1.24393, abs=5e-6)      # Handrechnung des Plans (V1)
    g_hand = min(2 / (k2 - k1), 4 / (2 * k2 - k1))                    # 2/0,82185 = 2,4336 und 4/2,06578 = 1,9363
    assert (2 / (k2 - k1)) == pytest.approx(2.4336, abs=5e-4) and (4 / (2 * k2 - k1)) == pytest.approx(1.9363, abs=5e-5)
    assert M.jury_window(a, b, k) == pytest.approx((0.0, g_hand), abs=1e-12)
    assert M.min_return(a, b, k)[0] == pytest.approx(1 - (2 * k2 - k1) / 4, abs=1e-9)       # 1 − 0,51644 = 0,4836
    assert np.allclose(sorted(np.abs(np.linalg.eigvals(a - b @ k))), [math.sqrt(0.17815358648)] * 2, atol=1e-6)   # Pole 0,378 ± 0,188 i: Betrag² = 0,17815


def test_wagen_matrices_and_forms():
    a, b = D.wagen(0.5, "euler")
    assert np.array_equal(a, [[1.0, 0.5], [0.0, 1.0]]) and np.array_equal(b, [[0.0], [0.5]])
    a2, b2 = D.wagen(0.5, "halten")
    assert np.array_equal(b2, [[0.125], [0.5]]) and np.array_equal(a2, a)                    # Ts²/2 = 0,125
    with pytest.raises(ValueError):
        D.wagen(1.0, "heun")
    with pytest.raises(ValueError):
        D.dlqr(a, b, np.eye(2), 0.0)


# --- H4: Doyle q = σ = 5 -----------------------------------------------------------------------------------------------------------------

def test_h4_doyle5_polynomial_and_window():
    g, k = M.doyle_gains(5.0, 5.0)
    assert np.allclose(g, 5.0) and np.allclose(k, 5.0)                 # f = d = 2 + √9 = 5
    for u in (F(0), F(-1, 10), F(3, 100)):
        c = np.poly(M.doyle_closed_loop(1.0 + float(u), g, k))
        exp = X.doyle5_polynomial(u)
        assert np.allclose(c[1:], [float(x) for x in exp], atol=1e-9)
    # Hurwitz: H(u) = 324 + 3600u − 2500u² (Handrechnung), Nullstelle u = (18 − 9√5)/25
    c3, c2 = F(6), F(11)
    for u in (F(0), F(-1, 10), F(3, 100)):
        c1, c0 = F(6) + 50 * u, F(1) - 25 * u
        assert c1 * (c3 * c2 - c1) - c3 ** 2 * c0 == 324 + 3600 * u - 2500 * u * u
    lo, hi = X.doyle5_window()
    assert lo == (43 - 9 * S5) / 25 and hi == X.QS(F(26, 25))
    wlo, whi = M.doyle_window_hurwitz(5.0)
    assert _close(wlo, float(lo), 1e-9) and _close(whi, 1.04, 1e-12)
    for m, ok in ((0.91, False), (0.92, True), (1.039, True), (1.041, False)):
        assert M.doyle_stable(m, g, k) == ok
    assert M.doyle_return0(5.0) == pytest.approx(1 / 26, abs=1e-15)
    assert M.doyle_lq_lower(5.0) == pytest.approx(0.4, abs=1e-15)     # A − m·B·g: tr = 2 − 5m < 0 ⇔ m > 0,4, det = 1


def test_h4_doyle_lq_alone_is_stable_exactly_above_two_over_f():
    g, _ = M.doyle_gains(5.0, 5.0)
    a, b = M.A_DOYLE, M.B_DOYLE
    for m, ok in ((0.39, False), (0.41, True), (1000.0, True)):
        assert (np.max(np.linalg.eigvals(a - m * b @ g).real) < 0) == ok


# --- H5: stetiger Gewinn, abgetastet gehalten --------------------------------------------------------------------------------------------

def test_h5_continuous_gain_held_limit_is_two_over_k2():
    ac, bc = D.wagen_stetig()
    _, kc = D.clqr(ac, bc, np.eye(2), 1.0)
    assert kc == pytest.approx(np.array([[1.0, math.sqrt(3.0)]]), abs=1e-12)        # CARE-Handwert: K = (1, √3)
    tmax = D.held_continuous_gain_ts_max(kc)
    assert tmax == pytest.approx(2 / math.sqrt(3.0), abs=1e-12)                      # Ts < 2/√3 = 1,1547
    assert D.held_continuous_gain_radius(kc, 0.999 * tmax) < 1 < D.held_continuous_gain_radius(kc, 1.001 * tmax)
    assert D.held_continuous_gain_radius(kc, 1.0) == pytest.approx(0.6115, abs=1e-4)


# --- weitere Einheiten -------------------------------------------------------------------------------------------------------------------

def test_settle_by_hand_and_unstable_and_horizon():
    assert D.settle(np.array([[0.5]]), [1.0]) == (7, 1.0)            # 0,5^6 = 0,0156 > 0,01 ≥ 0,5^7: letzter Punkt über 1 % ist k = 6, also 7 Schritte
    steps, peak = D.settle(np.array([[2.0]]), [1.0])
    assert steps is None and peak > 1e12
    steps, peak = D.settle(np.array([[1.0]]), [1.0], horizon=50)     # klingt nie ab
    assert steps is None and peak == 1.0


def test_trajectory_and_closed_loop(wagen_r1):
    a, b, p, k = wagen_r1
    traj = D.trajectory(D.closed_loop(a, b, k), [1.0, 0.0], 3)
    assert traj.shape == (4, 2) and np.array_equal(traj[0], [1.0, 0.0])
    assert traj[1] == pytest.approx([1.0, -k[0, 0]])                 # x₁⁺ = x₁ + x₂ = 1, x₂⁺ = x₂ − K·x = −k₁
    assert np.array_equal(D.closed_loop(a, b, k, 0.0), a) and D.spectral_radius(D.closed_loop(a, b, k)) == pytest.approx(0.42208, abs=5e-6)
