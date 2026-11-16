"""Orakel-Tests (unabhängiger Rechenweg, ORACLE-PLAYBOOK §5, DEMO-PLAYBOOK §12) für den Kern der Reserven.

Jede Kernroutine wird auf einem anderen Weg nachgerechnet:
  * DARE: eigene Riccati-Rekursion gegen `scipy.linalg.solve_discrete_are`; stetig: geschlossene Form für den Doppelintegrator und das CARE-Residuum;
  * Fenster nach oben: Jury (Formeln) gegen Bisektion auf dem Spektralradius gegen den Nyquist-Schnitt −1/L(−1);
  * Nyquist-Abstand: Gitter mit Verfeinerung gegen 1 − 1/g_oben am Wagen und gegen die Rückführdifferenz-Gleichung (aus P, Q, nicht aus K);
  * Zeitbereich: Sinus-Störung am Streckeneingang, stationäre Amplitude per Kleinste Quadrate gegen 1/|1 + L|;
  * LQG: Frequenzgang gegen den Determinantenquotienten, Trennungsprinzip (Pole);
  * Doyle: np.poly des Regelkreises gegen die Formeln, Hurwitz-Fenster gegen Eigenwert-Fenster.
Plattform: Bisektions- und Gittergrößen als Band (1e-6 bis 1e-9), Zählungen nie exakt an einer Schwelle. Die Handwerte stehen in tests/test_dynamics.py."""

import math

import numpy as np
import pytest

import res_dynamics as D
import res_margins as M
from res_rng import SplitMix64

I2 = np.eye(2)
R_SET = (1.0, 0.1, 0.01, 0.001)


def _wagen(r, ts=1.0, form="euler"):
    a, b = D.wagen(ts, form)
    p, k = D.dlqr(a, b, I2, r)
    return a, b, p, k


# --- DARE und CARE ----------------------------------------------------------------------------------------------------------------------

def test_dare_riccati_iteration_agrees_with_scipy_on_the_wagen():
    for r in R_SET:
        a, b = D.wagen(1.0, "euler")
        p_it, steps = D.riccati_iteration(a, b, I2, r)
        p_sp, _ = D.dlqr(a, b, I2, r)
        assert np.max(np.abs(p_it - p_sp)) < 1e-9 * np.max(np.abs(p_sp)), r
        assert steps > 1


def test_dare_riccati_iteration_agrees_with_scipy_on_random_systems():
    rng = SplitMix64(4)
    checked = 0
    for _ in range(25):
        a, b, q, r = M.random_system(rng)
        if not M._controllable(a, b) or np.max(np.abs(np.linalg.eigvals(a))) > 1.5:
            continue                                                   # die Rekursion braucht bei stark instabilem A sehr viele Schritte
        p_it, _ = D.riccati_iteration(a, b, q, r)
        p_sp, k = D.dlqr(a, b, q, r)
        assert np.max(np.abs(p_it - p_sp)) < 1e-8 * max(1.0, np.max(np.abs(p_sp)))
        checked += 1
    assert checked >= 12


def test_continuous_lqr_double_integrator_closed_form_and_care_residual():
    ac, bc = D.wagen_stetig()
    for r in (1.0, 0.1, 0.01):
        pc, kc = D.clqr(ac, bc, I2, r)
        # geschlossene Form (hier hergeleitet): k₁ = 1/√R, k₂ = √(1/R + 2/√R)
        assert kc[0, 0] == pytest.approx(1 / math.sqrt(r), rel=1e-9) and kc[0, 1] == pytest.approx(math.sqrt(1 / r + 2 / math.sqrt(r)), rel=1e-9)
        res = ac.T @ pc + pc @ ac - pc @ bc @ bc.T @ pc / r + I2
        assert np.max(np.abs(res)) < 1e-9 * max(1.0, np.max(np.abs(pc)))


# --- Fenster nach oben: drei Wege ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ts,form", [(1.0, "euler"), (0.5, "halten"), (0.1, "euler")])
def test_window_upper_edge_three_ways_agree(ts, form):
    for r in R_SET:
        a, b, p, k = _wagen(r, ts, form)
        lo, hi = M.jury_window(a, b, k)
        assert lo == 0.0
        assert abs(hi - M.g_upper_bisect(a, b, k)) < 1e-9 * hi
        assert abs(hi - M.g_nyquist_pi(a, b, k)) < 1e-9 * hi
        assert M.g_lower_bisect(a, b, k) < 1e-9                         # nach unten: jedes g > 0 (die Bisektion endet an der Genauigkeit des Stabilitätstests, etwa 1e-11)


def test_window_edges_by_direct_stability_probe():
    a, b, p, k = _wagen(0.1)
    _, hi = M.jury_window(a, b, k)
    assert D.is_stable(D.closed_loop(a, b, k, hi * (1 - 1e-7))) and not D.is_stable(D.closed_loop(a, b, k, hi * (1 + 1e-7)))
    assert D.is_stable(D.closed_loop(a, b, k, 1e-4))


# --- Nyquist-Abstand ---------------------------------------------------------------------------------------------------------------------

def test_min_return_equals_one_minus_inverse_upper_gain_at_pi():
    for r in R_SET:
        a, b, p, k = _wagen(r)
        mn, wm = M.min_return(a, b, k)
        _, hi = M.jury_window(a, b, k)
        assert mn == pytest.approx(1 - 1 / hi, abs=1e-9)
        assert wm == pytest.approx(math.pi, abs=1e-5)                    # am Wagen mit Ts = 1: Tiefstwert bei der Nyquist-Frequenz


def test_min_return_from_the_return_difference_equation_independent_of_k():
    """Zweiter Weg: |1 + L|² = (R + ‖Φ·B‖²_Q)/(R + BᵀPB) mit Φ = (zI − A)⁻¹, nur aus A, B, Q, R und P (nicht aus K)."""
    for r in R_SET:
        a, b, p, k = _wagen(r)
        w = np.linspace(1e-6, math.pi, 40001)
        z = np.exp(1j * w)
        phi_b = np.linalg.solve(z[:, None, None] * I2[None] - a[None], np.broadcast_to(b, (len(z), 2, 1)))[:, :, 0]
        num = r + np.real(np.einsum("ni,ij,nj->n", phi_b.conj(), I2, phi_b))
        den = r + float((b.T @ p @ b)[0, 0])
        via_equation = float(np.sqrt(np.min(num / den)))
        assert via_equation == pytest.approx(M.min_return(a, b, k)[0], abs=1e-8)


def test_return_difference_equation_on_random_systems():
    rng = SplitMix64(41)
    n_ok = 0
    for _ in range(30):
        a, b, q, r = M.random_system(rng)
        if not M._controllable(a, b):
            continue
        p, k = D.dlqr(a, b, q, r)
        if M.dare_residual(a, b, q, p, k) > 1e-10:
            continue
        w = np.array([math.pi * rng.random() for _ in range(25)])
        assert M.return_difference_residual(a, b, q, p, k, r, np.exp(1j * w)) < 1e-8
        assert M.det_relation_residual(a, b, p, k, r) < 1e-8
        n_ok += 1
    assert n_ok >= 20


# --- Zeitbereich: Sinus-Störung ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("r", [1.0, 0.1])
def test_sine_disturbance_amplitude_equals_inverse_return_difference(r):
    a, b, p, k = _wagen(r)
    for w0 in (math.pi - 1e-2, 0.3, 1.0, 2.5):
        x = np.zeros((2, 1))
        n = 4000
        s = np.empty(n)
        for i in range(n):
            d = math.sin(w0 * i)
            u = -(k @ x)[0, 0]
            s[i] = u + d                                               # Streckeneingang
            x = a @ x + b * (u + d)
        t = np.arange(n - 1500, n)
        design = np.column_stack([np.sin(w0 * t), np.cos(w0 * t)])
        coef, *_ = np.linalg.lstsq(design, s[t], rcond=None)
        expected = 1 / abs(1 + M.L_of(np.exp(1j * w0), a, b, k)[0])
        assert np.hypot(*coef) == pytest.approx(expected, abs=1e-9)


# --- LQG ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("form,gkey,w", [("schaetz", "b", 1e4), ("vorhersage", "b", 1.0), ("schaetz", "m3", 100.0), ("vorhersage", "m1", 0.01)])
def test_lqg_frequency_response_agrees_with_determinant_quotient_and_separation_holds(form, gkey, w):
    a, b = D.wagen(1.0, "euler")
    _, k = D.dlqr(a, b, I2, 1.0)
    g = b.copy() if gkey == "b" else np.array({"m1": [[1.0], [-1.0]], "m3": [[1.0], [-3.0]]}[gkey])
    res = M.lqg_analyse(a, b, k, g, w, form, n_freq=4001)
    assert res["route2_deviation"] < 1e-5                                # gemessen bis 7e-6 (schlecht konditioniert bei großem W)
    assert res["separation_deviation"] < 1e-6


def test_lqg_separation_principle_poles_directly():
    a, b = D.wagen(1.0, "euler")
    _, k = D.dlqr(a, b, I2, 1.0)
    pp = M.lqg_predictor_covariance(a, b, 100.0)
    s = float((M.C_POS @ pp @ M.C_POS.T)[0, 0]) + 1.0
    lp = a @ pp @ M.C_POS.T / s
    for form in M.LQG_FORMS:
        m0, m1 = M.lqg_matrices(a, b, M.lqg_controller(form, a, b, k, pp))
        poles = np.linalg.eigvals(m0 + m1)
        expected = np.concatenate([np.linalg.eigvals(a - b @ k), np.linalg.eigvals(a - lp @ M.C_POS)])
        assert M.eig_distance(poles, expected) < 1e-6
    assert M.eig_distance([1.0, 2.0], [1.0, 2.5]) == pytest.approx(0.5)  # die Abstandsfunktion selbst, von Hand


def test_lqg_rejects_unstabilizable_noise_direction_and_nonpositive_w():
    a, b = D.wagen(1.0, "euler")
    with pytest.raises(ValueError):
        M.lqg_predictor_covariance(a, b, 0.0)
    with pytest.raises(ValueError):
        M.lqg_predictor_covariance(a, np.array([[1.0], [0.0]]), 1.0)     # G = (1, 0)ᵀ: (A, G) nicht stabilisierbar, die DARE hat keine stabilisierende Lösung


# --- Doyle --------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [5.0, 12.0, 100.0])
def test_doyle_polynomial_coefficients_agree_with_the_formulas(q):
    g, k = M.doyle_gains(q, q)
    f = M.doyle_f(q)
    assert np.allclose(g, f, atol=1e-9) and np.allclose(k, f, atol=1e-9)
    for m in (0.5, 0.9, 1.0, 1.003, 1.5, 3.0):
        c = np.poly(M.doyle_closed_loop(m, g, k))
        assert c[3] == pytest.approx(2 * f - 4 + 2 * (m - 1) * f * f, rel=1e-9, abs=1e-9)
        assert c[4] == pytest.approx(1 + (1 - m) * f * f, rel=1e-9, abs=1e-9)
        assert c[1] == pytest.approx(2 * (f - 2), abs=1e-9) and c[2] == pytest.approx(q + 6, abs=1e-9)    # c₃, c₂: Beobachtung für diese q, nicht allgemein bewiesen


@pytest.mark.parametrize("q", [1.0, 5.0, 12.0, 100.0, 1596.0, 10000.0, 1000000.0])
def test_doyle_hurwitz_window_agrees_with_eigenvalue_window(q):
    lo_h, hi_h = M.doyle_window_hurwitz(q)
    lo_e, hi_e, changes = M.doyle_window_eigen(q)
    width = hi_h - lo_h
    assert abs(lo_h - lo_e) < 1e-6 * width and abs(hi_h - hi_e) < 1e-6 * width
    assert changes == 2                                                  # auf dem Gitter genau ein stabiles Fenster
    assert hi_h == pytest.approx(1 + 1 / M.doyle_f(q) ** 2, rel=1e-12)   # obere Kante exakt 1 + 1/(d·f)


@pytest.mark.parametrize("q", [1.0, 5.0, 100.0, 1596.0])
def test_doyle_return0_two_ways(q):
    w = np.array([1e-3, 0.1, 1.0, 10.0])
    direct = np.abs(1 + M.doyle_L(w, q))
    quotient = np.abs(M.doyle_return_det_quotient(w, q))
    assert np.max(np.abs(direct - quotient)) < 1e-9
    l0 = M.doyle_L(np.array([1e-9]), q)[0]
    assert abs(1 + l0) == pytest.approx(M.doyle_return0(q), rel=1e-5)    # bei ω → 0: 1/(1 + d·f)
