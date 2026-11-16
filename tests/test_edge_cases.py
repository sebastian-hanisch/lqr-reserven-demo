"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): extreme R und Ts, Fensterkanten (nur als Band), q = σ = 10⁶, unzulässige Rauscheingänge, entartete Skalare.

Grenzen von Fenstern gelten offen (am Rand ist ρ = 1): getestet wird immer mit 1 ± 1e-6 relativ, nie mit „==“ (ρ = 1 ± 1e-9 ist Rundungsrauschen)."""

import math

import numpy as np
import pytest

import res_dynamics as D
import res_evaluation as ev
import res_margins as M

I2 = np.eye(2)


def _wagen(r, ts=1.0, form="euler"):
    a, b = D.wagen(ts, form)
    p, k = D.dlqr(a, b, I2, r)
    return a, b, p, k


def test_very_small_r_converges_and_the_reserve_stays_near_the_limit_of_the_series():
    """R = 10⁻⁶ und 10⁻⁴: K nähert sich (0,618; 1,618), der Abstand ändert sich kaum (0,3458 bei R = 10⁻³); ein Grenzwert ist damit nicht bewiesen."""
    for r in (1e-4, 1e-6):
        a, b, p, k = _wagen(r)
        assert k[0] == pytest.approx([0.618, 1.618], abs=1e-3)
        assert M.min_return(a, b, k)[0] == pytest.approx(0.3455, abs=2e-4)
        assert M.jury_window(a, b, k)[1] == pytest.approx(1.5279, abs=2e-4)


def test_very_large_r_gives_a_weak_gain_and_a_reserve_close_to_but_below_one():
    a, b, p, k = _wagen(1e6)
    assert np.max(np.abs(k)) < 0.05
    mn, wm = M.min_return(a, b, k)
    assert 0.97 < mn < 1.0 and mn == pytest.approx(1 - (2 * k[0, 1] - k[0, 0]) / 4, abs=1e-6)     # Tiefstwert 1 − (2k₂ − k₁)/4
    assert M.jury_window(a, b, k)[1] > 40


def test_ts_towards_zero_the_reserve_approaches_one_from_below():
    values = []
    for ts in (0.01, 0.0025, 0.001):
        a, b, p, k = _wagen(1.0, ts)
        values.append(M.min_return(a, b, k, n=100001)[0])
    assert values == sorted(values) and 0.99 < values[0] and values[-1] < 1.0


def test_window_edge_as_a_band_never_exact():
    a, b, p, k = _wagen(1.0)
    hi = M.jury_window(a, b, k)[1]
    assert D.spectral_radius(D.closed_loop(a, b, k, hi * (1 - 1e-6))) < 1 < D.spectral_radius(D.closed_loop(a, b, k, hi * (1 + 1e-6)))
    assert abs(D.spectral_radius(D.closed_loop(a, b, k, hi)) - 1.0) < 1e-9          # auf der Kante selbst: Rundungsrauschen um 1


def test_doyle_with_q_sigma_one_million_has_a_window_one_millionth_wide():
    lo, hi = M.doyle_window_hurwitz(1e6)
    assert 0.9989 < lo < 0.9991 and 1.0 < hi < 1.00001
    g, k = M.doyle_gains(1e6, 1e6)
    assert M.doyle_stable(1.0, g, k) and not M.doyle_stable(0.99, g, k) and not M.doyle_stable(1.01, g, k)
    assert M.doyle_stable(0.5 * (lo + hi), g, k)


def test_noise_direction_g_equal_to_one_zero_is_rejected_and_w_zero_too():
    a, b = D.wagen(1.0, "euler")
    _, k = D.dlqr(a, b, I2, 1.0)
    with pytest.raises(ValueError):
        M.lqg_analyse(a, b, k, np.array([[1.0], [0.0]]), 10.0, "schaetz")
    with pytest.raises(ValueError):
        M.lqg_analyse(a, b, k, b.copy(), 0.0, "schaetz")
    with pytest.raises(ValueError):
        M.lqg_controller("heun", a, b, k, np.eye(2))


def test_lqg_window_is_none_when_the_loop_is_unstable_at_m_equal_one():
    a, b = D.wagen(1.0, "euler")
    pp = M.lqg_predictor_covariance(a, b, 10.0)
    ctrl = M.lqg_controller("schaetz", a, b, np.zeros((1, 2)), pp)      # K = 0: der Regelkreis ist die offene Strecke mit ρ = 1
    m0, m1 = M.lqg_matrices(a, b, ctrl)
    assert M.lqg_window(m0, m1) is None


def test_scalar_stable_plant_has_no_crossing_and_the_window_starts_at_zero():
    """a = 0,5: |L| ≤ K/(1 − a) = 0,53 < 1, also keine Durchtrittsstelle und keine Phasenreserve; das Fenster ist (0, (1 + a)/K)."""
    a, b = D.skalar(0.5)
    p, k = D.dlqr(a, b, np.array([[1.0]]), 1.0)
    assert M.phase_margins(a, b, k) == [] and M.worst_phase_margin([]) == (None, None)
    lo, hi = M.window(a, b, k)
    assert lo == 0.0 and hi == pytest.approx(1.5 / k[0, 0], rel=1e-12)
    assert abs(k[0, 0]) / (1 - 0.5) < 1


def test_scalar_a_minus_two_mirrors_a_plus_two():
    """a → −a: K → −K und L(z) → L(−z): derselbe Abstand und dasselbe Fenster."""
    res = {}
    for a_val in (2.0, -2.0):
        a, b = D.skalar(a_val)
        p, k = D.dlqr(a, b, np.array([[1.0]]), 1.0)
        res[a_val] = (k[0, 0], M.min_return(a, b, k)[0], M.window(a, b, k))
    assert res[-2.0][0] == pytest.approx(-res[2.0][0], rel=1e-12)
    assert res[-2.0][1] == pytest.approx(res[2.0][1], abs=1e-9)
    assert res[-2.0][2] == pytest.approx(res[2.0][2], rel=1e-9)


def test_degenerate_windows():
    assert M.scalar_window(0.5, 0.0) == (0.0, math.inf) and M.scalar_window(2.0, 0.0) is None
    a, b = D.wagen(1.0, "euler")
    assert M.jury_window(a, b, np.array([[-1.0, -1.0]])) is None        # Rückführung mit falschem Vorzeichen: für kein g > 0 stabil
    with pytest.raises(ValueError):
        M.jury_window(np.eye(3), np.ones((3, 1)), np.ones((1, 3)))
    with pytest.raises(ValueError):
        M.g_upper_bisect(a, b, np.zeros((1, 2)))                        # g = 1 nicht stabil


def test_bound_gain_interval_edges():
    assert M.gain_interval_from_bound(0.5) == (pytest.approx(2 / 3), pytest.approx(2.0))
    assert M.gain_interval_from_bound(1.0)[1] == math.inf


def test_formatting_helpers_have_german_decimal_commas():
    assert ev.de(0.4836, 3) == "0,484" and ev.de(-1.5, 1) == "−1,5" and ev.de(math.inf) == "∞" and ev.de(0.3, 1, sign=True) == "+0,3"
    assert ev.de_g(0.001) == "0,001" and ev.de_g(100.0) == "100" and ev.fmt_int(9161) == "9 161"
    assert ev.pow10(10000.0) == "10⁴" and ev.pow10(1e6) == "10⁶" and ev.pow10(100.0) == "100" and ev.pow10(5.0) == "5" and ev.pow10(1e8) == "10⁸"
    assert ev.omega_text(math.pi) == "π" and ev.omega_text(1.0) == "1,000"


def test_scientific_notation_helper_has_decimal_comma_and_superscripts():
    assert ev.sci(4.2e-15) == "4,2·10⁻¹⁵" and ev.sci(0.0) == "0" and ev.sci(2.9e-6) == "2,9·10⁻⁶" and ev.sci(1.0e6, 0) == "1·10⁶"
