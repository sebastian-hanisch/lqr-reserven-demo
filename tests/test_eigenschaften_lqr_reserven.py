"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2): Aussagen, die ohne bekanntes Ergebnis gelten müssen.

Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed (SplitMix64), plus die Randfälle. Jede Eigenschaft nennt ihre Voraussetzung: Was nur am Wagen gemessen ist
(R wachsend, Abtastzeit-Reihe, Tiefstwert bei ω = π), wird auch nur dort geprüft und steht in der README als „am Wagen“."""

import math

import numpy as np
import pytest

import res_dynamics as D
import res_margins as M
from res_rng import SplitMix64

I2 = np.eye(2)


def _systems(seed, count, max_rho=2.5):
    rng = SplitMix64(seed)
    out = []
    while len(out) < count:
        a, b, q, r = M.random_system(rng)
        if M._controllable(a, b) and np.max(np.abs(np.linalg.eigvals(a))) < max_rho:
            out.append((a, b, q, r, rng))
    return out


def test_scaling_q_and_r_together_changes_neither_gain_nor_reserve():
    for a, b, q, r, rng in _systems(21, 25):
        c = 0.1 + 20.0 * rng.random()
        p1, k1 = D.dlqr(a, b, q, r)
        p2, k2 = D.dlqr(a, b, c * q, c * r)
        assert np.allclose(k1, k2, atol=1e-8 * max(1.0, np.max(np.abs(k1))))
        assert M.bound_r0(b, p1, r) == pytest.approx(M.bound_r0(b, p2, c * r), rel=1e-8)    # r₀ hängt nur am Verhältnis, nicht an der Skala
        assert M.min_return(a, b, k1, n=2001)[0] == pytest.approx(M.min_return(a, b, k2, n=2001)[0], rel=1e-7)


def test_scaling_input_matrix_with_inverse_gain_leaves_the_loop_unchanged():
    """Algebraisch, ohne DARE: B → c·B und K → K/c ändern L(z) = K(zI − A)⁻¹B nicht."""
    for a, b, q, r, rng in _systems(22, 25):
        k = np.array([[rng.normal() for _ in range(a.shape[0])]])
        c = 0.2 + 5.0 * rng.random()
        z = np.exp(1j * np.array([math.pi * rng.random() for _ in range(10)]))
        assert np.allclose(M.L_of(z, a, b, k), M.L_of(z, a, c * b, k / c), rtol=1e-9, atol=1e-12)


def test_similarity_transformation_leaves_the_return_difference_unchanged():
    for a, b, q, r, rng in _systems(23, 25):
        n = a.shape[0]
        t = np.array([[rng.normal() for _ in range(n)] for _ in range(n)]) + 3.0 * np.eye(n)
        if np.linalg.cond(t) > 1e3:
            continue
        p, k = D.dlqr(a, b, q, r)
        ti = np.linalg.inv(t)
        z = np.exp(1j * np.array([math.pi * rng.random() for _ in range(10)]))
        assert np.allclose(M.L_of(z, a, b, k), M.L_of(z, t @ a @ ti, t @ b, k @ ti), rtol=1e-7, atol=1e-9)


def test_bound_holds_on_random_systems_with_eigenvalues_inside_and_outside_the_unit_circle():
    """Gegenprobe für „die Schranke gilt immer“ mit gelockerten Annahmen: A beliebig (stabil und instabil), R über vier Zehnerpotenzen; nur Eingang 1, steuerbar."""
    inside = outside = 0
    for a, b, q, r, rng in _systems(24, 50, max_rho=3.0):
        p, k = D.dlqr(a, b, q, r)
        if M.dare_residual(a, b, q, p, k) > 1e-8:
            continue
        mn, _ = M.min_return(a, b, k, n=4001)
        assert mn >= M.bound_r0(b, p, r) - 1e-9
        if np.max(np.abs(np.linalg.eigvals(a))) < 1.0:
            inside += 1
        else:
            outside += 1
    assert inside >= 5 and outside >= 5


def test_continuous_lqr_return_difference_is_at_least_one_on_random_systems():
    """Stetig: min |1 + L(jω)| ≥ 1 − 1e-9 (Gitter 10⁻³ … 10³): zusätzliche Eigenschaft der stetigen Rechnung, Eingang 1, steuerbar."""
    w = np.logspace(-3, 3, 2001)
    checked = 0
    for a, b, q, r, rng in _systems(25, 30):
        try:
            _, kc = D.clqr(a, b, q, r)
        except (np.linalg.LinAlgError, ValueError):
            continue
        assert np.min(np.abs(1 + M.L_of(1j * w, a, b, kc))) >= 1 - 1e-9
        checked += 1
    assert checked >= 25


@pytest.mark.parametrize("form", ["euler", "halten"])
def test_wagen_reserve_does_not_shrink_when_r_grows(form):
    """Nur am Wagen (Ts = 1, Q = I) belegt, nicht allgemein: kleineres R heißt weniger Reserve."""
    values = []
    for r in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        a, b = D.wagen(1.0, form)
        _, k = D.dlqr(a, b, I2, r)
        values.append(M.min_return(a, b, k, n=4001)[0])
    assert all(x <= y + 1e-12 for x, y in zip(values, values[1:]))
    assert values[0] < values[-1]


@pytest.mark.parametrize("form", ["euler", "halten"])
def test_wagen_reserve_grows_when_ts_shrinks_on_the_measured_series(form):
    """Nur die gemessene Reihe Ts = 1 … 0,01 (R = 1, Q = I), kein Satz für alle Ts."""
    values = []
    for ts in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
        a, b = D.wagen(ts, form)
        _, k = D.dlqr(a, b, I2, 1.0)
        values.append(M.min_return(a, b, k, n=20001)[0])
    assert all(x < y for x, y in zip(values, values[1:])) and values[-1] < 1.0


def test_upper_gain_equals_inverse_of_one_minus_min_return_at_the_wagen():
    """Aussage nur am Wagen (Euler und Halten, R = 1, Ts = 1 … 0,01): g_oben = 1/(1 − min|1 + L|), weil der Tiefstwert bei ω ≈ π liegt (bei Ts = 0,01 geprüft)."""
    for form in ("euler", "halten"):
        for ts in (1.0, 0.5, 0.1, 0.01):
            a, b = D.wagen(ts, form)
            _, k = D.dlqr(a, b, I2, 1.0)
            mn, wm = M.min_return(a, b, k, n=20001)
            _, hi = M.jury_window(a, b, k)
            assert 1 / (1 - mn) == pytest.approx(hi, rel=1e-6), (form, ts)
            assert abs(wm - math.pi) < 1e-3, (form, ts)


def test_same_zufalls_seed_gives_the_same_systems_and_counts():
    first = M.bound_check(77, 40)
    second = M.bound_check(77, 40)
    assert first == second
    assert M.bound_check(78, 40)["tightest_ratio"] != first["tightest_ratio"]
    r1, r2 = SplitMix64(5), SplitMix64(5)
    a1, a2 = M.random_system(r1), M.random_system(r2)
    assert all(np.array_equal(x, y) for x, y in zip(a1[:3], a2[:3])) and a1[3] == a2[3]


def test_lqg_distance_falls_with_growing_noise_strength_on_the_measured_configuration():
    """Nur die gemessene Konfiguration (Ts = 1, Euler, Q = I, R = 1, G = B, Schätzform), kein Satz: der Abstand fällt mit wachsendem W (0,01 … 10⁴); bei kleinem W liegt er
    sogar über dem des Zustands-LQR (LQG kann besser oder schlechter sein), ab W = 1 darunter."""
    a, b = D.wagen(1.0, "euler")
    _, k = D.dlqr(a, b, I2, 1.0)
    base = M.min_return(a, b, k)[0]
    values = [M.lqg_analyse(a, b, k, b.copy(), w, "schaetz", n_freq=4001)["min_return"] for w in (0.01, 1.0, 100.0, 1e4)]
    assert all(x > y for x, y in zip(values, values[1:]))
    assert values[0] > base and all(v < base for v in values[1:])
