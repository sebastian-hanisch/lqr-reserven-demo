"""Kern 1: die Strecken (Wagen, Skalar), der LQR-Gewinn (DARE und CARE) und der Zeitbereich.

Konvention überall: u = −K·x (Faktor ½ in den Kosten nicht benutzt), Zustandsgleichung x⁺ = A·x + B·u, Kosten Σ xᵀQx + uᵀRu mit Q = I und einem Eingang (R eine Zahl).
Der Wagen ist der Doppelintegrator mit Abtastzeit Ts: A = [[1, Ts], [0, 1]]; „Euler-Form“ B = (0, Ts)ᵀ (wie in den anderen Regelungs-Demos), „Halte-Form“ B = (Ts²/2, Ts)ᵀ (exakte
Diskretisierung bei konstanter Stellgröße). Kosten Ts·(Q, R) und (Q, R) geben denselben Gewinn: Q und R werden gemeinsam skaliert, K ändert sich nicht.

Herkunft: neu geschrieben; nichts aus anderen Demos importiert. Die DARE löst `scipy.linalg.solve_discrete_are`; die eigene
Riccati-Rekursion (`riccati_iteration`) ist der zweite Rechenweg für die Orakel-Tests und die Anzeige der Schrittzahl."""

import numpy as np
import scipy.linalg as la

STABLE_EPS = 1e-12      # „stabil“ heißt Spektralradius < 1 − 1e-12 (nie == 1: ρ = 1 ± 1e-9 ist Rundungsrauschen)
FORMS = ("euler", "halten")


def wagen(ts: float = 1.0, form: str = "euler"):
    """Doppelintegrator abgetastet: (A, B). form = 'euler' (B = (0, Ts)ᵀ) oder 'halten' (B = (Ts²/2, Ts)ᵀ)."""
    if form not in FORMS:
        raise ValueError(f"Form von B muss eine von {FORMS} sein")
    a = np.array([[1.0, ts], [0.0, 1.0]])
    b = np.array([[0.0], [ts]]) if form == "euler" else np.array([[ts * ts / 2.0], [ts]])
    return a, b


def wagen_stetig():
    """Doppelintegrator in der stetigen Zeit: (Ac, Bc)."""
    return np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]])


def skalar(a: float):
    """Skalar x⁺ = a·x + u: (A, B) als 1×1-Matrizen."""
    return np.array([[float(a)]]), np.array([[1.0]])


def _rm(r: float) -> np.ndarray:
    if not r > 0:
        raise ValueError("R muss positiv sein")
    return np.array([[float(r)]])


def dlqr(a, b, q, r):
    """Diskreter LQR: (P, K) mit K = (R + BᵀPB)⁻¹·BᵀPA und der DARE-Lösung P (SciPy)."""
    rm = _rm(r)
    p = la.solve_discrete_are(a, b, q, rm)
    k = np.linalg.solve(rm + b.T @ p @ b, b.T @ p @ a)
    return p, k


def riccati_iteration(a, b, q, r, p0=None, tol: float = 1e-12, max_iter: int = 50000):
    """Eigener zweiter Weg zur DARE: Riccati-Rekursion von P₀ = Q (oder `p0`) aus; gibt (P, Schritte) zurück, Schritte = Zahl der Rekursionsschritte bis zur Toleranz."""
    rm = _rm(r)
    p = q.copy() if p0 is None else p0.copy()
    for step in range(1, max_iter + 1):
        pb = p @ b
        nxt = q + a.T @ p @ a - a.T @ pb @ np.linalg.solve(rm + b.T @ pb, pb.T @ a)
        nxt = 0.5 * (nxt + nxt.T)          # Symmetrie erzwingen: Rundungsreste würden sonst bei instabilem A aufschaukeln
        if np.max(np.abs(nxt - p)) <= tol * max(1.0, np.max(np.abs(nxt))):
            return nxt, step
        p = nxt
    raise RuntimeError("Riccati-Rekursion ist nicht konvergiert")


def clqr(ac, bc, q, r):
    """Stetiger LQR: (P, K) mit K = R⁻¹·BᵀP und der CARE-Lösung P."""
    rm = _rm(r)
    p = la.solve_continuous_are(ac, bc, q, rm)
    return p, (bc.T @ p) / float(r)


def closed_loop(a, b, k, g: float = 1.0):
    """Regelkreismatrix A − g·B·K; g ist der Faktor auf die Streckenverstärkung (Modellfehler: die Strecke wirkt g-mal so stark wie modelliert)."""
    return a - g * b @ k


def spectral_radius(m) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(m))))


def is_stable(m) -> bool:
    return spectral_radius(m) < 1.0 - STABLE_EPS


def trajectory(m, x0, steps: int) -> np.ndarray:
    """Freie Bahn x_{k+1} = M·x_k, Form (steps + 1, n); Zustand 0 ist x0."""
    x = np.asarray(x0, dtype=float).copy()
    out = np.empty((steps + 1, len(x)))
    out[0] = x
    for i in range(steps):
        x = m @ x
        out[i + 1] = x
    return out


def settle(m, x0, horizon: int = 20000, drop: float = 0.01):
    """(Schritte bis ‖x‖ dauerhaft unter `drop`·Spitze, Spitze): beobachtet bis `horizon`; None, wenn der Betrag nicht dauerhaft darunter fällt
    (instabil oder bis zum Horizont nicht abgeklungen) oder der Zustand über 1e12 wächst. Norm über die ersten beiden Komponenten (die Strecke)."""
    x = np.asarray(x0, dtype=float).copy()
    norms = np.empty(horizon)
    for i in range(horizon):
        norms[i] = np.linalg.norm(x[:2])
        x = m @ x
        if not np.all(np.isfinite(x)) or norms[i] > 1e12:
            return None, float(np.max(norms[: i + 1]))
    peak = float(norms.max())
    above = np.nonzero(norms > drop * peak)[0]
    last = int(above.max())
    return (last + 1 if last < horizon - 1 else None), peak


def held_gain_matrices(ts: float):
    """Exakte Halte-Diskretisierung des stetigen Wagens (Ad, Bd) bei Abtastzeit Ts, für den abgetastet gehaltenen stetigen Gewinn."""
    return wagen(ts, "halten")


def held_continuous_gain_radius(kc, ts: float) -> float:
    """Spektralradius des stetigen Gewinns Kc am Wagen, wenn er mit Abtastzeit Ts gehalten wird (exakte Halte-Diskretisierung)."""
    ad, bd = held_gain_matrices(ts)
    return spectral_radius(ad - bd @ kc)


def held_continuous_gain_ts_max(kc) -> float:
    """Größte Abtastzeit, bis zu der der gehaltene stetige Gewinn am Wagen stabil bleibt: Ts < 2/k₂ (Jury: P(−1) = 4 − 2·k₂·Ts)."""
    return 2.0 / float(kc[0, 1])
