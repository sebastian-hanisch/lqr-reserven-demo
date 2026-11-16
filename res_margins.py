"""Kern 2: Reserven im Frequenz- und Verstärkungsbereich, die Schranke, Doyles Gegenbeispiel und der abgetastete Wagen mit Schätzer (LQG).

Drei Bedeutungen von „Reserve“, immer mit Namen:
  * **Nyquist-Abstand** min |1 + L(e^{jω})| (Schleife am Streckeneingang geöffnet, L(z) = K·(zI − A)⁻¹·B; stetig L(jω) = K·(jωI − A)⁻¹·B),
  * **Verstärkungsfenster** (g_unten, g_oben): Bereich des Faktors g in u = −g·K·x, in dem die Schleife stabil bleibt (g = Streckenverstärkung des Modellfehlers),
  * **Phasenreserve** 180° + arg L bei |L| = 1 (kleinste Durchtrittsstelle) und **Verzugsreserve** = Phasenreserve / Durchtrittsfrequenz, in Perioden.

Herkunft: neu geschrieben nach Plan 04 der Linie „Regelung“ (die Wegwerf-Skripte `rg_v13_*` der Vormessung wurden gelesen, nicht kopiert). Die Doyle-Parameter stammen
aus dem Quellenbericht der Planung (Stufe W: Doyle-Scan als Bild gelesen); hier wird nur nachgerechnet. Eigene Herleitung, geprüft in diesem Modell (keine Neuheitsaussage):
Rückführdifferenz-Gleichung (R + BᵀPB)·|1 + L|² = R + ‖Q^{1/2}(zI − A)⁻¹B‖² und die Schranke min |1 + L| ≥ √(R/(R + BᵀPB)) für einen Eingang."""

import math

import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq, minimize_scalar

import res_dynamics as D
from res_rng import SplitMix64

INF = float("inf")


# --- Frequenzgang ----------------------------------------------------------------------------------------------------------------------

def L_of(z, a, b, k):
    """Schleifenverstärkung K·(zI − A)⁻¹·B an den Stellen z (Feld, auch stetig mit z = jω), Schleife am Streckeneingang geöffnet; ein Eingang."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    n = a.shape[0]
    m = z[:, None, None] * np.eye(n)[None] - a[None]
    x = np.linalg.solve(m, np.broadcast_to(b, (len(z), n, b.shape[1])))
    return np.einsum("ij,njk->nik", k, x)[:, 0, 0]


def min_return(a, b, k, n: int = 20001):
    """(min |1 + L(e^{jω})|, Ort ω) über ω in (0, π]: Gitter, Verfeinerung um den kleinsten Gitterpunkt und der Randpunkt ω = π (Nyquist-Frequenz) einzeln."""
    w = np.linspace(1e-6, math.pi, n)
    v = np.abs(1.0 + L_of(np.exp(1j * w), a, b, k))
    i = int(np.argmin(v))
    cands = [(float(v[i]), float(w[i]))]
    lo, hi = float(w[max(i - 1, 0)]), float(w[min(i + 1, n - 1)])
    res = minimize_scalar(lambda x: abs(1.0 + L_of(np.exp(1j * x), a, b, k)[0]), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
    cands.append((float(res.fun), float(res.x)))
    cands.append((float(abs(1.0 + L_of(np.array([-1.0 + 0j]), a, b, k)[0])), math.pi))
    return min(cands)


def sensitivity_peak(min_ret: float) -> float:
    """Empfindlichkeitsspitze max |S| = 1 / min |1 + L|."""
    return 1.0 / min_ret


def phase_margins(a, b, k, n: int = 20001, stetig: bool = False):
    """Alle Durchtrittsstellen |L| = 1 als Liste von Wörterbüchern {w, pm_deg, delay}: Phasenreserve in Grad (|180° + arg L|) und Verzugsreserve
    pm/ω in Perioden (stetig: in Zeiteinheiten). Gitter für die Vorzeichenwechsel von |L| − 1, dann Brent-Verfeinerung. Leere Liste, wenn |L| nie 1 kreuzt."""
    if stetig:
        w = np.logspace(-3, 3, n)
        point = lambda x: 1j * x
    else:
        w = np.linspace(1e-6, math.pi, n)
        point = lambda x: np.exp(1j * x)
    mag = np.abs(L_of(point(w), a, b, k))
    idx = np.nonzero(np.sign(mag[:-1] - 1.0) != np.sign(mag[1:] - 1.0))[0]
    out = []
    for i in idx:
        wc = brentq(lambda x: abs(L_of(point(x), a, b, k)[0]) - 1.0, float(w[i]), float(w[i + 1]), xtol=1e-14)
        lc = L_of(point(wc), a, b, k)[0]
        pm = abs(float(np.angle(-lc)))
        out.append({"w": float(wc), "pm_deg": math.degrees(pm), "delay": pm / wc})
    return out


def worst_phase_margin(margins):
    """Kleinste Phasenreserve (Grad) und kleinste Verzugsreserve der Liste; (None, None), wenn es keine Durchtrittsstelle gibt."""
    if not margins:
        return None, None
    return min(m["pm_deg"] for m in margins), min(m["delay"] for m in margins)


def cont_min_return(ac, bc, kc, n: int = 10001):
    """Kleinster Abstand |1 + L(jω)| des stetigen Entwurfs auf dem Gitter ω = 10⁻³ … 10³ (logarithmisch); gibt (Wert, Ort)."""
    w = np.logspace(-3, 3, n)
    v = np.abs(1.0 + L_of(1j * w, ac, bc, kc))
    i = int(np.argmin(v))
    return float(v[i]), float(w[i])


# --- Verstärkungsfenster ---------------------------------------------------------------------------------------------------------------

def jury_window(a, b, k):
    """Verstärkungsfenster (g_unten, g_oben) für Systeme zweiter Ordnung nach Jury, exakt in Formeln: A − g·B·K hat das Polynom z² − tr(g)·z + det(g) mit
    tr(g) = tr A − g·K·B und det(g) = det A − g·K·adj(A)·B; stabil genau für |det| < 1, 1 + tr + det > 0 und 1 − tr + det > 0 (vier Bedingungen c₀ + c₁·g > 0).
    Gibt (lo, hi) mit lo ≥ 0 und hi = ∞ möglich, oder None, wenn kein g > 0 stabil ist."""
    if a.shape != (2, 2):
        raise ValueError("Jury-Fenster nur für Systeme zweiter Ordnung")
    det_a, tr_a = float(np.linalg.det(a)), float(np.trace(a))
    adj = np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]])
    kb, kab = float((k @ b)[0, 0]), float((k @ adj @ b)[0, 0])
    conds = [(1.0 - det_a, kab), (1.0 + det_a, -kab), (1.0 + det_a + tr_a, -(kab + kb)), (1.0 + det_a - tr_a, kb - kab)]
    lo, hi = 0.0, INF
    for c0, c1 in conds:
        if c1 > 0:
            lo = max(lo, -c0 / c1)
        elif c1 < 0:
            hi = min(hi, -c0 / c1)
        elif c0 <= 0:
            return None
    return (lo, hi) if lo < hi else None


def scalar_window(a: float, k: float):
    """Fenster für den Skalar: |a − g·K| < 1, g > 0; (lo, hi) oder None."""
    if k == 0.0:
        return (0.0, INF) if abs(a) < 1.0 else None
    ends = sorted(((a - 1.0) / k, (a + 1.0) / k))
    lo, hi = max(0.0, ends[0]), ends[1]
    return (lo, hi) if lo < hi else None


def window(a, b, k):
    """Fenster je nach Ordnung (Skalar oder zweiter Ordnung in Formeln)."""
    if a.shape == (1, 1):
        return scalar_window(float(a[0, 0]), float(k[0, 0] * b[0, 0]))
    return jury_window(a, b, k)


def g_upper_bisect(a, b, k, hi: float = 1e6, iters: int = 200) -> float:
    """Zweiter Weg zum oberen Rand: Bisektion auf dem Spektralradius (g = 1 ist stabil, hi muss instabil sein, sonst ∞)."""
    lo = 1.0
    if not D.is_stable(D.closed_loop(a, b, k, lo)):
        raise ValueError("der Entwurf ist bei g = 1 nicht stabil")
    if D.is_stable(D.closed_loop(a, b, k, hi)):
        return INF
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if D.is_stable(D.closed_loop(a, b, k, mid)):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def g_lower_bisect(a, b, k, iters: int = 200, floor: float = 1e-12) -> float:
    """Unterer Rand durch Bisektion; 0, wenn schon g = `floor` stabil ist. Am Wagen ist jedes g > 0 stabil (Jury: P(1) = g·k₁ > 0); die Bisektion endet dort bei etwa
    1e-11, an der Genauigkeit des Stabilitätstests (ρ < 1 − 1e-12), nicht an einer echten Kante."""
    lo, hi = floor, 1.0
    if D.is_stable(D.closed_loop(a, b, k, lo)):
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if D.is_stable(D.closed_loop(a, b, k, mid)):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def g_nyquist_pi(a, b, k) -> float:
    """Dritter Weg: Schnitt der Ortskurve mit der negativen reellen Achse bei ω = π (z = −1): g = −1/L(−1), gilt, wenn L(−1) < 0 und dies die kleinste Kreuzung ist (am Wagen der Fall)."""
    lpi = L_of(np.array([-1.0 + 0j]), a, b, k)[0]
    return INF if lpi.real >= 0 else float(-1.0 / lpi.real)


# --- Schranke --------------------------------------------------------------------------------------------------------------------------

def bound_r0(b, p, r: float) -> float:
    """Untere Schranke min |1 + L| ≥ √(R/(R + BᵀPB)) für einen Eingang (aus der Rückführdifferenz-Gleichung, eigene Herleitung)."""
    return math.sqrt(r / (r + float((b.T @ p @ b)[0, 0])))


def gain_interval_from_bound(r0: float):
    """Aus der Schranke garantiertes Verstärkungsintervall (1/(1 + r₀), 1/(1 − r₀)) (obere Kante ∞ bei r₀ ≥ 1)."""
    return 1.0 / (1.0 + r0), (INF if r0 >= 1.0 else 1.0 / (1.0 - r0))


def return_difference_residual(a, b, q, p, k, r: float, z) -> float:
    """Größte relative Abweichung der Rückführdifferenz-Gleichung (R + BᵀPB)·|1 + L(z)|² = R + (zI − A)⁻¹B)ᴴ Q ((zI − A)⁻¹B) über die Stellen z."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    n = a.shape[0]
    rb = r + float((b.T @ p @ b)[0, 0])
    lw = L_of(z, a, b, k)
    lhs = rb * np.abs(1.0 + lw) ** 2
    xs = np.linalg.solve(z[:, None, None] * np.eye(n)[None] - a[None], np.broadcast_to(b, (len(z), n, 1)))[:, :, 0]
    rhs = r + np.real(np.einsum("ni,ij,nj->n", xs.conj(), q, xs))
    return float(np.max(np.abs(lhs - rhs) / np.maximum(1.0, np.abs(rhs))))


def det_relation_residual(a, b, p, k, r: float) -> float:
    """Relative Abweichung von det(A − BK) = det(A)·R/(R + BᵀPB)."""
    rb = r + float((b.T @ p @ b)[0, 0])
    d1 = float(np.linalg.det(a - b @ k))
    d2 = float(np.linalg.det(a)) * r / rb
    return abs(d1 - d2) / max(1.0, abs(d2))


def dare_residual(a, b, q, p, k) -> float:
    """Relative Residuum der DARE: ‖P − (Q + AᵀPA − AᵀPB·K)‖ / ‖P‖."""
    return float(np.linalg.norm(p - (q + a.T @ p @ a - a.T @ p @ b @ k)) / np.linalg.norm(p))


def _normal_matrix(rng: SplitMix64, rows: int, cols: int) -> np.ndarray:
    return np.array([[rng.normal() for _ in range(cols)] for _ in range(rows)])


_R_DECADES = (0.01, 0.1, 1.0, 10.0)   # Zehnerpotenzen als Literale: kein pow aus der C-Bibliothek


def random_system(rng: SplitMix64):
    """Ein Zufallssystem (n = 2 bis 4, ein Eingang): A, B, Q = MMᵀ + 0,05·I, R in [0,01; 100). Alles aus dem ganzzahligen SplitMix64 (Zufalls-Seed der Demo)."""
    n = 2 + rng.below(3)
    scale = 0.3 + 0.9 * rng.random()
    a = _normal_matrix(rng, n, n) * scale
    b = _normal_matrix(rng, n, 1)
    m = _normal_matrix(rng, n, n)
    q = m @ m.T + 0.05 * np.eye(n)
    r = (1.0 + 9.0 * rng.random()) * _R_DECADES[rng.below(4)]
    return a, b, q, r


def _controllable(a, b) -> bool:
    n = a.shape[0]
    ctrb = np.hstack([np.linalg.matrix_power(a, i) @ b for i in range(n)])
    return bool(np.linalg.matrix_rank(ctrb, tol=1e-6) == n and np.linalg.cond(ctrb) <= 1e7)


def bound_check(seed: int, count: int, n_freq: int = 40, n_grid: int = 4001, dare_tol: float = 1e-10) -> dict:
    """Prüft die Schranke an `count` Zufallssystemen (steuerbar und mit lösbarer DARE; die anderen werden übersprungen).

    Zählt: zulässige Systeme, solche mit DARE-Residuum < dare_tol (nur dort die Gleichung und die Determinantenbeziehung streng), Verletzungen der Schranke
    min|1 + L| ≥ r₀ (Toleranz 1e-9) und des Verstärkungsintervalls (drei Stichproben je System: untere und obere Kante, jeweils 0,1 % innen, und g = 1)."""
    rng = SplitMix64(seed)
    out = {"seed": seed, "requested": count, "ok": 0, "good": 0, "max_equation_good": 0.0, "max_det_good": 0.0, "max_dare_residual": 0.0,
           "violations_bound": 0, "violations_gain": 0, "tightest_ratio": INF}
    for _ in range(count):
        a, b, q, r = random_system(rng)
        w = np.array([math.pi * rng.random() for _ in range(n_freq)])   # immer ziehen, damit der Zufallsstrom nicht vom Ausgang abhängt
        if not _controllable(a, b):
            continue
        try:
            p, k = D.dlqr(a, b, q, r)
        except (np.linalg.LinAlgError, ValueError):
            continue
        out["ok"] += 1
        res = dare_residual(a, b, q, p, k)
        out["max_dare_residual"] = max(out["max_dare_residual"], res)
        if res < dare_tol:
            out["good"] += 1
            out["max_equation_good"] = max(out["max_equation_good"], return_difference_residual(a, b, q, p, k, r, np.exp(1j * w)))
            out["max_det_good"] = max(out["max_det_good"], det_relation_residual(a, b, p, k, r))
        r0 = bound_r0(b, p, r)
        mn, _ = min_return(a, b, k, n=n_grid)
        if mn < r0 - 1e-9:
            out["violations_bound"] += 1
        out["tightest_ratio"] = min(out["tightest_ratio"], mn / r0)
        lo, hi = gain_interval_from_bound(r0)
        samples = [lo * 1.001, 1.0] + ([hi * 0.999] if hi < INF else [5.0])
        for g in samples:
            if not D.is_stable(D.closed_loop(a, b, k, g)):
                out["violations_gain"] += 1
    return out


def kalman_check(seed: int, count: int, n_grid: int = 4001) -> dict:
    """Stetige Kalman-Ungleichung an Zufallssystemen: min |1 + L(jω)| ≥ 1 (Toleranz 1e-9) auf dem Gitter ω = 10⁻³ … 10³ für den stetigen LQR (CARE), ein Eingang."""
    rng = SplitMix64(seed)
    out = {"seed": seed, "requested": count, "ok": 0, "violations": 0, "smallest": INF}
    w = np.logspace(-3, 3, n_grid)
    for _ in range(count):
        a, b, q, r = random_system(rng)
        if not _controllable(a, b):
            continue
        try:
            _, kc = D.clqr(a, b, q, r)
        except (np.linalg.LinAlgError, ValueError):
            continue
        out["ok"] += 1
        mn = float(np.min(np.abs(1.0 + L_of(1j * w, a, b, kc))))
        out["smallest"] = min(out["smallest"], mn)
        if mn < 1.0 - 1e-9:
            out["violations"] += 1
    return out


# --- Doyles Gegenbeispiel (stetig) -----------------------------------------------------------------------------------------------------

A_DOYLE = np.array([[1.0, 1.0], [0.0, 1.0]])
B_DOYLE = np.array([[0.0], [1.0]])
G_DOYLE = np.array([[1.0], [1.0]])       # Rauscheingang
C_DOYLE = np.array([[1.0, 0.0]])


def doyle_gains(q: float, sigma: float):
    """Reglergewinn g (1×2, u = −g·x̂) aus der CARE mit Q = q·(1,1)ᵀ(1,1), R = 1 und Filtergewinn k (2×1) aus der CARE mit Intensität σ·(1,1)ᵀ(1,1), Messrauschen 1."""
    pc = la.solve_continuous_are(A_DOYLE, B_DOYLE, q * (G_DOYLE @ G_DOYLE.T), np.array([[1.0]]))
    pf = la.solve_continuous_are(A_DOYLE.T, C_DOYLE.T, sigma * (G_DOYLE @ G_DOYLE.T), np.array([[1.0]]))
    return B_DOYLE.T @ pc, pf @ C_DOYLE.T


def doyle_f(q: float) -> float:
    """Gewinnbetrag f = 2 + √(4 + q): g = f·(1, 1) (Formel des Quellenberichts, hier nachgerechnet); für das Filter d = doyle_f(σ)."""
    return 2.0 + math.sqrt(4.0 + q)


def doyle_closed_loop(m: float, g, k) -> np.ndarray:
    """Zustand (x, x̂): ẋ = A·x − m·B·g·x̂, x̂˙ = k·C·x + (A − B·g − k·C)·x̂. m = Streckenverstärkung, dem Filter unbekannt."""
    top = np.hstack([A_DOYLE, -m * B_DOYLE @ g])
    bot = np.hstack([k @ C_DOYLE, A_DOYLE - B_DOYLE @ g - k @ C_DOYLE])
    return np.vstack([top, bot])


def doyle_stable(m: float, g, k) -> bool:
    return bool(np.max(np.linalg.eigvals(doyle_closed_loop(m, g, k)).real) < -1e-12)


def doyle_window_hurwitz(q: float):
    """Fenster der Streckenverstärkung (m_unten, m_oben) für q = σ in Formeln: c₁(m) = d + f − 4 + 2(m − 1)·d·f, c₀(m) = 1 + (1 − m)·d·f (Formeln des
    Quellenberichts, hier gegen np.poly bestätigt), c₃ und c₂ hängen nicht von m ab. Hurwitz Grad 4: c₃, c₀ > 0, c₃·c₂ − c₁ > 0, H = c₁(c₃c₂ − c₁) − c₃²c₀ > 0;
    H ist in u = m − 1 quadratisch. Die obere Kante ist exakt 1 + 1/(d·f); die untere Kante der Quelle (nur c₁ > 0) ist nur notwendig."""
    g, k = doyle_gains(q, q)
    f = d = doyle_f(q)
    c = np.poly(doyle_closed_loop(1.0, g, k))
    c3, c2 = float(c[1]), float(c[2])
    c1_0, c0_0 = d + f - 4.0, 1.0
    alpha, beta = 2.0 * d * f, d * f          # c1 = c1_0 + alpha·u, c0 = c0_0 − beta·u
    lows, highs = [], []
    highs.append(c0_0 / beta)                                   # c0 > 0
    lows.append(-c1_0 / alpha)                                  # c1 > 0 (notwendig, Formel der Quelle)
    highs.append((c3 * c2 - c1_0) / alpha)                      # c3·c2 − c1 > 0
    # H(u) = −α²·u² + b·u + c0H > 0 (Nullstellen stabil berechnet)
    qa = -alpha * alpha
    qb = alpha * (c3 * c2 - c1_0) - alpha * c1_0 + c3 * c3 * beta
    qc = c1_0 * (c3 * c2 - c1_0) - c3 * c3 * c0_0
    disc = qb * qb - 4.0 * qa * qc
    if disc <= 0:
        return None
    t = -0.5 * (qb + math.copysign(math.sqrt(disc), qb))
    r1, r2 = sorted((t / qa, qc / t))
    lows.append(r1)
    highs.append(r2)
    lo, hi = max(lows), min(highs)
    return (1.0 + lo, 1.0 + hi) if lo < 0.0 < hi else None


def doyle_window_eigen(q: float, grid: int = 6001):
    """Zweiter Weg: Fenster aus den Eigenwerten (Gitter über m = 10⁻³ … 10³ logarithmisch, dann Bisektion an beiden Rändern; Zusammenhang des Fensters um m = 1 vorausgesetzt)."""
    g, k = doyle_gains(q, q)
    ms = np.logspace(-3, 3, grid)
    st = np.array([doyle_stable(m, g, k) for m in ms])

    def bis(good, bad):
        for _ in range(200):
            mid = 0.5 * (good + bad)
            if doyle_stable(mid, g, k):
                good = mid
            else:
                bad = mid
        return 0.5 * (good + bad)

    above = ms[(ms > 1.0) & ~st]
    below = ms[(ms < 1.0) & ~st]
    hi = bis(1.0, float(above.min())) if len(above) else INF
    lo = bis(1.0, float(below.max())) if len(below) else 0.0
    return lo, hi, int(np.sum(st[1:] != st[:-1]))


def doyle_lq_lower(q: float) -> float:
    """Reiner LQ-Regler (Zustandsrückführung ohne Filter): A − m·B·g ist stabil genau für m > 2/f (tr = 2 − m·f, det = 1); nach oben ohne Grenze."""
    return 2.0 / doyle_f(q)


def doyle_return0(q: float) -> float:
    """|1 + L(0)| des LQG-Kreises (Schleife am Streckeneingang) = 1/(1 + d·f), exakt (hier gegen den Frequenzgang geprüft)."""
    f = doyle_f(q)
    return 1.0 / (1.0 + f * f)


def doyle_L(w, q: float):
    """L(jω) = g·(jωI − A + B·g + k·C)⁻¹·k · C·(jωI − A)⁻¹·B des LQG-Kreises (m = 1)."""
    g, k = doyle_gains(q, q)
    ctl = A_DOYLE - B_DOYLE @ g - k @ C_DOYLE
    gz = doyle_plant(w)
    n = 2
    s = 1j * np.atleast_1d(np.asarray(w, dtype=float))
    m = s[:, None, None] * np.eye(n)[None] - ctl[None]
    x = np.linalg.solve(m, np.broadcast_to(k, (len(s), n, 1)))
    return np.einsum("ij,njk->nik", g, x)[:, 0, 0] * gz


def doyle_plant(w):
    """Strecke C·(jωI − A)⁻¹·B an den Stellen ω (Feld)."""
    s = 1j * np.atleast_1d(np.asarray(w, dtype=float))
    m = s[:, None, None] * np.eye(2)[None] - A_DOYLE[None]
    x = np.linalg.solve(m, np.broadcast_to(B_DOYLE, (len(s), 2, 1)))
    return np.einsum("ij,njk->nik", C_DOYLE, x)[:, 0, 0]


def doyle_return_det_quotient(w, q: float):
    """Zweiter Rechenweg für 1 + L: det(jωI − A_cl) / (det(jωI − A)·det(jωI − A + B·g + k·C)), A_cl bei m = 1."""
    g, k = doyle_gains(q, q)
    acl = doyle_closed_loop(1.0, g, k)
    ctl = A_DOYLE - B_DOYLE @ g - k @ C_DOYLE
    out = []
    for x in np.atleast_1d(w):
        s = 1j * float(x)
        out.append(np.linalg.det(s * np.eye(4) - acl) / (np.linalg.det(s * np.eye(2) - A_DOYLE) * np.linalg.det(s * np.eye(2) - ctl)))
    return np.array(out)


def doyle_min_return(q: float, n: int = 14001):
    """Kleinster Abstand |1 + L(jω)| auf dem Gitter ω = 10⁻³ … 10⁴ (logarithmisch); (Wert, Ort). Der Wert bei ω = 0 steht in `doyle_return0`."""
    w = np.logspace(-3, 4, n)
    v = np.abs(1.0 + doyle_L(w, q))
    i = int(np.argmin(v))
    return float(v[i]), float(w[i])


# --- Abgetasteter Wagen mit Schätzer (LQG) ---------------------------------------------------------------------------------------------

C_POS = np.array([[1.0, 0.0]])           # gemessen wird nur die Position
LQG_FORMS = ("schaetz", "vorhersage")
M_GRID = np.linspace(0.001, 8.0, 8000)    # Gitter der Streckenverstärkung für das Fenster; „0“ als Untergrenze heißt: stabil bis zum Gitterrand 0,001


def lqg_predictor_covariance(a, g, w: float) -> np.ndarray:
    """Vorhersage-Kovarianz Pp (vor der Messung) des Kalman-Filters: DARE mit Prozessrauschen W·G·Gᵀ und Messrauschen 1 (Position)."""
    if not w > 0:
        raise ValueError("Die Rauschstärke W muss positiv sein")
    if not _controllable(a, g):
        raise ValueError("Das Prozessrauschen erreicht nicht alle Zustände ((A, G) nicht steuerbar, z. B. G = (1, 0)ᵀ am Wagen): das Filter hätte keine stabilisierende Lösung")
    pp = la.solve_discrete_are(a.T, C_POS.T, w * (g @ g.T), np.array([[1.0]]))
    return pp


def lqg_controller(form: str, a, b, k, pp):
    """Regler als Zustandsraum (Ac, Bc, Cc, Dc) mit u = −(Dc·y + Cc·ξ), ξ⁺ = Ac·ξ + Bc·y.

    'vorhersage': u_k aus der Schätzung mit Messungen bis k − 1 (Ac = A − BK − Lp·C, Bc = Lp, Cc = K, Dc = 0, Lp = A·Pp·Cᵀ/S).
    'schaetz': u_k aus der Schätzung mit Messung k (erst vorhersagen, dann korrigieren, wie das Filter der Demo kalman-filter-demo): Lc = Pp·Cᵀ/S."""
    s = float((C_POS @ pp @ C_POS.T)[0, 0]) + 1.0
    n = a.shape[0]
    if form == "vorhersage":
        lp = a @ pp @ C_POS.T / s
        return a - b @ k - lp @ C_POS, lp, k.copy(), np.zeros((1, 1))
    if form == "schaetz":
        lc = pp @ C_POS.T / s
        ik = np.eye(n) - lc @ C_POS
        return (a - b @ k) @ ik, (a - b @ k) @ lc, k @ ik, k @ lc
    raise ValueError(f"Form muss eine von {LQG_FORMS} sein")


def lqg_matrices(a, b, ctrl):
    """Regelkreis x⁺ = A·x − m·B·(Dc·C·x + Cc·ξ), ξ⁺ = Bc·C·x + Ac·ξ als M0 + m·M1 (m = Streckenverstärkung, dem Schätzer unbekannt)."""
    ac, bc, cc, dc = ctrl
    n = a.shape[0]
    m0 = np.block([[a, np.zeros((n, n))], [bc @ C_POS, ac]])
    m1 = np.block([[-b @ dc @ C_POS, -b @ cc], [np.zeros((n, n)), np.zeros((n, n))]])
    return m0, m1


def _rho_stack(m0, m1, ms) -> np.ndarray:
    mats = m0[None] + ms[:, None, None] * m1[None]
    return np.max(np.abs(np.linalg.eigvals(mats)), axis=1)


def lqg_window(m0, m1, grid=M_GRID):
    """Fenster der Streckenverstärkung um m = 1: Gitter 0,001 … 8, an beiden Rändern per Bisektion verfeinert. (lo, hi): lo = 0 heißt stabil bis zum Gitterrand 0,001,
    hi = ∞ heißt stabil bis zum Gitterrand 8. None, wenn m = 1 selbst instabil ist."""
    st = _rho_stack(m0, m1, grid) < 1.0 - D.STABLE_EPS
    i1 = int(np.argmin(np.abs(grid - 1.0)))
    if not st[i1]:
        return None
    a = i1
    while a > 0 and st[a - 1]:
        a -= 1
    b = i1
    while b < len(grid) - 1 and st[b + 1]:
        b += 1

    def stable(m):
        return float(_rho_stack(m0, m1, np.array([m]))[0]) < 1.0 - D.STABLE_EPS

    def bis(good, bad):
        for _ in range(60):
            mid = 0.5 * (good + bad)
            if stable(mid):
                good = mid
            else:
                bad = mid
        return 0.5 * (good + bad)

    lo = 0.0 if a == 0 else bis(float(grid[a]), float(grid[a - 1]))
    hi = INF if b == len(grid) - 1 else bis(float(grid[b]), float(grid[b + 1]))
    return lo, hi


def lqg_loop_gain(a, b, ctrl, w):
    """L(e^{jω}) des LQG-Kreises (Schleife am Streckeneingang): Reglerübertragung (Dc + Cc·(zI − Ac)⁻¹·Bc) mal Strecke C·(zI − A)⁻¹·B."""
    ac, bc, cc, dc = ctrl
    z = np.exp(1j * np.atleast_1d(np.asarray(w, dtype=float)))
    n = a.shape[0]
    plant = L_of(z, a, b, C_POS)   # C·(zI − A)⁻¹·B, Zeile C statt K
    mm = z[:, None, None] * np.eye(ac.shape[0])[None] - ac[None]
    x = np.linalg.solve(mm, np.broadcast_to(bc, (len(z), ac.shape[0], 1)))
    cz = dc[0, 0] + np.einsum("ij,njk->nik", cc, x)[:, 0, 0]
    return cz * plant


def lqg_return_det_quotient(a, b, ctrl, w):
    """Zweiter Rechenweg: 1 + L = det(zI − A_cl)/(det(zI − A)·det(zI − Ac)), A_cl bei m = 1."""
    m0, m1 = lqg_matrices(a, b, ctrl)
    acl = m0 + m1
    ac = ctrl[0]
    out = []
    for x in np.atleast_1d(w):
        z = np.exp(1j * float(x))
        out.append(np.linalg.det(z * np.eye(acl.shape[0]) - acl) / (np.linalg.det(z * np.eye(a.shape[0]) - a) * np.linalg.det(z * np.eye(ac.shape[0]) - ac)))
    return np.array(out)


def eig_distance(e1, e2) -> float:
    """Größter Abstand zwischen zwei Eigenwertmengen (zuordnungsfrei: jeder Wert zum nächsten der anderen Menge, in beide Richtungen)."""
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d = np.abs(e1[:, None] - e2[None, :])
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def lqg_analyse(a, b, k, g, w: float, form: str, n_freq: int = 12001) -> dict:
    """Ein LQG-Entwurf am abgetasteten Wagen (Positionsmessung): Fenster der Streckenverstärkung, Nyquist-Abstand (zwei Rechenwege), Phasenreserve, Trennungsprinzip.

    k: Reglergewinn (aus dlqr), g: Rauscheingang (2×1), w: Rauschstärke W (Prozessrauschen W·G·Gᵀ, Messrauschen 1)."""
    pp = lqg_predictor_covariance(a, g, w)
    ctrl = lqg_controller(form, a, b, k, pp)
    m0, m1 = lqg_matrices(a, b, ctrl)
    win = lqg_window(m0, m1)
    om = np.linspace(1e-5, math.pi, n_freq)
    lw = lqg_loop_gain(a, b, ctrl, om)
    v = np.abs(1.0 + lw)
    i = int(np.argmin(v))
    sub = om[:: max(1, n_freq // 200)]
    dev = float(np.max(np.abs((1.0 + lqg_loop_gain(a, b, ctrl, sub)) - lqg_return_det_quotient(a, b, ctrl, sub))))
    mag = np.abs(lw)
    idx = np.nonzero(np.sign(mag[:-1] - 1.0) != np.sign(mag[1:] - 1.0))[0]
    pms = [math.degrees(abs(float(np.angle(-lw[j])))) for j in idx]
    s = float((C_POS @ pp @ C_POS.T)[0, 0]) + 1.0
    lp = a @ pp @ C_POS.T / s
    ev_cl = np.linalg.eigvals(m0 + m1)
    ev_sep = np.concatenate([np.linalg.eigvals(a - b @ k), np.linalg.eigvals(a - lp @ C_POS)])
    return {"window": win, "min_return": float(v[i]), "w_min": float(om[i]), "pm_deg": min(pms) if pms else None, "route2_deviation": dev,
            "separation_deviation": eig_distance(ev_cl, ev_sep), "rho_nominal": float(np.max(np.abs(ev_cl))), "ctrl": ctrl, "m0": m0, "m1": m1}
