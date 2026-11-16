"""Auswertungen der Experimente der App und ihre Verdikte (reine Funktionen: (Art, Text)).

Jede Auswertung ruft nur den Kern `res_dynamics` und `res_margins` auf; die App legt sie hinter `st.cache_data`. Zahlen in den Verdikten kommen aus der
Rechnung oder aus `data/res_frozen.json` (vorgerechnet von tools/make_frozen.py; in App und README als „vorgerechnet“ gekennzeichnet).
Modell (überall): Wagen mit Abtastzeit Ts, u = −K·x, Q = I (LQG: auch Q = 100·(1,1)ᵀ(1,1)), ein Eingang, Schleife am Streckeneingang geöffnet."""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import res_constants as C
import res_dynamics as D
import res_exact as X
import res_margins as M

FROZEN_PATH = Path(__file__).resolve().parent / "data" / "res_frozen.json"
I2 = np.eye(2)
INF = float("inf")
D_EPS = D.STABLE_EPS


@dataclass(frozen=True)
class Settings:
    r: float = C.DEFAULTS["r"]
    ts: float = C.DEFAULTS["ts"]
    bform: str = C.DEFAULTS["bform"]
    a: float = C.DEFAULTS["a"]
    m: float = C.DEFAULTS["m"]
    q: float = C.DEFAULTS["q"]
    dm: float = C.DEFAULTS["dm"]
    w: float = C.DEFAULTS["w"]
    lform: str = C.DEFAULTS["lform"]
    g: str = C.DEFAULTS["g"]
    lq: str = C.DEFAULTS["lq"]
    lm: float = C.DEFAULTS["lm"]
    seed: int = C.DEFAULTS["seed"]
    n: int = C.DEFAULTS["n"]


# --- Formatierung ----------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma und echtem Minuszeichen (Anzeige); ∞ als Zeichen."""
    if x == INF:
        return "∞"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def de_g(x: float, sig: int = 4) -> str:
    """Zahl mit `sig` gültigen Stellen und Dezimalkomma (für Zahlen über viele Größenordnungen: R, W, q)."""
    if x == INF:
        return "∞"
    return f"{x:.{sig}g}".replace(".", ",").replace("-", "−")


def fmt_int(n: int) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner (1722 -> 1 722)."""
    return f"{n:,}".replace(",", " ")


def pow10(x: float) -> str:
    """W und q als Zehnerpotenz, wenn es eine ist (10000 -> 10⁴), sonst als Zahl."""
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    if x >= 1000:
        e = round(math.log10(x))
        if abs(10.0 ** e - x) < 1e-9 * x:
            return "10" + str(e).translate(sup)
    return de_g(x)


def omega_text(w: float) -> str:
    return "π" if abs(w - math.pi) < 1e-4 else de(w, 3)


# --- Entwürfe und Auswertungen ----------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def design(r: float, ts: float, bform: str):
    """Wagen-Entwurf: (A, B, P, K) mit Q = I."""
    a, b = D.wagen(ts, bform)
    p, k = D.dlqr(a, b, I2, r)
    return a, b, p, k


def cont_design(r: float):
    ac, bc = D.wagen_stetig()
    pc, kc = D.clqr(ac, bc, I2, r)
    return ac, bc, kc


def nyquist_view(r: float, ts: float, bform: str) -> dict:
    """Experiment 1: Ortskurve, kleinster Abstand zu −1, Fenster, Reserven, Schranke; dazu der stetige Entwurf mit demselben R."""
    a, b, p, k = design(r, ts, bform)
    mn, wm = M.min_return(a, b, k)
    win = M.jury_window(a, b, k)
    pm_deg, delay = M.worst_phase_margin(M.phase_margins(a, b, k))
    w = np.concatenate([np.linspace(0.02, 3.0, 700), np.linspace(3.0, math.pi, 100)[1:]])
    lw = M.L_of(np.exp(1j * w), a, b, k)
    ac, bc, kc = cont_design(r)
    wc = np.logspace(-2, 3, 1500)
    lc = M.L_of(1j * wc, ac, bc, kc)
    cmin, _ = M.cont_min_return(ac, bc, kc)
    cpm, _ = M.worst_phase_margin(M.phase_margins(ac, bc, kc, stetig=True))
    return {"k": k.ravel().tolist(), "min_return": mn, "w_min": wm, "ms": 1.0 / mn, "window": win, "pm_deg": pm_deg, "delay": delay,
            "r0": M.bound_r0(b, p, r), "w": w, "L": lw, "wc": wc, "Lc": lc, "cont_min": cmin, "cont_pm": cpm, "kc": kc.ravel().tolist(),
            "rho_nominal": D.spectral_radius(D.closed_loop(a, b, k))}


def misses_guarantee(ts: float, bform: str, rs=(1.0, 0.1, 0.01)) -> int:
    """Zahl der R aus `rs`, bei denen der Nyquist-Abstand des abgetasteten Entwurfs unter 1 liegt (die stetige Garantie wird verfehlt)."""
    count = 0
    for r in rs:
        a, b, _, k = design(r, ts, bform)
        if M.min_return(a, b, k, n=4001)[0] < 1.0 - 1e-9:
            count += 1
    return count


def skalar_view(a_val: float) -> dict:
    """Handbeispiel Skalar x⁺ = a·x + u, Q = R = 1: K, Pol, Fenster, kleinster Abstand, Schranke, Phasen- und Verzugsreserve; exakte Werte für a = 1 und 2."""
    a, b = D.skalar(a_val)
    p, k = D.dlqr(a, b, np.array([[1.0]]), 1.0)
    mn, wm = M.min_return(a, b, k, n=20001)
    win = M.window(a, b, k)
    pm_deg, delay = M.worst_phase_margin(M.phase_margins(a, b, k))
    w = np.linspace(0.02, math.pi, 600)
    lw = M.L_of(np.exp(1j * w), a, b, k)
    out = {"a": a_val, "p": float(p[0, 0]), "K": float(k[0, 0]), "pole": a_val - float(k[0, 0]), "min_return": mn, "w_min": wm, "window": win,
           "pm_deg": pm_deg, "delay": delay, "r0": M.bound_r0(b, p, 1.0), "w": w, "L": lw, "exact": None}
    if a_val in (1.0, 2.0):
        ex = X.scalar_exact(int(a_val))
        out["exact"] = {"p": ex["p"], "K": ex["K"], "pole": ex["pole"], "g_lo": ex["g_lo"], "g_hi": ex["g_hi"], "min_return": ex["min_return"]}
    return out


M_GRID = np.round(np.arange(C.M_MIN, C.M_MAX + 1e-9, 0.01), 2)


def gain_view(r: float, ts: float, bform: str, m: float) -> dict:
    """Experiment 2: Streckenverstärkung m (u wirkt als m·u): Spektralradius, Abklingzeit, Fenster, Verzugsreserve, Bahn und ρ(m)-Kurve."""
    a, b, p, k = design(r, ts, bform)
    mat = D.closed_loop(a, b, k, m)
    rho = D.spectral_radius(mat)
    steps, peak = D.settle(mat, [1.0, 0.0])
    win = M.jury_window(a, b, k)
    _, delay = M.worst_phase_margin(M.phase_margins(a, b, k))
    mats = a[None] - M_GRID[:, None, None] * (b @ k)[None]
    rho_curve = np.max(np.abs(np.linalg.eigvals(mats)), axis=1)
    traj = D.trajectory(mat, [1.0, 0.0], 60)
    traj = np.where(np.abs(traj) > 1e6, np.sign(traj) * 1e6, traj)
    return {"rho": rho, "steps": steps, "peak": peak, "window": win, "delay": delay, "rho_curve": rho_curve, "traj": traj, "ts": ts,
            "stable": rho < 1.0 - D.STABLE_EPS}


def ts_rows(r: float) -> dict:
    """Experiment 3: Nyquist-Abstand und obere Fensterkante über die Abtastzeit für beide Formen von B; der stetige Gewinn (CARE) abgetastet gehalten."""
    rows = []
    for ts in C.TS_SERIES:
        row = {"ts": ts}
        for form in C.BFORMS:
            a, b, _, k = design(r, ts, form)
            mn, _ = M.min_return(a, b, k, n=20001)
            row[f"min_{form}"] = mn
            row[f"gup_{form}"] = M.jury_window(a, b, k)[1]
        rows.append(row)
    ac, bc, kc = cont_design(r)
    cmin, _ = M.cont_min_return(ac, bc, kc)
    held = [{"ts": ts, "rho": D.held_continuous_gain_radius(kc, ts)} for ts in C.TS_OPTIONS]
    return {"rows": rows, "cont_min": cmin, "kc": kc.ravel().tolist(), "ts_max": D.held_continuous_gain_ts_max(kc), "held": held}


@lru_cache(maxsize=64)
def doyle_view(q: float, dm: float) -> dict:
    """Experiment 4: Doyles Gegenbeispiel (stetig), q = σ: Fenster der Streckenverstärkung (Hurwitz), reiner LQ-Regler, |1 + L(0)|, Stabilität bei m = dm."""
    g, k = M.doyle_gains(q, q)
    lo, hi = M.doyle_window_hurwitz(q)
    r0 = M.doyle_return0(q)
    lq_lo = M.doyle_lq_lower(q)
    return {"q": q, "window": (lo, hi), "lq_lower": lq_lo, "return0": r0, "ms": 1.0 / r0, "stable_lqg": M.doyle_stable(dm, g, k),
            "stable_lq": dm > lq_lo, "min_return": M.doyle_min_return(q)[0]}


@lru_cache(maxsize=4)
def doyle_fan() -> list:
    rows = []
    for q in C.DOYLE_QS:
        lo, hi = M.doyle_window_hurwitz(q)
        rows.append({"q": q, "lo": lo, "hi": hi, "lq_lower": M.doyle_lq_lower(q), "return0": M.doyle_return0(q)})
    return rows


def lqg_inputs(r: float, ts: float, bform: str, lq: str, gkey: str):
    """(A, B, K, G, LQR-Nyquist-Abstand, LQR-Fenster) des LQG-Entwurfs am Wagen: Regler-Gewichte Q = I oder 100·(1,1)ᵀ(1,1), Rauscheingang G."""
    a, b = D.wagen(ts, bform)
    q = I2 if lq == "I" else 100.0 * np.ones((2, 2))
    _, k = D.dlqr(a, b, q, r)
    g = b.copy() if gkey == "b" else np.array(C.G_VECTORS[gkey]).reshape(2, 1)
    mn, _ = M.min_return(a, b, k, n=20001)
    return a, b, k, g, mn, M.jury_window(a, b, k)


def lqg_view(r: float, ts: float, bform: str, lq: str, gkey: str, w: float, lform: str, lm: float) -> dict:
    """Experiment 5: ein LQG-Entwurf am abgetasteten Wagen gegen den Zustands-LQR (gleicher Gewinn K) und die Bahn bei der Streckenverstärkung lm."""
    a, b, k, g, lqr_min, lqr_win = lqg_inputs(r, ts, bform, lq, gkey)
    res = M.lqg_analyse(a, b, k, g, w, lform)
    lqr_pm, _ = M.worst_phase_margin(M.phase_margins(a, b, k))
    mat_lqr = D.closed_loop(a, b, k, lm)
    mat_lqg = res["m0"] + lm * res["m1"]
    x0 = np.array([1.0, 0.0, 0.0, 0.0])
    tr_lqg = D.trajectory(mat_lqg, x0, 80)[:, 0]
    tr_lqr = D.trajectory(mat_lqr, x0[:2], 80)[:, 0]
    clip = lambda x: np.where(np.abs(x) > 1e6, np.sign(x) * 1e6, x)
    return {"window": res["window"], "min_return": res["min_return"], "w_min": res["w_min"], "pm_deg": res["pm_deg"], "separation": res["separation_deviation"],
            "route2": res["route2_deviation"], "rho_nominal": res["rho_nominal"], "lqr_min": lqr_min, "lqr_window": lqr_win, "lqr_pm": lqr_pm,
            "rho_lqg": D.spectral_radius(mat_lqg), "rho_lqr": D.spectral_radius(mat_lqr), "traj_lqg": clip(tr_lqg), "traj_lqr": clip(tr_lqr), "lm": lm}


def bound_rows(ts: float, bform: str) -> list:
    """Experiment 6, Wagen: Schranke r₀ gegen gemessenen Abstand und garantiertes Intervall gegen das Fenster, für R = 0,001 … 100."""
    rows = []
    for r in C.R_SERIES:
        a, b, p, k = design(r, ts, bform)
        mn, _ = M.min_return(a, b, k, n=20001)
        r0 = M.bound_r0(b, p, r)
        lo, hi = M.gain_interval_from_bound(r0)
        win = M.jury_window(a, b, k)
        rows.append({"r": r, "r0": r0, "min_return": mn, "guar_lo": lo, "guar_hi": hi, "win_lo": win[0], "win_hi": win[1],
                     "poles_product": abs(float(np.linalg.det(a - b @ k)))})
    return rows


def random_check_live(seed: int, count: int) -> dict:
    return M.bound_check(seed, count)


# --- Vorgerechnetes (data/res_frozen.json, tools/make_frozen.py) -------------------------------------------------------------------------

LQG_REGLER = (("I", 1.0), ("I", 0.1), ("rang1", 1.0))     # Regler der 90 Entwürfe: (Q-Wahl, R)


def lqg90_rows() -> list:
    """Die 90 LQG-Entwürfe am Wagen (Ts = 1, Euler-Form): 3 Regler × 2 Formen × 3 Rauschrichtungen × 5 Rauschstärken; Positionsmessung, Messrauschen 1."""
    rows = []
    for lq, r in LQG_REGLER:
        for lform in C.LQG_FORMS:
            for gkey in C.G_KEYS:
                a, b, k, g, lqr_min, lqr_win = lqg_inputs(r, 1.0, "euler", lq, gkey)
                for w in C.LQG_W_GRID:
                    res = M.lqg_analyse(a, b, k, g, w, lform)
                    win = res["window"]
                    rows.append({"lq": lq, "r": r, "form": lform, "g": gkey, "w": w, "lo": None if win is None else win[0],
                                 "hi": None if win is None or win[1] == INF else win[1], "min_return": res["min_return"], "w_min": res["w_min"],
                                 "pm_deg": res["pm_deg"], "separation": res["separation_deviation"], "route2": res["route2_deviation"],
                                 "lqr_min": lqr_min, "lqr_hi": lqr_win[1]})
    return rows


def lqg90_summary(rows: list) -> dict:
    below = [r["min_return"] < r["lqr_min"] - 1e-9 for r in rows]
    per = {}
    for lq, r in LQG_REGLER:
        sel = [b for b, row in zip(below, rows) if row["lq"] == lq and row["r"] == r]
        per[f"{lq}_{r}"] = [int(sum(sel)), len(sel)]
    upper_above = sum(1 for r in rows if (r["hi"] is None) or (r["hi"] > r["lqr_hi"] + 1e-9))
    return {"count": len(rows), "below_lqr": int(sum(below)), "below_per_regler": per, "upper_above_lqr": int(upper_above),
            "min_return_min": min(r["min_return"] for r in rows), "min_return_max": max(r["min_return"] for r in rows),
            "pm_min": min(r["pm_deg"] for r in rows if r["pm_deg"] is not None), "separation_max": max(r["separation"] for r in rows),
            "route2_max": max(r["route2"] for r in rows)}


@lru_cache(maxsize=1)
def load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


def lqg90_lookup(lq: str, r: float, form: str, gkey: str, w: float):
    """Zeile der 90 vorgerechneten Entwürfe oder None."""
    for row in load_frozen()["lqg90"]:
        if (row["lq"], row["r"], row["form"], row["g"], row["w"]) == (lq, r, form, gkey, w):
            return row
    return None


# --- Verdikte --------------------------------------------------------------------------------------------------------------------------
# Jedes Verdikt ist eine reine Funktion (Art, Text); tests/test_verdicts.py prüft Schwellen und jede Zahl im Text an handgesetzten Werten.

def verdict_nyquist(r: float, ts: float, min_ret: float, w_min: float, cont_min: float, n_missed: int, n_total: int = 3) -> tuple[str, str]:
    base = (f"Wagen, Ts = {de_g(ts)}, Q = I, R = {de_g(r)}: Der kleinste Abstand der Ortskurve zu −1 ist **{de(min_ret, 3)}** (bei ω = {omega_text(w_min)}, "
            f"Empfindlichkeitsspitze {de(1.0 / min_ret)}); der stetige Entwurf mit demselben R liegt bei {de(cont_min, 3)}. ")
    tail = (f"Die stetige Garantie (Abstand ≥ 1) wird bei diesem Ts in {n_missed} von {n_total} gemessenen R (1 / 0,1 / 0,01) verfehlt.")
    if min_ret >= 1.0 - 1e-6:
        return "success", base + tail
    if min_ret < 0.5:
        return "warning", base + tail
    return "info", base + tail


def verdict_gain(r: float, m: float, window, rho: float, steps, horizon: int = 20000) -> tuple[str, str]:
    hi = window[1] if window else 0.0
    head = f"R = {de_g(r)}: Die Schleife ist stabil für 0 < m < {de(hi, 4)}. Bei m = {de(m)} ist ρ = {de(rho, 3)}"
    if rho >= 1.0 - D_EPS:
        return "error", head + ": die Bahn wächst (instabil)."
    if steps is None:
        return "warning", head + f": stabil, aber die Bahn ist nach {fmt_int(horizon)} Perioden nicht auf 1 % abgeklungen."
    kind = "warning" if rho >= 0.9 else "success"
    return kind, head + f": die Bahn klingt in {fmt_int(steps)} Perioden auf 1 % der Spitze ab."





def verdict_ts(rows: list, ts_cur: float, held_rho: float, ts_max: float, r: float) -> tuple[str, str]:
    by = {row["ts"]: row for row in rows}
    parts = ", ".join(f"Ts = {de_g(t)}: {de(by[t]['min_euler'], 3)}" for t in (1.0, 0.1, 0.01) if t in by)
    state = (f"bei Ts = {de_g(ts_cur)} stabil (ρ = {de(held_rho, 3)})" if held_rho < 1.0 - D_EPS
             else f"bei Ts = {de_g(ts_cur)} instabil (ρ = {de(held_rho, 3)})")
    return "info", (f"Nyquist-Abstand der Euler-Form (Q = I, R = {de_g(r)}) — {parts}; stetig gleich 1. Der stetige Gewinn, mit exakter Halte-Diskretisierung "
                    f"auf den Wagen gegeben, ist {state} und stabil nur für Ts < {de(ts_max, 4)} (= 2/k₂).")


def verdict_doyle(q: float, dm: float, window, lq_lower: float, return0: float, stable_lqg: bool, stable_lq: bool) -> tuple[str, str]:
    lo, hi = window
    base = (f"q = σ = {pow10(q)}: Der Kreis aus LQ-Regler und Filter ist stabil nur für {de(lo, 6)} < m < {de(hi, 6)}; der LQ-Regler allein für jedes m > {de(lq_lower, 4)} "
            f"(nach oben ohne Grenze). Bei m = {de(dm, 3)} ist der Kreis mit Filter {'stabil' if stable_lqg else 'instabil'}, der LQ-Regler allein "
            f"{'stabil' if stable_lq else 'instabil'}. |1 + L(0)| = {de_g(return0, 4)}.")
    return ("success" if stable_lqg else "warning"), base


def verdict_lqg(res: dict, w: float, lform: str, frozen_summary: dict) -> tuple[str, str]:
    win = res["window"]
    if win is None:
        wtxt = "instabil bei m = 1"
    else:
        lo_txt = "0" if win[0] == 0.0 else de(win[0], 3)
        wtxt = f"({lo_txt}; {de(win[1], 3)})"
    lqr_win = res["lqr_window"]
    form_txt = "Schätzform" if lform == "schaetz" else "Vorhersage-Form"
    text = (f"{form_txt}, W = {pow10(w)}: Fenster der Streckenverstärkung {wtxt}, Nyquist-Abstand **{de(res['min_return'], 3)}**; der Zustands-LQR mit demselben K hat "
            f"(0; {de(lqr_win[1], 3)}) und {de(res['lqr_min'], 3)}. Über {frozen_summary['count']} vorgerechnete Entwürfe liegt der Abstand in {frozen_summary['below_lqr']} "
            f"unter dem des Zustands-LQR (vorgerechnet); eine Garantie gibt es nicht.")
    return ("warning" if res["min_return"] < res["lqr_min"] - 1e-9 else "info"), text


def verdict_bound_wagen(rows: list) -> tuple[str, str]:
    worst = min(rows, key=lambda r: r["r0"])
    return "info", (f"Die Schranke r₀ = √(R/(R + BᵀPB)) gilt, sagt aber für kleines R wenig: bei R = {de_g(worst['r'])} nur {de(worst['r0'], 4)} (gemessen {de(worst['min_return'], 4)}), "
                    f"garantiertes Intervall ({de(worst['guar_lo'], 3)}; {de(worst['guar_hi'], 3)}) statt ({de(worst['win_lo'], 0)}; {de(worst['win_hi'], 3)}). "
                    "Eine Reserve nach dem Entwurf prüfen, nicht aus der Schranke ablesen.")


def verdict_bound_random(check: dict, frozen: dict) -> tuple[str, str]:
    live = (f"Live: {check['violations_bound']} Verletzungen der Schranke in {check['ok']} gültigen Zufallssystemen (Zufalls-Seed {check['seed']}, "
            f"{check['good']} mit DARE-Residuum < 10⁻¹⁰).")
    froz = (f" Vorgerechnet: {frozen['violations_bound']} Verletzungen in {fmt_int(frozen['ok'])} von {fmt_int(frozen['requested'])} Zufallssystemen "
            f"(Zufalls-Seed {frozen['seed']}).")
    kind = "success" if check["violations_bound"] == 0 else "error"
    return kind, live + froz


def verdict_window(r: float, window, pm_deg, delay) -> tuple[str, str]:
    """Kernfrage 2: Verstärkungsfenster, Phasen- und Verzugsreserve des abgetasteten Entwurfs."""
    if window is None:
        return "error", f"R = {de_g(r)}: Der Entwurf ist bei m = 1 nicht stabil."
    hi = window[1]
    db = 20.0 * math.log10(hi) if hi < INF else INF
    pm_txt = "keine Durchtrittsstelle (|L| kreuzt 1 nicht)" if pm_deg is None else f"Phasenreserve {de(pm_deg, 1)}°, Verzugsreserve {de(delay, 3)} Perioden"
    return "info", (f"R = {de_g(r)}: stabil für 0 < g < **{de(hi, 4)}** ({de(db, 2)} dB); nach unten für jedes g > 0 (bis zur Rechengenauigkeit). {pm_txt}.")


def verdict_garantie(min_ret: float, cont_min: float, ts: float, doyle_window, summary: dict) -> tuple[str, str]:
    """Kernfrage 3: gilt die stetige Garantie (Abstand ≥ 1) noch, wenn abgetastet oder mit Schätzer?"""
    lo, hi = doyle_window
    return "warning", (f"Stetig: Abstand {de(cont_min, 3)}. Abgetastet mit Ts = {de_g(ts)}: {de(min_ret, 3)}. Mit Schätzer: Doyles Gegenbeispiel (q = σ = 100) ist nur für "
                       f"{de(lo, 3)} < m < {de(hi, 4)} stabil; am abgetasteten Wagen liegt der Abstand in {summary['below_lqr']} von {summary['count']} vorgerechneten "
                       f"LQG-Entwürfen unter dem des Zustands-LQR (kleinster Wert {de(summary['min_return_min'], 3)}). Eine Abstandsgarantie von 1 gibt es nur im stetigen Entwurf; "
                       "abgetastet bleibt die schwache Schranke r₀ (Experiment 6), mit Schätzer keine.")


def sci(x: float, nd: int = 1) -> str:
    """Zahl in wissenschaftlicher Schreibweise mit Dezimalkomma: 4,2·10⁻¹⁵ (Anzeige von Rundungsabweichungen)."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.{nd}e}".split("e")
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{mant.replace('.', ',')}·10{str(int(exp)).translate(sup)}"
