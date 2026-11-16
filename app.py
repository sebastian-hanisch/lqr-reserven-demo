"""Reserven: Wie viel Modellfehler hält ein LQR aus? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Linie „Regelung (von LQR bis MPC)“: der LQR am Wagen (Doppelintegrator), gemessen an drei Reserven (Nyquist-Abstand, Verstärkungsfenster,
Phasenreserve): stetig (Garantie), abgetastet (keine) und mit Schätzer (LQG, Doyles Gegenbeispiel). Alles eigene Nachrechnung in diesem Modell.

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import res_constants as C
import res_evaluation as ev
import res_exact as X
from res_evaluation import de, de_g, fmt_int, pow10
from res_layout import dunkles_design, kopf_css
from res_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from res_visualization import (
    build_bound,
    build_doyle,
    build_held,
    build_lqg90,
    build_lqg_traj,
    build_nyquist,
    build_rho,
    build_skalar,
    build_trajectory,
    build_ts,
)

st.set_page_config(page_title="Reserven – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _nyquist(r, ts, bform):
    return ev.nyquist_view(r, ts, bform)


@st.cache_data(show_spinner=False)
def _missed(ts, bform):
    return ev.misses_guarantee(ts, bform)


@st.cache_data(show_spinner=False)
def _skalar(a):
    return ev.skalar_view(a)


@st.cache_data(show_spinner=False)
def _gain(r, ts, bform, m):
    return ev.gain_view(r, ts, bform, m)


@st.cache_data(show_spinner=False)
def _ts(r):
    return ev.ts_rows(r)


@st.cache_data(show_spinner=False)
def _doyle(q, dm):
    return ev.doyle_view(q, dm)


@st.cache_data(show_spinner=False)
def _fan():
    return ev.doyle_fan()


@st.cache_data(show_spinner="Rechne den LQG-Entwurf …")
def _lqg(r, ts, bform, lq, g, w, lform, lm):
    return ev.lqg_view(r, ts, bform, lq, g, w, lform, lm)


@st.cache_data(show_spinner=False)
def _bound(ts, bform):
    return ev.bound_rows(ts, bform)


@st.cache_data(show_spinner="Rechne die Zufallssysteme …")
def _random(seed, count):
    return ev.random_check_live(seed, count)


def _table(header, rows):
    """Markdown-Tabelle (Zahlen sind schon mit Dezimalkomma formatiert; kein senkrechter Strich in den Zellen)."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


FROZEN = ev.load_frozen()
SUMMARY = FROZEN["lqg90_summary"]

# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in res_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎛️ Reserven: Wie viel Modellfehler hält ein LQR aus?")
st.markdown(
    """
Ein **LQR** liefert eine Verstärkung K und ruhige Pole. Die Frage dieser Demo ist, **wie weit der Regelkreis von der Instabilität entfernt ist**, und sie wird auf drei Arten
gemessen, immer mit Namen: als **Nyquist-Abstand** der Ortskurve zum kritischen Punkt −1, als **Verstärkungsfenster** (wie viel stärker oder schwächer die Strecke wirken darf
als modelliert) und als **Phasenreserve**. Verglichen werden der **stetige** Entwurf, derselbe Entwurf **abgetastet** und der Entwurf **mit Schätzer** (LQG, **Doyles Gegenbeispiel**).
Alles gilt in diesem Modell: Wagen (Doppelintegrator), Abtastzeit 1, Euler-Form, Q = I, ein Eingang; die Zahlen sind eigene Nachrechnung, keine Quelle gibt für den abgetasteten Fall eine Garantie an.

Weiter unten: das **Nyquist-Bild**, die **Kernfragen** mit Verdikt, sechs Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Reserve", expanded=True):
    st.markdown(
        """
1. **Konvention:** Strecke x⁺ = A·x + B·u, Regler **u = −K·x**, Kosten Σ xᵀQx + uᵀRu mit Q = I (kein Faktor ½). Der Wagen hat A = [[1, Ts], [0, 1]] und in der Euler-Form (der Form der Linie) B = (0, Ts)ᵀ; die Halte-Form hat B = (Ts²/2, Ts)ᵀ.
2. **Schleife öffnen:** Aufgetrennt am Streckeneingang ist L(z) = K·(zI − A)⁻¹·B. Je näher die Ortskurve L(e^{jω}) an **−1** kommt, desto weniger Modellfehler hält der Kreis aus.
3. **Drei Reserven:** der **Nyquist-Abstand** min |1 + L|, das **Verstärkungsfenster** (Bereich des Faktors g in u = −g·K·x, in dem die Schleife stabil bleibt) und die **Phasenreserve** (180° + arg L bei |L| = 1). Die **Verzugsreserve** ist die Phasenreserve geteilt durch die Durchtrittsfrequenz, in Perioden.
4. **Stetig gegen abgetastet:** Beim stetigen LQR zeigt die Messung in dieser Demo überall Abstand ≥ 1 (nach Doyle, 1978, laut Quellenbericht 60° Phase und nach oben unbegrenzte Verstärkung). Für den abgetasteten Entwurf gibt keine gelesene Quelle eine solche Garantie; hier wird gemessen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Entwurf am Wagen** (gilt für Ergebnis, Kernfragen und die Experimente 2, 3, 5 und 6)")
    r = st.select_slider("Stellgewicht R", options=C.R_OPTIONS, key="r_select", format_func=de_g, help="Kosten Σ xᵀx + R·u²: kleines R heißt starker Eingriff und schnellen Entwurf.")
    ts = st.select_slider("Abtastzeit Ts", options=C.TS_OPTIONS, key="ts_select", format_func=de_g, help="Voreinstellung 1: die Messungen der README gelten dort. Kleine Ts nähern den stetigen Entwurf an.")
    bform = st.radio("Form von B", C.BFORMS, key="bform_radio", format_func=C.BFORM_LABELS.get, help="Euler: B = (0, Ts)ᵀ, die Form der Linie. Halten: exakte Diskretisierung bei konstanter Stellgröße.")

sync_query_params({"r_select": r, "ts_select": ts, "bform_radio": bform})

view = _nyquist(r, ts, bform)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: Das Nyquist-Bild des Wagens")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Nyquist-Abstand min |1 + L|", de(view["min_return"], 3), delta=f"{de(view['min_return'] - view['cont_min'], 3, sign=True)} gegen stetig", delta_color="normal",
          help="Kleinster Abstand der Ortskurve zu −1. Stetig gilt in dieser Messung ≥ 1.")
win = view["window"]
m2.metric("Verstärkungsfenster oben", de(win[1], 3) if win else "–", delta=f"{de(20 * math.log10(win[1]), 2)} dB" if win and win[1] != ev.INF else None, delta_color="off",
          help="Größter Faktor g auf die Streckenverstärkung, bei dem die Schleife noch stabil ist (nach unten: jedes g > 0).")
m3.metric("Phasenreserve", f"{de(view['pm_deg'], 1)}°" if view["pm_deg"] is not None else "–", delta=f"stetig {de(view['cont_pm'], 1)}°" if view["cont_pm"] is not None else None, delta_color="off")
m4.metric("Verzugsreserve", f"{de(view['delay'], 3)} Perioden" if view["delay"] is not None else "–", delta_color="off",
          help="Phasenreserve geteilt durch die Durchtrittsfrequenz: so viel zusätzlicher Verzug (in Perioden) wäre an der Grenze. Der Verzug selbst gehört dem Stück „Verzug“ der Linie.")
st.plotly_chart(build_nyquist(view), width="stretch", key="nyquist_chart")
st.caption("Grün: Ortskurve des abgetasteten Entwurfs (gepunktet: negative Frequenzen), rot: Kreis um −1 mit dem kleinsten Abstand, grau gepunktet: Kreis der Schranke r₀ (Experiment 6), "
           "orange: der stetige Entwurf mit demselben R und der Kreis vom Radius 1. Kurvenstücke jenseits von Betrag 2,5 sind abgeschnitten (für ω → 0 läuft die Kurve ins Unendliche).")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Wie weit liegt die Schleife vom kritischen Punkt −1?**")
kind, text = ev.verdict_nyquist(r, ts, view["min_return"], view["w_min"], view["cont_min"], _missed(ts, bform))
getattr(st, kind)(text)
st.markdown("**2. Wie viel Modellfehler hält sie aus?**")
kind, text = ev.verdict_window(r, view["window"], view["pm_deg"], view["delay"])
getattr(st, kind)(text)
st.markdown("**3. Gilt die stetige Garantie noch, wenn abgetastet oder mit Schätzer?**")
kind, text = ev.verdict_garantie(view["min_return"], view["cont_min"], ts, _doyle(100.0, C.DEFAULTS["dm"])["window"], SUMMARY)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1: Skalar von Hand ---------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Von Hand nachgerechnet, der Skalar x⁺ = a·x + u")
st.caption("Q = R = 1, Eingang 1. Für a = 1 und a = 2 stehen die Werte auch exakt in Q(√5) (Brüche); die Zahlen daneben sind Gleitkomma. Der Skalar ist so klein, dass die ganze Rechnung auf eine Seite passt.")
a_val = st.select_slider("Strecke x⁺ = a·x + u, Faktor a", options=C.A_OPTIONS, key="a_select", format_func=lambda x: de(x, 1),
                         help="a = 1: Integrator, a = 2: instabile Strecke, |a| < 1: stabile Strecke. a = 0 fehlt: dort ist K = 0 und es gibt nichts zu messen.")
sv = _skalar(a_val)
c1, c2 = st.columns([1, 1])
with c1:
    st.plotly_chart(build_skalar(sv), width="stretch", key="skalar_chart")
with c2:
    sw = sv["window"]
    ex = sv["exact"]
    rows = [
        ["DARE-Lösung p", de(sv["p"], 4), X.qs_str(ex["p"]) if ex else "–"],
        ["Gewinn K", de(sv["K"], 4), X.qs_str(ex["K"]) if ex else "–"],
        ["Pol a − K", de(sv["pole"], 4), X.qs_str(ex["pole"]) if ex else "–"],
        ["Fenster unten", de(sw[0], 4) if sw else "–", X.qs_str(ex["g_lo"]) if ex else "–"],
        ["Fenster oben", de(sw[1], 4) if sw else "–", X.qs_str(ex["g_hi"]) if ex else "–"],
        ["kleinster Abstand (kleinster Betrag von 1 + L)", de(sv["min_return"], 4), X.qs_str(ex["min_return"]) if ex else "–"],
        ["Schranke r₀ = √(1/(1 + p))", de(sv["r0"], 4), "–"],
        ["Phasenreserve", f"{de(sv['pm_deg'], 2)}°" if sv["pm_deg"] is not None else "–", "–"],
    ]
    st.markdown(_table(["Größe", "Gleitkomma", "exakt"], rows))
if a_val == 1.0:
    st.success("a = 1: Der kleinste Abstand ist (5 − √5)/4 = 0,691 bei ω = π, das Fenster ist 0 < g < 1 + √5 = 3,236 und die Phasenreserve ist genau 72°. Phasenreserve über 60°, trotzdem endet die Verstärkung bei 3,24: im stetigen Fall hätte sie hier keine obere Grenze.")
elif a_val == 2.0:
    st.warning("a = 2: K = (1 + √5)/2 = 1,618, Fenster 0,618 < g < 1,854, kleinster Abstand (5 − √5)/6 = 0,461 und Phasenreserve 29,8°; eine Periode Verzug macht den Kreis instabil (|Pole|² = K > 1). "
               "Dieselbe Methode, nur abgetastet: das Fenster hat eine endliche obere Grenze und die Phasenreserve liegt unter 60°.")
else:
    pm_txt = "keine Durchtrittsstelle (|L| kreuzt 1 nicht)" if sv["pm_deg"] is None else f"Phasenreserve {de(sv['pm_deg'], 1)}°"
    st.info(f"a = {de(a_val, 1)}: K = {de(sv['K'], 4)}, Fenster ({de(sw[0], 3)}; {de(sw[1], 3)}), kleinster Abstand {de(sv['min_return'], 4)}, Schranke {de(sv['r0'], 4)}, {pm_txt}. "
            "Exakte Werte gibt es hier nur für a = 1 und a = 2.")

st.markdown("---")

# --- Experiment 2: Verstärkungsfehler ------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie viel Verstärkungsfehler hält der Wagen aus?")
st.caption("Die Strecke wirkt m-mal so stark wie modelliert (u wird zu m·u), der Regler weiß es nicht. R, Ts und die Form von B stehen in der Seitenleiste; Anfangszustand (1, 0), Zähleinheit: Periode.")
m_val = st.slider("Streckenverstärkung m", C.M_MIN, C.M_MAX, key="m_slider", step=C.M_STEP, format="%.2f",
                  help="m = 1: das Modell stimmt. Das Fenster des Entwurfs steht im Verdikt; nahe der Kante klingt die Bahn kaum ab.")
gv = _gain(r, ts, bform, m_val)
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(build_rho(gv, m_val, r), width="stretch", key="rho_chart")
with g2:
    st.plotly_chart(build_trajectory(gv, m_val), width="stretch", key="traj_chart")
kind, text = ev.verdict_gain(r, m_val, gv["window"], gv["rho"], gv["steps"])
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3: Abtastzeit --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Reserve und Abtastzeit")
st.caption("Kosten pro Periode gleich bleibend (Q = I, R aus der Seitenleiste; Q und R gemeinsam mit Ts zu skalieren gäbe denselben Gewinn). Zwei Formen von B; grau gepunktet die gewählte Abtastzeit.")
tsd = _ts(r)
t1, t2 = st.columns(2)
with t1:
    st.plotly_chart(build_ts(tsd, ts), width="stretch", key="ts_chart")
with t2:
    st.plotly_chart(build_held(tsd), width="stretch", key="held_chart")
st.markdown(_table(["Ts", "Abstand Euler", "Abstand Halten", "g oben Euler", "g oben Halten"],
                   [[de_g(row["ts"]), de(row["min_euler"], 4), de(row["min_halten"], 4), de(row["gup_euler"], 2), de(row["gup_halten"], 2)] for row in tsd["rows"]]))
held_now = next(h["rho"] for h in tsd["held"] if h["ts"] == ts)
kind, text = ev.verdict_ts(tsd["rows"], ts, held_now, tsd["ts_max"], r)
getattr(st, kind)(text)
st.caption("Rechts: der stetige Gewinn K_c (CARE, gleiches R) am Wagen mit exakter Halte-Diskretisierung. Bei R = 0,1 und 0,01 ist er schon bei Ts = 1 instabil.")

st.markdown("---")

# --- Experiment 4: Doyle -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Doyles Gegenbeispiel (stetig): LQ-Regler mit Filter")
st.caption("Strecke ẋ = [[1, 1], [0, 1]]·x + (0, 1)ᵀ·u + (1, 1)ᵀ·w, Messung y = x₁ + v; Güte Q = q·(1, 1)ᵀ(1, 1), R = 1, Rauschintensitäten σ und 1, hier q = σ. "
           "Die Parameter stammen aus einem Quellenbericht (Doyle 1978, Scan als Bild gelesen) und werden hier nur nachgerechnet. m ist die Streckenverstärkung, die das Filter nicht kennt.")
d1, d2 = st.columns(2)
with d1:
    q_val = st.select_slider("Gewicht q = Rauschintensität σ", options=C.Q_OPTIONS, key="q_select", format_func=pow10,
                             help="Je größer q = σ, desto kleiner das Fenster der Streckenverstärkung des Kreises mit Filter.")
with d2:
    dm_val = st.slider("Streckenverstärkung m", C.DM_MIN, C.DM_MAX, key="dm_slider", step=C.DM_STEP, format="%.3f",
                       help="Der Kreis mit Filter ist nur in einem schmalen Fenster um m = 1 stabil, der LQ-Regler allein für jedes m > 2/f.")
dv = _doyle(q_val, dm_val)
st.plotly_chart(build_doyle(_fan(), q_val, dm_val), width="stretch", key="doyle_chart")
kind, text = ev.verdict_doyle(q_val, dm_val, dv["window"], dv["lq_lower"], dv["return0"], dv["stable_lqg"], dv["stable_lq"])
getattr(st, kind)(text)
st.caption(f"Empfindlichkeitsspitze 1/|1 + L(0)| = {de_g(dv['ms'], 4)} gegen 1 beim LQ-Regler allein. Die untere Kante kommt aus der Hurwitz-Bedingung vierten Grades; "
           "die Formel des Quellenberichts für die untere Kante ist nur die notwendige Bedingung c₁ > 0 und liegt darunter (Befunde und Korrekturen in der README).")

st.markdown("---")

# --- Experiment 5: LQG am Wagen ------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: LQG am abgetasteten Wagen")
st.caption("Gemessen wird die Position, Messrauschen 1, Prozessrauschen W·G·Gᵀ. Zwei Formen: **Schätzform** (u_k mit der Messung k, wie das Filter der Demo „Kalman-Filter“: erst vorhersagen, dann korrigieren) "
           "und **Vorhersage** (u_k mit Messungen bis k − 1). Der Gewinn K ist der des Zustands-LQR; R, Ts und die Form von B stehen in der Seitenleiste (die Messung der README: Ts = 1, Euler-Form).")
l1, l2, l3 = st.columns(3)
with l1:
    w_val = st.select_slider("Rauschstärke W", options=C.W_OPTIONS, key="w_select", format_func=pow10, help="Je größer W, desto mehr misstraut das Filter dem Modell und desto größer ist sein Gewinn.")
    lq_val = st.selectbox("Gewichte des Reglers", C.LQ_KEYS, key="lqgq_select", format_func=C.LQ_LABELS.get)
with l2:
    lform_val = st.radio("Form des Reglers", C.LQG_FORMS, key="lqgform_radio", format_func=C.LQG_FORM_LABELS.get)
    g_val = st.selectbox("Rauschrichtung G", C.G_KEYS, key="g_select", format_func=C.G_LABELS.get, help="Die Richtung (1, 0)ᵀ fehlt: dort ist (A, G) nicht steuerbar.")
with l3:
    lm_val = st.slider("Streckenverstärkung m (Zeitbereich)", C.LQM_MIN, C.LQM_MAX, key="lqgm_slider", step=C.LQM_STEP, format="%.2f", help="Nur für die Bahn links; das Fenster steht im Verdikt.")
lv = _lqg(r, ts, bform, lq_val, g_val, w_val, lform_val, lm_val)
q1, q2 = st.columns(2)
with q1:
    st.plotly_chart(build_lqg_traj(lv), width="stretch", key="lqg_traj_chart")
with q2:
    st.plotly_chart(build_lqg90(FROZEN["lqg90"]), width="stretch", key="lqg90_chart")
st.caption("Links: Position bei der Streckenverstärkung m für den Zustands-LQR (ohne Schätzer) und den LQG-Kreis, Anfangszustand (1, 0), Schätzer bei 0. Rechts: die 90 vorgerechneten Entwürfe der Messreihe (Ts = 1, Euler-Form); "
           "gestrichelt der Zustands-LQR des jeweiligen Reglers, durchgezogen die Schätzform, gepunktet die Vorhersage.")
kind, text = ev.verdict_lqg(lv, w_val, lform_val, SUMMARY)
getattr(st, kind)(text)
st.caption(f"„0“ als untere Fensterkante heißt: stabil bis zum Gitterrand 0,001. Trennungsprinzip (Pole der Schleife gleich den Polen von A − BK und des Filters): größte Abweichung dieses Entwurfs {ev.sci(lv['separation'])}; "
           f"zweiter Rechenweg für |1 + L|: {ev.sci(lv['route2'])}. Bei m = {de(lm_val)} ist ρ = {de(lv['rho_lqg'], 3)} mit Schätzer und {de(lv['rho_lqr'], 3)} ohne.")
with st.expander("Alle 90 vorgerechneten Entwürfe (Tabelle)"):
    frame = pd.DataFrame(FROZEN["lqg90"])
    frame = frame.assign(Regler=frame["lq"].map(C.LQ_LABELS) + ", R = " + frame["r"].map(de_g), Form=frame["form"].map(C.LQG_FORM_LABELS), Rauschrichtung=frame["g"].map(C.G_LABELS))
    frame = frame.rename(columns={"w": "W", "lo": "m unten", "hi": "m oben", "min_return": "min |1+L|", "pm_deg": "Phasenreserve (°)", "lqr_min": "Zustands-LQR"})
    st.dataframe(frame[["Regler", "Form", "Rauschrichtung", "W", "m unten", "m oben", "min |1+L|", "Phasenreserve (°)", "Zustands-LQR"]], hide_index=True, width="stretch")
    st.caption(f"Vorgerechnet von tools/make_frozen.py. „m unten = 0“ heißt: stabil bis zum Gitterrand 0,001; leeres „m oben“: stabil bis zum Gitterrand 8. "
               f"Abstand unter dem des Zustands-LQR in {SUMMARY['below_lqr']} von {SUMMARY['count']} Entwürfen.")

st.markdown("---")

# --- Experiment 6: Schranke ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Die Schranke")
st.caption("Aus der DARE folgt für einen Eingang die Rückführdifferenz-Gleichung (R + BᵀPB)·|1 + L|² = R + ‖Q^{1/2}(zI − A)⁻¹B‖² (eigene Herleitung, in diesem Modell geprüft, keine Neuheitsaussage) und daraus min |1 + L| ≥ r₀ = √(R/(R + BᵀPB)).")
brows = _bound(ts, bform)
b1, b2 = st.columns([1, 1])
with b1:
    st.plotly_chart(build_bound(brows), width="stretch", key="bound_chart")
with b2:
    st.markdown(_table(["R", "r₀", "gemessen", "garantiert für g", "tatsächlich"],
                       [[de_g(row["r"]), de(row["r0"], 4), de(row["min_return"], 4), f"({de(row['guar_lo'], 3)}; {de(row['guar_hi'], 3)})", f"({de(row['win_lo'], 0)}; {de(row['win_hi'], 3)})"] for row in brows]))
kind, text = ev.verdict_bound_wagen(brows)
getattr(st, kind)(text)
st.markdown("**Zufallssysteme** (n = 2 bis 4 Zustände, ein Eingang, Q positiv definit, R von 0,01 bis 100): die Schranke wird an jedem Zufallssystem nachgeprüft.")
s1, s2, s3 = st.columns([2, 2, 1])
with s1:
    seed_val = st.number_input("Zufalls-Seed der Zufallssysteme", *bounds("seed_input"), key="seed_input", step=1)
with s2:
    n_val = st.slider("Zahl der Zufallssysteme (live)", *bounds("rs_slider"), key="rs_slider", step=C.RS_STEP)
with s3:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)
chk = _random(int(seed_val), int(n_val))
kind, text = ev.verdict_bound_random(chk, FROZEN["bound"])
getattr(st, kind)(text)
st.caption(f"Kleinster Quotient gemessener Abstand / Schranke in diesem Lauf: {de(chk['tightest_ratio'], 4)} (auf dem Gitter, also eine obere Näherung). "
           f"Vorgerechnet über {fmt_int(FROZEN['bound']['requested'])} Systeme: {de(FROZEN['bound']['tightest_ratio'], 5)}; die Schranke ist also bei einzelnen Systemen fast scharf, bei kleinem R am Wagen aber leer.")

sync_query_params({"a_select": a_val, "m_slider": m_val, "q_select": q_val, "dm_slider": dm_val, "w_select": w_val, "lqgform_radio": lform_val, "g_select": g_val,
                   "lqgq_select": lq_val, "lqgm_slider": lm_val, "seed_input": int(seed_val), "rs_slider": int(n_val)})

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Strecke:** Wagen (und ein Skalar), Q = I, ein Eingang | Andere Strecken (mit Nullstelle, Mehrgrößen) sind nicht gemessen; die Zahlen gelten nur für den Wagen mit Ts = 1 in Euler-Form (Zählungen und Tabellen) | – |
| **Lineares Gebiet:** keine Stellgrenzen | Mit aktiver Stellgrenze ist die Schleife nichtlinear; ein Nyquist-Abstand gilt nur im linearen Gebiet | [mpc-stellgrenzen-demo](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo), [windup-pi-demo](https://github.com/sebastian-hanisch/windup-pi-demo) |
| **K aus dem unendlichen Horizont** | Die Reserve eines Gesetzes mit endlichem Horizont N ist nicht gemessen und kann abweichen | [horizont-falle-demo](https://github.com/sebastian-hanisch/horizont-falle-demo) |
| **Verzug nur als Zahl** | Die Verzugsreserve ist Phasenreserve / Durchtrittsfrequenz; wie ein ganzer Verzug den Entwurf kippt, misst eine eigene Demo | [regelung-verzug-demo](https://github.com/sebastian-hanisch/regelung-verzug-demo) |
| **Schätzer nur mit Positionsmessung** | Keine Geschwindigkeitsmessung, keine Polvorgabe; warum die Reserve bei G = B und W → ∞ nicht auf die des Zustands-LQR zurückgeht, ist nicht untersucht | [offsetfrei-demo](https://github.com/sebastian-hanisch/offsetfrei-demo) |
| **Phasenreserve klassisch** | Nur die kleinste Durchtrittsstelle; keine Scheibenreserve, keine strukturierte Singulärwertgrenze; Verstärkungsfehler und Verzug nicht gemeinsam | – |
| **Zeitbereich ohne Rauschen** | Die Bahnen sind frei (Anfangszustand, kein Rauschen); die Abklingzeit nahe der Fensterkante ist lang | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist die Strecke $x^+ = Ax + Bu$ mit einem Eingang, die Gewichte $Q = I$ und $R$ und der LQR-Gewinn $K$ (Regler $u = -Kx$). Gesucht ist, wie weit die Schleife
$L(z) = K(zI - A)^{-1}B$ von $-1$ entfernt ist und wie viel Modellfehler der Kreis aushält. Dies ist eine **Skizze der Kennzahlen**: Die Herleitungen der Gleichungen stehen nicht hier; sie sind in diesem Modell
an 3 000 Zufallssystemen (vorgerechnet) und an den Handbeispielen geprüft.

| Symbol | Bedeutung |
|---|---|
| $A, B$ | Strecke, abgetastet; Wagen: $A = \begin{pmatrix}1 & T_s\\0 & 1\end{pmatrix}$, $B = (0,\,T_s)^\top$ (Euler) bzw. $(T_s^2/2,\,T_s)^\top$ (Halten) |
| $K, P$ | LQR-Gewinn und Lösung der Riccati-Gleichung (DARE) |
| $L(z)$ | Schleifenverstärkung, am Streckeneingang aufgetrennt |
| $g$ | Faktor auf die Streckenverstärkung (Modellfehler, in $u = -gKx$) |
| $r_0$ | Schranke für den Abstand $\min_\omega\lvert 1 + L\rvert$ |
| $q, \sigma$ | Doyles Gegenbeispiel: Reglergewicht und Rauschintensität |
| $W$ | Stärke des Prozessrauschens beim LQG am Wagen |

**Der Entwurf.**

$$K = (R + B^\top P B)^{-1} B^\top P A, \qquad P = Q + A^\top P A - A^\top P B\,K$$

**Die drei Reserven** (Nyquist-Abstand, Verstärkungsfenster, Phasenreserve):

$$\min_{\omega}\,\lvert 1 + L(e^{j\omega})\rvert, \qquad \{g > 0 : A - gBK \text{ ist stabil}\}, \qquad 180^\circ + \arg L(e^{j\omega_c}) \ \text{ mit } \lvert L(e^{j\omega_c})\rvert = 1$$

**Rückführdifferenz-Gleichung und Schranke** (ein Eingang, eigene Herleitung, in diesem Modell geprüft):

$$(R + B^\top P B)\,\lvert 1 + L(e^{j\omega})\rvert^2 = R + \big\lVert Q^{1/2}(e^{j\omega}I - A)^{-1}B\big\rVert^2$$

$$\min_\omega \lvert 1 + L(e^{j\omega})\rvert \ \ge\ r_0 = \sqrt{\frac{R}{R + B^\top P B}}$$

**Wagen, Fenster nach Jury** (Gewinn $K = (k_1, k_2)$, Euler-Form mit $T_s = 1$):

$$g_{\max} = \min\!\Big(\frac{2}{k_2 - k_1},\ \frac{4}{2k_2 - k_1}\Big), \qquad \lvert 1 + L(-1)\rvert = 1 - \frac{2k_2 - k_1}{4}$$

**Doyles Gegenbeispiel** ($q = \sigma$, stetig): Reglergewinn $g = f\,(1,\,1)$ mit $f = 2 + \sqrt{4 + q}$, Filtergewinn $k = d\,(1,\,1)^\top$ mit $d = 2 + \sqrt{4 + \sigma}$. Mit der Streckenverstärkung $m$ ist das
charakteristische Polynom $s^4 + c_3 s^3 + c_2 s^2 + c_1 s + c_0$ mit

$$c_1 = d + f - 4 + 2(m - 1)\,d f, \qquad c_0 = 1 + (1 - m)\,d f$$

und $c_3, c_2$ unabhängig von $m$. Die obere Fensterkante ist $m = 1 + 1/(df)$, die untere folgt aus der Hurwitz-Bedingung $c_1(c_3 c_2 - c_1) - c_3^2 c_0 > 0$. Bei $\omega = 0$ gilt

$$\lvert 1 + L(0)\rvert = \frac{1}{1 + d f}$$

**Literatur.** Doyle, J. C. (1978): Guaranteed margins for LQG regulators. IEEE Transactions on Automatic Control 23(4), 756–757 (Scan als Bild gelesen; die Parameter des Gegenbeispiels sind hier nachgerechnet,
die Aussage „keine garantierten Reserven“ steht dort). Für den abgetasteten LQR ist keine Reservegarantie in einer gelesenen Quelle belegt; die Rückführdifferenz-Gleichung ist hier eigene Herleitung, ob sie in Lehrbüchern steht, ist ungelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
