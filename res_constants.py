"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Modell (überall): der Wagen (Doppelintegrator) mit Abtastzeit Ts, u = −K·x, Q = I, ein Eingang; Voreinstellung Ts = 1 und Euler-Form B = (0, Ts)ᵀ, R = 1.
Messwerte der README stehen nicht hier, sondern in den Tests (tests/test_claims.py) und in data/res_frozen.json (vorgerechnet von tools/make_frozen.py)."""

R_OPTIONS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
TS_OPTIONS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 1.2)
BFORMS = ("euler", "halten")
BFORM_LABELS = {"euler": "Euler: B = (0, Ts)ᵀ", "halten": "Halten: B = (Ts²/2, Ts)ᵀ"}
A_OPTIONS = (-2.0, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0)     # Skalar x⁺ = a·x + u (a = 0 gäbe K = 0: nichts zu messen)
M_MIN, M_MAX, M_STEP = 0.05, 2.2, 0.01                  # Streckenverstärkung beim LQR (Modellfehler)
Q_OPTIONS = (1.0, 5.0, 12.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)   # Doyle: q = σ
DM_MIN, DM_MAX, DM_STEP = 0.90, 1.10, 0.001             # Streckenverstärkung beim Doyle-Kreis
W_OPTIONS = (0.01, 1.0, 100.0, 10000.0, 100000000.0)   # Rauschstärke W des Prozessrauschens beim LQG
LQG_FORMS = ("schaetz", "vorhersage")
LQG_FORM_LABELS = {"schaetz": "Schätzform (u_k mit Messung k)", "vorhersage": "Vorhersage (u_k mit Messungen bis k − 1)"}
G_KEYS = ("b", "m1", "m3")                              # Rauschrichtung G des Prozessrauschens
G_LABELS = {"b": "G = B (Rauschen wie die Stellgröße)", "m1": "G = (1, −1)ᵀ", "m3": "G = (1, −3)ᵀ"}
G_VECTORS = {"m1": (1.0, -1.0), "m3": (1.0, -3.0)}      # G = B ist ts-abhängig (B = (0, Ts)ᵀ bzw. (Ts²/2, Ts)ᵀ)
LQ_KEYS = ("I", "rang1")                                # Gewichte des Reglers: Q = I oder Q = 100·(1, 1)ᵀ(1, 1) (Doyle-artig)
LQ_LABELS = {"I": "Q = I", "rang1": "Q = 100·(1, 1)ᵀ(1, 1)"}
LQM_MIN, LQM_MAX, LQM_STEP = 0.2, 2.2, 0.01             # Streckenverstärkung beim LQG (Zeitbereich)
SEED_MAX, DEFAULT_SEED = 999999, 15
RS_MIN, RS_MAX, RS_STEP, DEFAULT_RS = 50, 300, 50, 200  # Zahl der Zufallssysteme im Live-Lauf (der volle Lauf mit 3 000 ist vorgerechnet)

DEFAULTS = {"r": 1.0, "ts": 1.0, "bform": "euler", "a": 1.0, "m": 1.9, "q": 5.0, "dm": 1.02, "w": 1.0, "lform": "schaetz", "g": "b", "lq": "I",
            "lm": 1.3, "seed": DEFAULT_SEED, "n": DEFAULT_RS}

TS_SERIES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)           # Abtastzeit-Reihe (Experiment 3), wie in der Vormessung
R_SERIES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)         # Experiment 6 (Schranke am Wagen)
DOYLE_QS = (1.0, 5.0, 12.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)
LQG_W_GRID = (0.01, 1.0, 100.0, 10000.0, 100000000.0)   # die 90 vorgerechneten Entwürfe: 3 Regler × 2 Formen × 3 Rauschrichtungen × 5 Rauschstärken
FROZEN_SEED_BOUND = 15                                  # Zufalls-Seed der vorgerechneten 3 000 Zufallssysteme
FROZEN_COUNT = 3000

COLOR_MAIN = "#2F6B65"
COLOR_BAD = "#b23a2a"
COLOR_LINE = "#4c78a8"
COLOR_ORANGE = "#c9742a"
COLOR_GREY = "rgba(150,150,150,0.8)"


def _preset(**over):
    p = dict(DEFAULTS)
    p.update(over)
    return p


PRESETS = {
    "Wagen R = 1, Skalar a = 1 (Voreinstellung)": _preset(),
    "Von Hand: Skalar a = 2": _preset(a=2.0),
    "Wagen R = 0,01 (aggressiv)": _preset(r=0.01, m=1.5),
    "Fein abgetastet (Ts = 0,05)": _preset(ts=0.05),
    "Doyle q = σ = 100": _preset(q=100.0),
    "LQG am Wagen (W = 10⁴)": _preset(w=10000.0),
}
PRESET_HELP = {
    "Wagen R = 1, Skalar a = 1 (Voreinstellung)": "Der Wagen mit Abtastzeit 1 und R = 1, dazu das Handbeispiel Skalar x⁺ = x + u. Der stetige Entwurf hätte Abstand 1 zu −1; hier sind es 0,484.",
    "Von Hand: Skalar a = 2": "Das Handbeispiel mit der instabilen Strecke x⁺ = 2x + u: K = 1,618, Fenster 0,618 < g < 1,854, kleinster Abstand (5 − √5)/6 = 0,461, Phasenreserve 29,8°.",
    "Wagen R = 0,01 (aggressiv)": "Starker Eingriff (R = 0,01): kleinster Abstand 0,348, Verstärkungsfenster bis 1,535. Die Streckenverstärkung m = 1,5 liegt knapp darunter.",
    "Fein abgetastet (Ts = 0,05)": "Abtastzeit 0,05: der Abstand ist 0,958 (Euler-Form), also fast der stetige Wert 1; bei Ts = 1 sind es 0,484.",
    "Doyle q = σ = 100": "Doyles Gegenbeispiel mit q = σ = 100: der Kreis aus Regler und Filter verträgt nur 0,938 < m < 1,0067; die Streckenverstärkung m = 1,02 macht ihn instabil.",
    "LQG am Wagen (W = 10⁴)": "Abgetasteter Wagen mit Kalman-Filter, Rauschstärke W = 10⁴: das Verstärkungsfenster schrumpft gegenüber dem Zustands-LQR (0 bis 1,936).",
}
